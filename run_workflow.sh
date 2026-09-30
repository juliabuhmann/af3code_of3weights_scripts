#!/usr/bin/env bash
# Orchestrator for the AF3-code / OpenFold3-openbind-weights comparison.
#
# One stage per invocation, in this order:
#
#   fetch-benchmark      Download Zenodo 15502855 multimer.tar.gz and unpack it.
#   build-dataset        multimer/ -> entries.tsv, sequences.tsv, chains.tsv.
#   prepare-msa          Index an existing AF3 MSA cache by SEQUENCE, symlink hits
#                        into work/msa_store/, write monomer inputs for the misses.
#   submit-msa           sbatch the CPU array for those misses (skip if none).
#   build-jsons          Complex AF3 input JSONs with MSAs + templates inline.
#   build-of3-inputs     Convert those JSONs into OpenFold3 query JSONs (MSAs only).
#   submit-predictions   sbatch the GPU array for ONE config (run once per arm).
#                        Dispatches on the config's `engine`: af3 -> predict.sbatch,
#                        of3 -> predict_of3.sbatch, af3torch_kit -> predict_af3torch_kit.sbatch
#                        (Anthropic's af3_torch kit from its image, on the af3 inputs).
#   status               How many predictions each arm has finished.
#   collect-profiling    Inference time -> results/profiling/<arm>.tsv, one per arm.
#   score                DockQ vs cleanNative -> results/dockq/<arm>.tsv (runs HERE).
#   link-paper           Expose Zheng & Peng's own AF3-standalone predictions as a
#                        third scoreable arm (symlinks; a sanity check on the harness).
#   submit-score         The same, as a SLURM job — 2 x <entries> DockQ calls is
#                        too much for a shared login node.
#   plot                 Figures into <repo>/results/benchmark/sweep/<arms joined>/,
#                        named after the arms passed with --arms.
#   publish-results      Copy the result tables into <repo>/results/, minus the
#                        columns that name this cluster (for a public repo).
#
# Stages up to build-of3-inputs plus everything from collect-profiling on run in
# the pixi analysis environment; the two submit- stages need SLURM (login node).
#
# RESULTS ARE STORED PER ARM
# --------------------------
# Both result tables are split one file per arm:
#
#   <base>/results/dockq/<arm>.tsv        from --stage score
#   <base>/results/profiling/<arm>.tsv    from --stage collect-profiling
#
# so learning about one pipeline never costs a rescore of the others. Two knobs make
# that concrete:
#
#   --config <arm>   score/collect only that arm. Adding a fifth arm to four
#                    existing ones costs that arm's entries, not 5 x 206 DockQ calls.
#   (default)        all arms, but within each arm only entries not already recorded
#                    `ok` — so a re-run costs whatever is genuinely outstanding.
#                    Entries whose status was a timeout or a missing prediction are
#                    retried; --rescore forces everything.
#
# No merged table is kept on disk. --stage plot and --stage publish-results take the
# directory and concatenate on the fly, so a stale merge cannot contradict the
# per-arm tables. publish-results is what flattens them into the single sanitised
# results/dockq.tsv and results/profiling.tsv committed to the repository.
#
# GPU PACKING — --chunk-size
# --------------------------
# A GPU allocation is expensive to obtain and cheap to keep. On the completed
# 203-task af3_openbind array, 7.7 GPU-hours of real work spanned 2.6 hours of wall
# clock at --array-limit 20 — about 6.8x inflation, essentially all of it waiting for
# SLURM to hand each two-minute task its own GPU.
#
# --chunk-size N submits ceil(entries/N) array tasks instead of one per entry, and
# each task works through N entries in the allocation it holds. The launchers deal
# the todo list out by STRIDE, not in contiguous blocks: the list is sorted
# largest-complex-first, so contiguous blocks would hand task 1 every big complex.
#
# Startup cost is amortised too, because the compiled-kernel caches are created once
# per task: AF3's --cache_dir, and OF3's TRITON_CACHE_DIR / TORCH_EXTENSIONS_DIR.
# That matters most for OF3, where a single-entry task measured 31.9 s of model time
# inside 117 s of wall clock.
#
# Per-entry outputs do not change — still one <entry>.log and one
# <entry>.meta.json — so stages 04 and 05 are unaffected either way. slurm.time
# applies per TASK, so raise it in the config when raising --chunk-size; this script
# warns when the budget cannot cover the chunk.
#
# ARMS AND THE TEMPLATES SWITCH
# -----------------------------
# An arm is a config in configs/, and it declares three axes — `engine` (af3|of3|af3torch_kit),
# `weights` (af3_native|openbind) and `use_templates` (true|false) — plus a
# `name_tag` that names its output directory. All three are required; nothing is
# guessed. The config is copied to predictions/<name_tag>/config.json at submit
# time, and stages 04/05 label every result row from it, so a table can never be
# read as comparing two arms that in fact differed in their templates.
#
# `use_templates` reaches the two engines differently, because they differ:
#   of3   passed straight through as --use-templates true|false.
#   af3   AlphaFold3 has no inference-time template flag, so slurm/predict.sbatch
#         applies it by writing a `templates: []` copy of the input JSON into the
#         task's node-local scratch. work/af3_input/ always holds the templates-ON
#         superset and is shared by every AF3 arm — which is what guarantees that
#         a templates-on and a templates-off arm ran on identical MSAs.
# Either way the launcher counts what the model actually received and records it in
# predictions/<name_tag>/_profiling/<entry>.meta.json.
#
# Options:
#   --stage STAGE        Required.
#   --base-dir PATH      Where ALL generated data goes. Nothing is written into the
#                        repo. Default: $DEFAULT_BASE_DIR below.
#   --config FILE        configs/*.json — required for submit-predictions.
#   --subset dimers|all  Which entries to work on (default: dimers, n_chains == 2).
#   --msa-cache DIR      Existing AF3 MSA cache to reuse (repeatable).
#                        Default: $DEFAULT_MSA_CACHE below.
#   --db-dir DIR         AF3 search databases for submit-msa.
#   --array-limit N      Max concurrently running array tasks (default 20).
#   --chunk-size N       How many entries ONE array task runs inside the GPU
#                        allocation it already holds (default 1 = one entry per
#                        task, the original behaviour). See GPU PACKING below.
#   --per-entry-reserve S
#                        Seconds a task must have left before it starts another
#                        entry (default 1200). Entries that do not fit are skipped
#                        and named in the task log, never half-written.
#   --arms "A B ..."     Which arms `plot` compares, by the `short` in each arm's
#                        config (af3-native-tmpl, af3-ob-tmpl, af3-ob, of3-ob,
#                        zheng). The first is the baseline. Required.
#   --figures-dir DIR    Root `plot` writes under, as results/benchmark/{sweep,
#                        individual_runs}/. Default: <repo>/results, so the figures
#                        are committable; pass a path under --base-dir to keep them out.
#   --score-cpus N       cpus-per-task for submit-score (default 32); the scoring
#                        script's --jobs is set from it.
#   --select-by X        best (AF3 ranking_score, default) | maxiptm.
#   --rescore            Re-run DockQ for entries already recorded ok. Use when the
#                        scoring changed, not when the predictions did.
#   --array-spec SPEC    Submit only these array indices, e.g. '127,128' or '1-10'.
#                        Indices are line numbers in work/predict_todo.txt (or
#                        msa_todo.txt), which is sorted LARGEST COMPLEX FIRST — so
#                        the last indices are the small, fast ones to smoke-test on.
#                        Default: all tasks, throttled by --array-limit.
#   --dry-run            Print the sbatch command instead of running it.
#   --pixi-run           Prefix python stages with `pixi run` (default: on if a
#                        .pixi env exists and pixi is on PATH).
#
# Layout under --base-dir:
#   benchmark_iv/          the Zenodo archive + extracted multimer/
#   dataset/               entries.tsv, sequences.tsv, chains.tsv
#   work/                  msa_store/, msa_input/, af3_input/, the array lists
#   predictions/<name_tag>/  AF3 outputs + _profiling/
#   logs/<stage>/          SLURM .out files
#   results/               profiling.tsv, dockq.tsv
# and, in the repo itself:
#   <repo>/results/benchmark/sweep/<set>/        the PNGs and tables from --stage plot
#   <repo>/results/benchmark/individual_runs/<arm>/   the same, one arm at a time

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# SITE CONFIGURATION — set these for your cluster, or pass the matching flags.
# They can also be set in the environment, which is the least invasive way to keep
# a local setup out of your diffs:
#
#   export AF3OF3_BASE_DIR=/scratch/me/af3code_of3weights
#   export AF3OF3_DB_DIR=/path/to/alphafold3_databases
#   export AF3OF3_IMAGE=/path/to/alphafold3_openbind_1fff5c4_weights.sif
#
# BASE_DIR is the only one that is always needed: everything this pipeline
# generates goes under it (see the layout at the bottom of this comment block).
DEFAULT_BASE_DIR="${AF3OF3_BASE_DIR:-/path/to/workdir/af3code_of3weights}"

# OPTIONAL. An existing AlphaFold3 MSA/template cache of <id>_data.json.gz to reuse.
# Entries are matched BY SEQUENCE, so whatever ids the producing run used are
# irrelevant. Leave this unset (or point it at an empty directory) to compute every
# MSA from scratch via --stage submit-msa; set it to skip work you have already done.
DEFAULT_MSA_CACHE="${AF3OF3_MSA_CACHE:-}"

# The full AlphaFold3 sequence and template databases (uniref90, mgnify, BFD,
# uniprot, PDB seqres + mmCIFs). Only needed for MSA cache misses. See
# fetch_databases.sh in the AlphaFold 3 repository.
DEFAULT_DB_DIR="${AF3OF3_DB_DIR:-/path/to/alphafold3_databases}"
# AF3's own default. Benchmark-IV entries were all released after 2024-01-01, so
# this cutoff excludes the targets themselves from the template set.
MAX_TEMPLATE_DATE="2021-09-30"

STAGE=""
BASE_DIR="$DEFAULT_BASE_DIR"
CONFIG=""
SUBSET="dimers"
MSA_CACHES=()
DB_DIR="$DEFAULT_DB_DIR"
ARRAY_LIMIT=20
ARRAY_SPEC=""
CHUNK_SIZE=1
PER_ENTRY_RESERVE=1200
RESCORE=""
SCORE_CPUS=32
FIGURES_DIR_OVERRIDE=""
PLOT_ARMS=""
SELECT_BY="best"
DRY_RUN=0
PIXI_RUN=""

_val() { [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 1; }; printf '%s' "$2"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)       STAGE=$(_val "$@"); shift 2 ;;
        --base-dir)    BASE_DIR=$(_val "$@"); shift 2 ;;
        --config)      CONFIG=$(_val "$@"); shift 2 ;;
        --subset)      SUBSET=$(_val "$@"); shift 2 ;;
        --msa-cache)   MSA_CACHES+=("$(_val "$@")"); shift 2 ;;
        --db-dir)      DB_DIR=$(_val "$@"); shift 2 ;;
        --array-limit) ARRAY_LIMIT=$(_val "$@"); shift 2 ;;
        --chunk-size)  CHUNK_SIZE=$(_val "$@"); shift 2 ;;
        --per-entry-reserve) PER_ENTRY_RESERVE=$(_val "$@"); shift 2 ;;
        --array-spec)  ARRAY_SPEC=$(_val "$@"); shift 2 ;;
        --score-cpus)  SCORE_CPUS=$(_val "$@"); shift 2 ;;
        --figures-dir) FIGURES_DIR_OVERRIDE=$(_val "$@"); shift 2 ;;
        --arms)        PLOT_ARMS=$(_val "$@"); shift 2 ;;
        --select-by)   SELECT_BY=$(_val "$@"); shift 2 ;;
        --rescore)     RESCORE=yes; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --pixi-run)    PIXI_RUN="yes"; shift ;;
        --no-pixi-run) PIXI_RUN="no"; shift ;;
        -h|--help)     awk 'NR>1 && /^set -euo/{exit} NR>1' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$STAGE" ]] || { echo "ERROR: --stage is required (see --help)" >&2; exit 1; }
if [[ "$BASE_DIR" == /path/to/* ]]; then
    echo "ERROR: --base-dir is still the placeholder ($BASE_DIR)." >&2
    echo "       Pass --base-dir, or set AF3OF3_BASE_DIR; see SITE CONFIGURATION in this script." >&2
    exit 1
fi
# An empty default means "no cache to reuse" — not a directory called "".
[[ ${#MSA_CACHES[@]} -gt 0 ]] || { [[ -n "$DEFAULT_MSA_CACHE" ]] && MSA_CACHES=("$DEFAULT_MSA_CACHE") || MSA_CACHES=(); }

BENCH_DIR="$BASE_DIR/benchmark_iv"
DATASET_DIR="$BASE_DIR/dataset"
WORK_DIR="$BASE_DIR/work"
PRED_ROOT="$BASE_DIR/predictions"
LOG_ROOT="$BASE_DIR/logs"
RESULTS_DIR="$BASE_DIR/results"
# Figures go INTO THE REPO by default: they are small, they are the thing worth
# sharing, and keeping them next to the code that produced them is the point of
# a self-contained repository. Override with --figures-dir to keep them out.
FIGURES_DIR="${FIGURES_DIR_OVERRIDE:-$SCRIPT_DIR/results}"

# Run python through pixi so the analysis stages use the locked environment.
if [[ -z "$PIXI_RUN" ]]; then
    if command -v pixi >/dev/null 2>&1 && [[ -f "$SCRIPT_DIR/pixi.toml" ]]; then
        PIXI_RUN="yes"
    else
        PIXI_RUN="no"
    fi
fi
if [[ "$PIXI_RUN" == "yes" ]]; then
    PY=(pixi run --manifest-path "$SCRIPT_DIR/pixi.toml" python)
else
    PY=(python3)
fi

# Read a field out of a config JSON without depending on jq.
cfg() {
    python3 -c "import json,sys
d=json.load(open(sys.argv[1]))
keys=sys.argv[2].split('.')
for k in keys:
    d=d.get(k) if isinstance(d,dict) else None
    if d is None: break
print('' if d is None else (str(d).lower() if isinstance(d,bool) else d))" "$CONFIG" "$1"
}

# "4:00:00" / "1-00:30:00" -> seconds. Used to give each task a time budget it can
# check before starting another entry, instead of being killed part-way through one.
hms_to_s() {
    python3 -c "
import sys
value = sys.argv[1]
days, _, clock = value.rpartition('-')
parts = [float(p) for p in clock.split(':')]
while len(parts) < 3: parts.insert(0, 0.0)
print(int((int(days) if days else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]))" "$1"
}

# Build RUN_ARGS (--run LABEL DIR ...) for the analysis stages. With --config it is
# just that ONE arm — which is what makes scoring a single pipeline cheap, since the
# tables are per arm — and without it, every arm on disk. $1, when given, names a
# subdirectory an arm must have to qualify (e.g. _profiling).
arm_run_args() {
    local need="${1:-}"
    RUN_ARGS=()
    if [[ -n "$CONFIG" ]]; then
        require_config
        local tag dir
        tag=$(cfg name_tag)
        dir="$PRED_ROOT/$tag"
        [[ -d "$dir" ]] || { echo "ERROR: no predictions for arm '$tag': $dir" >&2; exit 1; }
        RUN_ARGS=(--run "$tag" "$dir")
    else
        for d in "$PRED_ROOT"/*/; do
            [[ -d "$d" ]] || continue
            [[ -z "$need" || -d "$d/$need" ]] || continue
            RUN_ARGS+=(--run "$(basename "$d")" "${d%/}")
        done
    fi
    [[ ${#RUN_ARGS[@]} -gt 0 ]] || {
        echo "ERROR: no arms under $PRED_ROOT${need:+ with a $need directory}" >&2; exit 1; }
    # RUN_ARGS holds three elements per arm (--run LABEL DIR).
    echo "arms          : $(( ${#RUN_ARGS[@]} / 3 ))"
}

# Turn N entries + --chunk-size into a task count and the stride the launchers use.
# Sets NTASKS and CHUNK_STRIDE. CHUNK_STRIDE=0 tells the launcher "one entry, at line
# $SLURM_ARRAY_TASK_ID" — byte-for-byte the original behaviour.
chunk_plan() {
    local n="$1"
    if [[ "$CHUNK_SIZE" -le 1 ]]; then
        NTASKS="$n"; CHUNK_STRIDE=0
    else
        NTASKS=$(( (n + CHUNK_SIZE - 1) / CHUNK_SIZE ))
        # The launcher walks lines i, i+NTASKS, i+2*NTASKS, ..., so the stride IS the
        # number of tasks; that is what makes every task get a size-balanced mix out
        # of a list sorted largest-first.
        CHUNK_STRIDE="$NTASKS"
        local per_entry=$(( TIME_BUDGET_S / CHUNK_SIZE ))
        if [[ "$per_entry" -lt "$PER_ENTRY_RESERVE" ]]; then
            echo "WARNING: slurm.time is $TIME (${TIME_BUDGET_S}s) for --chunk-size $CHUNK_SIZE," >&2
            echo "         i.e. ${per_entry}s per entry, below --per-entry-reserve ${PER_ENTRY_RESERVE}s." >&2
            echo "         Tasks will skip their last entries (named in the task log)." >&2
            echo "         Raise slurm.time in the config, or lower --chunk-size." >&2
        fi
    fi
    echo "entries       : $n"
    echo "chunking      : --chunk-size $CHUNK_SIZE -> $NTASKS array task(s), stride $CHUNK_STRIDE"
}

# Array spec for sbatch: an explicit --array-spec wins; otherwise all N tasks
# throttled to --array-limit concurrent.
array_spec() {
    if [[ -n "$ARRAY_SPEC" ]]; then printf '%s' "$ARRAY_SPEC"
    else printf '1-%s%%%s' "$1" "$ARRAY_LIMIT"; fi
}

# Run an sbatch command (or print it under --dry-run) and say plainly what happened.
# The earlier version printed the command through `printf '%q'`, which renders every
# comma in --export as `\,`. That is only a display artifact — the array passed the
# value unescaped and submission worked — but it made a successful submit look like a
# quoting bug. Printing one argument per line, and stating the outcome explicitly,
# removes the ambiguity.
submit() {
    echo
    echo "sbatch command:"
    printf '  %s' "$1"; shift
    printf ' \\\n    %s' "$@"; echo
    echo
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "--dry-run: NOT submitted."
        return 0
    fi
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "ERROR: sbatch not found on PATH — run this from a login node." >&2
        return 127
    fi
    local out rc
    out=$(sbatch "$@" 2>&1); rc=$?
    echo "$out"
    if [[ $rc -ne 0 ]]; then
        echo "ERROR: sbatch exited $rc — nothing was submitted." >&2
        return "$rc"
    fi
    echo "Check with:  squeue -u \$USER"
    return 0
}

require_config() {
    [[ -n "$CONFIG" ]] || { echo "ERROR: --config is required for stage '$STAGE'" >&2; exit 1; }
    [[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
}

# Expand a $HOME / ~ prefix so configs are not tied to one user's home.
expand_home() { local v="$1"; v="${v/#\$HOME/$HOME}"; printf '%s' "${v/#\~/$HOME}"; }

# `engine`, defaulting to af3 for configs written before the OF3 arm existed.
cfg_engine() { local e; e=$(cfg engine); printf '%s' "${e:-af3}"; }

# `use_templates` as 1/0. Required and never guessed: an arm whose template state
# is unknown cannot be compared with one whose is, which is the whole point of
# recording it. Emits 1/0 on stdout.
cfg_use_templates() {
    local v; v=$(cfg use_templates)
    case "$v" in
        true)  printf '1' ;;
        false) printf '0' ;;
        *) echo "ERROR: $CONFIG must declare \"use_templates\": true or false" >&2
           echo "       (it is the templates axis of the comparison and is recorded" >&2
           echo "        in every output row; see ARMS AND THE TEMPLATES SWITCH)" >&2
           return 1 ;;
    esac
}

case "$STAGE" in

fetch-benchmark)
    "${PY[@]}" "$SCRIPT_DIR/scripts/00_fetch_benchmark.py" --out-dir "$BENCH_DIR"
    ;;

build-dataset)
    "${PY[@]}" "$SCRIPT_DIR/scripts/01_build_dataset.py" \
        --multimer-dir "$BENCH_DIR/multimer" --out-dir "$DATASET_DIR"
    ;;

prepare-msa)
    CACHE_ARGS=()
    for c in "${MSA_CACHES[@]}"; do CACHE_ARGS+=(--msa-cache "$c"); done
    "${PY[@]}" "$SCRIPT_DIR/scripts/02_prepare_msa.py" \
        --dataset-dir "$DATASET_DIR" --work-dir "$WORK_DIR" \
        --subset "$SUBSET" "${CACHE_ARGS[@]}"
    ;;

submit-msa)
    TODO="$WORK_DIR/msa_todo.txt"
    [[ -f "$TODO" ]] || { echo "ERROR: $TODO not found — run --stage prepare-msa first" >&2; exit 1; }
    N=$(grep -c . "$TODO" || true)
    if [[ "$N" -eq 0 ]]; then
        echo "Nothing to compute: the MSA store is already complete."
        echo "Next: --stage build-jsons"
        exit 0
    fi
    # The data pipeline never touches the weights, so this stage is arm-independent:
    # fall back to af3_native.json's image when no --config is given.
    if [[ "$DB_DIR" == /path/to/* || ! -d "$DB_DIR" ]]; then
        echo "ERROR: --db-dir is not a directory: $DB_DIR" >&2
        echo "       The AlphaFold3 search databases are needed to build MSAs." >&2
        exit 1
    fi
    IMAGE_CONFIG="${CONFIG:-$SCRIPT_DIR/configs/af3_native.json}"
    IMAGE=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['image'])" "$IMAGE_CONFIG")
    LOGS="$LOG_ROOT/msa"; mkdir -p "$LOGS" "$WORK_DIR/msa_store"
    # Resources match batch-infer's proven alphafold3_msas rule: its
    # workflow/rules/alphafold3_msas.smk pins mem_mb=65536 on the first attempt, and
    # its config comment records that threads=4 gives 90% CPU utilisation against
    # 30-50% at 8 — jackhmmer does not scale past 4 here. 12h rather than
    # batch-infer's 4h first attempt, because this array job has no retry ladder.
    CMD=(sbatch
        --job-name=af3msa
        --array="$(array_spec "$N")"
        --cpus-per-task=4
        --mem-per-cpu=16g
        --time=12:00:00
        --output="$LOGS/%x_%A_%a.out"
        --export="ALL,WORK_DIR=$WORK_DIR,IMAGE=$IMAGE,DB_DIR=$DB_DIR,MAX_TEMPLATE_DATE=$MAX_TEMPLATE_DATE"
        "$SCRIPT_DIR/slurm/msa.sbatch")
    echo "Sequences needing an MSA: $N"
    submit "${CMD[@]}"
    ;;

build-jsons)
    require_config
    "${PY[@]}" "$SCRIPT_DIR/scripts/03_build_af3_jsons.py" \
        --dataset-dir "$DATASET_DIR" --work-dir "$WORK_DIR" \
        --config "$CONFIG" --subset "$SUBSET"
    ;;

build-of3-inputs)
    require_config
    [[ "$(cfg_engine)" == "of3" ]] || {
        echo "ERROR: --config must be an of3-engine config for this stage" >&2
        echo "       (got engine='$(cfg_engine)' in $CONFIG)" >&2
        exit 1; }
    # Reads work/af3_input/ — the SAME JSONs the AF3 arms run on — so the two
    # engines see the same alignments without recomputing anything.
    "${PY[@]}" "$SCRIPT_DIR/scripts/09_build_of3_inputs.py" \
        --af3-input-dir "$WORK_DIR/af3_input" \
        --out-dir "$WORK_DIR/of3_input" \
        --todo "$WORK_DIR/of3_todo.txt" \
        --seed "$(cfg model_seed)" --quiet
    ;;

submit-predictions)
    require_config
    ENGINE=$(cfg_engine)
    USE_TEMPLATES=$(cfg_use_templates) || exit 1
    NAME_TAG=$(cfg name_tag)
    SAMPLES=$(cfg num_diffusion_samples)
    GPU=$(cfg slurm.gpu)
    GPUMEM=$(cfg slurm.gpumem)
    CPUS=$(cfg slurm.cpus)
    MEM=$(cfg slurm.mem_per_cpu)
    TIME=$(cfg slurm.time)

    OUT_DIR="$PRED_ROOT/$NAME_TAG"
    LOGS="$LOG_ROOT/$NAME_TAG"
    mkdir -p "$OUT_DIR/_profiling" "$LOGS" "$RESULTS_DIR"
    # Keep the exact config next to the outputs it produced: stages 04 and 05 read
    # engine/weights/use_templates back out of it to label every result row.
    cp "$CONFIG" "$OUT_DIR/config.json"

    TIME_BUDGET_S=$(hms_to_s "$TIME")
    [[ "$CHUNK_SIZE" -ge 1 ]] || { echo "ERROR: --chunk-size must be >= 1" >&2; exit 1; }

    echo "arm           : $NAME_TAG  ($(cfg label))"
    echo "engine        : $ENGINE   weights: $(cfg weights)   use_templates: $(cfg use_templates)"

    case "$ENGINE" in
    af3)
        TODO="$WORK_DIR/predict_todo.txt"
        [[ -f "$TODO" ]] || { echo "ERROR: $TODO not found — run --stage build-jsons first" >&2; exit 1; }
        N=$(grep -c . "$TODO" || true)
        [[ "$N" -gt 0 ]] || { echo "ERROR: $TODO is empty" >&2; exit 1; }
        chunk_plan "$N"

        # AF3OF3_IMAGE overrides the config, so the .sif can move without a config edit.
        IMAGE="${AF3OF3_IMAGE:-$(cfg image)}"
        MODEL_DIR=$(expand_home "$(cfg model_dir)")
        OF3=$(cfg of3_weights); [[ "$OF3" == "true" ]] && OF3=1 || OF3=0
        RECYCLES=$(cfg num_recycles)

        CMD=(sbatch
            --job-name="$NAME_TAG"
            --array="$(array_spec "$NTASKS")"
            --gpus="$GPU"
            --gres="gpumem:$GPUMEM"
            --cpus-per-task="$CPUS"
            --mem-per-cpu="$MEM"
            --time="$TIME"
            --output="$LOGS/%x_%A_%a.out"
            --export="ALL,WORK_DIR=$WORK_DIR,OUT_DIR=$OUT_DIR,REPO_DIR=$SCRIPT_DIR,IMAGE=$IMAGE,MODEL_DIR=$MODEL_DIR,OF3_WEIGHTS=$OF3,USE_TEMPLATES=$USE_TEMPLATES,NUM_RECYCLES=$RECYCLES,NUM_DIFFUSION_SAMPLES=$SAMPLES,CHUNK_STRIDE=$CHUNK_STRIDE,TIME_BUDGET_S=$TIME_BUDGET_S,PER_ENTRY_RESERVE_S=$PER_ENTRY_RESERVE"
            "$SCRIPT_DIR/slurm/predict.sbatch")

        echo "array         : submitting $(array_spec "$NTASKS")"
        echo "image         : $IMAGE"
        echo "model_dir     : $MODEL_DIR   of3_weights=$OF3"
        echo "out_dir       : $OUT_DIR"
        submit "${CMD[@]}"
        ;;
    of3)
        # scripts/09_build_of3_inputs.py converts MSAs only. An of3 arm that declared
        # use_templates=true would therefore run with none and still report success,
        # so refuse rather than produce a result that cannot mean what it claims.
        if [[ "$USE_TEMPLATES" == "1" ]]; then
            echo "ERROR: $CONFIG sets use_templates=true, but scripts/09_build_of3_inputs.py" >&2
            echo "       converts MSAs only — the queries carry no template_cif_paths and" >&2
            echo "       OpenFold3 would silently run without templates. See that script's" >&2
            echo "       docstring for where the template-aware converter lives." >&2
            exit 1
        fi
        TODO="$WORK_DIR/of3_todo.txt"
        [[ -f "$TODO" ]] || { echo "ERROR: $TODO not found — run --stage build-of3-inputs first" >&2; exit 1; }
        N=$(grep -c . "$TODO" || true)
        [[ "$N" -gt 0 ]] || { echo "ERROR: $TODO is empty" >&2; exit 1; }
        chunk_plan "$N"

        # Same override idea as AF3OF3_IMAGE: the OF3 checkout, its pixi env and the
        # checkpoint can move without editing the config.
        OF3_SRC="${AF3OF3_OF3_SRC:-$(cfg of3_src)}"
        OF3_ENV="${AF3OF3_OF3_ENV:-$(cfg of3_env)}"
        CKPT=$(expand_home "${AF3OF3_OF3_CKPT:-$(cfg ckpt)}")
        RUNNER_YAML=$(cfg runner_yaml)
        # A relative runner_yaml is repo-relative, so the config stays portable.
        [[ "$RUNNER_YAML" == /* ]] || RUNNER_YAML="$SCRIPT_DIR/$RUNNER_YAML"
        for v in OF3_SRC OF3_ENV CKPT RUNNER_YAML; do
            [[ -n "${!v}" ]] || { echo "ERROR: $CONFIG is missing $(printf '%s' "$v" | tr 'A-Z' 'a-z')" >&2; exit 1; }
        done

        CMD=(sbatch
            --job-name="$NAME_TAG"
            --array="$(array_spec "$NTASKS")"
            --gpus="$GPU"
            --gres="gpumem:$GPUMEM"
            --cpus-per-task="$CPUS"
            --mem-per-cpu="$MEM"
            --time="$TIME"
            --output="$LOGS/%x_%A_%a.out"
            --export="ALL,WORK_DIR=$WORK_DIR,OUT_DIR=$OUT_DIR,REPO_DIR=$SCRIPT_DIR,OF3_SRC=$OF3_SRC,OF3_ENV=$OF3_ENV,CKPT=$CKPT,RUNNER_YAML=$RUNNER_YAML,USE_TEMPLATES=$USE_TEMPLATES,NUM_DIFFUSION_SAMPLES=$SAMPLES,TODO_FILE=$TODO,CHUNK_STRIDE=$CHUNK_STRIDE,TIME_BUDGET_S=$TIME_BUDGET_S,PER_ENTRY_RESERVE_S=$PER_ENTRY_RESERVE"
            "$SCRIPT_DIR/slurm/predict_of3.sbatch")

        echo "array         : submitting $(array_spec "$NTASKS")"
        echo "of3_src       : $OF3_SRC"
        echo "of3_env       : $OF3_ENV"
        echo "ckpt          : $CKPT"
        echo "runner yaml   : $RUNNER_YAML"
        echo "out_dir       : $OUT_DIR"
        submit "${CMD[@]}"
        ;;
    af3torch_kit)
        # Anthropic's af3_torch optimization kit (xfold) from its container image: the af3 inputs and
        # protocol (predict_todo.txt, templates stripped per USE_TEMPLATES, AlphaFold 3's output layout),
        # the kit's `pred --mode` entry point. See configs/af3torchkit_*.json and slurm/predict_af3torch_kit.sbatch.
        TODO="$WORK_DIR/predict_todo.txt"
        [[ -f "$TODO" ]] || { echo "ERROR: $TODO not found — run --stage build-jsons first" >&2; exit 1; }
        N=$(grep -c . "$TODO" || true)
        [[ "$N" -gt 0 ]] || { echo "ERROR: $TODO is empty" >&2; exit 1; }
        chunk_plan "$N"

        KIT_KIND=$(cfg launcher.kind)
        [[ "$KIT_KIND" == "af3_torch_kit" ]] || { echo "ERROR: engine af3torch_kit needs launcher.kind af3_torch_kit in $CONFIG (got '$KIT_KIND')" >&2; exit 1; }
        # AF3OF3_KIT_IMAGE overrides the config, as AF3OF3_IMAGE does for the af3 arms.
        KIT_IMAGE="${AF3OF3_KIT_IMAGE:-$(expand_home "$(cfg launcher.image)")}"
        KIT_MODE=$(cfg launcher.mode); KIT_CONFIG=$(cfg launcher.kit_config); KIT_LEVERS_OFF=$(cfg launcher.levers_off)
        MODEL_DIR=$(expand_home "$(cfg model_dir)")
        RECYCLES=$(cfg num_recycles)
        [[ -n "$KIT_MODE" ]] || { echo "ERROR: launcher.mode missing in $CONFIG" >&2; exit 1; }
        [[ -f "$KIT_IMAGE" ]] || { echo "ERROR: kit image not found: $KIT_IMAGE (launcher.image / AF3OF3_KIT_IMAGE)" >&2; exit 1; }
        [[ -f "$MODEL_DIR/of3_ported_weights.bin.zst" ]] || { echo "ERROR: no converted weights at $MODEL_DIR (the kit's 'run.sh install --weights DIR --fetch --variant $(cfg launcher.variant)' output)" >&2; exit 1; }

        CMD=(sbatch
            --job-name="$NAME_TAG"
            --array="$(array_spec "$NTASKS")"
            --gpus="$GPU"
            --gres="gpumem:$GPUMEM"
            --cpus-per-task="$CPUS"
            --mem-per-cpu="$MEM"
            --time="$TIME"
            --output="$LOGS/%x_%A_%a.out"
            --export="ALL,WORK_DIR=$WORK_DIR,OUT_DIR=$OUT_DIR,REPO_DIR=$SCRIPT_DIR,IMAGE=$KIT_IMAGE,MODEL_DIR=$MODEL_DIR,KIT_MODE=$KIT_MODE,KIT_CONFIG=${KIT_CONFIG:-h100},KIT_LEVERS_OFF=$KIT_LEVERS_OFF,USE_TEMPLATES=$USE_TEMPLATES,NUM_RECYCLES=$RECYCLES,NUM_DIFFUSION_SAMPLES=$SAMPLES,CHUNK_STRIDE=$CHUNK_STRIDE,TIME_BUDGET_S=$TIME_BUDGET_S,PER_ENTRY_RESERVE_S=$PER_ENTRY_RESERVE"
            "$SCRIPT_DIR/slurm/predict_af3torch_kit.sbatch")

        echo "array         : submitting $(array_spec "$NTASKS")"
        echo "kit image     : $KIT_IMAGE  ($(cfg launcher.image_ref))"
        echo "kit           : mode=$KIT_MODE  config=${KIT_CONFIG:-h100}  variant=$(cfg launcher.variant)  levers_off=${KIT_LEVERS_OFF:-none}"
        echo "model_dir     : $MODEL_DIR"
        echo "out_dir       : $OUT_DIR"
        submit "${CMD[@]}"
        ;;
    *)
        echo "ERROR: unknown engine '$ENGINE' in $CONFIG (expected af3, of3 or af3torch_kit)" >&2
        exit 1
        ;;
    esac
    ;;
status)
    TODO="$WORK_DIR/predict_todo.txt"
    EXPECTED=$( [[ -f "$TODO" ]] && grep -c . "$TODO" || echo 0 )
    echo "expected entries: $EXPECTED"
    for d in "$PRED_ROOT"/*/; do
        [[ -d "$d" ]] || continue
        tag=$(basename "$d")
        metas=$(find "$d/_profiling" -name '*.meta.json' 2>/dev/null | wc -l)
        ok=$(grep -l '"exit_code": 0' "$d"/_profiling/*.meta.json 2>/dev/null | wc -l)
        cifs=$(find "$d" -mindepth 2 -name '*_model.cif' 2>/dev/null | wc -l)
        # The arm's axes, from the config copied in at submit time. An arm with no
        # config.json predates the copy and is reported as unknown rather than guessed.
        axes="engine=? templates=?"
        if [[ -f "$d/config.json" ]]; then
            axes=$(CONFIG="$d/config.json" python3 -c "import json,os
d=json.load(open(os.environ['CONFIG']))
t=d.get('use_templates')
print(f\"engine={d.get('engine','af3')} weights={d.get('weights','?')} \"
      f\"templates={'?' if t is None else str(t).lower()}\")")
        fi
        printf '  %-26s profiled=%-4s exit0=%-4s model_cifs=%-6s %s\n' \
            "$tag" "$metas" "$ok" "$cifs" "$axes"
    done
    ;;

collect-profiling)
    arm_run_args _profiling
    "${PY[@]}" "$SCRIPT_DIR/scripts/04_collect_profiling.py" \
        --dataset-dir "$DATASET_DIR" "${RUN_ARGS[@]}" \
        --out-dir "$RESULTS_DIR/profiling" --sacct
    ;;

score)
    arm_run_args
    "${PY[@]}" "$SCRIPT_DIR/scripts/05_score_dockq.py" \
        --dataset-dir "$DATASET_DIR" "${RUN_ARGS[@]}" \
        --subset "$SUBSET" --select-by "$SELECT_BY" \
        --out-dir "$RESULTS_DIR/dockq" ${RESCORE:+--rescore}
    ;;

link-paper)
    "${PY[@]}" "$SCRIPT_DIR/scripts/07_link_paper_predictions.py" \
        --multimer-dir "$BENCH_DIR/multimer" \
        --out-dir "$PRED_ROOT/zheng_af3_local"
    ;;

submit-score)
    [[ -d "$PRED_ROOT" ]] || { echo "ERROR: no predictions under $PRED_ROOT" >&2; exit 1; }
    LOGS="$LOG_ROOT/score"; mkdir -p "$LOGS" "$RESULTS_DIR/dockq"
    # ARMS is empty for "every arm on disk"; with --config it is that one arm, so the
    # job scores it alone and leaves every other arm's table untouched.
    ARMS=""
    if [[ -n "$CONFIG" ]]; then
        require_config
        ARMS=$(cfg name_tag)
    fi
    CMD=(sbatch
        --job-name=af3dockq
        --cpus-per-task="$SCORE_CPUS"
        --mem-per-cpu=4g
        --time=8:00:00
        --output="$LOGS/%x_%j.out"
        --export="ALL,REPO_DIR=$SCRIPT_DIR,DATASET_DIR=$DATASET_DIR,PRED_ROOT=$PRED_ROOT,OUT_DIR=$RESULTS_DIR/dockq,SUBSET=$SUBSET,SELECT_BY=$SELECT_BY,ARMS=$ARMS,RESCORE=${RESCORE:-}"
        "$SCRIPT_DIR/slurm/score.sbatch")
    echo "subset    : $SUBSET   select-by: $SELECT_BY   cpus: $SCORE_CPUS"
    echo "arms      : ${ARMS:-all arms under $PRED_ROOT}"
    echo "rescore   : ${RESCORE:-no (entries already ok are carried over)}"
    echo "tables    : $RESULTS_DIR/dockq/<arm>.tsv"
    submit "${CMD[@]}"
    ;;

plot)
    ARGS=()
    # Prefer the per-arm directories; fall back to a single combined table so an
    # older results/ layout still plots.
    if [[ -d "$RESULTS_DIR/dockq" ]]; then ARGS+=(--dockq "$RESULTS_DIR/dockq")
    elif [[ -f "$RESULTS_DIR/dockq.tsv" ]]; then ARGS+=(--dockq "$RESULTS_DIR/dockq.tsv"); fi
    if [[ -d "$RESULTS_DIR/profiling" ]]; then ARGS+=(--profile "$RESULTS_DIR/profiling")
    elif [[ -f "$RESULTS_DIR/profiling.tsv" ]]; then ARGS+=(--profile "$RESULTS_DIR/profiling.tsv"); fi
    [[ ${#ARGS[@]} -gt 0 ]] || { echo "ERROR: no dockq or profiling results in $RESULTS_DIR" >&2; exit 1; }
    [[ -n "$PLOT_ARMS" ]] || { echo "ERROR: --arms is required, e.g. --arms 'af3-ob of3-ob'" >&2
                               "${PY[@]}" "$SCRIPT_DIR/scripts/06_plot_comparison.py" --arms
                               exit 1; }
    # shellcheck disable=SC2086 -- a space-separated list of short arm names, on purpose
    "${PY[@]}" "$SCRIPT_DIR/scripts/06_plot_comparison.py" \
        "${ARGS[@]}" --arms $PLOT_ARMS --results-root "$FIGURES_DIR"
    ;;

publish-results)
    "${PY[@]}" "$SCRIPT_DIR/scripts/08_publish_results.py" \
        --results-dir "$RESULTS_DIR" --out-dir "$SCRIPT_DIR/results"
    ;;

*)
    echo "Unknown stage: $STAGE" >&2
    awk '/^#   fetch-benchmark/,/^#   publish-results/' "${BASH_SOURCE[0]}" >&2
    exit 1
    ;;
esac
