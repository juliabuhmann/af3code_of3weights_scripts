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
#   submit-predictions   sbatch the GPU array for ONE config (run once per arm).
#   status               How many predictions each arm has finished.
#   collect-profiling    Inference time + peak GPU memory -> results/profiling.tsv.
#   score                DockQ vs cleanNative -> results/dockq.tsv (runs HERE).
#   link-paper           Expose Zheng & Peng's own AF3-standalone predictions as a
#                        third scoreable arm (symlinks; a sanity check on the harness).
#   submit-score         The same, as a SLURM job — 2 x <entries> DockQ calls is
#                        too much for a shared login node.
#   plot                 Figures into <repo>/results/figures/.
#   publish-results      Copy the result tables into <repo>/results/, minus the
#                        columns that name this cluster (for a public repo).
#
# Stages up to build-jsons plus everything from collect-profiling on run in the
# pixi analysis environment; the two submit- stages need SLURM (login node).
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
#   --arm-a / --arm-b    Which two runs `plot` compares (defaults af3_native vs
#                        af3_openbind). Use --arm-b zheng_af3_local for the
#                        harness sanity check against the paper's own predictions.
#   --figures-dir DIR    Where `plot` writes its PNGs. Default: <repo>/results/figures,
#                        so the figures are committable; pass a path under --base-dir
#                        to keep them out of the repo.
#   --score-cpus N       cpus-per-task for submit-score (default 32); the scoring
#                        script's --jobs is set from it.
#   --select-by X        best (AF3 ranking_score, default) | maxiptm.
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
#   <repo>/results/figures/  the PNGs from --stage plot (see --figures-dir)

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
SCORE_CPUS=32
FIGURES_DIR_OVERRIDE=""
ARM_A="af3_native"
ARM_B="af3_openbind"
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
        --array-spec)  ARRAY_SPEC=$(_val "$@"); shift 2 ;;
        --score-cpus)  SCORE_CPUS=$(_val "$@"); shift 2 ;;
        --figures-dir) FIGURES_DIR_OVERRIDE=$(_val "$@"); shift 2 ;;
        --arm-a)       ARM_A=$(_val "$@"); shift 2 ;;
        --arm-b)       ARM_B=$(_val "$@"); shift 2 ;;
        --select-by)   SELECT_BY=$(_val "$@"); shift 2 ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --pixi-run)    PIXI_RUN="yes"; shift ;;
        --no-pixi-run) PIXI_RUN="no"; shift ;;
        -h|--help)     sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
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
FIGURES_DIR="${FIGURES_DIR_OVERRIDE:-$SCRIPT_DIR/results/figures}"

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

submit-predictions)
    require_config
    TODO="$WORK_DIR/predict_todo.txt"
    [[ -f "$TODO" ]] || { echo "ERROR: $TODO not found — run --stage build-jsons first" >&2; exit 1; }
    N=$(grep -c . "$TODO" || true)
    [[ "$N" -gt 0 ]] || { echo "ERROR: $TODO is empty" >&2; exit 1; }

    NAME_TAG=$(cfg name_tag)
    # AF3OF3_IMAGE overrides the config, so the .sif can move without a config edit.
    IMAGE="${AF3OF3_IMAGE:-$(cfg image)}"
    MODEL_DIR=$(cfg model_dir)
    # Configs may write $HOME or ~ so they are not tied to one user's home.
    MODEL_DIR="${MODEL_DIR/#\$HOME/$HOME}"; MODEL_DIR="${MODEL_DIR/#\~/$HOME}"
    OF3=$(cfg of3_weights); [[ "$OF3" == "true" ]] && OF3=1 || OF3=0
    RECYCLES=$(cfg num_recycles)
    SAMPLES=$(cfg num_diffusion_samples)
    GPU=$(cfg slurm.gpu)
    GPUMEM=$(cfg slurm.gpumem)
    CPUS=$(cfg slurm.cpus)
    MEM=$(cfg slurm.mem_per_cpu)
    TIME=$(cfg slurm.time)

    OUT_DIR="$PRED_ROOT/$NAME_TAG"
    LOGS="$LOG_ROOT/$NAME_TAG"
    mkdir -p "$OUT_DIR/_profiling" "$LOGS" "$RESULTS_DIR"
    # Keep the exact config next to the outputs it produced.
    cp "$CONFIG" "$OUT_DIR/config.json"

    CMD=(sbatch
        --job-name="af3_$NAME_TAG"
        --array="$(array_spec "$N")"
        --gpus="$GPU"
        --gres="gpumem:$GPUMEM"
        --cpus-per-task="$CPUS"
        --mem-per-cpu="$MEM"
        --time="$TIME"
        --output="$LOGS/%x_%A_%a.out"
        --export="ALL,WORK_DIR=$WORK_DIR,OUT_DIR=$OUT_DIR,IMAGE=$IMAGE,MODEL_DIR=$MODEL_DIR,OF3_WEIGHTS=$OF3,NUM_RECYCLES=$RECYCLES,NUM_DIFFUSION_SAMPLES=$SAMPLES"
        "$SCRIPT_DIR/slurm/predict.sbatch")

    echo "arm       : $NAME_TAG  ($(cfg label))"
    echo "tasks     : $N total, submitting array $(array_spec "$N")"
    echo "image     : $IMAGE"
    echo "model_dir : $MODEL_DIR   of3_weights=$OF3"
    echo "out_dir   : $OUT_DIR"
    submit "${CMD[@]}"
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
        printf '  %-16s profiled=%-4s exit0=%-4s model_cifs=%s\n' "$tag" "$metas" "$ok" "$cifs"
    done
    ;;

collect-profiling)
    RUN_ARGS=()
    for d in "$PRED_ROOT"/*/; do
        [[ -d "$d/_profiling" ]] && RUN_ARGS+=(--run "$(basename "$d")" "${d%/}")
    done
    [[ ${#RUN_ARGS[@]} -gt 0 ]] || { echo "ERROR: no runs with a _profiling dir under $PRED_ROOT" >&2; exit 1; }
    "${PY[@]}" "$SCRIPT_DIR/scripts/04_collect_profiling.py" \
        --dataset-dir "$DATASET_DIR" "${RUN_ARGS[@]}" \
        --out "$RESULTS_DIR/profiling.tsv" --sacct
    ;;

score)
    RUN_ARGS=()
    for d in "$PRED_ROOT"/*/; do
        [[ -d "$d" ]] && RUN_ARGS+=(--run "$(basename "$d")" "${d%/}")
    done
    [[ ${#RUN_ARGS[@]} -gt 0 ]] || { echo "ERROR: no prediction dirs under $PRED_ROOT" >&2; exit 1; }
    "${PY[@]}" "$SCRIPT_DIR/scripts/05_score_dockq.py" \
        --dataset-dir "$DATASET_DIR" "${RUN_ARGS[@]}" \
        --subset "$SUBSET" --out "$RESULTS_DIR/dockq.tsv"
    ;;

link-paper)
    "${PY[@]}" "$SCRIPT_DIR/scripts/07_link_paper_predictions.py" \
        --multimer-dir "$BENCH_DIR/multimer" \
        --out-dir "$PRED_ROOT/zheng_af3_local"
    ;;

submit-score)
    [[ -d "$PRED_ROOT" ]] || { echo "ERROR: no predictions under $PRED_ROOT" >&2; exit 1; }
    LOGS="$LOG_ROOT/score"; mkdir -p "$LOGS" "$RESULTS_DIR"
    CMD=(sbatch
        --job-name=af3dockq
        --cpus-per-task="$SCORE_CPUS"
        --mem-per-cpu=4g
        --time=8:00:00
        --output="$LOGS/%x_%j.out"
        --export="ALL,REPO_DIR=$SCRIPT_DIR,DATASET_DIR=$DATASET_DIR,PRED_ROOT=$PRED_ROOT,OUT_TSV=$RESULTS_DIR/dockq.tsv,SUBSET=$SUBSET,SELECT_BY=$SELECT_BY"
        "$SCRIPT_DIR/slurm/score.sbatch")
    echo "subset    : $SUBSET   select-by: $SELECT_BY   cpus: $SCORE_CPUS"
    echo "out       : $RESULTS_DIR/dockq.tsv"
    submit "${CMD[@]}"
    ;;

plot)
    ARGS=()
    [[ -f "$RESULTS_DIR/dockq.tsv" ]]     && ARGS+=(--dockq "$RESULTS_DIR/dockq.tsv")
    [[ -f "$RESULTS_DIR/profiling.tsv" ]] && ARGS+=(--profile "$RESULTS_DIR/profiling.tsv")
    [[ ${#ARGS[@]} -gt 0 ]] || { echo "ERROR: neither dockq.tsv nor profiling.tsv in $RESULTS_DIR" >&2; exit 1; }
    "${PY[@]}" "$SCRIPT_DIR/scripts/06_plot_comparison.py" \
        "${ARGS[@]}" --arm-a "$ARM_A" --arm-b "$ARM_B" --out-dir "$FIGURES_DIR"
    ;;

publish-results)
    "${PY[@]}" "$SCRIPT_DIR/scripts/08_publish_results.py" \
        --results-dir "$RESULTS_DIR" --out-dir "$SCRIPT_DIR/results"
    ;;

*)
    echo "Unknown stage: $STAGE" >&2
    sed -n '4,16p' "${BASH_SOURCE[0]}" >&2
    exit 1
    ;;
esac
