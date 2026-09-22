#!/usr/bin/env bash
# Submit (or run) the VRAM-scaling sweep for ONE arm.
#
#   bash profiling/run_sweep.sh --config configs/af3_openbind_templateoff.json
#   bash profiling/run_sweep.sh --config configs/of3_openbind_templateoff.json
#
# The arm's own config supplies the engine, the weights, the template setting, the
# recycles/samples and the GPU request — the same file the benchmark ran from — so the
# sweep measures the configuration this repository actually uses and not a variant of
# it. Only the wall-clock limit differs, because one job now walks a whole token grid
# instead of predicting one complex (--time, default below).
#
# The pool inputs are built once and shared by every arm: profiling/build_pool_inputs.py
# for the AF3 side, scripts/09_build_of3_inputs.py for the OF3 queries derived from
# them. This script builds them if they are missing.
#
# Options:
#   --config PATH      arm config under configs/ (required)
#   --base-dir DIR     benchmark base dir (default: $AF3OF3_BASE_DIR)
#   --input-dir DIR    pool inputs (default: <base-dir>/profiling/inputs)
#   --out-dir DIR      results (default: <base-dir>/profiling/<arm>)
#   --time HH:MM:SS    wall-clock limit for the sweep job (default: $DEFAULT_TIME)
#   --max-tokens N     stop the grid at N tokens even if larger pools exist
#   --mem-fraction F   AlphaFold3's XLA arena as a fraction of the card (default 0.95,
#                      matching the openbind runs). Ignored by the OF3 engine.
#   --end N            largest token count to BUILD (default: 7000)
#   --step N           token-count increment (default: 500)
#   --local            run here instead of submitting — for a session that already
#                      holds a GPU. Blocks until the sweep finishes.
#   --dry-run          print what would be submitted and exit
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_TIME="12:00:00"

CONFIG=""
BASE_DIR="${AF3OF3_BASE_DIR:-/path/to/workdir/af3code_of3weights}"
INPUT_DIR=""
OUT_DIR=""
TIME="$DEFAULT_TIME"
MAX_TOKENS=0
MEM_FRACTION=0.95
GRID_END=7000
GRID_STEP=500
LOCAL=0
DRY_RUN=0

_val() { [[ $# -ge 2 && "$2" != --* ]] || { echo "ERROR: $1 needs a value" >&2; exit 1; }; printf '%s' "$2"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)     CONFIG=$(_val "$@");     shift 2 ;;
        --base-dir)   BASE_DIR=$(_val "$@");   shift 2 ;;
        --input-dir)  INPUT_DIR=$(_val "$@");  shift 2 ;;
        --out-dir)    OUT_DIR=$(_val "$@");    shift 2 ;;
        --time)       TIME=$(_val "$@");       shift 2 ;;
        --max-tokens) MAX_TOKENS=$(_val "$@"); shift 2 ;;
        --mem-fraction) MEM_FRACTION=$(_val "$@"); shift 2 ;;
        --end)        GRID_END=$(_val "$@");   shift 2 ;;
        --step)       GRID_STEP=$(_val "$@");  shift 2 ;;
        --local)      LOCAL=1;                 shift   ;;
        --dry-run)    DRY_RUN=1;               shift   ;;
        -h|--help)    sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$CONFIG" ]] || { echo "ERROR: --config is required" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
if [[ "$BASE_DIR" == /path/to/* ]]; then
    echo "ERROR: --base-dir is still the placeholder ($BASE_DIR)." >&2
    echo "       Pass --base-dir, or set AF3OF3_BASE_DIR; see SITE CONFIGURATION in run_workflow.sh." >&2
    exit 1
fi
[[ -d "$BASE_DIR" ]] || { echo "ERROR: no such base dir: $BASE_DIR" >&2; exit 1; }

# Same helpers as run_workflow.sh, so a config is read identically by both.
cfg() {
    python3 -c "import json,sys
d=json.load(open(sys.argv[1]))
keys=sys.argv[2].split('.')
for k in keys:
    d=d.get(k) if isinstance(d,dict) else None
    if d is None: break
print('' if d is None else (str(d).lower() if isinstance(d,bool) else d))" "$CONFIG" "$1"
}
hms_to_s() {
    python3 -c "
import sys
value = sys.argv[1]
days, _, clock = value.rpartition('-')
parts = [float(p) for p in clock.split(':')]
while len(parts) < 3: parts.insert(0, 0.0)
print(int((int(days) if days else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]))" "$1"
}
expand_home() { local v="$1"; v="${v/#\$HOME/$HOME}"; printf '%s' "${v/#\~/$HOME}"; }

ARM=$(cfg name_tag)
ENGINE=$(cfg engine)
WEIGHTS=$(cfg weights)
USE_TEMPLATES=$(cfg use_templates); [[ "$USE_TEMPLATES" == "true" ]] && USE_TEMPLATES=1 || USE_TEMPLATES=0
SAMPLES=$(cfg num_diffusion_samples)
RECYCLES=$(cfg num_recycles)
# warmup_runs (optional, default 0): untimed runs of each size before the recorded one, for arms
# whose kernels compile or autotune per tensor shape inside the timed loop (the of3kit_* arms).
WARMUP_RUNS=$(cfg warmup_runs); WARMUP_RUNS=${WARMUP_RUNS:-0}
GPU=$(cfg slurm.gpu)
GPUMEM=$(cfg slurm.gpumem)
CPUS=$(cfg slurm.cpus)
MEM=$(cfg slurm.mem_per_cpu)

[[ -n "$INPUT_DIR" ]] || INPUT_DIR="$BASE_DIR/profiling/inputs"
[[ -n "$OUT_DIR" ]]   || OUT_DIR="$BASE_DIR/profiling/$ARM"
LOGS="$BASE_DIR/logs/profiling"
mkdir -p "$OUT_DIR" "$LOGS"
# The config next to the results, for the same reason run_workflow.sh does it: a row
# is labelled by what actually ran.
cp "$CONFIG" "$OUT_DIR/config.json"

echo "arm           : $ARM  ($(cfg label))"
echo "engine        : $ENGINE   weights: $WEIGHTS   use_templates: $(cfg use_templates)   warmup_runs: $WARMUP_RUNS"
echo "base dir      : $BASE_DIR"
echo "inputs        : $INPUT_DIR"
echo "out dir       : $OUT_DIR"

# ── Pool inputs, built once and shared by every arm ──────────────────────────────
if [[ ! -f "$INPUT_DIR/pool_manifest.tsv" ]]; then
    echo
    echo "Building pool inputs (none at $INPUT_DIR) ..."
    python3 "$SCRIPT_DIR/profiling/build_pool_inputs.py" \
        --base-dir "$BASE_DIR" --out-dir "$INPUT_DIR" \
        --end "$GRID_END" --step "$GRID_STEP"
fi
if [[ "$ENGINE" == "of3" && ! -d "$INPUT_DIR/of3_pool" ]]; then
    echo
    echo "Converting the pool inputs to OF3 queries ..."
    python3 "$SCRIPT_DIR/scripts/09_build_of3_inputs.py" \
        --af3-input-dir "$INPUT_DIR/af3_pool" --out-dir "$INPUT_DIR/of3_pool" --quiet
fi

# ── Engine-specific environment, resolved exactly as run_workflow.sh resolves it ──
# One KEY=VALUE per element: joined with commas for sbatch --export, passed as-is to
# `env` under --local, so the two paths cannot drift.
VARS=("ARM=$ARM" "ENGINE=$ENGINE" "WEIGHTS=$WEIGHTS" "USE_TEMPLATES=$USE_TEMPLATES" "WARMUP_RUNS=$WARMUP_RUNS"
      "INPUT_DIR=$INPUT_DIR" "OUT_DIR=$OUT_DIR" "REPO_DIR=$SCRIPT_DIR"
      "NUM_DIFFUSION_SAMPLES=$SAMPLES" "NUM_RECYCLES=$RECYCLES"
      "TIME_BUDGET_S=$(hms_to_s "$TIME")" "MAX_TOKENS=$MAX_TOKENS"
      "MEM_FRACTION=$MEM_FRACTION")

case "$ENGINE" in
af3)
    IMAGE="${AF3OF3_IMAGE:-$(cfg image)}"
    MODEL_DIR=$(expand_home "$(cfg model_dir)")
    OF3=$(cfg of3_weights); [[ "$OF3" == "true" ]] && OF3=1 || OF3=0
    VARS+=("MODEL_DIR=$MODEL_DIR" "OF3_WEIGHTS=$OF3")
    echo "model_dir     : $MODEL_DIR   of3_weights=$OF3"
    echo "xla arena     : preallocate=true  mem_fraction=$MEM_FRACTION"
    # Two launchers for the one engine. Without a `launcher` block the arm is the production image
    # (singularity exec ... run_alphafold.py). With launcher.kind "af3_jax_kit" it is Anthropic's
    # af3_jax optimization kit, run from ITS container image (launcher.image: a .sif built from the
    # kit's own environment/Dockerfile via GitHub Actions -> ghcr -> `apptainer build`; the image
    # carries the stock venv, the kit venv and the pinned AlphaFold 3 checkout, so nothing is built
    # or patched on the host). The kit's entry point inside it: `run.sh pred --mode M`. Same engine,
    # same af3_pool inputs, same log timers; see configs/af3kit_*.json and profiling/README.md.
    KIT_KIND=$(cfg launcher.kind)
    if [[ -n "$KIT_KIND" ]]; then
        [[ "$KIT_KIND" == "af3_jax_kit" ]] || { echo "ERROR: unknown launcher.kind '$KIT_KIND' for engine af3 in $CONFIG" >&2; exit 1; }
        KIT_IMAGE="${AF3OF3_KIT_IMAGE:-$(expand_home "$(cfg launcher.image)")}"
        KIT_CONFIG=$(cfg launcher.kit_config); KIT_MODE=$(cfg launcher.mode); KIT_VARIANT=$(cfg launcher.variant); KIT_LEVERS_OFF=$(cfg launcher.levers_off)
        KIT_ENV_VARS=$(python3 -c "import json,sys; e=json.load(open(sys.argv[1])).get('launcher',{}).get('env',{}); print(';'.join(f'{k}={v}' for k,v in e.items()))" "$CONFIG")
        [[ -n "$KIT_MODE" ]] || { echo "ERROR: launcher.mode missing in $CONFIG" >&2; exit 1; }
        [[ -f "$KIT_IMAGE" ]] || { echo "ERROR: kit image not found: $KIT_IMAGE (launcher.image / AF3OF3_KIT_IMAGE; built with the fork's 'kit image' GitHub Action, then apptainer build -- profiling/README.md)" >&2; exit 1; }
        [[ -f "$MODEL_DIR/of3_ported_weights.bin.zst" ]] || { echo "ERROR: no converted weights at $MODEL_DIR (the kit's 'run.sh install --weights' output for variant ${KIT_VARIANT:-p2})" >&2; exit 1; }
        VARS+=("KIT_KIND=$KIT_KIND" "KIT_IMAGE=$KIT_IMAGE" "KIT_CONFIG=${KIT_CONFIG:-a100}"
               "KIT_MODE=$KIT_MODE" "KIT_VARIANT=${KIT_VARIANT:-p2}" "KIT_LEVERS_OFF=$KIT_LEVERS_OFF" "KIT_ENV_VARS=$KIT_ENV_VARS")
        echo "launcher      : $KIT_KIND  mode=$KIT_MODE  variant=${KIT_VARIANT:-p2}  levers_off=${KIT_LEVERS_OFF:-none}  card file=${KIT_CONFIG:-a100}.env"
        echo "kit image     : $KIT_IMAGE  ($(cfg launcher.image_ref))"
    else
        VARS+=("IMAGE=$IMAGE")
        echo "image         : $IMAGE"
    fi
    ;;
of3)
    if [[ "$USE_TEMPLATES" == "1" ]]; then
        echo "ERROR: $CONFIG sets use_templates=true, but the OF3 queries carry no" >&2
        echo "       template_cif_paths and OpenFold3 would silently run without them." >&2
        echo "       See scripts/09_build_of3_inputs.py." >&2
        exit 1
    fi
    CKPT=$(expand_home "${AF3OF3_OF3_CKPT:-$(cfg ckpt)}")
    RUNNER_YAML=$(cfg runner_yaml)
    [[ "$RUNNER_YAML" == /* ]] || RUNNER_YAML="$SCRIPT_DIR/$RUNNER_YAML"
    VARS+=("CKPT=$CKPT" "RUNNER_YAML=$RUNNER_YAML")
    echo "ckpt          : $CKPT"
    echo "runner yaml   : $RUNNER_YAML"
    # Two launchers for the one engine. Without a `launcher` block the arm is OpenFold3 from a
    # source checkout (of3_src) in its own environment (of3_env). With launcher.kind
    # "openfold3_ob0_kit" it is Anthropic's optimization kit: the kit's venv, its per-card
    # environment file, its `pred --mode` entry point (or upstream's for mode off). Same
    # engine, same queries, same counters; see configs/of3kit_*.json and setup_kit_env.sh.
    KIT_KIND=$(cfg launcher.kind)
    if [[ -n "$KIT_KIND" ]]; then
        [[ "$KIT_KIND" == "openfold3_ob0_kit" ]] || { echo "ERROR: unknown launcher.kind '$KIT_KIND' in $CONFIG" >&2; exit 1; }
        KIT_DIR="${AF3OF3_KIT_DIR:-$(cfg launcher.kit_dir)}"
        KIT_ENV="${AF3OF3_KIT_ENV:-$(cfg launcher.kit_env)}"
        KIT_CUDA_HOME="${AF3OF3_KIT_CUDA_HOME:-$(cfg launcher.cuda_home)}"
        KIT_LD="${AF3OF3_KIT_LD:-$(cfg launcher.ld_library_path)}"
        KIT_CONFIG=$(cfg launcher.kit_config); KIT_MODE=$(cfg launcher.mode); KIT_LEVERS_OFF=$(cfg launcher.levers_off)
        # launcher.env (optional): variables exported into the kit process, e.g. the kit's own gate
        # overrides (OPENFOLD3_OB0_OPT_REACH_GATE, OF3O_MIN_TOKENS); the same map the af3 launcher takes.
        KIT_ENV_VARS=$(python3 -c "import json,sys; e=json.load(open(sys.argv[1])).get('launcher',{}).get('env',{}); print(';'.join(f'{k}={v}' for k,v in e.items()))" "$CONFIG")
        [[ -n "$KIT_MODE" ]] || { echo "ERROR: launcher.mode missing in $CONFIG" >&2; exit 1; }
        [[ -d "$KIT_DIR" && -x "$KIT_ENV/bin/python" ]] || { echo "ERROR: kit dir or venv missing ($KIT_DIR, $KIT_ENV) -- profiling/setup_kit_env.sh" >&2; exit 1; }
        VARS+=("KIT_KIND=$KIT_KIND" "KIT_DIR=$KIT_DIR" "KIT_ENV=$KIT_ENV" "KIT_CONFIG=${KIT_CONFIG:-h100}" "KIT_MODE=$KIT_MODE"
               "KIT_LEVERS_OFF=$KIT_LEVERS_OFF" "KIT_CUDA_HOME=$KIT_CUDA_HOME" "KIT_LD=$KIT_LD" "KIT_ENV_VARS=$KIT_ENV_VARS")
        echo "launcher      : $KIT_KIND  mode=$KIT_MODE  levers_off=${KIT_LEVERS_OFF:-none}  card file=${KIT_CONFIG:-h100}.env${KIT_ENV_VARS:+  env=$KIT_ENV_VARS}"
        echo "kit dir       : $KIT_DIR  ($(cfg launcher.kit_commit))"
        echo "kit env       : $KIT_ENV"
    else
        OF3_SRC="${AF3OF3_OF3_SRC:-$(cfg of3_src)}"
        OF3_ENV="${AF3OF3_OF3_ENV:-$(cfg of3_env)}"
        VARS+=("OF3_SRC=$OF3_SRC" "OF3_ENV=$OF3_ENV")
        echo "of3_src       : $OF3_SRC"
    fi
    ;;
*)
    echo "ERROR: unknown engine '$ENGINE' in $CONFIG (expected af3 or of3)" >&2; exit 1 ;;
esac

SWEEP="$SCRIPT_DIR/profiling/sweep.sbatch"
EXPORTS="ALL,$(IFS=,; printf '%s' "${VARS[*]}")"

# ── Run here, or submit ──────────────────────────────────────────────────────────
if [[ "$LOCAL" -eq 1 ]]; then
    echo
    echo "--local: running in this allocation (needs a GPU to itself)."
    if [[ "$DRY_RUN" -eq 1 ]]; then echo "--dry-run: NOT started."; exit 0; fi
    env "${VARS[@]}" bash "$SWEEP"
    exit $?
fi

PART=$(cfg slurm.partition); PART_ARG=(); [[ -n "$PART" ]] && PART_ARG=(--partition="$PART")
CMD=(sbatch "${PART_ARG[@]}"
    --job-name="vram_$ARM"
    --gpus="$GPU"
    --gres="gpumem:$GPUMEM"
    --cpus-per-task="$CPUS"
    --mem-per-cpu="$MEM"
    --time="$TIME"
    --output="$LOGS/%x_%j.out"
    --export="$EXPORTS"
    "$SWEEP")

echo
echo "sbatch command:"
printf '  %s' "${CMD[0]}"; printf ' \\\n    %s' "${CMD[@]:1}"; echo
echo
if [[ "$DRY_RUN" -eq 1 ]]; then echo "--dry-run: NOT submitted."; exit 0; fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not found on PATH — run this from a login node, or pass --local" >&2
    exit 127
fi
"${CMD[@]}"
echo "Check with:  squeue -u \$USER"
