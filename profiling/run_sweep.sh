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
echo "engine        : $ENGINE   weights: $WEIGHTS   use_templates: $(cfg use_templates)"
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
VARS=("ARM=$ARM" "ENGINE=$ENGINE" "WEIGHTS=$WEIGHTS" "USE_TEMPLATES=$USE_TEMPLATES"
      "INPUT_DIR=$INPUT_DIR" "OUT_DIR=$OUT_DIR" "REPO_DIR=$SCRIPT_DIR"
      "NUM_DIFFUSION_SAMPLES=$SAMPLES" "NUM_RECYCLES=$RECYCLES"
      "TIME_BUDGET_S=$(hms_to_s "$TIME")" "MAX_TOKENS=$MAX_TOKENS"
      "MEM_FRACTION=$MEM_FRACTION")

case "$ENGINE" in
af3)
    IMAGE="${AF3OF3_IMAGE:-$(cfg image)}"
    MODEL_DIR=$(expand_home "$(cfg model_dir)")
    OF3=$(cfg of3_weights); [[ "$OF3" == "true" ]] && OF3=1 || OF3=0
    VARS+=("IMAGE=$IMAGE" "MODEL_DIR=$MODEL_DIR" "OF3_WEIGHTS=$OF3")
    echo "image         : $IMAGE"
    echo "model_dir     : $MODEL_DIR   of3_weights=$OF3"
    echo "xla arena     : preallocate=true  mem_fraction=$MEM_FRACTION"
    ;;
of3)
    if [[ "$USE_TEMPLATES" == "1" ]]; then
        echo "ERROR: $CONFIG sets use_templates=true, but the OF3 queries carry no" >&2
        echo "       template_cif_paths and OpenFold3 would silently run without them." >&2
        echo "       See scripts/09_build_of3_inputs.py." >&2
        exit 1
    fi
    OF3_SRC="${AF3OF3_OF3_SRC:-$(cfg of3_src)}"
    OF3_ENV="${AF3OF3_OF3_ENV:-$(cfg of3_env)}"
    CKPT=$(expand_home "${AF3OF3_OF3_CKPT:-$(cfg ckpt)}")
    RUNNER_YAML=$(cfg runner_yaml)
    [[ "$RUNNER_YAML" == /* ]] || RUNNER_YAML="$SCRIPT_DIR/$RUNNER_YAML"
    VARS+=("OF3_SRC=$OF3_SRC" "OF3_ENV=$OF3_ENV" "CKPT=$CKPT" "RUNNER_YAML=$RUNNER_YAML")
    echo "of3_src       : $OF3_SRC"
    echo "ckpt          : $CKPT"
    echo "runner yaml   : $RUNNER_YAML"
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

CMD=(sbatch
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
