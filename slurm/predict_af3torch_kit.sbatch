#!/usr/bin/env bash
#SBATCH --job-name=af3torchkit
#SBATCH --output=%x_%A_%a.out
#SBATCH --ntasks=1
#
# Benchmark-IV inference through Anthropic's af3_torch optimization kit (xfold, the PyTorch port of
# AlphaFold 3) from its container image, one or more complexes per task from $WORK_DIR/predict_todo.txt.
# The protocol is predict.sbatch's — same todo list, CHUNK_STRIDE dealing, USE_TEMPLATES stripping,
# time budget, one <entry>.log + <entry>.meta.json per entry under $OUT_DIR/_profiling — so stages 04
# and 05 read this arm like the af3 ones. What differs is the call:
#
#   singularity run --nv --cleanenv <image> pred --config $KIT_CONFIG --mode $KIT_MODE --allow-partial
#       --json_path=<entry>.json --output_dir=<node-local dir> --model_dir=$MODEL_DIR
#       --norun_data_pipeline --num_recycles=N --num_diffusion_samples=S
#
# The image's runscript is the kit's `run.sh`; `pred` is its inference verb, the flags after it are
# xfold's own (the kit's --num_recycles default is 10, hence the explicit value). --allow-partial: on a
# card outside the kit's certified set (this one, cc 12.0) a kernel row may refuse by name, which the kit
# otherwise turns into exit 3; the DONE line then says partial=allowed:<what>. AF3_TORCH_PARAMS_DIR names
# the directory holding of3_ported_weights.bin.zst; the kit reads the weights VARIANT (preview-2 or
# openbind) off the converted records themselves and refuses a mismatch by name.
#
# OUTPUT DIRECTORY — one kit call per directory. The kit keeps step reports at a fixed path under its
# --output_dir (.af3_torch_opt_work) and removes them when done, so array tasks must never share one.
# Each entry therefore runs into $SCRATCH/out/<entry>/ and its result directory <entry>/ (AlphaFold 3's
# layout: <entry>_model.cif, <entry>_summary_confidences.json, seed-*_sample-*/) is moved into $OUT_DIR
# afterwards, replacing an earlier attempt. The kit names the item after the input FILE STEM, which is the
# entry id for both the stage-03 JSON and its stripped copy.
#
# CACHES. MODEL_OPT_JIT_ROOT (Triton / Inductor kernel caches, NFS-hostile) and TMPDIR go to node-local
# scratch shared by every entry of this task: the first entry of a task compiles the kernels for its
# padded shape inside its forward, later entries of the same shape find them.
#
# TIMING — the kit's own lines in <entry>.log, read by scripts/04_collect_profiling.py:
#   STEP step=featurise ... wall_s=F      featurisation (the fork's data pipeline on CPU)
#   ITEM name=<entry> ... forward_s=T     the model forward on the GPU (tokens=, peak_gb= on the same line)
#   STEP step=postprocess ... wall_s=W    the writers
#   DONE ok=1 rc=0 ... partial=...        the run's verdict
# meta.json carries wall_s around the singularity call (container start, model build and load included).
#
# Submitted by run_workflow.sh, which supplies resources and these variables:
#   WORK_DIR OUT_DIR IMAGE MODEL_DIR KIT_MODE KIT_CONFIG KIT_LEVERS_OFF NUM_RECYCLES NUM_DIFFUSION_SAMPLES
#   USE_TEMPLATES REPO_DIR CHUNK_STRIDE TIME_BUDGET_S PER_ENTRY_RESERVE_S

set -euo pipefail

: "${WORK_DIR:?WORK_DIR must be set}"
: "${OUT_DIR:?OUT_DIR must be set}"
: "${IMAGE:?IMAGE must be set}"
: "${MODEL_DIR:?MODEL_DIR must be set}"
KIT_MODE="${KIT_MODE:-big}"
KIT_CONFIG="${KIT_CONFIG:-h100}"
KIT_LEVERS_OFF="${KIT_LEVERS_OFF:-}"
USE_TEMPLATES="${USE_TEMPLATES:-1}"
NUM_RECYCLES="${NUM_RECYCLES:-3}"
NUM_DIFFUSION_SAMPLES="${NUM_DIFFUSION_SAMPLES:-5}"
INPUT_SUBDIR="${INPUT_SUBDIR:-af3_input}"
CHUNK_STRIDE="${CHUNK_STRIDE:-0}"
TIME_BUDGET_S="${TIME_BUDGET_S:-0}"
PER_ENTRY_RESERVE_S="${PER_ENTRY_RESERVE_S:-1200}"
SINGULARITY="${SINGULARITY:-singularity}"    # the runtime binary (apptainer works too); overridable for a local dry test

TODO="$WORK_DIR/predict_todo.txt"
[[ -f "$TODO" ]] || { echo "ERROR: no such file: $TODO" >&2; exit 1; }
[[ -f "$IMAGE" ]] || { echo "ERROR: kit image not found: $IMAGE (launcher.image / AF3OF3_KIT_IMAGE)" >&2; exit 1; }
[[ -f "$MODEL_DIR/of3_ported_weights.bin.zst" ]] || { echo "ERROR: no of3_ported_weights.bin.zst under $MODEL_DIR (the kit's converted weights)" >&2; exit 1; }
N_TODO=$(grep -c . "$TODO")
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

INDICES=()
if (( CHUNK_STRIDE > 0 )); then
    i=$TASK_ID
    while (( i <= N_TODO )); do INDICES+=("$i"); i=$(( i + CHUNK_STRIDE )); done
else
    INDICES=("$TASK_ID")
fi
[[ ${#INDICES[@]} -gt 0 ]] || { echo "ERROR: task $TASK_ID maps to no line of $TODO ($N_TODO lines)" >&2; exit 1; }

PROF_DIR="$OUT_DIR/_profiling"
mkdir -p "$OUT_DIR" "$PROF_DIR"

SCRATCH="${TMPDIR:-/tmp}/af3torchkit_${SLURM_JOB_ID:-0}_${TASK_ID}"
JIT_ROOT="$SCRATCH/jit"
mkdir -p "$JIT_ROOT" "$SCRATCH/tmp" "$SCRATCH/input" "$SCRATCH/out"
trap 'rm -rf "$SCRATCH"' EXIT

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unknown)

STRIPPER="${REPO_DIR:-}/slurm/strip_templates.py"
[[ -f "$STRIPPER" ]] || { echo "ERROR: no such file: $STRIPPER (set REPO_DIR)" >&2; exit 1; }
run_stripper() { python3 "$STRIPPER" "$@"; }

# The container call shared by the check and every entry: --cleanenv so the host's python / CUDA variables
# never reach the kit's venvs, the kit's variables passed explicitly (the image presets its interpreters).
kit() {
    local env_levers=()
    [[ -n "$KIT_LEVERS_OFF" ]] && env_levers=(--env "MODEL_OPT_LEVERS_OFF=$KIT_LEVERS_OFF")
    "$SINGULARITY" run --nv --cleanenv \
        --bind "$WORK_DIR:$WORK_DIR" \
        --bind "$SCRATCH:$SCRATCH" \
        --bind "$MODEL_DIR:$MODEL_DIR:ro" \
        --env "AF3_TORCH_PARAMS_DIR=$MODEL_DIR" \
        --env "MODEL_OPT_JIT_ROOT=$JIT_ROOT" \
        --env "TMPDIR=$SCRATCH/tmp" \
        --env "PYTHONHASHSEED=0" \
        ${env_levers[@]+"${env_levers[@]}"} \
        "$IMAGE" "$@"
}

echo "=== af3_torch kit inference ==="
echo "task          : array id $TASK_ID, ${#INDICES[@]} entry/entries (stride ${CHUNK_STRIDE:-0} over $N_TODO)"
echo "entries       : $(for i in "${INDICES[@]}"; do printf '%s ' "$(sed -n "${i}p" "$TODO")"; done)"
echo "image         : $IMAGE"
echo "kit           : mode=$KIT_MODE  config=$KIT_CONFIG  levers_off=${KIT_LEVERS_OFF:-none}"
echo "model_dir     : $MODEL_DIR  (variant marker: $(cat "$MODEL_DIR/of3_variant" 2>/dev/null || echo 'none; the kit reads the records'))"
echo "use_templates : $USE_TEMPLATES"
echo "recycles      : $NUM_RECYCLES   diffusion samples: $NUM_DIFFUSION_SAMPLES"
echo "node          : $(hostname)   gpu: $GPU_NAME   driver: $DRIVER"
echo "time budget   : ${TIME_BUDGET_S}s   per-entry reserve: ${PER_ENTRY_RESERVE_S}s"
echo "started       : $(date -Is)"
echo
# The kit's dry check once per task: pins, interpreters, the weights digest and variant, the ACTIVE line
# (no launch). A refusal here would fail every entry, so it fails the task by name instead.
echo "--- kit check ---"
set +e
CHECK_OUT=$(kit check --config "$KIT_CONFIG" --mode "$KIT_MODE" 2>&1); CHECK_RC=$?
set -e
grep -E "ACTIVE|weights|WEIGHTS|PINS" <<< "$CHECK_OUT" | cut -c1-400 || true
[[ $CHECK_RC -eq 0 ]] || { echo "ERROR: the kit's check refused (rc=$CHECK_RC); its full output:" >&2; echo "$CHECK_OUT" >&2; exit 1; }

run_entry() {
    local entry_id="$1"
    local input="$WORK_DIR/$INPUT_SUBDIR/$entry_id.json"
    local log="$PROF_DIR/$entry_id.log"
    local meta="$PROF_DIR/$entry_id.meta.json"
    [[ -f "$input" ]] || { echo "ERROR: no input JSON: $input" >&2; return 1; }

    local input_used="$input" counts
    if [[ "$USE_TEMPLATES" == "0" ]]; then
        input_used="$SCRATCH/input/$entry_id.json"
        counts=$(run_stripper --input "$input" --output "$input_used") || { echo "ERROR: stripping templates failed for $entry_id" >&2; return 1; }
    else
        counts=$(run_stripper --input "$input") || { echo "ERROR: counting templates failed for $entry_id" >&2; return 1; }
    fi
    local n_in n_used n_entities n_with
    read -r n_in n_used n_entities n_with <<< "$counts"

    local kit_out="$SCRATCH/out/$entry_id"
    rm -rf "$kit_out"; mkdir -p "$kit_out"

    echo
    echo "--- $entry_id ---"
    echo "input       : $input"
    echo "input used  : $input_used"
    echo "templates   : use_templates=$USE_TEMPLATES  in_input=$n_in  used=$n_used  (entities=$n_entities, $n_with with templates)"
    echo "started     : $(date -Is)"

    local t_start t_end exit_code
    t_start=$(date +%s)
    set +e
    kit pred --config "$KIT_CONFIG" --mode "$KIT_MODE" --allow-partial \
        --json_path="$input_used" \
        --output_dir="$kit_out" \
        --model_dir="$MODEL_DIR" \
        --norun_data_pipeline \
        --num_recycles="$NUM_RECYCLES" \
        --num_diffusion_samples="$NUM_DIFFUSION_SAMPLES" \
        2>&1 | tee "$log"
    exit_code=${PIPESTATUS[0]}
    set -e
    t_end=$(date +%s)

    # The result directory into the shared layout; the kit's step reports (small) beside the log on failure.
    if [[ $exit_code -eq 0 && -d "$kit_out/$entry_id" ]]; then
        rm -rf "${OUT_DIR:?}/$entry_id"
        mv "$kit_out/$entry_id" "$OUT_DIR/$entry_id"
    else
        [[ $exit_code -eq 0 ]] && { echo "ERROR: kit exited 0 but wrote no $kit_out/$entry_id" >&2; exit_code=1; }
        rm -rf "$PROF_DIR/$entry_id.kit_work"
        if [[ -d "$kit_out/.af3_torch_opt_work" ]]; then cp -r "$kit_out/.af3_torch_opt_work" "$PROF_DIR/$entry_id.kit_work"; fi
    fi
    rm -rf "$kit_out"

    # The kit's own numbers for this entry, off its log lines (null when a line is missing).
    local item forward_s tokens peak_gb feat_s post_s partial
    item=$(grep -E "^\[af3-torch-opt\] ITEM name=$entry_id " "$log" | tail -1 || true)
    forward_s=$(sed -n 's/.* forward_s=\([0-9.]*\).*/\1/p' <<< "$item"); tokens=$(sed -n 's/.* tokens=\([0-9]*\).*/\1/p' <<< "$item")
    peak_gb=$(sed -n 's/.* peak_gb=\([0-9.]*\).*/\1/p' <<< "$item")
    feat_s=$(grep -E "^\[af3-torch-opt\] STEP step=featurise " "$log" | tail -1 | sed -n 's/.* wall_s=\([0-9.]*\).*/\1/p' || true)
    post_s=$(grep -E "^\[af3-torch-opt\] STEP step=postprocess " "$log" | tail -1 | sed -n 's/.* wall_s=\([0-9.]*\).*/\1/p' || true)
    partial=$(grep -E "^\[af3-torch-opt\] DONE " "$log" | tail -1 | sed -n 's/.* partial=\([^ ]*\).*/\1/p' || true)

    cat > "$meta" <<JSON
{
  "entry_id": "$entry_id",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "slurm_array_job_id": "${SLURM_ARRAY_JOB_ID:-}",
  "slurm_array_task_id": "${TASK_ID}",
  "node": "$(hostname)",
  "gpu_name": "$GPU_NAME",
  "driver_version": "$DRIVER",
  "image": "$IMAGE",
  "model_dir": "$MODEL_DIR",
  "of3_weights": true,
  "engine": "af3torch_kit",
  "kit_mode": "$KIT_MODE",
  "kit_config": "$KIT_CONFIG",
  "kit_levers_off": "$KIT_LEVERS_OFF",
  "kit_featurise_s": ${feat_s:-null},
  "kit_forward_s": ${forward_s:-null},
  "kit_postprocess_s": ${post_s:-null},
  "kit_tokens": ${tokens:-null},
  "kit_peak_gb": ${peak_gb:-null},
  "kit_partial": "${partial:-}",
  "use_templates": $([[ "$USE_TEMPLATES" == "1" ]] && echo true || echo false),
  "n_templates_in_input": $n_in,
  "n_templates_used": $n_used,
  "n_entities": $n_entities,
  "n_entities_with_templates": $n_with,
  "num_recycles": $NUM_RECYCLES,
  "num_diffusion_samples": $NUM_DIFFUSION_SAMPLES,
  "chunk_stride": $CHUNK_STRIDE,
  "wall_start_epoch_s": $t_start,
  "wall_end_epoch_s": $t_end,
  "wall_s": $((t_end - t_start)),
  "exit_code": $exit_code
}
JSON

    echo "finished    : $(date -Is)   exit=$exit_code   wall=$((t_end - t_start))s   forward=${forward_s:-?}s tokens=${tokens:-?} partial=${partial:-?}"
    return "$exit_code"
}

T_TASK_START=$(date +%s)
DONE=(); FAILED=(); SKIPPED=()
for idx in "${INDICES[@]}"; do
    ENTRY_ID=$(sed -n "${idx}p" "$TODO")
    [[ -n "$ENTRY_ID" ]] || { echo "WARNING: line $idx of $TODO is empty, skipping" >&2; continue; }
    if (( TIME_BUDGET_S > 0 )); then
        ELAPSED=$(( $(date +%s) - T_TASK_START ))
        if (( TIME_BUDGET_S - ELAPSED < PER_ENTRY_RESERVE_S )); then SKIPPED+=("$ENTRY_ID"); continue; fi
    fi
    if run_entry "$ENTRY_ID"; then DONE+=("$ENTRY_ID"); else FAILED+=("$ENTRY_ID"); fi
done

echo
echo "=== task $TASK_ID summary ==="
echo "wall          : $(( $(date +%s) - T_TASK_START ))s"
echo "ok            : ${#DONE[@]}${DONE[*]:+  ${DONE[*]}}"
echo "failed        : ${#FAILED[@]}${FAILED[*]:+  ${FAILED[*]}}"
echo "skipped       : ${#SKIPPED[@]}${SKIPPED[*]:+  ${SKIPPED[*]}}"
[[ ${#SKIPPED[@]} -eq 0 ]] || echo "NOTE: skipped entries ran out of time budget — resubmit them, or raise slurm.time / lower --chunk-size."
[[ ${#FAILED[@]} -eq 0 ]] || exit 1
exit 0
