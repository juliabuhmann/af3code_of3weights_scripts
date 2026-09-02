#!/usr/bin/env bash
#SBATCH --job-name=af3pred
#SBATCH --output=%x_%A_%a.out
#SBATCH --ntasks=1
#
# AlphaFold3 inference for ONE complex — one array task per line of
# $WORK_DIR/predict_todo.txt.
#
# The input JSON already carries MSAs and templates (stage 03), so this runs with
# --norun_data_pipeline: no database search, no jackhmmer, and the timings below
# measure the model rather than the pipeline.
#
# The two weight arms differ ONLY in MODEL_DIR and OF3_WEIGHTS:
#   native    MODEL_DIR=/cluster/home/.../.alphafold3_model_dir   OF3_WEIGHTS=0
#   openbind  MODEL_DIR=/models  (inside the image)               OF3_WEIGHTS=1
# Same container, same input JSON, same recycles/samples/seed.
#
# TIMING — what is and is not measured
# ------------------------------------
#   the log    AF3's own timers: "Featurising data ..." and "Running model
#              inference and extracting output structures ...". These exclude
#              container startup and parameter loading.
#   meta.json  node, GPU model, driver, and the wall clock around the singularity
#              call — which DOES include startup and parameter loading.
#
# CONTAINER SCRATCH — two things that must be handled explicitly, both of which bit
# on the first attempt:
#   * ptxas (invoked by JAX to compile kernels) writes temporaries to $TMPDIR. The
#     host $TMPDIR leaks into the container via --export=ALL, but the path is not
#     bind-mounted there, so ptxas dies with "Could not open output file". TMPDIR is
#     therefore bound at its own path AND re-exported into the container.
#   * /tmp/alphafold_cache already EXISTS inside this image, so binding onto it
#     fails with "mkdir /tmp/alphafold_cache: file exists". We leave the image's
#     /tmp alone and point AF3 at a host cache with --cache_dir.
#
# --force_output_dir is passed for a third reason: without it AF3 refuses to reuse a
# non-empty <output_dir>/<name> and silently writes to <name>_<timestamp> instead, so
# a retry after a failed attempt would land where the scoring stage does not look.
# With the flag, one entry always maps to exactly one output directory.
#
# Submitted by run_workflow.sh, which supplies resources and these variables:
#   WORK_DIR OUT_DIR IMAGE MODEL_DIR OF3_WEIGHTS NUM_RECYCLES NUM_DIFFUSION_SAMPLES

set -euo pipefail

: "${WORK_DIR:?WORK_DIR must be set}"
: "${OUT_DIR:?OUT_DIR must be set}"
: "${IMAGE:?IMAGE must be set}"
: "${MODEL_DIR:?MODEL_DIR must be set}"
OF3_WEIGHTS="${OF3_WEIGHTS:-0}"
NUM_RECYCLES="${NUM_RECYCLES:-3}"
NUM_DIFFUSION_SAMPLES="${NUM_DIFFUSION_SAMPLES:-5}"
INPUT_SUBDIR="${INPUT_SUBDIR:-af3_input}"

TODO="$WORK_DIR/predict_todo.txt"
[[ -f "$TODO" ]] || { echo "ERROR: no such file: $TODO" >&2; exit 1; }

ENTRY_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TODO")
[[ -n "$ENTRY_ID" ]] || { echo "ERROR: array index $SLURM_ARRAY_TASK_ID beyond $TODO" >&2; exit 1; }

INPUT="$WORK_DIR/$INPUT_SUBDIR/$ENTRY_ID.json"
[[ -f "$INPUT" ]] || { echo "ERROR: no input JSON: $INPUT" >&2; exit 1; }

PROF_DIR="$OUT_DIR/_profiling"
mkdir -p "$OUT_DIR" "$PROF_DIR"
LOG="$PROF_DIR/$ENTRY_ID.log"
META="$PROF_DIR/$ENTRY_ID.meta.json"

# Node-local scratch for this task. $TMPDIR is node-local on Euler; SLURM sets it,
# an interactive shell may not. See CONTAINER SCRATCH above.
SCRATCH="${TMPDIR:-/tmp}/af3pred_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
CACHE_DIR="$SCRATCH/alphafold_cache"
mkdir -p "$CACHE_DIR" "$SCRATCH/tmp"
trap 'rm -rf "$SCRATCH"' EXIT

# GPU identity is recorded because a timing comparison across different GPU models
# would be meaningless — stage 04 writes it into every row so that stays checkable.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unknown)

echo "=== AF3 inference ==="
echo "entry_id    : $ENTRY_ID"
echo "input       : $INPUT"
echo "image       : $IMAGE"
echo "model_dir   : $MODEL_DIR"
echo "of3_weights : $OF3_WEIGHTS"
echo "recycles    : $NUM_RECYCLES   diffusion samples: $NUM_DIFFUSION_SAMPLES"
echo "node        : $(hostname)   gpu: $GPU_NAME   driver: $DRIVER"
echo "started     : $(date -Is)"

# --of3_weights is a boolean flag: pass it for the openbind arm, omit it entirely
# for the native arm. The image's /models carries an `of3_variant` marker, so no
# --of3_openbind is needed on top.
OF3_FLAG=()
[[ "$OF3_WEIGHTS" == "1" ]] && OF3_FLAG=(--of3_weights)

# The weights live inside the image for the openbind arm, so only bind MODEL_DIR
# when it is a real host path.
MODEL_BIND=()
[[ -d "$MODEL_DIR" && "$MODEL_DIR" != /models ]] && MODEL_BIND=(--bind "$MODEL_DIR:$MODEL_DIR:ro")

T_START=$(date +%s)
set +e
singularity exec --nv \
    --bind "$WORK_DIR:$WORK_DIR" \
    --bind "$OUT_DIR:$OUT_DIR" \
    --bind "$SCRATCH:$SCRATCH" \
    --env "TMPDIR=$SCRATCH/tmp" \
    "${MODEL_BIND[@]}" \
    "$IMAGE" \
    python /app/alphafold/run_alphafold.py \
        --json_path="$INPUT" \
        --output_dir="$OUT_DIR" \
        --model_dir="$MODEL_DIR" \
        --cache_dir="$CACHE_DIR" \
        --force_output_dir \
        --num_recycles="$NUM_RECYCLES" \
        --num_diffusion_samples="$NUM_DIFFUSION_SAMPLES" \
        --norun_data_pipeline \
        "${OF3_FLAG[@]}" \
    2>&1 | tee "$LOG"
EXIT_CODE=${PIPESTATUS[0]}
set -e
T_END=$(date +%s)

cat > "$META" <<JSON
{
  "entry_id": "$ENTRY_ID",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "slurm_array_job_id": "${SLURM_ARRAY_JOB_ID:-}",
  "slurm_array_task_id": "${SLURM_ARRAY_TASK_ID:-}",
  "node": "$(hostname)",
  "gpu_name": "$GPU_NAME",
  "driver_version": "$DRIVER",
  "image": "$IMAGE",
  "model_dir": "$MODEL_DIR",
  "of3_weights": $([[ "$OF3_WEIGHTS" == "1" ]] && echo true || echo false),
  "num_recycles": $NUM_RECYCLES,
  "num_diffusion_samples": $NUM_DIFFUSION_SAMPLES,
  "wall_start_epoch_s": $T_START,
  "wall_end_epoch_s": $T_END,
  "wall_s": $((T_END - T_START)),
  "exit_code": $EXIT_CODE
}
JSON

echo "finished    : $(date -Is)   exit=$EXIT_CODE   wall=$((T_END - T_START))s"
echo "profile     : $META"
exit "$EXIT_CODE"
