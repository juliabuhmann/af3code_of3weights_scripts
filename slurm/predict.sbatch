#!/usr/bin/env bash
#SBATCH --job-name=af3pred
#SBATCH --output=%x_%A_%a.out
#SBATCH --ntasks=1
#
# AlphaFold3 inference for one or more complexes, from $WORK_DIR/predict_todo.txt.
#
# The input JSON already carries MSAs and templates (stage 03), so this runs with
# --norun_data_pipeline: no database search, no jackhmmer, and the timings below
# measure the model rather than the pipeline.
#
# The two weight arms differ ONLY in MODEL_DIR and OF3_WEIGHTS:
#   native    MODEL_DIR=<dir holding af3.bin.zst>          OF3_WEIGHTS=0
#   openbind  MODEL_DIR=/models  (inside the image)               OF3_WEIGHTS=1
# Same container, same recycles/samples/seed.
#
# ONE ALLOCATION, MANY ENTRIES — CHUNK_STRIDE
# -------------------------------------------
# A GPU allocation is expensive to obtain and cheap to keep: on the completed
# 203-task af3_openbind array, 7.7 GPU-hours of real work took 2.6 hours of wall
# clock at --array-limit 20, i.e. ~6.8x inflation, essentially all of it waiting for
# SLURM to hand each two-minute task its own GPU. So a task can process several
# entries in the allocation it already holds:
#
#   CHUNK_STRIDE unset/0   this task runs exactly the entry at line
#                          $SLURM_ARRAY_TASK_ID — the original behaviour.
#   CHUNK_STRIDE=T         this task runs lines $SLURM_ARRAY_TASK_ID, +T, +2T, ...
#
# STRIDE, NOT A CONTIGUOUS BLOCK. predict_todo.txt is sorted largest-complex-first,
# so contiguous blocks would give task 1 every big complex and the last task every
# small one — the array would finish only when the slowest block did. Striding deals
# the sorted list out round-robin, so every task gets a comparable size mix.
#
# Two further gains come for free, because the caches below are created once per
# TASK rather than once per entry: AF3's --cache_dir keeps its compiled-kernel cache
# warm across the chunk, and the container's weights stay in the page cache.
#
# Per-entry outputs are unchanged — one <entry>.log and one <entry>.meta.json each —
# so stages 04 and 05 need no knowledge of chunking.
#
# FAILURE ISOLATION. One entry failing does not abandon the rest of the chunk: each
# is run, its exit code recorded in its own meta.json, and the task exits non-zero
# if any entry failed. A summary of done/failed/skipped is printed at the end.
#
# TIME BUDGET. Before starting an entry the task checks it has PER_ENTRY_RESERVE_S
# left of TIME_BUDGET_S; if not it stops cleanly and NAMES the entries it skipped,
# rather than being killed by SLURM part-way through writing one.
#
# TEMPLATES — USE_TEMPLATES=1|0
# -----------------------------
# AlphaFold3 has no inference-time template flag: the only template flag in
# run_alphafold.py is --max_template_date, which configures the data pipeline this
# job does not run. Templates reach the model solely through the input JSON, so the
# arm's setting is applied HERE rather than by forking work/af3_input/:
#
#   USE_TEMPLATES=1   the stage-03 JSON is used as written, templates inline.
#   USE_TEMPLATES=0   slurm/strip_templates.py writes a copy with `templates: []`
#                     — AlphaFold3's explicit "use no templates" value — into this
#                     task's node-local scratch, and that copy is what runs.
#
# Both arms therefore read the same file for their MSAs, so the alignments are
# identical between templates-on and templates-off by construction. What actually
# happened is counted, not assumed: n_templates_in_input and n_templates_used go
# into meta.json, and the stripper exits non-zero if any template survives.
#
# TIMING — what is and is not measured
# ------------------------------------
#   the log    AF3's own timers: "Featurising data ..." and "Running model
#              inference and extracting output structures ...". These exclude
#              container startup and parameter loading.
#   meta.json  node, GPU model, driver, and the wall clock around the singularity
#              call — which DOES include startup and parameter loading. Note that
#              with a warm chunk cache, later entries in a task have a smaller
#              startup component than the first one.
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
#   USE_TEMPLATES REPO_DIR CHUNK_STRIDE TIME_BUDGET_S PER_ENTRY_RESERVE_S

set -euo pipefail

: "${WORK_DIR:?WORK_DIR must be set}"
: "${OUT_DIR:?OUT_DIR must be set}"
: "${IMAGE:?IMAGE must be set}"
: "${MODEL_DIR:?MODEL_DIR must be set}"
OF3_WEIGHTS="${OF3_WEIGHTS:-0}"
# Default 1 = use the JSON as stage 03 wrote it. Only an explicit 0 strips.
USE_TEMPLATES="${USE_TEMPLATES:-1}"
NUM_RECYCLES="${NUM_RECYCLES:-3}"
NUM_DIFFUSION_SAMPLES="${NUM_DIFFUSION_SAMPLES:-5}"
INPUT_SUBDIR="${INPUT_SUBDIR:-af3_input}"
CHUNK_STRIDE="${CHUNK_STRIDE:-0}"
TIME_BUDGET_S="${TIME_BUDGET_S:-0}"
PER_ENTRY_RESERVE_S="${PER_ENTRY_RESERVE_S:-1200}"

TODO="$WORK_DIR/predict_todo.txt"
[[ -f "$TODO" ]] || { echo "ERROR: no such file: $TODO" >&2; exit 1; }
N_TODO=$(grep -c . "$TODO")
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

# Which lines of the todo list belong to this task (see CHUNK_STRIDE above).
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

# Node-local scratch for this TASK — shared by every entry it runs, which is what
# keeps the compiled-kernel cache warm across the chunk. $TMPDIR is node-local on
# Euler; SLURM sets it, an interactive shell may not. See CONTAINER SCRATCH above.
SCRATCH="${TMPDIR:-/tmp}/af3pred_${SLURM_JOB_ID:-0}_${TASK_ID}"
CACHE_DIR="$SCRATCH/alphafold_cache"
mkdir -p "$CACHE_DIR" "$SCRATCH/tmp" "$SCRATCH/input"
trap 'rm -rf "$SCRATCH"' EXIT

# GPU identity is recorded because a timing comparison across different GPU models
# would be meaningless — stage 04 writes it into every row so that stays checkable.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unknown)

# strip_templates.py is stdlib-only; run it with the node's python3 when there is
# one, otherwise with the container's interpreter, fed on stdin so the repository
# does not need to be bind-mounted for this one call.
STRIPPER="${REPO_DIR:-}/slurm/strip_templates.py"
[[ -f "$STRIPPER" ]] || { echo "ERROR: no such file: $STRIPPER (set REPO_DIR)" >&2; exit 1; }
run_stripper() {
    if command -v python3 >/dev/null 2>&1; then
        python3 "$STRIPPER" "$@"
    else
        singularity exec --bind "$WORK_DIR:$WORK_DIR" --bind "$SCRATCH:$SCRATCH" \
            "$IMAGE" python - "$@" < "$STRIPPER"
    fi
}

# --of3_weights is a boolean flag: pass it for the openbind arm, omit it entirely
# for the native arm. The image's /models carries an `of3_variant` marker, so no
# --of3_openbind is needed on top.
OF3_FLAG=()
[[ "$OF3_WEIGHTS" == "1" ]] && OF3_FLAG=(--of3_weights)

# The weights live inside the image for the openbind arm, so only bind MODEL_DIR
# when it is a real host path.
MODEL_BIND=()
[[ -d "$MODEL_DIR" && "$MODEL_DIR" != /models ]] && MODEL_BIND=(--bind "$MODEL_DIR:$MODEL_DIR:ro")

echo "=== AF3 inference ==="
echo "task          : array id $TASK_ID, ${#INDICES[@]} entry/entries (stride ${CHUNK_STRIDE:-0} over $N_TODO)"
echo "entries       : $(for i in "${INDICES[@]}"; do printf '%s ' "$(sed -n "${i}p" "$TODO")"; done)"
echo "image         : $IMAGE"
echo "model_dir     : $MODEL_DIR   of3_weights=$OF3_WEIGHTS"
echo "use_templates : $USE_TEMPLATES"
echo "recycles      : $NUM_RECYCLES   diffusion samples: $NUM_DIFFUSION_SAMPLES"
echo "node          : $(hostname)   gpu: $GPU_NAME   driver: $DRIVER"
echo "time budget   : ${TIME_BUDGET_S}s   per-entry reserve: ${PER_ENTRY_RESERVE_S}s"
echo "started       : $(date -Is)"

# Run ONE entry. Returns AF3's exit code; never aborts the surrounding loop.
run_entry() {
    local entry_id="$1"
    local input="$WORK_DIR/$INPUT_SUBDIR/$entry_id.json"
    local log="$PROF_DIR/$entry_id.log"
    local meta="$PROF_DIR/$entry_id.meta.json"

    if [[ ! -f "$input" ]]; then
        echo "ERROR: no input JSON: $input" >&2
        return 1
    fi

    # Apply USE_TEMPLATES for this entry.
    local input_used="$input" counts
    if [[ "$USE_TEMPLATES" == "0" ]]; then
        input_used="$SCRATCH/input/$entry_id.json"
        if ! counts=$(run_stripper --input "$input" --output "$input_used"); then
            echo "ERROR: stripping templates failed for $entry_id" >&2
            return 1
        fi
    else
        if ! counts=$(run_stripper --input "$input"); then
            echo "ERROR: counting templates failed for $entry_id" >&2
            return 1
        fi
    fi
    local n_in n_used n_entities n_with
    read -r n_in n_used n_entities n_with <<< "$counts"

    echo
    echo "--- $entry_id ---"
    echo "input       : $input"
    echo "input used  : $input_used"
    echo "templates   : use_templates=$USE_TEMPLATES  in_input=$n_in  used=$n_used  (entities=$n_entities, $n_with with templates)"
    echo "started     : $(date -Is)"

    local t_start t_end exit_code
    t_start=$(date +%s)
    set +e
    singularity exec --nv \
        --bind "$WORK_DIR:$WORK_DIR" \
        --bind "$OUT_DIR:$OUT_DIR" \
        --bind "$SCRATCH:$SCRATCH" \
        --env "TMPDIR=$SCRATCH/tmp" \
        "${MODEL_BIND[@]}" \
        "$IMAGE" \
        python /app/alphafold/run_alphafold.py \
            --json_path="$input_used" \
            --output_dir="$OUT_DIR" \
            --model_dir="$MODEL_DIR" \
            --cache_dir="$CACHE_DIR" \
            --force_output_dir \
            --num_recycles="$NUM_RECYCLES" \
            --num_diffusion_samples="$NUM_DIFFUSION_SAMPLES" \
            --norun_data_pipeline \
            "${OF3_FLAG[@]}" \
        2>&1 | tee "$log"
    exit_code=${PIPESTATUS[0]}
    set -e
    t_end=$(date +%s)

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
  "of3_weights": $([[ "$OF3_WEIGHTS" == "1" ]] && echo true || echo false),
  "engine": "af3",
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

    echo "finished    : $(date -Is)   exit=$exit_code   wall=$((t_end - t_start))s"
    return "$exit_code"
}

T_TASK_START=$(date +%s)
DONE=(); FAILED=(); SKIPPED=()
for idx in "${INDICES[@]}"; do
    ENTRY_ID=$(sed -n "${idx}p" "$TODO")
    if [[ -z "$ENTRY_ID" ]]; then
        echo "WARNING: line $idx of $TODO is empty, skipping" >&2
        continue
    fi
    # Stop cleanly rather than be killed part-way through writing an entry.
    if (( TIME_BUDGET_S > 0 )); then
        ELAPSED=$(( $(date +%s) - T_TASK_START ))
        if (( TIME_BUDGET_S - ELAPSED < PER_ENTRY_RESERVE_S )); then
            SKIPPED+=("$ENTRY_ID")
            continue
        fi
    fi
    if run_entry "$ENTRY_ID"; then DONE+=("$ENTRY_ID"); else FAILED+=("$ENTRY_ID"); fi
done

echo
echo "=== task $TASK_ID summary ==="
echo "wall          : $(( $(date +%s) - T_TASK_START ))s"
echo "ok            : ${#DONE[@]}${DONE[*]:+  ${DONE[*]}}"
echo "failed        : ${#FAILED[@]}${FAILED[*]:+  ${FAILED[*]}}"
echo "skipped       : ${#SKIPPED[@]}${SKIPPED[*]:+  ${SKIPPED[*]}}"
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo "NOTE: skipped entries ran out of time budget, not out of inputs — resubmit"
    echo "      them, or raise slurm.time / lower --chunk-size."
fi
[[ ${#FAILED[@]} -eq 0 ]] || exit 1
exit 0
