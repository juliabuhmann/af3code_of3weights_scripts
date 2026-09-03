#!/usr/bin/env bash
#SBATCH --job-name=of3pred
#SBATCH --output=%x_%A_%a.out
#SBATCH --ntasks=1
#
# OpenFold3 inference for one or more complexes, from the todo list (default
# $WORK_DIR/of3_todo.txt) — the OF3 counterpart of slurm/predict.sbatch.
#
# Inputs are the query JSONs written by scripts/09_build_of3_inputs.py from the very
# same stage-03 AF3 input JSONs the AF3 arms consume, so the alignments are shared
# across engines and nothing is recomputed here.
#
# ONE ALLOCATION, MANY ENTRIES — CHUNK_STRIDE
# -------------------------------------------
# Same mechanism and the same reasoning as predict.sbatch:
#   CHUNK_STRIDE unset/0   run only the entry at line $SLURM_ARRAY_TASK_ID.
#   CHUNK_STRIDE=T         run lines $SLURM_ARRAY_TASK_ID, +T, +2T, ...
# Stride rather than a contiguous block, because of3_todo.txt is sorted
# largest-complex-first and contiguous blocks would concentrate the big complexes in
# the first task.
#
# Chunking pays off MORE here than on the AF3 side. Measured on 8x8t: of3_runtime_s
# 31.9 against wall_s 117 — roughly three quarters of a single-entry task is
# interpreter start, checkpoint load and Triton compilation. Because
# TRITON_CACHE_DIR and TORCH_EXTENSIONS_DIR below are created once per TASK, every
# entry after the first in a chunk reuses that compilation.
#
# Per-entry outputs are unchanged — one <entry>.log, <entry>.meta.json and
# <entry>.summary.txt each — so stages 04 and 05 need no knowledge of chunking.
# Failure isolation and the TIME_BUDGET_S guard also work as in predict.sbatch.
#
# TEMPLATES — USE_TEMPLATES=1|0
# -----------------------------
# Unlike AlphaFold3, OpenFold3 has a real flag: --use-templates true|false. It is
# passed straight through from the arm's config. What it cannot do is protect you
# from a mismatch, because OF3 discards templates SILENTLY — it will happily report
# `Preprocessing templates: 100%` and exit 0 having used none. So each query JSON is
# checked against the declared setting first (slurm/of3_query_info.py), before any
# GPU time is spent, and the template count is recorded in meta.json.
#
# scripts/09_build_of3_inputs.py converts MSAs only, so its queries carry no
# template_cif_paths and USE_TEMPLATES=0 is the only setting they satisfy.
#
# OUTPUT DIRECTORY — WHY A SCRATCH DETOUR
# ---------------------------------------
# OF3 writes summary.txt (and, when enabled, template_logs/) to the ROOT of its
# --output-dir. Pointed at the shared arm directory those root-level files would race
# between concurrent tasks, and the per-query failure check below would be reading
# someone else's result. So every entry runs into its OWN node-local output directory
# and is copied into $OUT_DIR when finished, with summary.txt kept per entry under
# _profiling/. This also keeps OF3's incremental writes off shared storage.
#
# NON-OBVIOUS ENVIRONMENT — both ported from
# ~/src/pooled_ppi_comparison/slurm/run_yeast_of3_bw.sh, the version known to run on
# these nodes:
#   ptxas-blackwell        Triton looks for a ptxas named with the arch suffix; the
#                          pixi env only ships plain `ptxas`, so a symlink is made.
#   expandable_segments    The confidence-scoring step allocates a large N_atom^2
#                          distance matrix that OOMs on a fragmented heap even with
#                          plenty of free VRAM. A/B-tested on a borderline pool in
#                          the yeast benchmark: default allocator 0/5 samples,
#                          expandable segments 5/5.
#
# RECYCLES. OF3 exposes no recycle flag; num_recycles is fixed at 3 in its
# architecture config, which matches what the AF3 arms pass. Nothing to set here.
#
# Submitted by run_workflow.sh, which supplies resources and these variables:
#   WORK_DIR OUT_DIR REPO_DIR OF3_SRC OF3_ENV CKPT RUNNER_YAML
#   NUM_DIFFUSION_SAMPLES USE_TEMPLATES OF3_INPUT_SUBDIR TODO_FILE
#   CHUNK_STRIDE TIME_BUDGET_S PER_ENTRY_RESERVE_S

set -euo pipefail

module load eth_proxy 2>/dev/null || true

: "${WORK_DIR:?WORK_DIR must be set}"
: "${OUT_DIR:?OUT_DIR must be set}"
: "${OF3_SRC:?OF3_SRC must be set}"
: "${OF3_ENV:?OF3_ENV must be set}"
: "${CKPT:?CKPT must be set}"
: "${RUNNER_YAML:?RUNNER_YAML must be set}"
: "${REPO_DIR:?REPO_DIR must be set}"
NUM_DIFFUSION_SAMPLES="${NUM_DIFFUSION_SAMPLES:-5}"
USE_TEMPLATES="${USE_TEMPLATES:-0}"
OF3_INPUT_SUBDIR="${OF3_INPUT_SUBDIR:-of3_input}"
TODO="${TODO_FILE:-$WORK_DIR/of3_todo.txt}"
CHUNK_STRIDE="${CHUNK_STRIDE:-0}"
TIME_BUDGET_S="${TIME_BUDGET_S:-0}"
PER_ENTRY_RESERVE_S="${PER_ENTRY_RESERVE_S:-1200}"

[[ -f "$TODO" ]]        || { echo "ERROR: no such file: $TODO" >&2; exit 1; }
[[ -d "$OF3_ENV" ]]     || { echo "ERROR: no such pixi env: $OF3_ENV" >&2; exit 1; }
[[ -f "$CKPT" ]]        || { echo "ERROR: no such checkpoint: $CKPT" >&2; exit 1; }
[[ -f "$RUNNER_YAML" ]] || { echo "ERROR: no such runner yaml: $RUNNER_YAML" >&2; exit 1; }

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

# Node-local scratch for this TASK. The Torch/Triton caches live here and are shared
# by every entry the task runs, which is where the chunking win comes from.
SCRATCH="${TMPDIR:-/tmp}/of3pred_${SLURM_JOB_ID:-0}_${TASK_ID}"
mkdir -p "$SCRATCH/out" "$SCRATCH/torch_ext" "$SCRATCH/triton"
trap 'rm -rf "$SCRATCH"' EXIT

# Create the arch-suffixed ptxas Triton looks for, but only if absent: this path is
# INSIDE the shared OF3 env, so with many concurrent array tasks an unconditional
# `ln -sf` (unlink + symlink) can race and kill a task under `set -e`. Idempotent
# check first, and a failure here is not fatal — a concurrent task made the same link.
if [[ ! -e "$OF3_ENV/bin/ptxas-blackwell" ]]; then
    ln -sf "$OF3_ENV/bin/ptxas" "$OF3_ENV/bin/ptxas-blackwell" || true
fi

export PATH="$OF3_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$OF3_ENV/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="$OF3_ENV"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_EXTENSIONS_DIR="$SCRATCH/torch_ext"
export TRITON_CACHE_DIR="$SCRATCH/triton"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
USE_TEMPLATES_BOOL=$([[ "$USE_TEMPLATES" == "1" ]] && echo true || echo false)

echo "=== OF3 inference ==="
echo "task          : array id $TASK_ID, ${#INDICES[@]} entry/entries (stride ${CHUNK_STRIDE:-0} over $N_TODO)"
echo "entries       : $(for i in "${INDICES[@]}"; do printf '%s ' "$(sed -n "${i}p" "$TODO")"; done)"
echo "use_templates : $USE_TEMPLATES_BOOL"
echo "ckpt          : $CKPT"
echo "runner yaml   : $RUNNER_YAML"
echo "samples       : $NUM_DIFFUSION_SAMPLES"
echo "out_dir       : $OUT_DIR   (via $SCRATCH/out)"
echo "node          : $(hostname)   gpu: $GPU_NAME   driver: $DRIVER"
echo "time budget   : ${TIME_BUDGET_S}s   per-entry reserve: ${PER_ENTRY_RESERVE_S}s"
echo "started       : $(date -Is)"

# Run ONE entry. Returns non-zero on failure; never aborts the surrounding loop.
run_entry() {
    local entry_id="$1"
    local query="$WORK_DIR/$OF3_INPUT_SUBDIR/$entry_id/query.json"
    local log="$PROF_DIR/$entry_id.log"
    local meta="$PROF_DIR/$entry_id.meta.json"
    local scratch_out="$SCRATCH/out/$entry_id"

    if [[ ! -f "$query" ]]; then
        echo "ERROR: no query JSON: $query" >&2
        return 1
    fi

    # Verify the query matches the arm's declared template setting BEFORE GPU work.
    local info
    if ! info=$("$OF3_ENV/bin/python" "$REPO_DIR/slurm/of3_query_info.py" \
                    --query "$query" --require-templates "$USE_TEMPLATES_BOOL"); then
        echo "ERROR: query check failed for $entry_id" >&2
        return 1
    fi
    local n_queries n_chains n_template_paths n_msa_paths seeds
    read -r n_queries n_chains n_template_paths n_msa_paths seeds <<< "$info"

    mkdir -p "$scratch_out"
    echo
    echo "--- $entry_id ---"
    echo "query       : $query"
    echo "query holds : $n_queries query, $n_chains chains, $n_msa_paths msa file(s), $n_template_paths template cif(s), seeds $seeds"
    echo "started     : $(date -Is)"

    local t_start t_end exit_code
    t_start=$(date +%s)
    set +e
    PYTHONPATH="$OF3_SRC" "$OF3_ENV/bin/python" -m openfold3.run_openfold predict \
        --query-json            "$query" \
        --use-msa-server=false \
        --use-templates="$USE_TEMPLATES_BOOL" \
        --runner-yaml           "$RUNNER_YAML" \
        --inference-ckpt-path   "$CKPT" \
        --output-dir            "$scratch_out" \
        --num-diffusion-samples "$NUM_DIFFUSION_SAMPLES" \
        2>&1 | tee "$log"
    exit_code=${PIPESTATUS[0]}
    set -e
    t_end=$(date +%s)

    # OF3 exits 0 even when every query fails during featurisation: the process-level
    # status covers the harness only, while per-query failures are reported in
    # summary.txt. Without this check a total failure looks like a successful run.
    if [[ -f "$scratch_out/summary.txt" ]]; then
        cp "$scratch_out/summary.txt" "$PROF_DIR/$entry_id.summary.txt"
        if grep -qE "Failed Queries:[[:space:]]+[1-9]" "$scratch_out/summary.txt"; then
            echo "ERROR: OF3 reported failed queries for $entry_id despite exit $exit_code:" >&2
            sed -n "/PREDICTION SUMMARY/,/^===/p" "$scratch_out/summary.txt" >&2
            [[ "$exit_code" == "0" ]] && exit_code=2
        fi
    fi

    # Template selection logs, when the runner yaml enables create_logs. Kept because
    # they are the ONLY evidence that a templates-on OF3 run used any template.
    if [[ -d "$scratch_out/template_logs" ]]; then
        rm -rf "$PROF_DIR/${entry_id}_template_logs"
        cp -r "$scratch_out/template_logs" "$PROF_DIR/${entry_id}_template_logs"
    fi

    # OF3's own per-seed timer, summed over seeds. This is the number comparable to
    # AlphaFold3's featurisation+inference: it covers the predict loop for this query
    # and excludes interpreter startup and checkpoint loading.
    local runtime_s
    runtime_s=$("$OF3_ENV/bin/python" -c "
import glob, json, sys
values = [json.load(open(p)).get('runtime_s') for p in glob.glob(sys.argv[1])]
values = [v for v in values if isinstance(v, (int, float))]
print(f'{sum(values):.3f}' if values else '')
" "$scratch_out/$entry_id/seed_*/timing.json")

    # Move the finished entry onto shared storage. The entry directory is replaced
    # wholesale so a retry after a failure cannot leave a half-written mixture of two
    # attempts — the same reasoning as --force_output_dir on the AF3 side.
    local n_models=0
    if [[ -d "$scratch_out/$entry_id" ]]; then
        rm -rf "$OUT_DIR/$entry_id"
        cp -r "$scratch_out/$entry_id" "$OUT_DIR/$entry_id"
        n_models=$(find "$OUT_DIR/$entry_id" -name '*_model.cif' | wc -l)
    else
        echo "WARNING: OF3 wrote no directory for $entry_id under $scratch_out" >&2
        [[ "$exit_code" == "0" ]] && exit_code=3
    fi
    # Reclaim the node-local copy now: a whole chunk of these would fill $TMPDIR.
    rm -rf "$scratch_out"

    cat > "$meta" <<JSON
{
  "entry_id": "$entry_id",
  "engine": "of3",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "slurm_array_job_id": "${SLURM_ARRAY_JOB_ID:-}",
  "slurm_array_task_id": "${TASK_ID}",
  "node": "$(hostname)",
  "gpu_name": "$GPU_NAME",
  "driver_version": "$DRIVER",
  "ckpt": "$CKPT",
  "runner_yaml": "$RUNNER_YAML",
  "use_templates": $USE_TEMPLATES_BOOL,
  "n_template_paths_in_query": $n_template_paths,
  "n_msa_paths_in_query": $n_msa_paths,
  "n_chains": $n_chains,
  "seeds": "$seeds",
  "num_diffusion_samples": $NUM_DIFFUSION_SAMPLES,
  "n_model_cifs": $n_models,
  "of3_runtime_s": ${runtime_s:-null},
  "chunk_stride": $CHUNK_STRIDE,
  "wall_start_epoch_s": $t_start,
  "wall_end_epoch_s": $t_end,
  "wall_s": $((t_end - t_start)),
  "exit_code": $exit_code
}
JSON

    echo "finished    : $(date -Is)   exit=$exit_code   wall=$((t_end - t_start))s   of3_runtime=${runtime_s:-?}s   models=$n_models"
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
