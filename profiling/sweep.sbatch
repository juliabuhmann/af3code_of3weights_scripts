#!/usr/bin/env bash
#SBATCH --job-name=vram_sweep
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#
# VRAM-scaling sweep for ONE arm: hold one GPU, walk the token grid upwards, stop
# where it stops.
#
# ONE ALLOCATION, MANY SIZES. Every pool runs inside the SAME allocation, ascending,
# and the sweep ends at the first out-of-memory — so the boundary is found in one job
# rather than in a dozen queued ones. That is the same reasoning as CHUNK_STRIDE in
# slurm/predict.sbatch (per-task GPU allocation, not compute, dominates these jobs),
# and it has a second benefit here: every size is measured on the same physical card,
# so the curve cannot be an artefact of two different GPUs.
#
# Ascending, not descending, and the sweep does NOT continue past a failure: the point
# is the largest pool that RUNS, and everything above the boundary is uninformative
# GPU time. A row is appended per size as it completes, so a job that dies still
# leaves everything it measured, and re-running skips sizes already in the table.
#
# THE MEASUREMENT is two numbers per size, because neither one alone is both correct
# and comparable:
#   peak_alloc_mib   the framework's own live-buffer high-water — JAX
#                    peak_bytes_in_use, PyTorch max_memory_allocated — read by
#                    profiling/peak_mem.py, which wraps the engine's entry point. THIS
#                    is the curve: it is the same quantity for both engines and it does
#                    not depend on how much the allocator reserved from the driver.
#   peak_vram_mib    nvidia-smi polled once a second (profiling/run_with_trace.py).
#                    Whole-device, so it is only meaningful with the card to itself,
#                    and under AF3 preallocation it just reports the arena. Kept as the
#                    record of what the card actually had to hand over.
#
# AF3 MEMORY ENVIRONMENT — matched to production, which is load-bearing:
#   XLA_PYTHON_CLIENT_PREALLOCATE=true, XLA_CLIENT_MEM_FRACTION=0.95 is what the
#       openbind runs use (pooled_ppi_comparison/scripts/2026aug_af3_openbind_weights).
#       Do NOT "improve" this by turning preallocation off to make nvidia-smi follow
#       demand: that caps the arena at JAX's default 0.75 fraction AND leaves BFC
#       growing in regions, so a single large block (41.6 GiB at 4000 tokens) cannot be
#       placed. Measured: 4000 tokens OOMs with preallocation off and runs in 476 s
#       with it on. Set MEM_FRACTION to change the fraction.
#       XLA_PYTHON_CLIENT_MEM_FRACTION is NOT set — jaxlib rejects both at once.
#   TF_FORCE_UNIFIED_MEMORY is left UNSET. AlphaFold3 recommends it above ~5120
#       tokens, and it works by spilling to host RAM — which would turn the GPU
#       boundary this sweep is looking for into a much slower run that never OOMs.
# AF3 also pads token counts to its default buckets (512, 1024, ... 5120), so its
# curve is a staircase and a 500-token pool is really a 512-token forward pass.
#
# Runs as an sbatch script, or directly inside an allocation that already has a GPU:
#   ARM=... ENGINE=... [see below] bash profiling/sweep.sbatch
#
# Submitted by profiling/run_sweep.sh, which supplies:
#   ARM ENGINE WEIGHTS USE_TEMPLATES INPUT_DIR OUT_DIR REPO_DIR
#   NUM_DIFFUSION_SAMPLES NUM_RECYCLES TIME_BUDGET_S PER_SIZE_RESERVE_S MAX_TOKENS
#   MEM_FRACTION (af3, default 0.95)
#   af3: IMAGE MODEL_DIR OF3_WEIGHTS
#   of3: OF3_SRC OF3_ENV CKPT RUNNER_YAML

set -euo pipefail

module load eth_proxy 2>/dev/null || true

: "${ARM:?ARM must be set}"
: "${ENGINE:?ENGINE must be set (af3|of3)}"
: "${INPUT_DIR:?INPUT_DIR must be set (the pool input root)}"
: "${OUT_DIR:?OUT_DIR must be set}"
: "${REPO_DIR:?REPO_DIR must be set}"
WEIGHTS="${WEIGHTS:-}"
USE_TEMPLATES="${USE_TEMPLATES:-0}"
NUM_DIFFUSION_SAMPLES="${NUM_DIFFUSION_SAMPLES:-5}"
NUM_RECYCLES="${NUM_RECYCLES:-3}"
TIME_BUDGET_S="${TIME_BUDGET_S:-0}"
PER_SIZE_RESERVE_S="${PER_SIZE_RESERVE_S:-1800}"
MAX_TOKENS="${MAX_TOKENS:-0}"
# AlphaFold3's arena as a fraction of the card. See AF3 MEMORY ENVIRONMENT above.
MEM_FRACTION="${MEM_FRACTION:-0.95}"

MANIFEST="$INPUT_DIR/pool_manifest.tsv"
[[ -f "$MANIFEST" ]] || { echo "ERROR: no such file: $MANIFEST — run profiling/build_pool_inputs.py first" >&2; exit 1; }

mkdir -p "$OUT_DIR"
TABLE="$OUT_DIR/vram_sweep.tsv"
COLUMNS="arm	engine	weights	use_templates	tokens	n_chains	status	peak_alloc_mib	peak_alloc_gib	peak_reserved_mib	peak_vram_mib	peak_vram_gib	gpu_total_mib	wall_s	model_s	exit_code	num_diffusion_samples	num_recycles	mem_fraction	gpu_name	driver	node	slurm_job_id"
[[ -f "$TABLE" ]] || printf '%s\n' "$COLUMNS" > "$TABLE"

# Node-local scratch, shared by every size this job runs — which is what keeps the
# compiled-kernel caches warm across the sweep, as in the two prediction launchers.
SCRATCH="${TMPDIR:-/tmp}/vram_sweep_${ARM}_${SLURM_JOB_ID:-0}"
mkdir -p "$SCRATCH/out" "$SCRATCH/tmp" "$SCRATCH/input" "$SCRATCH/torch_ext" \
         "$SCRATCH/triton" "$SCRATCH/alphafold_cache"
trap 'rm -rf "$SCRATCH"' EXIT

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unknown)

case "$ENGINE" in
af3)
    : "${IMAGE:?IMAGE must be set for engine af3}"
    : "${MODEL_DIR:?MODEL_DIR must be set for engine af3}"
    OF3_WEIGHTS="${OF3_WEIGHTS:-0}"
    POOL_DIR="$INPUT_DIR/af3_pool"
    # See AF3 MEMORY ENVIRONMENT above.
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_CLIENT_MEM_FRACTION="$MEM_FRACTION"
    OF3_FLAG=()
    [[ "$OF3_WEIGHTS" == "1" ]] && OF3_FLAG=(--of3_weights)
    MODEL_BIND=()
    [[ -d "$MODEL_DIR" && "$MODEL_DIR" != /models ]] && MODEL_BIND=(--bind "$MODEL_DIR:$MODEL_DIR:ro")
    STRIPPER="$REPO_DIR/slurm/strip_templates.py"
    [[ -f "$STRIPPER" ]] || { echo "ERROR: no such file: $STRIPPER" >&2; exit 1; }
    cp "$REPO_DIR/profiling/peak_mem.py" "$SCRATCH/peak_mem.py"
    ;;
of3)
    : "${OF3_SRC:?OF3_SRC must be set for engine of3}"
    : "${OF3_ENV:?OF3_ENV must be set for engine of3}"
    : "${CKPT:?CKPT must be set for engine of3}"
    : "${RUNNER_YAML:?RUNNER_YAML must be set for engine of3}"
    POOL_DIR="$INPUT_DIR/of3_pool"
    # Ported verbatim from slurm/predict_of3.sbatch — see its NON-OBVIOUS ENVIRONMENT
    # note for why each of these is needed on these nodes.
    if [[ ! -e "$OF3_ENV/bin/ptxas-blackwell" ]]; then
        ln -sf "$OF3_ENV/bin/ptxas" "$OF3_ENV/bin/ptxas-blackwell" || true
    fi
    export PATH="$OF3_ENV/bin:$PATH"
    export LD_LIBRARY_PATH="$OF3_ENV/lib:${LD_LIBRARY_PATH:-}"
    export CUDA_HOME="$OF3_ENV"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TORCH_EXTENSIONS_DIR="$SCRATCH/torch_ext"
    export TRITON_CACHE_DIR="$SCRATCH/triton"
    ;;
*)
    echo "ERROR: unknown ENGINE '$ENGINE' (expected af3 or of3)" >&2; exit 1 ;;
esac
[[ -d "$POOL_DIR" ]] || { echo "ERROR: no such directory: $POOL_DIR" >&2; exit 1; }

echo "=== VRAM sweep: $ARM ==="
echo "engine        : $ENGINE   weights: $WEIGHTS   use_templates: $USE_TEMPLATES"
echo "inputs        : $POOL_DIR"
echo "table         : $TABLE"
echo "samples       : $NUM_DIFFUSION_SAMPLES   recycles: $NUM_RECYCLES"
[[ "$ENGINE" == "af3" ]] && echo "xla arena     : preallocate=true  mem_fraction=$MEM_FRACTION"
echo "node          : $(hostname)   gpu: $GPU_NAME   driver: $DRIVER"
echo "time budget   : ${TIME_BUDGET_S}s   per-size reserve: ${PER_SIZE_RESERVE_S}s"
echo "started       : $(date -Is)"

# Run ONE pool: trace it, append its row to the table. Never aborts the surrounding
# loop; the status is read back from results.json by the caller.
run_size() {
    local tokens="$1" n_chains="$2"
    local name trace_dir status
    name=$(printf 'pool_%05daa' "$tokens")
    trace_dir="$OUT_DIR/$name"
    mkdir -p "$trace_dir"

    echo
    echo "--- $tokens tokens ($n_chains chains) ---"
    echo "started     : $(date -Is)"

    # Removed before the run, never after: a size that dies before the counter can be
    # read must report a missing peak, not the previous size's.
    local peak_json="$SCRATCH/peak_mem.json"
    rm -f "$peak_json"

    local -a trace_args=(--trace-dir "$trace_dir" --engine "$ENGINE"
                         --peak-mem-json "$peak_json")
    local -a cmd=()

    if [[ "$ENGINE" == "af3" ]]; then
        local input="$POOL_DIR/$name.json" input_used counts
        [[ -f "$input" ]] || { echo "ERROR: no such input: $input" >&2; return 1; }
        input_used="$input"
        if [[ "$USE_TEMPLATES" == "0" ]]; then
            # Same mechanism as slurm/predict.sbatch: AlphaFold3 has no inference-time
            # template flag, so templates-off is a `templates: []` copy of the input.
            input_used="$SCRATCH/input/$name.json"
            if ! counts=$(python3 "$STRIPPER" --input "$input" --output "$input_used"); then
                echo "ERROR: stripping templates failed for $name" >&2; return 1
            fi
            echo "templates   : stripped ($counts)"
        fi
        cmd=(singularity exec --nv
             --bind "$INPUT_DIR:$INPUT_DIR"
             --bind "$SCRATCH:$SCRATCH"
             --env "TMPDIR=$SCRATCH/tmp"
             --env "XLA_PYTHON_CLIENT_PREALLOCATE=true"
             --env "XLA_CLIENT_MEM_FRACTION=$MEM_FRACTION"
             "${MODEL_BIND[@]}"
             "$IMAGE"
             python "$SCRATCH/peak_mem.py"
                 --out "$peak_json" --engine af3
                 --script /app/alphafold/run_alphafold.py
                 --
                 --json_path="$input_used"
                 --output_dir="$SCRATCH/out"
                 --model_dir="$MODEL_DIR"
                 --cache_dir="$SCRATCH/alphafold_cache"
                 --force_output_dir
                 --num_recycles="$NUM_RECYCLES"
                 --num_diffusion_samples="$NUM_DIFFUSION_SAMPLES"
                 --norun_data_pipeline
                 "${OF3_FLAG[@]}")
    else
        local query="$POOL_DIR/$name/query.json"
        [[ -f "$query" ]] || { echo "ERROR: no such query: $query" >&2; return 1; }
        local use_templates_bool
        use_templates_bool=$([[ "$USE_TEMPLATES" == "1" ]] && echo true || echo false)
        trace_args+=(--timing-glob "$SCRATCH/out/$name/seed_*/timing.json"
                     --summary "$SCRATCH/out/summary.txt")
        cmd=(env "PYTHONPATH=$OF3_SRC" "$OF3_ENV/bin/python"
             "$REPO_DIR/profiling/peak_mem.py"
             --out "$peak_json" --engine of3 --module openfold3.run_openfold
             --
             predict
             --query-json            "$query"
             --use-msa-server=false
             --use-templates="$use_templates_bool"
             --runner-yaml           "$RUNNER_YAML"
             --inference-ckpt-path   "$CKPT"
             --output-dir            "$SCRATCH/out"
             --num-diffusion-samples "$NUM_DIFFUSION_SAMPLES")
    fi

    # </dev/null: the size loop reads the manifest on stdin, and a child process
    # that consumed it would silently truncate the sweep.
    set +e
    python3 "$REPO_DIR/profiling/run_with_trace.py" "${trace_args[@]}" -- "${cmd[@]}" </dev/null
    set -e

    # The structures are pools of identical chains and are never scored; only the
    # trace, the log and results.json are kept, so a 24-chain run does not leave
    # gigabytes behind.
    rm -rf "$SCRATCH/out"/* 2>/dev/null || true

    local results="$trace_dir/results.json"
    [[ -f "$results" ]] || { echo "ERROR: run_with_trace.py wrote no results.json" >&2; return 1; }
    python3 - "$results" "$TABLE" "$ARM" "$ENGINE" "$WEIGHTS" "$USE_TEMPLATES" \
             "$tokens" "$n_chains" "$NUM_DIFFUSION_SAMPLES" "$NUM_RECYCLES" \
             "$MEM_FRACTION" "$GPU_NAME" "$DRIVER" "$(hostname)" <<'ROWEOF'
import json, sys
(results, table, arm, engine, weights, use_templates, tokens, n_chains,
 samples, recycles, mem_fraction, gpu, driver, node) = sys.argv[1:15]
r = json.load(open(results))


def mib(key):
    value = r.get(key)
    return str(value) if isinstance(value, (int, float)) else ""


def gib(key):
    value = r.get(key)
    return f"{value / 1024:.2f}" if isinstance(value, (int, float)) and value else ""


row = [arm, engine, weights,
       "true" if use_templates == "1" else "false",
       tokens, n_chains, r.get("status", "?"),
       mib("peak_alloc_mib"), gib("peak_alloc_mib"), mib("peak_reserved_mib"),
       mib("peak_vram_mib"), gib("peak_vram_mib"),
       str(r.get("gpu_total_mib") or ""),
       str(r.get("wall_time_s") or ""),
       "" if r.get("model_time_s") is None else f"{r['model_time_s']:.1f}",
       str(r.get("exit_code", "")), samples, recycles,
       mem_fraction if engine == "af3" else "",
       gpu, driver, node, r.get("slurm_job_id", "")]
with open(table, "a") as fh:
    fh.write("\t".join(row) + "\n")
ROWEOF
}

T_START=$(date +%s)
LAST_OK=0
BOUNDARY=""
STOPPED=""
while IFS=$'\t' read -r name tokens n_chains rest; do
    [[ "$name" == "name" ]] && continue          # header
    [[ -n "$tokens" ]] || continue
    if (( MAX_TOKENS > 0 && tokens > MAX_TOKENS )); then continue; fi

    # Resume: a size already in the table is not re-run.
    if awk -F'\t' -v t="$tokens" 'NR>1 && $5==t {found=1} END {exit !found}' "$TABLE"; then
        echo "--- $tokens tokens: already in $TABLE, skipping ---"
        continue
    fi
    if (( TIME_BUDGET_S > 0 )); then
        ELAPSED=$(( $(date +%s) - T_START ))
        if (( TIME_BUDGET_S - ELAPSED < PER_SIZE_RESERVE_S )); then
            STOPPED="time budget: stopped before $tokens tokens with $(( TIME_BUDGET_S - ELAPSED ))s left"
            break
        fi
    fi

    run_size "$tokens" "$n_chains" || true
    STATUS=$(python3 -c "
import json, sys
try:
    print(json.load(open(sys.argv[1])).get('status', '?'))
except Exception:
    print('error')" "$OUT_DIR/$(printf 'pool_%05daa' "$tokens")/results.json")
    echo "finished    : $tokens tokens -> $STATUS   $(date -Is)"
    if [[ "$STATUS" == "ok" ]]; then
        LAST_OK="$tokens"
    else
        BOUNDARY="$tokens"
        STOPPED="stopped at $tokens tokens with status $STATUS"
        break
    fi
done < <(sort -t$'\t' -k2,2n "$MANIFEST")

echo
echo "=== $ARM summary ==="
echo "wall          : $(( $(date +%s) - T_START ))s"
echo "largest ok    : ${LAST_OK} tokens"
[[ -n "$BOUNDARY" ]] && echo "first failure : ${BOUNDARY} tokens"
[[ -n "$STOPPED" ]]  && echo "note          : $STOPPED"
echo "table         : $TABLE"
exit 0
