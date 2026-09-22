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
#   WARMUP_RUNS (default 0: untimed runs of each size before the recorded one)
#   af3: MODEL_DIR OF3_WEIGHTS + IMAGE (production image) OR KIT_KIND KIT_IMAGE KIT_CONFIG KIT_MODE
#        KIT_VARIANT KIT_LEVERS_OFF KIT_ENV_VARS (Anthropic's af3_jax kit from its image; configs/af3kit_*.json)
#   of3: CKPT RUNNER_YAML, then EITHER OF3_SRC OF3_ENV (a source checkout in its own env)
#        OR KIT_KIND KIT_DIR KIT_ENV KIT_CONFIG KIT_MODE KIT_LEVERS_OFF KIT_CUDA_HOME KIT_LD KIT_ENV_VARS
#        (Anthropic's openfold3_ob0 kit; configs/of3kit_*.json, profiling/setup_kit_env.sh)

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
WARMUP_RUNS="${WARMUP_RUNS:-0}"
# AlphaFold3's arena as a fraction of the card. See AF3 MEMORY ENVIRONMENT above.
MEM_FRACTION="${MEM_FRACTION:-0.95}"

MANIFEST="$INPUT_DIR/pool_manifest.tsv"
[[ -f "$MANIFEST" ]] || { echo "ERROR: no such file: $MANIFEST — run profiling/build_pool_inputs.py first" >&2; exit 1; }

mkdir -p "$OUT_DIR"
TABLE="$OUT_DIR/vram_sweep.tsv"
COLUMNS="arm	engine	weights	use_templates	tokens	n_chains	status	peak_alloc_mib	peak_alloc_gib	peak_reserved_mib	peak_vram_mib	peak_vram_gib	gpu_total_mib	wall_s	model_s	featurisation_s	model_source	exit_code	num_diffusion_samples	num_recycles	mem_fraction	gpu_name	driver	node	slurm_job_id"
if [[ -f "$TABLE" ]]; then
    # A table written before featurisation_s/model_source existed holds model times that
    # mean something different for of3 (its predict loop, without featurisation). Appending
    # a wider row to it would put every later field under the wrong heading and mix two
    # definitions in one column, so the sweep stops and says what to do.
    if [[ "$(head -1 "$TABLE")" != "$COLUMNS" ]]; then
        echo "ERROR: $TABLE was written with a different column set." >&2
        echo "       Move it aside and re-sweep this arm:" >&2
        echo "         mkdir -p $OUT_DIR/_superseded && mv $TABLE $OUT_DIR/pool_* $OUT_DIR/_superseded/" >&2
        exit 1
    fi
else
    printf '%s\n' "$COLUMNS" > "$TABLE"
fi

# Node-local scratch, shared by every size this job runs — which is what keeps the
# compiled-kernel caches warm across the sweep, as in the two prediction launchers.
SCRATCH="${TMPDIR:-/tmp}/vram_sweep_${ARM}_${SLURM_JOB_ID:-0}"
mkdir -p "$SCRATCH/out" "$SCRATCH/tmp" "$SCRATCH/input" "$SCRATCH/torch_ext" \
         "$SCRATCH/triton" "$SCRATCH/alphafold_cache"
trap 'rm -rf "$SCRATCH"' EXIT

# The job's own cgroup (cgroup v2, as SLURM sets it up on these nodes): memory.events'
# oom_kill counter says whether the host OOM killer struck during a size, and memory.stat's
# anon line is the process-owned host memory (memory.current also counts reclaimable page
# cache -- the 2.3 GB checkpoint, the venv -- and reads ~29 GiB even for a small pool).
# Unreadable outside SLURM; every use below tolerates that.
CGROUP="/sys/fs/cgroup/system.slice/slurmstepd.scope/job_${SLURM_JOB_ID:-0}"
cg_oom_kills() { awk '$1=="oom_kill"{print $2}' "$CGROUP/memory.events" 2>/dev/null || echo 0; }
cg_anon_mib()  { awk '$1=="anon"{printf "%d", $2/1048576}' "$CGROUP/memory.stat" 2>/dev/null || echo ""; }

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unknown)

case "$ENGINE" in
af3)
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
    if [[ -n "${KIT_KIND:-}" ]]; then
        # Anthropic's af3_jax kit from its own container image (configs/af3kit_*.json). The image
        # (the kit's environment/Dockerfile, built by the fork's GitHub Action) holds the stock venv,
        # the kit venv first on PATH and the pinned checkout at /app/alphafold, so the host adds
        # nothing: no venv, no libstdc++, no PYTHONPATH. Caches go to node-local scratch -- the
        # image's /opt/jit_cache is empty and read-only, and run.sh refuses an NFS cache root. The
        # kit fills the three XLA variables only where unset, so the arena settings passed with
        # --env stay in force. launcher.env (KEY=VALUE;...) rides in as --env too.
        : "${KIT_IMAGE:?}" "${KIT_MODE:?}"
        [[ -f "$KIT_IMAGE" ]] || { echo "ERROR: kit image not found: $KIT_IMAGE" >&2; exit 1; }
        KIT_EXEC=(singularity exec --nv
                  --bind "$INPUT_DIR:$INPUT_DIR"
                  --bind "$SCRATCH:$SCRATCH"
                  --env "TMPDIR=$SCRATCH/tmp"
                  --env "XLA_PYTHON_CLIENT_PREALLOCATE=true"
                  --env "XLA_CLIENT_MEM_FRACTION=$MEM_FRACTION"
                  --env "AF3_JAX_CACHE_ROOT=$SCRATCH/af3_jax_cache"
                  --env "AF3_JAX_PARAMS_ROOT=$(dirname "$MODEL_DIR")"
                  "${MODEL_BIND[@]}")
        [[ -n "${KIT_LEVERS_OFF:-}" && "$KIT_MODE" != "off" ]] && KIT_EXEC+=(--env "MODEL_OPT_LEVERS_OFF=$KIT_LEVERS_OFF")
        if [[ -n "${KIT_ENV_VARS:-}" ]]; then IFS=';' read -ra _kv <<< "$KIT_ENV_VARS"; for kv in "${_kv[@]}"; do KIT_EXEC+=(--env "$kv"); done; fi
        KIT_EXEC+=("$KIT_IMAGE" bash /kit/af3_jax/run.sh)
        KIT_ARGS=(--config "${KIT_CONFIG:-a100}" --variant "${KIT_VARIANT:-p2}" --mode "$KIT_MODE")
        mkdir -p "$SCRATCH/af3_jax_cache" "$SCRATCH/tmp"
        # The kit's dry run: exit 3 = the mode cannot engage on this GPU/stack (NOT ACTIVE, named);
        # 0 = ACTIVE. Gate on that code -- the kit prints to stderr, so a grep of stdout is empty.
        if ! "${KIT_EXEC[@]}" check "${KIT_ARGS[@]}" > "$SCRATCH/kit_check.log" 2>&1; then
            cat "$SCRATCH/kit_check.log" >&2; echo "ERROR: the kit refuses mode $KIT_MODE here (above)" >&2; exit 1
        fi
        grep -E "ACTIVE mode=|WEIGHTS" "$SCRATCH/kit_check.log" | cut -c1-200
    else
        : "${IMAGE:?IMAGE must be set for engine af3 without a kit launcher}"
    fi
    ;;
of3)
    : "${CKPT:?CKPT must be set for engine of3}"
    : "${RUNNER_YAML:?RUNNER_YAML must be set for engine of3}"
    POOL_DIR="$INPUT_DIR/of3_pool"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TORCH_EXTENSIONS_DIR="$SCRATCH/torch_ext"
    export TRITON_CACHE_DIR="$SCRATCH/triton"
    if [[ -n "${KIT_KIND:-}" ]]; then
        # Anthropic's openfold3_ob0 kit (configs/of3kit_*.json). The kit's venv holds openfold3
        # 0.5.0 at the kit's pin plus the kit packages; the kit's own pin check confirms that
        # install file for file before any GPU time is spent. The card file is the kit's
        # environment bootstrap (MODEL_OPT, the stack key, the cache dirs under MODEL_OPT_JIT_ROOT);
        # it refuses with exit 3 when the package is not importable. KIT_CUDA_HOME and KIT_LD are
        # the two host workarounds profiling/setup_kit_env.sh explains (deviations 2 and 3).
        : "${KIT_DIR:?KIT_DIR must be set for the kit launcher}"
        : "${KIT_ENV:?KIT_ENV must be set for the kit launcher}"
        : "${KIT_MODE:?KIT_MODE must be set for the kit launcher}"
        export PATH="$KIT_ENV/bin:$PATH"
        [[ -n "${KIT_CUDA_HOME:-}" ]] && export CUDA_HOME="$KIT_CUDA_HOME"
        [[ -n "${KIT_LD:-}" ]] && export LD_LIBRARY_PATH="$KIT_LD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export MODEL_OPT_JIT_ROOT="$SCRATCH/jit"; mkdir -p "$MODEL_OPT_JIT_ROOT"
        export OPENFOLD3_OB0_CKPT="$CKPT"
        # The venv carries the kit's autoload hook: ANY python in it engages a kit mode when
        # OPENFOLD3_OB0_OPT is set. The mode is selected by --mode below, never by the environment.
        unset OPENFOLD3_OB0_OPT MODEL_OPT_LEVERS_OFF
        python -I "$KIT_DIR/stock/check_pins.py" --quiet || { echo "ERROR: the kit venv's openfold3 is not at the kit's pin (stock/check_pins.py above)" >&2; exit 1; }
        # shellcheck disable=SC1090
        source "$KIT_DIR/configs/${KIT_CONFIG:-h100}.env"
        [[ -n "${KIT_LEVERS_OFF:-}" && "$KIT_MODE" != "off" ]] && export MODEL_OPT_LEVERS_OFF="$KIT_LEVERS_OFF"
        # launcher.env from the config (KEY=VALUE;KEY=VALUE): the kit's own overrides, exported last so
        # they win over the card file's defaults. Used by of3kit_big_reachoff to hold the reach gate open.
        if [[ -n "${KIT_ENV_VARS:-}" ]]; then IFS=';' read -ra _kv <<< "$KIT_ENV_VARS"; for kv in "${_kv[@]}"; do export "$kv"; echo "kit env       : $kv"; done; fi
    else
        : "${OF3_SRC:?OF3_SRC must be set for engine of3 (or a launcher block)}"
        : "${OF3_ENV:?OF3_ENV must be set for engine of3 (or a launcher block)}"
        # Ported verbatim from slurm/predict_of3.sbatch — see its NON-OBVIOUS ENVIRONMENT
        # note for why each of these is needed on these nodes.
        if [[ ! -e "$OF3_ENV/bin/ptxas-blackwell" ]]; then
            ln -sf "$OF3_ENV/bin/ptxas" "$OF3_ENV/bin/ptxas-blackwell" || true
        fi
        export PATH="$OF3_ENV/bin:$PATH"
        export LD_LIBRARY_PATH="$OF3_ENV/lib:${LD_LIBRARY_PATH:-}"
        export CUDA_HOME="$OF3_ENV"
    fi
    ;;
*)
    echo "ERROR: unknown ENGINE '$ENGINE' (expected af3 or of3)" >&2; exit 1 ;;
esac
[[ -d "$POOL_DIR" ]] || { echo "ERROR: no such directory: $POOL_DIR" >&2; exit 1; }

echo "=== VRAM sweep: $ARM ==="
echo "engine        : $ENGINE   weights: $WEIGHTS   use_templates: $USE_TEMPLATES"
echo "inputs        : $POOL_DIR"
echo "table         : $TABLE"
echo "samples       : $NUM_DIFFUSION_SAMPLES   recycles: $NUM_RECYCLES   warmup runs: $WARMUP_RUNS"
[[ -n "${KIT_KIND:-}" ]] && echo "launcher      : $KIT_KIND  mode=$KIT_MODE  levers_off=${KIT_LEVERS_OFF:-${MODEL_OPT_LEVERS_OFF:-none}}  card file=${KIT_CONFIG:-h100}.env  kit=${KIT_IMAGE:-$KIT_DIR}"
[[ "$ENGINE" == "af3" ]] && echo "xla arena     : preallocate=true  mem_fraction=$MEM_FRACTION"
echo "node          : $(hostname)   gpu: $GPU_NAME   driver: $DRIVER"
echo "time budget   : ${TIME_BUDGET_S}s   per-size reserve: ${PER_SIZE_RESERVE_S}s"
echo "started       : $(date -Is)"

# Run ONE pool: trace it, append its row to the table. Never aborts the surrounding
# loop; the status is read back from results.json by the caller.
run_size() {
    local tokens="$1" n_chains="$2" warm="${3:-}"
    local name trace_dir status
    name=$(printf 'pool_%05daa' "$tokens")
    # A warm-up run keeps its trace in scratch and appends no row: it exists only to compile
    # and autotune the kernels the recorded run will then find in the caches.
    if [[ -n "$warm" ]]; then trace_dir="$SCRATCH/warm/$name"; else trace_dir="$OUT_DIR/$name"; fi
    mkdir -p "$trace_dir"

    echo
    echo "--- $tokens tokens ($n_chains chains)${warm:+  [warm-up $warm/$WARMUP_RUNS, not recorded]} ---"
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
        if [[ -n "${KIT_KIND:-}" ]]; then
            # The kit's entry point inside its image. The model runs in a child process peak_mem.py
            # cannot wrap; run_with_trace.py reads the kit's PEAK-NOTE line instead (same JAX counter).
            # --cache_dir is passed to every mode, off included (see the config's notes).
            cmd=("${KIT_EXEC[@]}" pred "${KIT_ARGS[@]}"
                 --model_dir="$MODEL_DIR"
                 --json_path="$input_used"
                 --output_dir="$SCRATCH/out"
                 --cache_dir="$SCRATCH/alphafold_cache"
                 --force_output_dir
                 --num_recycles="$NUM_RECYCLES"
                 --num_diffusion_samples="$NUM_DIFFUSION_SAMPLES"
                 --norun_data_pipeline)
        else
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
        fi
    else
        local query="$POOL_DIR/$name/query.json"
        [[ -f "$query" ]] || { echo "ERROR: no such query: $query" >&2; return 1; }
        local use_templates_bool
        use_templates_bool=$([[ "$USE_TEMPLATES" == "1" ]] && echo true || echo false)
        trace_args+=(--timing-glob "$SCRATCH/out/$name/seed_*/timing.json"
                     --summary "$SCRATCH/out/summary.txt")
        if [[ -n "${KIT_KIND:-}" && "$KIT_MODE" != "off" ]]; then
            # The kit's entry point: `python -m openfold3_ob0_opt pred --mode M`, which is what
            # the kit's run.sh execs after its own checks (done above, once). peak_mem.py wraps it
            # like any module; the kit prints its ACTIVE and LEVER lines into inference.log.
            cmd=(python "$REPO_DIR/profiling/peak_mem.py"
                 --out "$peak_json" --engine of3 --module openfold3_ob0_opt
                 --
                 pred --mode "$KIT_MODE"
                 --query-json            "$query"
                 --output-dir            "$SCRATCH/out"
                 --use-msa-server        false
                 --use-templates         "$use_templates_bool"
                 --runner-yaml           "$RUNNER_YAML"
                 --num-diffusion-samples "$NUM_DIFFUSION_SAMPLES")
        elif [[ -n "${KIT_KIND:-}" ]]; then
            # mode off: stock OpenFold3 as the kit defines it. The kit's own `pred --mode off`
            # runs stock in a clean subprocess that peak_mem.py cannot wrap, so upstream's entry
            # point is called directly on the kit's stock yaml (configs/of3kit_stock_runner.yml),
            # with the kit's mode variable unset so the venv's autoload hook stays inert.
            cmd=(env -u OPENFOLD3_OB0_OPT -u MODEL_OPT_LEVERS_OFF python
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
        else
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
    fi

    # </dev/null: the size loop reads the manifest on stdin, and a child process
    # that consumed it would silently truncate the sweep.
    local oom_before host_log host_peak oom_after
    oom_before=$(cg_oom_kills); host_log="$trace_dir/host_mem.log"; : > "$host_log"
    ( while :; do cg_anon_mib >> "$host_log"; echo >> "$host_log"; sleep 1; done ) 2>/dev/null &
    local host_sampler=$!
    set +e
    python3 "$REPO_DIR/profiling/run_with_trace.py" "${trace_args[@]}" -- "${cmd[@]}" </dev/null
    set -e
    kill "$host_sampler" 2>/dev/null || true; wait "$host_sampler" 2>/dev/null || true
    oom_after=$(cg_oom_kills); host_peak=$(sort -n "$host_log" 2>/dev/null | tail -1)
    # Fold the host side into results.json: peak anonymous host memory, and a definitive
    # cpu_oom when the cgroup's OOM-kill counter moved during this size (a SIGKILL alone
    # is ambiguous: the OOM killer and SLURM's limits both send it).
    if [[ -f "$trace_dir/results.json" ]]; then
        python3 - "$trace_dir/results.json" "${host_peak:-}" "${oom_before:-0}" "${oom_after:-0}" <<'PYEOF'
import json, sys
path, host_peak, before, after = sys.argv[1:5]
r = json.load(open(path))
r["peak_host_anon_mib"] = int(host_peak) if host_peak.strip().isdigit() else None
r["cgroup_oom_kills_during_size"] = int(after) - int(before)
if r["cgroup_oom_kills_during_size"] > 0 and r.get("status") in ("killed", "error"):
    r["status"] = "cpu_oom"
json.dump(r, open(path, "w"), indent=2)
PYEOF
    fi

    # OpenFold3 writes the traceback behind an "OOM for query_id" line to its own
    # <output>/logs/predict_err_rank0.log, not to stdout, and labels every caught
    # exception that way; without this copy a failed size is undiagnosable once the
    # scratch is gone.
    if [[ "$ENGINE" == "of3" && -d "$SCRATCH/out/logs" ]]; then
        cp -r "$SCRATCH/out/logs" "$trace_dir/of3_logs" 2>/dev/null || true
    fi
    # The structures are pools of identical chains and are never scored; only the
    # trace, the log and results.json are kept, so a 24-chain run does not leave
    # gigabytes behind.
    rm -rf "$SCRATCH/out"/* 2>/dev/null || true

    local results="$trace_dir/results.json"
    [[ -f "$results" ]] || { echo "ERROR: run_with_trace.py wrote no results.json" >&2; return 1; }
    if [[ -n "$warm" ]]; then
        echo "warm-up     : $(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(r.get('status'), 'model', r.get('model_time_s'), 's')" "$results")"
        return 0
    fi
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
       "" if r.get("featurisation_s") is None else f"{r['featurisation_s']:.1f}",
       str(r.get("model_source") or ""),
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

    for (( w = 1; w <= WARMUP_RUNS; w++ )); do run_size "$tokens" "$n_chains" "$w" || true; done
    run_size "$tokens" "$n_chains" || true
    STATUS=$(python3 -c "
import json, sys
try:
    print(json.load(open(sys.argv[1])).get('status', '?'))
except Exception:
    print('error')" "$OUT_DIR/$(printf 'pool_%05daa' "$tokens")/results.json")
    echo "finished    : $tokens tokens -> $STATUS   host anon peak $(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('peak_host_anon_mib'))" "$OUT_DIR/$(printf 'pool_%05daa' "$tokens")/results.json" 2>/dev/null) MiB   $(date -Is)"
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
