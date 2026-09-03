#!/usr/bin/env bash
#SBATCH --job-name=af3dockq
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#
# DockQ-score predictions, as one CPU job — the login node is shared and this is up
# to <arms> x <entries> DockQ calls, the slowest of which are the many-chain
# complexes where DockQ searches chain permutations.
#
# ONE job, not an array: 05_score_dockq.py already parallelises over entries with a
# process pool, so --jobs is set to this job's own --cpus-per-task. Each worker
# shells out to `python -m DockQ` with BLAS pinned to one thread (the script sets
# that itself), so cpus-per-task is the real parallelism.
#
# WHAT IT SCORES — and what it deliberately does not
# --------------------------------------------------
# Results are stored ONE TABLE PER ARM under $OUT_DIR/<arm>.tsv, so:
#
#   ARMS unset   every arm under $PRED_ROOT is considered, but within each arm only
#                the entries not already recorded `ok` are scored. Re-running after
#                adding a handful of entries therefore costs a handful of DockQ
#                calls, not a full rescore.
#   ARMS set     a space-separated list of arm names; only those are touched. This
#                is what run_workflow.sh --stage submit-score --config <arm> sends,
#                and it is the cheap way to add one pipeline to an existing set.
#   RESCORE      non-empty re-runs entries already recorded ok, for when the scoring
#                itself changed rather than the predictions.
#
# Submitted by run_workflow.sh --stage submit-score, which supplies:
#   REPO_DIR DATASET_DIR PRED_ROOT OUT_DIR SUBSET SELECT_BY ARMS RESCORE

set -euo pipefail

: "${REPO_DIR:?REPO_DIR must be set}"
: "${DATASET_DIR:?DATASET_DIR must be set}"
: "${PRED_ROOT:?PRED_ROOT must be set}"
: "${OUT_DIR:?OUT_DIR must be set}"
SUBSET="${SUBSET:-all}"
SELECT_BY="${SELECT_BY:-best}"
ARMS="${ARMS:-}"
RESCORE="${RESCORE:-}"
JOBS="${SLURM_CPUS_PER_TASK:-8}"

# Discover the arms from the prediction tree, exactly as the login-node stage does,
# so the job needs no list passed in and cannot disagree with what is on disk.
# A non-empty $ARMS restricts that to the named arms.
RUN_ARGS=()
if [[ -n "$ARMS" ]]; then
    for tag in $ARMS; do
        d="$PRED_ROOT/$tag"
        [[ -d "$d" ]] || { echo "ERROR: no such arm directory: $d" >&2; exit 1; }
        RUN_ARGS+=(--run "$tag" "$d")
    done
else
    for d in "$PRED_ROOT"/*/; do
        [[ -d "$d" ]] || continue
        RUN_ARGS+=(--run "$(basename "$d")" "${d%/}")
    done
fi
[[ ${#RUN_ARGS[@]} -gt 0 ]] || { echo "ERROR: no prediction dirs under $PRED_ROOT" >&2; exit 1; }

mkdir -p "$OUT_DIR"

echo "=== DockQ scoring ==="
# RUN_ARGS holds three elements per arm (--run LABEL DIR).
echo "arms      : $(( ${#RUN_ARGS[@]} / 3 ))   ${ARMS:-(all under $PRED_ROOT)}"
echo "subset    : $SUBSET   select-by: $SELECT_BY"
echo "rescore   : ${RESCORE:-no (entries already ok are carried over)}"
echo "jobs      : $JOBS (from --cpus-per-task)"
echo "tables    : $OUT_DIR/<arm>.tsv"
echo "node      : $(hostname)"
echo "started   : $(date -Is)"

# DockQ comes from the pixi analysis environment (pinned 2.1.3 in pixi.toml).
# PIXI_HOME is inherited via --export=ALL, so the environment must live on shared
# storage that compute nodes can read (not a node-local or login-only path).
if command -v pixi >/dev/null 2>&1; then
    PY=(pixi run --manifest-path "$REPO_DIR/pixi.toml" python)
else
    echo "WARNING: pixi not on PATH; falling back to python3 (DockQ must be importable)" >&2
    PY=(python3)
fi

"${PY[@]}" "$REPO_DIR/scripts/05_score_dockq.py" \
    --dataset-dir "$DATASET_DIR" \
    "${RUN_ARGS[@]}" \
    --subset "$SUBSET" \
    --select-by "$SELECT_BY" \
    --jobs "$JOBS" \
    --out-dir "$OUT_DIR" \
    ${RESCORE:+--rescore}

echo "finished  : $(date -Is)"
