#!/usr/bin/env bash
#SBATCH --job-name=af3dockq
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#
# DockQ-score every prediction, as one CPU job — the login node is shared and this
# is 2 x <n entries> DockQ calls, the slowest of which are the many-chain complexes
# where DockQ searches chain permutations.
#
# ONE job, not an array: 05_score_dockq.py already parallelises over entries with a
# process pool, so --jobs is set to this job's own --cpus-per-task and the whole
# table is written by a single process. That keeps the output a single file with no
# merge step, which an array would need.
#
# Each worker shells out to `python -m DockQ` with BLAS pinned to one thread (the
# script sets that itself), so cpus-per-task is the real parallelism.
#
# Submitted by run_workflow.sh --stage submit-score, which supplies:
#   REPO_DIR DATASET_DIR PRED_ROOT OUT_TSV SUBSET SELECT_BY

set -euo pipefail

: "${REPO_DIR:?REPO_DIR must be set}"
: "${DATASET_DIR:?DATASET_DIR must be set}"
: "${PRED_ROOT:?PRED_ROOT must be set}"
: "${OUT_TSV:?OUT_TSV must be set}"
SUBSET="${SUBSET:-all}"
SELECT_BY="${SELECT_BY:-best}"
JOBS="${SLURM_CPUS_PER_TASK:-8}"

# Discover the arms from the prediction tree, exactly as the login-node stage does,
# so the job needs no list passed in and cannot disagree with what is on disk.
RUN_ARGS=()
for d in "$PRED_ROOT"/*/; do
    [[ -d "$d" ]] || continue
    RUN_ARGS+=(--run "$(basename "$d")" "${d%/}")
done
[[ ${#RUN_ARGS[@]} -gt 0 ]] || { echo "ERROR: no prediction dirs under $PRED_ROOT" >&2; exit 1; }

echo "=== DockQ scoring ==="
# RUN_ARGS holds three elements per arm (--run LABEL DIR).
echo "arms      : $(( ${#RUN_ARGS[@]} / 3 ))"
echo "subset    : $SUBSET   select-by: $SELECT_BY"
echo "jobs      : $JOBS (from --cpus-per-task)"
echo "out       : $OUT_TSV"
echo "node      : $(hostname)"
echo "started   : $(date -Is)"

# DockQ comes from the pixi analysis environment (pinned 2.1.3 in pixi.toml).
# PIXI_HOME is inherited via --export=ALL; the environments themselves live under
# /cluster/project, which compute nodes can read.
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
    --out "$OUT_TSV"

echo "finished  : $(date -Is)"
