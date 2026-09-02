#!/usr/bin/env bash
#SBATCH --job-name=af3msa
#SBATCH --output=%x_%A_%a.out
#SBATCH --ntasks=1
#
# AlphaFold3 data pipeline (MSA + template search) for ONE sequence — CPU only.
#
# Array job: one task per line of $WORK_DIR/msa_todo.txt, the gap list stage 02
# writes for sequences that were not found in any --msa-cache. Everything else
# in the store is a symlink into an existing cache and never reaches this script.
#
# The job runs run_alphafold.py with --norun_inference, so it needs the full AF3
# search databases ($DB_DIR) and no GPU. Its output, <name>_data.json, is gzipped
# into $WORK_DIR/msa_store/<seq_id>.json.gz — the same layout stage 03 reads for
# cache hits, so a computed entry and a reused one are interchangeable downstream.
#
# Resources and paths come in as environment variables from run_workflow.sh; the
# #SBATCH lines above hold only what does not vary. Submitted as:
#
#   sbatch --array=... --cpus-per-task=... --time=... --mem-per-cpu=... \
#          --export=ALL,WORK_DIR=...,IMAGE=...,DB_DIR=...,MAX_TEMPLATE_DATE=... \
#          slurm/msa.sbatch

set -euo pipefail

: "${WORK_DIR:?WORK_DIR must be set}"
: "${IMAGE:?IMAGE must be set}"
: "${DB_DIR:?DB_DIR must be set}"
MAX_TEMPLATE_DATE="${MAX_TEMPLATE_DATE:-2021-09-30}"
JACKHMMER_CPU="${JACKHMMER_CPU:-${SLURM_CPUS_PER_TASK:-8}}"

TODO="$WORK_DIR/msa_todo.txt"
[[ -f "$TODO" ]] || { echo "ERROR: no such file: $TODO" >&2; exit 1; }

SEQ_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TODO")
[[ -n "$SEQ_ID" ]] || { echo "ERROR: array index $SLURM_ARRAY_TASK_ID beyond $TODO" >&2; exit 1; }

INPUT="$WORK_DIR/msa_input/$SEQ_ID.json"
STORE="$WORK_DIR/msa_store"
# Node-local scratch, bound into the container AT ITS OWN PATH. Binding it somewhere
# else (e.g. /tmp/af3scratch) while still passing --output_dir=$SCRATCH/out makes AF3
# try to create /scratch/... inside the container, where /scratch is read-only:
#   OSError: [Errno 30] Read-only file system: '/scratch'
# TMPDIR is re-exported for the same reason the prediction job does it — tools the
# pipeline shells out to write temporaries there.
SCRATCH="${TMPDIR:-/tmp}/af3msa_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$STORE" "$SCRATCH/out" "$SCRATCH/tmp"
trap 'rm -rf "$SCRATCH"' EXIT

echo "=== AF3 data pipeline ==="
echo "seq_id      : $SEQ_ID"
echo "input       : $INPUT"
echo "image       : $IMAGE"
echo "db_dir      : $DB_DIR"
echo "template cut: $MAX_TEMPLATE_DATE"
echo "cpus        : $JACKHMMER_CPU"
echo "node        : $(hostname)"
echo "started     : $(date -Is)"

# --norun_inference: search only, no model, no GPU. The container is the same one
# the prediction stage uses, so the data JSON it emits is exactly what that stage
# expects to consume.
singularity exec \
    --bind "$WORK_DIR:$WORK_DIR" \
    --bind "$DB_DIR:$DB_DIR:ro" \
    --bind "$SCRATCH:$SCRATCH" \
    --env "TMPDIR=$SCRATCH/tmp" \
    "$IMAGE" \
    python /app/alphafold/run_alphafold.py \
        --json_path="$INPUT" \
        --output_dir="$SCRATCH/out" \
        --db_dir="$DB_DIR" \
        --max_template_date="$MAX_TEMPLATE_DATE" \
        --jackhmmer_n_cpu="$JACKHMMER_CPU" \
        --nhmmer_n_cpu="$JACKHMMER_CPU" \
        --norun_inference

# run_alphafold.py writes <output_dir>/<lowercased name>/<name>_data.json.
DATA_JSON=$(find "$SCRATCH/out" -name '*_data.json' -print -quit)
[[ -n "$DATA_JSON" ]] || { echo "ERROR: no *_data.json produced for $SEQ_ID" >&2; exit 1; }

# Write via a temp file then rename, so a killed job cannot leave a half-written
# entry that stage 03 would happily read.
gzip -c "$DATA_JSON" > "$STORE/.$SEQ_ID.json.gz.part"
mv "$STORE/.$SEQ_ID.json.gz.part" "$STORE/$SEQ_ID.json.gz"

echo "wrote       : $STORE/$SEQ_ID.json.gz ($(stat -c%s "$STORE/$SEQ_ID.json.gz") bytes)"
echo "finished    : $(date -Is)"
