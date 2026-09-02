#!/usr/bin/env python3
"""
Stage 08 — copy the result tables into the repository for sharing.

The authoritative tables are written under --base-dir, outside the repository. This
stage copies them into <repo>/results/ so the repository shows what it produced,
dropping the columns that name the cluster this happened to run on:

    dockq.tsv       drops `model_path`   (absolute paths into the prediction tree)
    profiling.tsv   drops `node`, `slurm_job_id`   (compute host names, local job ids)

Nothing scientific is removed — every score, timing and status column is kept. The
dropped columns are only useful on the machine that produced them, and this
repository is public.

Usage:
    python scripts/08_publish_results.py \
        --results-dir <base>/results \
        --out-dir     results
"""

import argparse
import csv
import sys
from pathlib import Path

# table -> columns to omit from the shared copy
DROP_COLUMNS = {
    "dockq.tsv": ["model_path"],
    "profiling.tsv": ["node", "slurm_job_id"],
}


def publish(src: Path, dest: Path, drop: list[str]) -> tuple[int, list[str]]:
    with open(src) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        return 0, []
    columns = [c for c in rows[0] if c not in drop]
    dropped = [c for c in rows[0] if c in drop]
    with open(dest, "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(c, "")) for c in columns) + "\n")
    return len(rows), dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, type=Path,
                    help="<base>/results, holding dockq.tsv and profiling.tsv")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Destination inside the repository, normally results/")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for name, drop in DROP_COLUMNS.items():
        src = args.results_dir / name
        if not src.is_file():
            missing.append(name)
            continue
        n_rows, dropped = publish(src, args.out_dir / name, drop)
        print(f"{name}: {n_rows} rows -> {args.out_dir / name}"
              f"{'  (dropped ' + ', '.join(dropped) + ')' if dropped else ''}")

    if missing:
        print(f"\nnot found in {args.results_dir}: {', '.join(missing)}")
        print("Run --stage collect-profiling and --stage score first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
