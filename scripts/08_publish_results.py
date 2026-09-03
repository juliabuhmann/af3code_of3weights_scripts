#!/usr/bin/env python3
"""
Stage 08 — flatten the per-arm result tables into the repository for sharing.

The authoritative tables live under --results-dir, outside the repository, ONE PER
ARM (see scripts/result_tables.py):

    <base>/results/dockq/<arm>.tsv
    <base>/results/profiling/<arm>.tsv

This stage concatenates each set into the single table the repository publishes,
dropping the columns that name the cluster this happened to run on:

    dockq.tsv       drops `model_path`   (absolute paths into the prediction tree)
    profiling.tsv   drops `node`, `slurm_job_id`   (compute host names, local job ids)

Nothing scientific is removed — every score, timing and status column is kept, and so
are the pipeline axes (engine, weights, use_templates) that say which arm each row
came from. The dropped columns are only useful on the machine that produced them, and
this repository is public.

Concatenating HERE rather than keeping a merged file on disk means the published table
cannot be a stale version of the per-arm tables it summarises. Columns are the union
across arms, so an arm scored before a column existed does not truncate the others.

A per-arm summary is printed, because the whole point of the split is to make it
obvious which pipelines are present and how complete each one is.

Usage:
    python scripts/08_publish_results.py \
        --results-dir <base>/results \
        --out-dir     results
"""

import argparse
import sys
from pathlib import Path

from arm_metadata import AXIS_COLUMNS
from result_tables import read_rows, union_columns, write_rows

# published table -> (per-arm subdirectory, columns to omit from the shared copy)
TABLES = {
    "dockq.tsv": ("dockq", ["model_path"]),
    "profiling.tsv": ("profiling", ["node", "slurm_job_id"]),
}


def publish(src_dir: Path, dest: Path, drop: list[str]) -> tuple[int, list[str], dict]:
    rows = read_rows(src_dir)
    if not rows:
        return 0, [], {}
    columns = [c for c in union_columns(rows) if c not in drop]
    dropped = [c for c in union_columns(rows) if c in drop]
    # Stable order so a re-publish with no new data produces no diff.
    rows.sort(key=lambda r: (r.get("run", ""), r.get("entry_id", "")))
    write_rows(dest, columns, rows)

    # "Good" means different things in the two tables: dockq rows carry a `status`,
    # profiling rows carry the launcher's `exit_code`. Counting rows and calling it
    # ok would report a validity check that had not been made.
    if "status" in columns:
        good, good_of = lambda r: r.get("status") == "ok", "ok"
    elif "exit_code" in columns:
        good, good_of = lambda r: str(r.get("exit_code")) == "0", "exit0"
    else:
        good, good_of = lambda r: True, "rows"

    per_arm: dict[str, dict] = {}
    for row in rows:
        arm = per_arm.setdefault(row.get("run", "?"),
                                 {"n": 0, "good": 0, "good_of": good_of,
                                  **{k: row.get(k, "") for k in AXIS_COLUMNS}})
        arm["n"] += 1
        if good(row):
            arm["good"] += 1
    return len(rows), dropped, per_arm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, type=Path,
                    help="<base>/results, holding the dockq/ and profiling/ "
                         "per-arm table directories")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Destination inside the repository, normally results/")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for name, (subdir, drop) in TABLES.items():
        src = args.results_dir / subdir
        if not src.is_dir():
            missing.append(f"{subdir}/")
            continue
        n_rows, dropped, per_arm = publish(src, args.out_dir / name, drop)
        if not n_rows:
            missing.append(f"{subdir}/ (no .tsv files)")
            continue
        n_files = len(sorted(src.glob("*.tsv")))
        print(f"{name}: {n_rows} rows from {n_files} arm table(s) "
              f"-> {args.out_dir / name}"
              f"{'  (dropped ' + ', '.join(dropped) + ')' if dropped else ''}")
        for arm, info in sorted(per_arm.items()):
            axes = " ".join(f"{k}={info[k] or '?'}" for k in AXIS_COLUMNS)
            print(f"    {arm:28s} {info['good']:4}/{info['n']:<4} "
                  f"{info['good_of']:<6} {axes}")

    if missing:
        print(f"\nnot found in {args.results_dir}: {', '.join(missing)}")
        print("Run --stage collect-profiling and --stage score first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
