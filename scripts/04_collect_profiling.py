#!/usr/bin/env python3
"""
Stage 04 — collect per-complex inference time into one table.

Reads what slurm/predict.sbatch left in <run-dir>/_profiling/ for each entry:

    <entry>.log        AF3 stdout — its own timers
    <entry>.meta.json  node, GPU model, wall clock around the singularity call

Columns written (--out, default <run-dir>/profiling.tsv)

    run, entry_id, category, n_chains, total_len
    featurisation_s   AF3's "Featurising data ... took X seconds".
                      Small here: MSAs and templates are already inline, so this is
                      tokenisation and feature assembly, not a database search.
    inference_s       AF3's "Running model inference and extracting output
                      structures ... took X seconds", over all diffusion samples.
                      EXCLUDES container startup and parameter loading. This is the
                      number to compare between the two weight arms.
    wall_s            wall clock around the singularity call, from meta.json.
                      INCLUDES container startup and parameter loading.
    startup_s         wall_s - (featurisation_s + inference_s)
    gpu_name, node, exit_code
                      gpu_name is carried on every row because a timing comparison
                      across different GPU models would be meaningless.

--sacct joins SLURM's own accounting (elapsed, MaxRSS = host RAM, queue wait). Those
records are pruned after a few months, so run it while the jobs are recent.

Usage:
    python scripts/04_collect_profiling.py \
        --dataset-dir <base>/dataset \
        --run af3_native   <base>/predictions/af3_native \
        --run af3_openbind <base>/predictions/af3_openbind \
        --out <base>/results/profiling.tsv --sacct
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

# AF3's aggregate timers. The seed clause wording varies between container
# versions, so both spellings are accepted; the per-seed lines ("with seed 1")
# deliberately do not match, only the aggregate.
_SEED_CLAUSE = r"(?:for seeds \([^)]*\)|with \d+ seed\(s\))"
FEATURISE_RE = re.compile(rf"Featurising data {_SEED_CLAUSE} took\s+([\d.]+) seconds")
INFERENCE_RE = re.compile(
    rf"Running model inference and extracting output structures {_SEED_CLAUSE} "
    rf"took\s+([\d.]+) seconds")

MIB_PER_GIB = 1024.0


def read_tsv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def parse_log(path: Path) -> tuple[float | None, float | None]:
    if not path.is_file():
        return None, None
    text = path.read_text(errors="replace")
    feat = FEATURISE_RE.search(text)
    infer = INFERENCE_RE.search(text)
    return (float(feat.group(1)) if feat else None,
            float(infer.group(1)) if infer else None)


def sacct_lookup(job_ids: set[str]) -> dict[str, dict]:
    """job id -> {elapsed_s, max_rss_gib, queue_wait_s, state} via sacct."""
    if not job_ids:
        return {}
    fields = "JobID,State,Submit,Start,End,Elapsed,MaxRSS,ReqMem,AllocTRES"
    try:
        out = subprocess.check_output(
            ["sacct", "-n", "-P", "--units=M", f"--format={fields}",
             "-j", ",".join(sorted(job_ids))],
            text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"WARNING: sacct unavailable ({exc}); skipping --sacct columns")
        return {}

    from datetime import datetime

    def ts(value: str) -> float | None:
        try:
            return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").timestamp()
        except ValueError:
            return None

    def hms(value: str) -> float | None:
        parts = value.split("-")
        days = int(parts[0]) if len(parts) == 2 else 0
        clock = parts[-1].split(":")
        try:
            nums = [float(x) for x in clock]
        except ValueError:
            return None
        while len(nums) < 3:
            nums.insert(0, 0.0)
        return days * 86400 + nums[0] * 3600 + nums[1] * 60 + nums[2]

    result: dict[str, dict] = {}
    for line in out.strip().splitlines():
        cols = line.split("|")
        if len(cols) < 9:
            continue
        job_id, state, submit, start, end, elapsed, max_rss = cols[:7]
        base = job_id.split(".")[0]
        record = result.setdefault(base, {})
        if "." not in job_id:   # the parent step carries the timestamps
            record["state"] = state
            record["elapsed_s"] = hms(elapsed)
            t_submit, t_start = ts(submit), ts(start)
            record["queue_wait_s"] = (t_start - t_submit
                                      if t_submit and t_start else None)
        if max_rss.endswith("M"):
            mib = float(max_rss[:-1])
            record["max_rss_gib"] = max(record.get("max_rss_gib") or 0.0,
                                        mib / MIB_PER_GIB)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "DIR"),
                    required=True, help="Run label and its prediction directory "
                                        "(repeatable, one per weight arm)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sacct", action="store_true",
                    help="Join SLURM accounting (billed elapsed, host MaxRSS, queue wait)")
    args = ap.parse_args()

    entries = {e["entry_id"]: e for e in read_tsv(args.dataset_dir / "entries.tsv")}

    rows: list[dict] = []
    job_ids: set[str] = set()
    for label, run_dir_s in args.run:
        prof_dir = Path(run_dir_s) / "_profiling"
        if not prof_dir.is_dir():
            print(f"WARNING: no _profiling dir for run '{label}': {prof_dir}")
            continue
        metas = sorted(prof_dir.glob("*.meta.json"))
        print(f"{label}: {len(metas)} profiled entries in {prof_dir}")

        for meta_path in metas:
            entry_id = meta_path.name[: -len(".meta.json")]
            meta = json.loads(meta_path.read_text())
            entry = entries.get(entry_id, {})

            feat_s, infer_s = parse_log(prof_dir / f"{entry_id}.log")

            wall_s = meta.get("wall_s")
            model_s = (feat_s or 0.0) + (infer_s or 0.0)
            startup_s = (wall_s - model_s
                         if wall_s is not None and model_s > 0 else None)

            job_id = str(meta.get("slurm_job_id") or "")
            if job_id:
                job_ids.add(job_id)

            rows.append({
                "run": label,
                "entry_id": entry_id,
                "category": entry.get("category", ""),
                "n_chains": entry.get("n_chains", ""),
                "total_len": entry.get("total_len", ""),
                "featurisation_s": f"{feat_s:.1f}" if feat_s is not None else "",
                "inference_s": f"{infer_s:.1f}" if infer_s is not None else "",
                "wall_s": wall_s if wall_s is not None else "",
                "startup_s": f"{startup_s:.1f}" if startup_s is not None else "",
                "gpu_name": meta.get("gpu_name", ""),
                "node": meta.get("node", ""),
                "slurm_job_id": job_id,
                "exit_code": meta.get("exit_code", ""),
            })

    columns = ["run", "entry_id", "category", "n_chains", "total_len",
               "featurisation_s", "inference_s", "wall_s", "startup_s",
               "gpu_name", "node", "slurm_job_id", "exit_code"]

    if args.sacct:
        acct = sacct_lookup(job_ids)
        if acct:
            columns += ["billed_elapsed_s", "queue_wait_s", "max_rss_gib", "slurm_state"]
            for row in rows:
                record = acct.get(row["slurm_job_id"], {})
                row["billed_elapsed_s"] = (f"{record['elapsed_s']:.0f}"
                                           if record.get("elapsed_s") else "")
                row["queue_wait_s"] = (f"{record['queue_wait_s']:.0f}"
                                       if record.get("queue_wait_s") else "")
                row["max_rss_gib"] = (f"{record['max_rss_gib']:.2f}"
                                      if record.get("max_rss_gib") else "")
                row["slurm_state"] = record.get("state", "")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(c, "")) for c in columns) + "\n")

    print(f"\nWrote {len(rows)} rows to {args.out}")

    # Per-run headline, over rows where the number exists — never over a silent zero.
    for label, _ in args.run:
        subset = [r for r in rows if r["run"] == label]
        if not subset:
            continue
        infer = sorted(float(r["inference_s"]) for r in subset if r["inference_s"])
        ok = sum(1 for r in subset if str(r["exit_code"]) == "0")
        print(f"\n{label}: {len(subset)} entries, {ok} exited 0")
        if infer:
            print(f"  inference_s   median {infer[len(infer) // 2]:.1f}  "
                  f"min {infer[0]:.1f}  max {infer[-1]:.1f}  (n={len(infer)})")
        gpus = {r["gpu_name"] for r in subset if r["gpu_name"]}
        print(f"  gpu(s): {', '.join(sorted(gpus))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
