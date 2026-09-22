#!/usr/bin/env python3
"""
Stage 04 — collect per-complex inference time into one table.

Reads what slurm/predict.sbatch left in <run-dir>/_profiling/ for each entry:

    <entry>.log        AF3 stdout — its own timers
    <entry>.meta.json  node, GPU model, wall clock around the singularity call

Output goes to --out-dir/<label>.tsv, ONE TABLE PER ARM, matching scripts/05 — so
collecting one arm never rewrites another's table. No merged file is written;
scripts/06 and scripts/08 take a directory and concatenate. See
scripts/result_tables.py.

Columns written

    run, engine, weights, use_templates
                      The arm's axes, read from predictions/<run>/config.json, so a
                      timing table can never be read as comparing two arms that in
                      fact differed in their templates. See scripts/arm_metadata.py.
    n_templates       How many templates the model ACTUALLY received for this entry,
                      from the launcher's own count in meta.json — not from the
                      config's claim. Blank for runs whose launcher predates it.
    entry_id, category, n_chains, total_len
    featurisation_s   AF3's "Featurising data ... took X seconds".
                      Small here: MSAs and templates are already inline, so this is
                      tokenisation and feature assembly, not a database search.
    inference_s       AF3's "Running model inference and extracting output
                      structures ... took X seconds", over all diffusion samples.
                      EXCLUDES container startup and parameter loading. This is the
                      number to compare between the two weight arms.
    wall_s            wall clock around the singularity call, from meta.json.
                      INCLUDES container startup and parameter loading.
    startup_s         wall_s - model_s
    model_s           THE CROSS-ENGINE COMPARABLE NUMBER: the engine's own timer for
                      featurisation + inference of this query, excluding interpreter
                      or container startup and parameter loading.
                        af3   featurisation_s + inference_s from the AF3 log.
                        of3   runtime_s from OF3's per-seed timing.json, summed over
                              seeds, which covers its predict loop for the query.
                      NOT YET IDENTICAL in what they include. OpenFold3 builds its
                      features in the DataLoader, BEFORE the predict loop its timer
                      covers, so runtime_s leaves featurisation out while the AF3
                      number counts it — measured on the token sweep, the gap grows
                      with input size (55 s of it at 5000 tokens). This stage has not
                      been fixed yet; profiling/peak_mem.py now measures the OF3
                      featurisation directly (a predict-timeline callback) and
                      profiling/run_with_trace.py uses it, so the sweep's model_s is
                      like-for-like from its next run onwards. Benchmark rows are not,
                      and featurisation_s/inference_s stay blank for of3 here rather
                      than being invented.
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
        --out-dir <base>/results/profiling --sacct
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

from arm_metadata import ARM_COLUMNS, AXIS_COLUMNS, read_arm
from result_tables import arm_table_path, union_columns, write_rows

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
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory for the per-arm tables, <out-dir>/<label>.tsv. "
                         "One file per arm, matching scripts/05; see "
                         "scripts/result_tables.py.")
    ap.add_argument("--sacct", action="store_true",
                    help="Join SLURM accounting (billed elapsed, host MaxRSS, queue wait)")
    args = ap.parse_args()

    entries = {e["entry_id"]: e for e in read_tsv(args.dataset_dir / "entries.tsv")}

    rows: list[dict] = []
    job_ids: set[str] = set()
    mismatches: list[str] = []
    for label, run_dir_s in args.run:
        prof_dir = Path(run_dir_s) / "_profiling"
        if not prof_dir.is_dir():
            print(f"WARNING: no _profiling dir for run '{label}': {prof_dir}")
            continue
        arm = read_arm(Path(run_dir_s))
        metas = sorted(prof_dir.glob("*.meta.json"))
        print(f"{label}: {len(metas)} profiled entries in {prof_dir}"
              f"   [engine={arm['engine'] or '?'} weights={arm['weights'] or '?'} "
              f"templates={arm['use_templates'] or '?'}]")

        for meta_path in metas:
            entry_id = meta_path.name[: -len(".meta.json")]
            meta = json.loads(meta_path.read_text())
            entry = entries.get(entry_id, {})

            feat_s, infer_s = parse_log(prof_dir / f"{entry_id}.log")

            wall_s = meta.get("wall_s")
            if meta.get("engine") == "of3":
                # OF3 writes its own per-seed timer; its log carries no parseable
                # featurisation/inference split. See `model_s` in the docstring.
                of3_runtime = meta.get("of3_runtime_s")
                model_s = float(of3_runtime) if isinstance(of3_runtime, (int, float)) else 0.0
            else:
                model_s = (feat_s or 0.0) + (infer_s or 0.0)
            startup_s = (wall_s - model_s
                         if wall_s is not None and model_s > 0 else None)

            # What the model actually got, as counted by the launcher. AF3 reports
            # how many survived its templates switch; OF3 how many the query listed.
            n_templates = meta.get("n_templates_used")
            if n_templates is None:
                n_templates = meta.get("n_template_paths_in_query")

            # A per-entry template state that contradicts the arm's config means the
            # table cannot be trusted to say what it appears to say — surface it.
            entry_templates = meta.get("use_templates")
            if (isinstance(entry_templates, bool) and arm["use_templates"]
                    and str(entry_templates).lower() != arm["use_templates"]):
                mismatches.append(f"{label}/{entry_id}: meta says "
                                  f"use_templates={str(entry_templates).lower()}, "
                                  f"config says {arm['use_templates']}")

            job_id = str(meta.get("slurm_job_id") or "")
            if job_id:
                job_ids.add(job_id)

            rows.append({
                "run": label,
                "tag": arm["tag"],
                "engine": arm["engine"] or meta.get("engine", ""),
                "weights": arm["weights"],
                "use_templates": arm["use_templates"],
                "n_templates": "" if n_templates is None else n_templates,
                "entry_id": entry_id,
                "category": entry.get("category", ""),
                "n_chains": entry.get("n_chains", ""),
                "total_len": entry.get("total_len", ""),
                "featurisation_s": f"{feat_s:.1f}" if feat_s is not None else "",
                "inference_s": f"{infer_s:.1f}" if infer_s is not None else "",
                "wall_s": wall_s if wall_s is not None else "",
                "startup_s": f"{startup_s:.1f}" if startup_s is not None else "",
                "model_s": f"{model_s:.1f}" if model_s > 0 else "",
                "gpu_name": meta.get("gpu_name", ""),
                "node": meta.get("node", ""),
                "slurm_job_id": job_id,
                "exit_code": meta.get("exit_code", ""),
            })

    columns = ["run", *ARM_COLUMNS, "n_templates",
               "entry_id", "category", "n_chains", "total_len",
               "featurisation_s", "inference_s", "model_s", "wall_s", "startup_s",
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

    # One table per arm. Unlike stage 05 there is no resume: this only parses text
    # logs, so regenerating an arm costs seconds. Splitting the files still matters,
    # because it means collecting one arm does not rewrite another's table.
    print()
    for label, _ in args.run:
        arm_rows = sorted((r for r in rows if r["run"] == label),
                          key=lambda r: r["entry_id"])
        if not arm_rows:
            continue
        path = arm_table_path(args.out_dir, label)
        write_rows(path, union_columns(arm_rows, columns), arm_rows)
        print(f"Wrote {len(arm_rows)} rows to {path}")

    if mismatches:
        print(f"\nWARNING: {len(mismatches)} entry/entries whose recorded template "
              f"state contradicts their arm's config:")
        for line in mismatches[:20]:
            print(f"  {line}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")

    # Per-run headline, over rows where the number exists — never over a silent zero.
    for label, _ in args.run:
        subset = [r for r in rows if r["run"] == label]
        if not subset:
            continue
        model = sorted(float(r["model_s"]) for r in subset if r["model_s"])
        ok = sum(1 for r in subset if str(r["exit_code"]) == "0")
        arm_note = " ".join(f"{k}={subset[0].get(k) or '?'}" for k in AXIS_COLUMNS)
        print(f"\n{label}: {len(subset)} entries, {ok} exited 0   [{arm_note}]")
        if model:
            print(f"  model_s       median {model[len(model) // 2]:.1f}  "
                  f"min {model[0]:.1f}  max {model[-1]:.1f}  (n={len(model)})")
        gpus = {r["gpu_name"] for r in subset if r["gpu_name"]}
        print(f"  gpu(s): {', '.join(sorted(gpus))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
