#!/usr/bin/env python3
"""
Stage 05 — DockQ-score each predicted complex against its Benchmark-IV native.

Reference structure is `cleanNative.pdb` from the Zenodo archive — the same file
Zheng & Peng scored against, so numbers here are on the paper's own scale.

Scoring goes through the DockQ CLI (`python -m DockQ`) rather than the Python API:
the CLI is what applies DockQ's chain-permutation search, which is what makes a
homodimer's A-B interface score correctly regardless of which predicted chain
corresponds to which native chain. 113 of the 131 two-chain entries are homodimers,
so that is not a corner case here.

Chain ids need no mapping: stage 01 assigns them positionally (A, B, C, ... in FASTA
order), which is how cleanNative.pdb is labelled, and verifies the correspondence by
sequence. Its `chains_match` column is carried into the output here so an entry where
that broke down stays visible rather than silently mis-scored.

Sample selection (--select-by), over the --num-diffusion-samples AF3 wrote:
    best      highest `ranking_score` — AF3's own top model. Default.
    maxiptm   highest `iptm` from the sample's summary_confidences.json.
Both are recorded in the output either way, so the choice can be revisited without
re-scoring.

ONE TABLE PER ARM, AND RESUME
-----------------------------
Output goes to --out-dir/<label>.tsv, one file per arm, NOT to a single combined
table. Scoring an arm therefore never rescores the others: adding a fifth arm costs
that arm's entries, not 5 x 206 DockQ calls. See scripts/result_tables.py.

Within an arm, entries already recorded with status `ok` are carried over from the
existing table and not re-run; entries that are missing, or whose status was a
timeout or a missing prediction, ARE retried. --rescore forces everything. So the
cost of a re-run is the work that is genuinely outstanding.

What is carried over is the MEASUREMENT. The label columns (tag and the pipeline
axes) are re-derived from the arm's config on every pass, because they describe the
arm rather than the scoring: renaming an arm's tag, or adding a label column that did
not exist when the table was written, then costs one cheap pass with zero DockQ calls
instead of a full rescore.

Nothing writes a merged file: scripts/06 and scripts/08 accept a directory and
concatenate on the fly, so no stale merge can contradict the per-arm tables.

Output columns, one row per (run, entry):
    run, engine, weights, use_templates,
        The arm's axes, read from predictions/<run>/config.json (see
        scripts/arm_metadata.py). Carried on every row so a DockQ table cannot be
        read as comparing two arms that in fact differed in their templates — which
        is exactly the mistake available when the arm names do not say so.
    entry_id, category, n_chains, total_len, chains_match,
    select_by, sample, ranking_score, iptm, ptm,
    dockq, dockq_min_interface, fnat, irmsd, lrmsd, f1, n_interfaces,
    model_path, status

`dockq` is DockQ v2's interface MEAN, so it is comparable across chain counts;
`fnat`/`irmsd`/`lrmsd`/`f1` are interface means too. `dockq_min_interface` is the
worst single interface, which the mean hides — a 4-chain complex can average well
while getting one interface badly wrong.

An entry with no prediction, or whose DockQ call fails, gets an explicit `status`
and is never silently dropped.

Usage:
    pixi run python scripts/05_score_dockq.py \
        --dataset-dir <base>/dataset \
        --run af3_native   <base>/predictions/af3_native \
        --run af3_openbind <base>/predictions/af3_openbind \
        --out-dir <base>/results/dockq --jobs 8

    # just the one new arm — the others keep their tables untouched
    pixi run python scripts/05_score_dockq.py \
        --dataset-dir <base>/dataset \
        --run of3_openbind_templateoff <base>/predictions/of3_openbind_templateoff \
        --out-dir <base>/results/dockq --subset all --jobs 32
"""

import os

# One BLAS thread per DockQ subprocess: we parallelise over entries ourselves, and
# numpy inside each worker would otherwise start ncpu threads apiece.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from arm_metadata import ARM_COLUMNS, AXIS_COLUMNS, read_arm
from result_tables import arm_table_path, load_scored, union_columns, write_rows

# DockQ v2 prints one block per native interface plus a total line.
#
# THE TOTAL IS THE MEAN over native interfaces, not a sum — verified against the
# tool on a 3-interface complex (8ee7): per-interface 0.619 / 0.804 / 0.804, total
# reported as 0.742 = their mean. So `dockq` is on the 0-1 scale for every entry
# and is directly comparable between a dimer and an 11-chain assembly.
#
# This is also exactly the convention Zheng & Peng use: their methods state DockQ
# was computed with "the package provided by [52], with default options", where
# [52] is DockQ v2 (Mirabello & Wallner 2024, Bioinformatics) — the package pinned
# in pixi.toml. Numbers here are therefore on the paper's scale.
TOTAL_RE = re.compile(r"Total DockQ over (\d+) native interface[s]?:\s*([\d.]+)")
# Per-interface metrics. ALL blocks are collected and averaged, so these are
# interface means, consistent with how `dockq` is defined above. For a dimer there
# is one block and the mean is that block.
METRIC_RES = {
    "fnat": re.compile(r"^\s*fnat:\s*([\d.]+)", re.MULTILINE),
    "irmsd": re.compile(r"^\s*iRMSD:\s*([\d.]+)", re.MULTILINE),
    "lrmsd": re.compile(r"^\s*LRMSD:\s*([\d.]+)", re.MULTILINE),
    "f1": re.compile(r"^\s*F1:\s*([\d.]+)", re.MULTILINE),
}
PER_INTERFACE_DOCKQ_RE = re.compile(r"^\s*DockQ:\s*([\d.]+)", re.MULTILINE)


def read_tsv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def find_samples(entry_dir: Path) -> list[tuple[str, Path, Path]]:
    """[(sample name, model .cif, summary_confidences.json)] for one AF3 output dir.

    AlphaFold3 writes each diffusion sample to seed-<N>_sample-<i>/, alongside a
    top-level copy of the best-ranked one. We enumerate the per-sample directories
    so that --select-by can choose, rather than trusting the top-level copy.
    """
    samples = []
    for sub in sorted(entry_dir.glob("seed-*_sample-*")):
        # Files inside a sample directory are prefixed with the fold-job name, e.g.
        # seed-42_sample-0/8x8t_seed-42_sample-0_model.cif — hence the glob rather
        # than a fixed filename.
        models = sorted(sub.glob("*_model.cif"))
        if not models:
            continue
        model = models[0]
        summary = model.parent / model.name.replace("_model.cif",
                                                    "_summary_confidences.json")
        samples.append((sub.name, model, summary))
    if not samples:
        # OpenFold3 layout: seed_<N>/<name>_seed_<N>_sample_<k>_model.cif. Note the
        # underscores (AF3 hyphenates), the extra directory level, 1-indexed sample
        # numbers, and confidences in *_confidences_aggregated.json.
        for sub in sorted(entry_dir.glob("seed_*")):
            for model in sorted(sub.glob("*_model.cif")):
                stem = model.name[: -len("_model.cif")]
                samples.append(
                    (stem, model, sub / f"{stem}_confidences_aggregated.json"))
    if not samples:
        # Flat layout (single sample, or an older container): the top-level copy.
        for model in sorted(entry_dir.glob("*_model.cif")):
            stem = model.name[: -len("_model.cif")]
            samples.append((stem, model, entry_dir / f"{stem}_summary_confidences.json"))
    return samples


def sample_scores(summary_path: Path) -> dict:
    if not summary_path.is_file():
        return {}
    try:
        data = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    scores = {k: data.get(k) for k in ("ranking_score", "iptm", "ptm")}
    # OpenFold3 calls it sample_ranking_score; iptm and ptm carry the same names.
    if scores["ranking_score"] is None:
        scores["ranking_score"] = data.get("sample_ranking_score")
    return scores


def pick_sample(samples: list[tuple[str, Path, Path]], select_by: str):
    """Return (name, model, scores). Ties and missing scores fall back to order."""
    scored = []
    for name, model, summary in samples:
        scores = sample_scores(summary)
        key = scores.get("ranking_score" if select_by == "best" else "iptm")
        scored.append((key if isinstance(key, (int, float)) else float("-inf"),
                       name, model, scores))
    scored.sort(key=lambda t: (-t[0], t[1]))
    _, name, model, scores = scored[0]
    return name, model, scores


def cif_with_occupancy(cif_text: str) -> str | None:
    """Add an `_atom_site.occupancy` column (all 1.00) if the mmCIF lacks one.

    OpenFold3's model.cif omits occupancy, which DockQ's Biopython MMCIFParser
    requires unconditionally (`mmcif_dict["_atom_site.occupancy"]`), so every OF3
    model otherwise dies with KeyError: '_atom_site.occupancy'. mmCIF loop columns
    are positional, so appending the tag after the last _atom_site.* header and one
    value per ATOM/HETATM row keeps them aligned.

    Returns None when nothing needs doing, so AF3 models are scored from their
    original file and this cannot perturb them. Ported from
    ~/src/pooled_ppi_comparison/benchmarks/zheng_dockq/04_analyze_dockq.py, where
    it was verified to reproduce scores on models the plain path already handled.
    """
    lines = cif_text.splitlines()
    if any(line.strip() == "_atom_site.occupancy" for line in lines):
        return None
    tag_idx = [i for i, line in enumerate(lines)
               if line.lstrip().startswith("_atom_site.")]
    if not tag_idx:
        return None
    last = tag_idx[-1]
    out = []
    for i, line in enumerate(lines):
        out.append(line)
        if i == last:
            out.append("_atom_site.occupancy")
    return "\n".join(
        (line.rstrip() + " 1.00") if line.lstrip().startswith(("ATOM ", "HETATM "))
        else line
        for line in out
    ) + "\n"


def run_dockq(dockq_cmd: list[str], model: Path, native: Path,
              timeout: int) -> tuple[dict, str]:
    # OF3 models need an injected occupancy column; the patched copy is written
    # beside nothing permanent, and AF3 models skip this entirely.
    tmp_model = None
    try:
        patched = cif_with_occupancy(model.read_text())
    except OSError:
        patched = None
    if patched is not None:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".cif", delete=False)
        tmp.write(patched)
        tmp.close()
        tmp_model = Path(tmp.name)
        model = tmp_model

    try:
        try:
            proc = subprocess.run(dockq_cmd + [str(model), str(native)],
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {}, f"dockq_timeout_{timeout}s"
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            tail = detail[-1][:160] if detail else "no output"
            return {}, f"dockq_failed:{tail}"

        out = proc.stdout
        result: dict = {}
        total = TOTAL_RE.search(out)
        if total:
            result["n_interfaces"] = int(total.group(1))
            result["dockq"] = float(total.group(2))

        per_interface = [float(v) for v in PER_INTERFACE_DOCKQ_RE.findall(out)]
        for key, pattern in METRIC_RES.items():
            values = [float(v) for v in pattern.findall(out)]
            if values:
                result[key] = sum(values) / len(values)

        if "dockq" not in result:
            # Single-interface runs of some versions omit the total line; fall back
            # to the one per-interface block rather than dropping the entry.
            if len(per_interface) == 1:
                result["dockq"] = per_interface[0]
                result["n_interfaces"] = 1
            else:
                return {}, "dockq_unparsed"
        result["dockq_min_interface"] = min(per_interface) if per_interface else ""
        return result, "ok"
    finally:
        if tmp_model is not None:
            tmp_model.unlink(missing_ok=True)


def score_one(task: dict) -> dict:
    entry = task["entry"]
    row = {
        "run": task["run"],
        **{key: task["arm"][key] for key in ARM_COLUMNS},
        "entry_id": entry["entry_id"],
        "category": entry["category"],
        "n_chains": entry["n_chains"],
        "total_len": entry["total_len"],
        "chains_match": entry["chains_match"],
        "select_by": task["select_by"],
        "sample": "", "ranking_score": "", "iptm": "", "ptm": "",
        "dockq": "", "dockq_min_interface": "",
        "fnat": "", "irmsd": "", "lrmsd": "", "f1": "",
        "n_interfaces": "", "model_path": "", "status": "",
    }

    native = Path(entry["native_pdb"]) if entry["native_pdb"] else None
    if native is None or not native.is_file():
        row["status"] = "no_native"
        return row

    # AF3 lowercases the input `name` for its output directory.
    entry_dir = Path(task["run_dir"]) / entry["entry_id"].lower()
    if not entry_dir.is_dir():
        entry_dir = Path(task["run_dir"]) / entry["entry_id"]
    if not entry_dir.is_dir():
        row["status"] = "no_prediction"
        return row

    samples = find_samples(entry_dir)
    if not samples:
        row["status"] = "no_samples"
        return row

    name, model, scores = pick_sample(samples, task["select_by"])
    row["sample"] = name
    row["model_path"] = str(model)
    for key in ("ranking_score", "iptm", "ptm"):
        value = scores.get(key)
        if isinstance(value, (int, float)):
            row[key] = f"{value:.4f}"

    metrics, status = run_dockq(task["dockq_cmd"], model, native, task["timeout"])
    row["status"] = status
    for key, value in metrics.items():
        row[key] = f"{value:.4f}" if isinstance(value, float) else value
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "DIR"),
                    required=True)
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Directory for the per-arm tables, <out-dir>/<label>.tsv. "
                         "One file per arm, so scoring one never rescores the rest.")
    ap.add_argument("--rescore", action="store_true",
                    help="Re-run DockQ for entries already recorded ok. Without it, "
                         "those are carried over from the arm's existing table and "
                         "only missing or failed entries are scored.")
    ap.add_argument("--select-by", choices=["best", "maxiptm"], default="best")
    ap.add_argument("--subset", choices=["dimers", "hetero", "all"], default="dimers",
                    help="Which entries to score. dimers (default): every 2-chain entry — one "
                         "interface per complex, so DockQ needs no averaging and the number means "
                         "the same thing on every row. hetero: heterodimer + heteromer, the entries "
                         "whose chains are not all one sequence. all: every entry of the dataset, "
                         "including the 3-to-11-chain homomers and heteromers, whose DockQ is a mean "
                         "over native interfaces.")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--timeout", type=int, default=1800,
                    help="Per-entry DockQ timeout in seconds")
    ap.add_argument("--dockq-cmd", default=f"{sys.executable} -m DockQ",
                    help="How to invoke DockQ (default: this interpreter's -m DockQ)")
    args = ap.parse_args()

    dockq_cmd = args.dockq_cmd.split()
    try:
        subprocess.run(dockq_cmd + ["--help"], capture_output=True, timeout=120, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"DockQ not runnable via '{args.dockq_cmd}': {exc}\n"
                         f"Install it with the analysis environment: pixi install")

    entries = read_tsv(args.dataset_dir / "entries.tsv")
    # The dataset holds no monomers: every entry has at least two chains. Its four categories are
    # homodimer (2 chains, 1 sequence; 113 entries), heterodimer (2 chains, 2 sequences; 18),
    # homomer (3-8 chains, 1 sequence; 57) and heteromer (3-11 chains, 2-4 sequences; 18).
    # DIMERS is the default: one native interface per complex, so its DockQ is that interface rather
    # than a mean over several, and the column means the same thing on every row. `hetero` narrows
    # further to the complexes whose chains are not all one sequence.
    if args.subset == "hetero":
        entries = [e for e in entries if e["category"] in ("heterodimer", "heteromer")]
    elif args.subset == "dimers":
        entries = [e for e in entries if int(e["n_chains"]) == 2]

    arms = {label: read_arm(Path(run_dir)) for label, run_dir in args.run}
    columns = ["run", *ARM_COLUMNS,
               "entry_id", "category", "n_chains", "total_len", "chains_match",
               "select_by", "sample", "ranking_score", "iptm", "ptm",
               "dockq", "dockq_min_interface",
               "fnat", "irmsd", "lrmsd", "f1", "n_interfaces",
               "model_path", "status"]

    # Work out what is actually outstanding BEFORE spending anything, per arm, so the
    # printed plan is the real cost rather than the worst case.
    carried: dict[str, dict[str, dict]] = {}
    tasks = []
    for label, run_dir in args.run:
        done = {} if args.rescore else load_scored(args.out_dir, label)
        # Carry the MEASUREMENTS but re-derive the LABELS. tag/engine/weights/
        # use_templates describe the arm, not the scoring, so a config whose tag or
        # recorded axes changed should be reflected without re-running DockQ — and a
        # table written before a label column existed gets it backfilled for free.
        for row in done.values():
            row["run"] = label
            row.update({key: arms[label][key] for key in ARM_COLUMNS})
        carried[label] = done
        for entry in entries:
            if entry["entry_id"] in done:
                continue
            tasks.append({"run": label, "run_dir": run_dir, "entry": entry,
                          "arm": arms[label], "select_by": args.select_by,
                          "dockq_cmd": dockq_cmd, "timeout": args.timeout})

    total_possible = len(entries) * len(args.run)
    print(f"{len(entries)} entries x {len(args.run)} arm(s) = {total_possible} pairs; "
          f"{len(tasks)} need DockQ, {total_possible - len(tasks)} carried over"
          f"{' (--rescore: nothing carried over)' if args.rescore else ''}")
    print(f"--jobs {args.jobs}, --select-by {args.select_by}, tables -> {args.out_dir}")
    for label, arm in arms.items():
        axes = " ".join(f"{k}={arm[k] or '?'}" for k in AXIS_COLUMNS)
        todo = sum(1 for t in tasks if t["run"] == label)
        print(f"  {label:28s} {axes}   to score: {todo}, carried: {len(carried[label])}")
        if not arm["engine"]:
            print(f"  {'':28s} NOTE: no config.json in this run directory, so its "
                  f"pipeline axes are unknown and left blank in the output.")

    fresh: list[dict] = []
    if tasks:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(score_one, t): t for t in tasks}
            for n, future in enumerate(as_completed(futures), 1):
                fresh.append(future.result())
                if n % 25 == 0 or n == len(tasks):
                    print(f"  {n}/{len(tasks)}", file=sys.stderr)

    # One table per arm: carried-over rows plus whatever was just scored.
    by_arm: dict[str, list[dict]] = {label: list(carried[label].values())
                                     for label, _ in args.run}
    for row in fresh:
        by_arm[row["run"]].append(row)

    print()
    for label, _ in args.run:
        rows = sorted(by_arm[label], key=lambda r: r["entry_id"])
        path = arm_table_path(args.out_dir, label)
        write_rows(path, union_columns(rows, columns), rows)
        n_new = sum(1 for r in fresh if r["run"] == label)
        print(f"Wrote {len(rows)} rows to {path}  ({n_new} newly scored)")

    for label, _ in args.run:
        subset = by_arm[label]
        scored = sorted(float(r["dockq"]) for r in subset if r["dockq"])
        axes = " ".join(f"{k}={arms[label][k] or '?'}" for k in AXIS_COLUMNS)
        print(f"\n{label}: {len(subset)} entries, {len(scored)} scored   [{axes}]")
        if scored:
            median = scored[len(scored) // 2]
            acceptable = sum(1 for d in scored if d >= 0.23)   # DockQ's own thresholds
            high = sum(1 for d in scored if d >= 0.80)
            print(f"  median DockQ {median:.3f}   mean {sum(scored) / len(scored):.3f}")
            print(f"  acceptable (>=0.23) {acceptable}/{len(scored)}   "
                  f"high (>=0.80) {high}/{len(scored)}")
        failures = {}
        for r in subset:
            if r["status"] != "ok":
                failures[r["status"].split(":")[0]] = failures.get(
                    r["status"].split(":")[0], 0) + 1
        if failures:
            print(f"  not scored: " +
                  ", ".join(f"{k}={v}" for k, v in sorted(failures.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
