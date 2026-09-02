#!/usr/bin/env python3
"""
Stage 07 — expose Zheng & Peng's own AlphaFold 3 predictions as a scoreable arm.

The Zenodo archive ships the authors' predictions next to each ground truth:

    multimer/<entry>/alphaFold3_local/seed-<N>_sample-<i>_model.cif   AF3 standalone
    multimer/<entry>/alphaFold3/rank_<n>.pdb                         AF3 server

`alphaFold3_local` is the useful one here: it is the AlphaFold 3 standalone package,
like our own runs, and its filenames are the layout AF3 writes natively — which is
the layout scripts/05_score_dockq.py already reads. So instead of parsing a published
table (Table S4 carries only PDB ids and a Homo/Hetero label, no DockQ), we score the
authors' structures with OUR DockQ call against the SAME natives. Any difference is
then a difference in predictions, not in scoring.

This script only builds the directory shape the scorer expects:

    <base>/predictions/<name>/<entry>  ->  <multimer>/<entry>/alphaFold3_local

symlinks, no copying. Afterwards the arm scores like any other:

    ./run_workflow.sh --stage submit-score --subset all

WHAT A DIFFERENCE MEANS. This is a sanity check on our harness, not a weights
comparison. The authors ran AlphaFold 3 with default parameters, i.e. 10 recycles,
and built MSAs with their own pipeline; we run 3 recycles on MSAs from a different
cache. Both use one seed and 5 diffusion samples (verified from their filenames), and
both take the top-ranked sample. So agreement means our pipeline reproduces published
AlphaFold 3 behaviour; disagreement is attributable to recycles and MSAs, and says
nothing about OpenFold3 weights.

Usage:
    python scripts/07_link_paper_predictions.py \
        --multimer-dir <base>/benchmark_iv/multimer \
        --out-dir      <base>/predictions/zheng_af3_local \
        --source       alphaFold3_local
"""

import argparse
import sys
from pathlib import Path

SOURCES = ("alphaFold3_local", "alphaFold3")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--multimer-dir", required=True, type=Path,
                    help="multimer/ tree from scripts/00_fetch_benchmark.py")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Prediction dir to create, e.g. <base>/predictions/zheng_af3_local")
    ap.add_argument("--source", default="alphaFold3_local", choices=SOURCES,
                    help="Which of the authors' prediction sets to expose "
                         "(default alphaFold3_local, the standalone package)")
    args = ap.parse_args()

    root: Path = args.multimer_dir
    if not root.is_dir():
        raise SystemExit(f"--multimer-dir not found: {root}")
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    linked, empty, missing = [], [], []
    for entry_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        src = entry_dir / args.source
        if not src.is_dir():
            missing.append(entry_dir.name)
            continue
        # The scorer needs at least one model file to select a sample from.
        if not any(src.glob("*_model.cif")) and not any(src.glob("*.pdb")):
            empty.append(entry_dir.name)
            continue
        dest = out_dir / entry_dir.name
        if dest.is_symlink() or dest.exists():
            dest.unlink() if dest.is_symlink() else None
        if not dest.exists():
            dest.symlink_to(src.resolve())
            linked.append(entry_dir.name)

    print(f"source      : {args.source}")
    print(f"linked      : {len(linked)} entries -> {out_dir}")
    if empty:
        print(f"no models   : {len(empty)} ({', '.join(empty[:8])}"
              f"{' ...' if len(empty) > 8 else ''})")
    if missing:
        print(f"no {args.source}: {len(missing)} "
              f"({', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''})")

    if args.source == "alphaFold3":
        print("\nNOTE: the server set is rank_<n>.pdb with no summary_confidences.json,")
        print("so sample selection cannot use ranking_score and 05_score_dockq.py will")
        print("fall back to filename order. alphaFold3_local is the comparable set.")

    print("\nNext: ./run_workflow.sh --stage submit-score --subset all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
