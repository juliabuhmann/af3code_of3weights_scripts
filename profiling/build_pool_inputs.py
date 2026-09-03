#!/usr/bin/env python3
"""
Build the input JSONs for the VRAM-scaling sweep: one pool per token count.

The question this is meant to answer is the one the top-level README asserts without
evidence — how far up the token axis each codebase gets on one RTX Pro 6000, and how
peak VRAM grows on the way there. So the inputs have to vary ONE thing, the number of
tokens, and hold everything else fixed.

CONSTRUCTION
------------
A pool of `n` copies of a single real benchmark sequence, taken from the MSA store
this repository already built (`<base>/work/msa_store/<seq_id>.json.gz`), with that
sequence's real `unpairedMsa`, `pairedMsa` and `templates` copied in verbatim. So:

  * the alignments are real, not a synthetic or single-sequence stand-in, and no MSA
    is recomputed — the store entry is reused byte-for-byte, exactly as stage 03 does
    it for the benchmark arms;
  * every pool sees the same alignment depth per chain, so a change in peak VRAM
    between two pools is a change in token count and nothing else;
  * for protein-only inputs AlphaFold3 tokenises one standard residue as one token,
    so tokens == residues == n_copies * len(monomer), on the nose.

MONOMER CHOICE is a rule, not a pick: the SHORTEST sequence in the store whose length
divides --step exactly (ties by seq_id), subject to --min-monomer-len. A length that
divides the step is what puts every pool exactly on the requested grid — 500, 1000,
1500 ... tokens with no rounding. If the store has no such sequence the script falls
back to the length closest to --step/2 and emits the achievable grid instead, printing
what it did.

WHAT THIS IS NOT. A homomeric pool of one sequence is not a biological complex and its
predicted structure is meaningless; nothing here scores anything. It is a token-count
dial with a realistic feature payload behind it. Two consequences worth naming:
a homomer's chains share one entity, so both engines expand the same alignment n
times, and the pool has no cross-species pairing signal a real heterocomplex would.

OUTPUT
------
    <out-dir>/af3_pool/pool_<NNNNN>aa.json   AlphaFold3 input, MSAs+templates inline
    <out-dir>/pool_manifest.tsv              tokens, n_chains, monomer, msa rows, ...

The OF3 side is NOT built here — it is built from these files by the repository's own
converter, so both engines provably read the same alignments:

    python scripts/09_build_of3_inputs.py \
        --af3-input-dir <out-dir>/af3_pool --out-dir <out-dir>/of3_pool

Usage:
    python profiling/build_pool_inputs.py --base-dir <base>
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Stage 03 already knows how to lift a protein entity out of the MSA store and which
# of its fields inference consumes. Loading it by path rather than restating those
# rules here keeps ONE definition of "what gets copied out of the store" — the module
# name starts with a digit, so it cannot be imported by name.
_spec = importlib.util.spec_from_file_location(
    "stage03", REPO / "scripts" / "03_build_af3_jsons.py")
stage03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage03)

# Matched to the benchmark arms (configs/*.json), so a pool is featurised the same way
# a benchmark entry was.
DEFAULT_SEED = 42


def chain_id(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA, ... — bijective base-26, so ids stay unique."""
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


def read_tsv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def pick_monomer(sequences: list[dict], store: Path, step: int,
                 min_len: int) -> tuple[dict, bool]:
    """(sequence row, divides_step). See MONOMER CHOICE in the module docstring."""
    available = [r for r in sequences
                 if (store / f"{r['seq_id']}.json.gz").exists()
                 or (store / f"{r['seq_id']}.json").exists()]
    if not available:
        sys.exit(f"ERROR: no sequence in {store} — has stage 02 run?")
    eligible = [r for r in available if int(r["length"]) >= min_len]
    if not eligible:
        sys.exit(f"ERROR: no store sequence is >= --min-monomer-len {min_len} aa "
                 f"(longest is {max(int(r['length']) for r in available)} aa)")

    exact = [r for r in eligible if step % int(r["length"]) == 0]
    if exact:
        exact.sort(key=lambda r: (int(r["length"]), r["seq_id"]))
        return exact[0], True
    eligible.sort(key=lambda r: (abs(int(r["length"]) - step / 2), r["seq_id"]))
    return eligible[0], False


def msa_rows(text: str | None) -> int:
    if not text:
        return 0
    return text.count("\n>") + (1 if text.startswith(">") else 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", type=Path,
                    default=Path(os.environ.get("AF3OF3_BASE_DIR", "")),
                    help="Benchmark base dir (holds dataset/ and work/msa_store/). "
                         "Defaults to $AF3OF3_BASE_DIR.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where the pool inputs go (default: <base-dir>/profiling/inputs)")
    ap.add_argument("--start", type=int, default=500, help="First token count")
    ap.add_argument("--end", type=int, default=7000, help="Last token count")
    ap.add_argument("--step", type=int, default=500, help="Token-count increment")
    ap.add_argument("--min-monomer-len", type=int, default=150,
                    help="Shortest monomer considered, so pools stay few-chained "
                         "(default: 150)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"modelSeeds value, matched to the arms (default: {DEFAULT_SEED})")
    args = ap.parse_args()

    if not args.base_dir or str(args.base_dir).startswith("/path/to"):
        sys.exit("ERROR: pass --base-dir or set AF3OF3_BASE_DIR "
                 "(see SITE CONFIGURATION in run_workflow.sh)")
    dataset_dir = args.base_dir / "dataset"
    store = args.base_dir / "work" / "msa_store"
    for path in (dataset_dir / "sequences.tsv", store):
        if not path.exists():
            sys.exit(f"ERROR: no such path: {path}")
    out_dir = args.out_dir or (args.base_dir / "profiling" / "inputs")
    af3_dir = out_dir / "af3_pool"
    af3_dir.mkdir(parents=True, exist_ok=True)

    sequences = read_tsv(dataset_dir / "sequences.tsv")
    row, divides = pick_monomer(sequences, store, args.step, args.min_monomer_len)
    monomer_len = int(row["length"])
    entity = stage03.load_store_entry(store, row["seq_id"])
    if entity is None:
        sys.exit(f"ERROR: {row['seq_id']} vanished from {store}")
    if entity["sequence"] != row["sequence"]:
        sys.exit(f"ERROR: {row['seq_id']} in the store does not match sequences.tsv")

    n_unpaired = msa_rows(entity.get("unpairedMsa"))
    n_paired = msa_rows(entity.get("pairedMsa"))
    n_templates = len(entity.get("templates") or [])

    print(f"monomer       : {row['seq_id']}  {monomer_len} aa")
    print(f"  alignments  : {n_unpaired} unpaired rows, {n_paired} paired rows, "
          f"{n_templates} template(s)")
    print(f"  chosen as   : shortest store sequence >= {args.min_monomer_len} aa "
          f"{'dividing' if divides else 'NOT dividing'} the {args.step}-token step")
    if not divides:
        print("  NOTE        : no store sequence divides the step, so the grid below "
              "is the achievable one, not the requested one.")

    # One pool per requested token count. n_copies is rounded, and the ACTUAL token
    # count (n_copies * monomer_len) is what everything downstream is keyed on — the
    # requested number is never assumed to have been met.
    seen: dict[int, int] = {}
    for target in range(args.start, args.end + 1, args.step):
        n_copies = max(1, round(target / monomer_len))
        seen.setdefault(n_copies * monomer_len, n_copies)

    manifest = []
    for tokens, n_copies in sorted(seen.items()):
        protein = {"id": [chain_id(i) for i in range(n_copies)],
                   "sequence": entity["sequence"]}
        for field in stage03.PIPELINE_FIELDS:
            if field in entity:
                protein[field] = entity[field]
        name = f"pool_{tokens:05d}aa"
        payload = {"dialect": "alphafold3",
                   "version": stage03.AF3_JSON_VERSION,
                   "name": name,
                   "sequences": [{"protein": protein}],
                   "modelSeeds": [args.seed]}
        path = af3_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        manifest.append({"name": name, "tokens": tokens, "n_chains": n_copies,
                         "monomer_seq_id": row["seq_id"], "monomer_len": monomer_len,
                         "n_templates_per_chain": n_templates,
                         "msa_rows_unpaired": n_unpaired,
                         "msa_rows_paired": n_paired,
                         "size_mb": round(path.stat().st_size / 1e6, 1)})

    columns = ["name", "tokens", "n_chains", "monomer_seq_id", "monomer_len",
               "n_templates_per_chain", "msa_rows_unpaired", "msa_rows_paired",
               "size_mb"]
    manifest_path = out_dir / "pool_manifest.tsv"
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\n{'tokens':>7} {'chains':>7} {'JSON MB':>8}")
    for m in manifest:
        print(f"{m['tokens']:>7} {m['n_chains']:>7} {m['size_mb']:>8.1f}")
    print(f"\nWrote {len(manifest)} pool JSONs to {af3_dir}")
    print(f"Manifest: {manifest_path}")
    print("\nNext, the OF3 queries — built from these same files by the repository's "
          "converter:")
    print(f"  python scripts/09_build_of3_inputs.py \\\n"
          f"      --af3-input-dir {af3_dir} --out-dir {out_dir / 'of3_pool'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
