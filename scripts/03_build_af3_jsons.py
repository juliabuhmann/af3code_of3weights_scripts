#!/usr/bin/env python3
"""
Stage 03 — build one AlphaFold3 input JSON per complex, with MSAs and templates inline.

The whole complex is folded as the benchmark defines it: every chain in the entry's
seq.fasta, carrying the chain ids stage 01 assigned positionally (A, B, C, ... in
FASTA order). Those are the ids cleanNative.pdb uses, and stage 01 verifies the
correspondence by sequence, so DockQ needs no mapping table. No pooling, no filler,
no pair decomposition.

Chains that share a sequence become ONE AlphaFold3 entity with a list of ids —
e.g. a homodimer is `{"protein": {"id": ["A", "B"], "sequence": ..., ...}}`. That is
AF3's own representation for copies of an entity, and it means the (large) MSA is
carried once per entity rather than once per chain.

MSAs and templates are copied verbatim out of <work-dir>/msa_store/<seq_id>.json.gz
(stage 02). Because the store is shared, the JSON written here is byte-identical for
both weight arms — the arms differ only in the container's --model_dir/--of3_weights.
Inference therefore runs with --norun_data_pipeline; no database search happens on
the GPU node.

Outputs (--work-dir):
    af3_input/<entry_id>.json    the prediction inputs
    predict_todo.txt             one entry_id per line — the sbatch array list
    af3_input_manifest.tsv       entry_id, n_chains, n_entities, total_len, status

Usage:
    python scripts/03_build_af3_jsons.py \
        --dataset-dir <base>/dataset \
        --work-dir    <base>/work \
        --config      configs/af3_openbind.json \
        --subset      dimers
"""

import argparse
import csv
import gzip
import json
import sys
from collections import OrderedDict
from pathlib import Path

# See scripts/02_prepare_msa.py for why 4.
AF3_JSON_VERSION = 4

SUBSETS = {
    "dimers": lambda e: int(e["n_chains"]) == 2,
    "all": lambda e: True,
}

# Fields of a protein entity that the data pipeline fills in and inference consumes.
# Copied straight across from the per-sequence store; anything else in the cached
# file (id, name, modelSeeds, ...) belongs to that monomer job, not to this complex.
PIPELINE_FIELDS = ("unpairedMsa", "pairedMsa", "unpairedMsaPath", "pairedMsaPath",
                   "templates", "modifications")


def read_tsv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def load_store_entry(store: Path, seq_id: str) -> dict | None:
    """Return the protein entity dict from <store>/<seq_id>.json.gz, or None."""
    path = store / f"{seq_id}.json.gz"
    if not path.exists():
        path = store / f"{seq_id}.json"
        if not path.exists():
            return None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:  # type: ignore[operator]
        data = json.load(fh)
    proteins = [entity["protein"] for entity in data.get("sequences", [])
                if "protein" in entity]
    if len(proteins) != 1:
        raise SystemExit(
            f"{path}: expected exactly one protein entity in an MSA-store file, "
            f"found {len(proteins)}")
    return proteins[0]


def build_entry_json(entry: dict, entry_chains: list[dict],
                     sequences: dict[str, dict], store: Path,
                     model_seed: int) -> tuple[dict | None, str]:
    """Return (payload, status). payload is None when an MSA is missing."""
    # Group chain ids by sequence, preserving first-appearance order so that entity
    # order (and therefore AF3's internal chain ordering) follows the FASTA.
    by_seq: "OrderedDict[str, list[str]]" = OrderedDict()
    for chain in entry_chains:
        by_seq.setdefault(chain["seq_id"], []).append(chain["chain_id"])

    entities = []
    for seq_id, chain_ids in by_seq.items():
        cached = load_store_entry(store, seq_id)
        if cached is None:
            return None, f"missing_msa:{seq_id}"
        if cached["sequence"] != sequences[seq_id]["sequence"]:
            return None, f"sequence_mismatch:{seq_id}"

        protein = {
            # A single-copy entity keeps a bare string id, matching AF3's examples;
            # multi-copy entities take the list form.
            "id": chain_ids if len(chain_ids) > 1 else chain_ids[0],
            "sequence": cached["sequence"],
        }
        for field in PIPELINE_FIELDS:
            if field in cached:
                protein[field] = cached[field]
        entities.append({"protein": protein})

    payload = {
        "dialect": "alphafold3",
        "version": AF3_JSON_VERSION,
        "name": entry["entry_id"],
        "sequences": entities,
        "modelSeeds": [model_seed],
    }
    return payload, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path,
                    help="configs/*.json — only model_seed is read here; the inputs "
                         "are otherwise identical across arms")
    ap.add_argument("--subset", choices=sorted(SUBSETS), default="dimers")
    ap.add_argument("--out-subdir", default="af3_input",
                    help="Directory under --work-dir for the JSONs (default af3_input)")
    args = ap.parse_args()

    config = json.loads(args.config.read_text())
    model_seed = int(config.get("model_seed", 42))

    dataset_dir: Path = args.dataset_dir
    work_dir: Path = args.work_dir
    store = work_dir / "msa_store"
    if not store.is_dir():
        raise SystemExit(f"MSA store not found: {store}\nRun scripts/02_prepare_msa.py first.")

    entries = read_tsv(dataset_dir / "entries.tsv")
    sequences = {r["seq_id"]: r for r in read_tsv(dataset_dir / "sequences.tsv")}
    chains_by_entry: dict[str, list[dict]] = {}
    for chain in read_tsv(dataset_dir / "chains.tsv"):
        chains_by_entry.setdefault(chain["entry_id"], []).append(chain)

    keep = SUBSETS[args.subset]
    selected = [e for e in entries if keep(e)]
    print(f"Subset '{args.subset}': {len(selected)} of {len(entries)} entries, "
          f"model seed {model_seed}")

    out_dir = work_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest, ready, failed = [], [], []
    for entry in selected:
        entry_chains = chains_by_entry.get(entry["entry_id"], [])
        payload, status = build_entry_json(entry, entry_chains, sequences,
                                           store, model_seed)
        n_entities = len(payload["sequences"]) if payload else 0
        if payload is not None:
            (out_dir / f"{entry['entry_id']}.json").write_text(
                json.dumps(payload, indent=2) + "\n")
            ready.append(entry["entry_id"])
        else:
            failed.append((entry["entry_id"], status))
        manifest.append({"entry_id": entry["entry_id"],
                         "n_chains": entry["n_chains"],
                         "n_entities": n_entities,
                         "total_len": entry["total_len"],
                         "category": entry["category"],
                         "status": status})

    with open(work_dir / "af3_input_manifest.tsv", "w") as fh:
        fh.write("entry_id\tn_chains\tn_entities\ttotal_len\tcategory\tstatus\n")
        for m in manifest:
            fh.write("\t".join(str(m[c]) for c in
                               ("entry_id", "n_chains", "n_entities",
                                "total_len", "category", "status")) + "\n")

    # Largest first: with an array job the long tail dominates wall time, so
    # starting the big complexes early shortens time-to-last-result.
    ready.sort(key=lambda eid: -int(next(m["total_len"] for m in manifest
                                         if m["entry_id"] == eid)))
    (work_dir / "predict_todo.txt").write_text("".join(f"{eid}\n" for eid in ready))

    print(f"\nWrote {len(ready)} input JSON(s) to {out_dir}")
    if failed:
        print(f"{len(failed)} entry/entries skipped:")
        for eid, status in failed[:20]:
            print(f"  {eid}: {status}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
        print("  (a missing_msa status means stage 02's todo list has not been run yet)")
    print(f"Array list: {work_dir / 'predict_todo.txt'} ({len(ready)} tasks)")
    print(f"\nSubmit:  ./run_workflow.sh --stage submit-predictions "
          f"--base-dir <base> --config {args.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
