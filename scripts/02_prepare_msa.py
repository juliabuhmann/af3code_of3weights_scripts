#!/usr/bin/env python3
"""
Stage 02 — assemble the MSA/template store, reusing an existing cache where possible.

Both prediction arms (native weights, openbind weights) must see BYTE-IDENTICAL
MSAs and templates, or a DockQ difference cannot be attributed to the weights.
This stage builds one store, keyed by sequence, that both arms then read.

    <work-dir>/msa_store/<seq_id>.json.gz    AlphaFold3 data-pipeline output for one
                                             sequence: unpairedMsa, pairedMsa, templates
                                             (templates carry the full mmCIF inline).
                                             A reused entry is a SYMLINK to the cache;
                                             a freshly computed one is a real file.
    <work-dir>/msa_input/<seq_id>.json       single-chain AF3 input for each sequence
                                             that is NOT in the cache — the array-job
                                             input for slurm/msa.sbatch.
    <work-dir>/msa_todo.txt                  one seq_id per line, the sbatch array list.
    <work-dir>/msa_manifest.tsv              seq_id, length, status, source — every
                                             sequence and where its MSA came from.

Cache matching is BY SEQUENCE, never by identifier. Existing AlphaFold3 MSA caches
in this group are keyed by whatever protein id the run that produced them used
(e.g. `7grsc0` from an earlier pair-decomposition benchmark), which has no relation
to this dataset's `seq_id`. This script reads the `sequence` field out of each
cached file and matches on its sha1, so the identifier scheme is irrelevant.

Reading the cache is done by STREAMING: a cached entry can be 100 000 paired-MSA
sequences (tens of MB decompressed), but the `sequence` field sits near the top of
the JSON, so only the first --peek-bytes of each decompressed file are read. The
resulting index is written to <work-dir>/msa_cache_index.tsv and reused on the next
run unless --reindex is passed.

Usage:
    python scripts/02_prepare_msa.py \
        --dataset-dir <base>/dataset \
        --work-dir    <base>/work \
        --msa-cache   /path/to/an/existing/alphafold3_msas \
        --subset all
"""

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

# Matches the "sequence" field of a protein entity in an AlphaFold3 input JSON.
# Anchored on the key so a residue string inside a template mmCIF cannot match
# first — the key/value pair appears before `templates` in the emitted layout.
SEQ_FIELD_RE = re.compile(r'"sequence"\s*:\s*"([A-Za-z]+)"')

# AlphaFold3 input-JSON version. The container accepts 1-4 and validates only
# membership; 4 is its current JSON_VERSION and what the reused cache entries
# carry, so emitting 4 keeps our inputs and the cache on the same dialect.
AF3_JSON_VERSION = 4

SUBSETS = {
    "dimers": lambda e: int(e["n_chains"]) == 2,
    "all": lambda e: True,
}


def sha1_full(sequence: str) -> str:
    return hashlib.sha1(sequence.encode()).hexdigest()


def read_tsv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def peek_sequence(path: Path, peek_bytes: int) -> str | None:
    """Read the first protein `sequence` out of a (possibly gzipped) AF3 JSON."""
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt") as fh:  # type: ignore[operator]
            head = fh.read(peek_bytes)
    except (OSError, EOFError, UnicodeDecodeError):
        return None
    match = SEQ_FIELD_RE.search(head)
    return match.group(1) if match else None


def build_cache_index(cache_dirs: list[Path], index_path: Path,
                      peek_bytes: int, reindex: bool) -> dict[str, Path]:
    """sha1(sequence) -> cached AF3 data JSON. Cached to index_path across runs."""
    if index_path.is_file() and not reindex:
        index: dict[str, Path] = {}
        for row in read_tsv(index_path):
            path = Path(row["path"])
            if path.exists():
                index[row["seq_sha1"]] = path
        print(f"Loaded cache index: {len(index)} sequences from {index_path}")
        print("  (pass --reindex to rebuild it after the cache changes)")
        return index

    files: list[Path] = []
    for cdir in cache_dirs:
        if not cdir.is_dir():
            print(f"WARNING: --msa-cache not a directory, skipping: {cdir}")
            continue
        files.extend(sorted(cdir.glob("*.json.gz")))
        files.extend(sorted(cdir.glob("*.json")))

    print(f"Indexing {len(files)} cached files by sequence "
          f"(streaming first {peek_bytes // 1024} KB of each) ...")
    index = {}
    collisions = 0
    for n, path in enumerate(files, 1):
        if n % 500 == 0:
            print(f"  {n}/{len(files)}", file=sys.stderr)
        sequence = peek_sequence(path, peek_bytes)
        if not sequence:
            continue
        key = sha1_full(sequence)
        if key in index:
            collisions += 1   # same sequence cached under two ids; first wins
            continue
        index[key] = path

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w") as fh:
        fh.write("seq_sha1\tpath\n")
        for key, path in sorted(index.items()):
            fh.write(f"{key}\t{path}\n")
    print(f"Indexed {len(index)} unique sequences "
          f"({collisions} duplicate entries ignored) -> {index_path}")
    return index


def write_monomer_json(path: Path, seq_id: str, sequence: str, model_seed: int) -> None:
    """Single-chain AlphaFold3 input, the unit the data pipeline runs on."""
    payload = {
        "dialect": "alphafold3",
        "version": AF3_JSON_VERSION,
        "name": seq_id,
        "sequences": [{"protein": {"id": "A", "sequence": sequence}}],
        "modelSeeds": [model_seed],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, type=Path,
                    help="Output of scripts/01_build_dataset.py")
    ap.add_argument("--work-dir", required=True, type=Path,
                    help="Where msa_store/, msa_input/ and the manifests are written")
    ap.add_argument("--msa-cache", type=Path, action="append", default=[],
                    help="Existing AF3 MSA cache directory of *_data.json.gz "
                         "(repeatable; searched in order)")
    ap.add_argument("--subset", choices=sorted(SUBSETS), default="dimers",
                    help="Which entries need MSAs (default: dimers = n_chains == 2)")
    ap.add_argument("--model-seed", type=int, default=1,
                    help="modelSeeds value in the monomer MSA inputs. Irrelevant to "
                         "the data pipeline (no inference runs) but AF3 requires it.")
    ap.add_argument("--peek-bytes", type=int, default=1 << 20,
                    help="Decompressed bytes read per cached file when indexing")
    ap.add_argument("--reindex", action="store_true",
                    help="Rebuild the cache index instead of reusing msa_cache_index.tsv")
    args = ap.parse_args()

    dataset_dir: Path = args.dataset_dir
    work_dir: Path = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    entries = read_tsv(dataset_dir / "entries.tsv")
    sequences = {r["seq_id"]: r for r in read_tsv(dataset_dir / "sequences.tsv")}
    chains = read_tsv(dataset_dir / "chains.tsv")

    keep = SUBSETS[args.subset]
    wanted_entries = {e["entry_id"] for e in entries if keep(e)}
    needed_seq_ids = sorted({c["seq_id"] for c in chains
                             if c["entry_id"] in wanted_entries},
                            key=lambda s: -int(sequences[s]["length"]))
    print(f"Subset '{args.subset}': {len(wanted_entries)} entries, "
          f"{len(needed_seq_ids)} unique sequences")

    index = build_cache_index(args.msa_cache,
                              work_dir / "msa_cache_index.tsv",
                              args.peek_bytes, args.reindex)

    store = work_dir / "msa_store"
    msa_input = work_dir / "msa_input"
    store.mkdir(exist_ok=True)
    msa_input.mkdir(exist_ok=True)

    manifest, todo = [], []
    n_cached, n_present, n_todo = 0, 0, 0
    for sid in needed_seq_ids:
        row = sequences[sid]
        sequence = row["sequence"]
        dest = store / f"{sid}.json.gz"

        if dest.exists():
            # Already in the store: computed by a previous run of slurm/msa.sbatch,
            # or a symlink created below on an earlier pass. Left untouched.
            status, source = "present", str(dest.resolve())
            n_present += 1
        elif (hit := index.get(sha1_full(sequence))) is not None:
            dest.symlink_to(hit)
            status, source = "cached", str(hit)
            n_cached += 1
        else:
            write_monomer_json(msa_input / f"{sid}.json", sid, sequence, args.model_seed)
            todo.append(sid)
            status, source = "todo", ""
            n_todo += 1

        manifest.append({"seq_id": sid, "length": row["length"],
                         "n_occurrences": row["n_occurrences"],
                         "status": status, "source": source})

    with open(work_dir / "msa_manifest.tsv", "w") as fh:
        fh.write("seq_id\tlength\tn_occurrences\tstatus\tsource\n")
        for m in manifest:
            fh.write("\t".join(str(m[c]) for c in
                               ("seq_id", "length", "n_occurrences", "status", "source")) + "\n")

    todo_path = work_dir / "msa_todo.txt"
    todo_path.write_text("".join(f"{s}\n" for s in todo))

    print(f"\nMSA store: {store}")
    print(f"  reused from cache : {n_cached}")
    print(f"  already in store  : {n_present}")
    print(f"  to compute        : {n_todo}")
    if todo:
        print(f"\n{n_todo} sequence(s) need the AlphaFold3 data pipeline:")
        for sid in todo:
            print(f"  {sid}  {sequences[sid]['length']} aa  "
                  f"(in {sequences[sid]['n_occurrences']} chain instances)")
        print(f"\nInputs written to {msa_input}, array list to {todo_path}")
        print("Submit:  ./run_workflow.sh --stage submit-msa --base-dir <base>")
    else:
        print("\nNothing to compute — the store is complete. "
              "Next: scripts/03_build_af3_jsons.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
