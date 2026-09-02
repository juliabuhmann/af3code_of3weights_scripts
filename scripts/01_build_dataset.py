#!/usr/bin/env python3
"""
Stage 01 — turn the extracted Benchmark-IV tree into this repo's dataset contract.

Input   the multimer/ tree from stage 00: multimer/<pdb>/{seq.fasta,cleanNative.pdb}
Output  (--out-dir)

    entries.tsv     one row per complex, the unit of prediction:
                      entry_id      PDB id, lowercase (e.g. 7grs)
                      n_chains      chain records in seq.fasta
                      n_uniq_seq    distinct sequences among them
                      category      homodimer | heterodimer | homomer | heteromer
                      total_len     summed chain length = AF3 token count for proteins
                      chain_ids     assigned AF3 chain ids, in seq.fasta order (A,B,...)
                      fasta_chains  the original FASTA header suffixes, same order
                      native_chains chain ids present in cleanNative.pdb, sorted
                      chains_match  yes|no — do chain_ids and native_chains agree?
                      seq_match_min per-chain agreement between the FASTA sequence and
                                    the modelled residues of the correspondingly-named
                                    native chain, minimum over chains (see below)
                      native_pdb    path to cleanNative.pdb

    sequences.tsv   one row per UNIQUE sequence across the whole benchmark:
                      seq_id (sha1[:12]), length, n_occurrences, sequence
                    This is the unit the MSA stage works on — a sequence that appears
                    in several complexes, or twice inside one homodimer, is searched once.

    chains.tsv      entry_id, chain_id, fasta_chain, seq_id, length.

    dataset_summary.json   counts, category breakdown, and any anomalies found.

NO PAIR DECOMPOSITION. Complexes are taken exactly as the benchmark defines them:
whatever chains seq.fasta lists is what gets folded, and cleanNative.pdb is the
reference. Selecting a subset (e.g. the 131 two-chain entries) happens downstream
via `category` / `n_chains`, not by rewriting the dataset.

CHAIN IDS — assigned POSITIONALLY (A, B, C, ... in seq.fasta order), not taken from
the FASTA header. The headers are `>{pdb}-assembly1_{chain}` carrying the *original
PDB* chain labels, which for 38 of the 206 entries are not A,B,C,... at all
(`A,B-2`, `H,L`, `A,C`, `D,A`, ...), while cleanNative.pdb has been relabelled
A,B,C,... in FASTA order. Assigning positionally therefore makes the predicted chain
ids equal the native ones, so DockQ needs no chain-mapping table.

That the relabelling really is positional is VERIFIED here rather than assumed:
`seq_match_min` is the fraction of each native chain's modelled residues that occur,
in order, in the FASTA sequence assigned the same id, minimised over the entry's
chains. A value near 1.0 means the assignment is right; a low value means the FASTA
order and the native order disagree for that entry, and it is listed as an anomaly.
(Native chains are missing unmodelled residues, so the comparison is a subsequence
test, not equality.)

Usage:
    python scripts/01_build_dataset.py \
        --multimer-dir <base>/benchmark_iv/multimer \
        --out-dir      <base>/dataset
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SEQ_ID_LEN = 12  # sha1 prefix; 12 hex chars over ~400 sequences has no realistic collision risk


def seq_id(sequence: str) -> str:
    return hashlib.sha1(sequence.encode()).hexdigest()[:SEQ_ID_LEN]


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Return [(header, sequence)] preserving file order."""
    records: list[tuple[str, list[str]]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                records.append((line[1:].strip(), []))
            elif records:
                records[-1][1].append(line)
    return [(h, "".join(parts)) for h, parts in records]


def fasta_chain_from_header(header: str) -> str:
    """`7grs-assembly1_A` -> `A`. The ORIGINAL PDB label, kept for provenance only."""
    return header.rsplit("_", 1)[-1] if "_" in header else header


def assign_chain_id(index: int) -> str:
    """AlphaFold3 chain id for the index-th chain: A..Z, then AA, AB, ...

    Benchmark-IV tops out at 11 chains, so in practice this is A..K; the
    two-letter continuation is there so the function is not silently wrong if the
    dataset is ever pointed at something larger.
    """
    letters = ""
    index += 1
    while index > 0:
        index, rem = divmod(index - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


# Modified residues that DockQ / the PDB map onto a standard parent. Only the ones
# that actually occur as CA-bearing residues in these natives need handling; anything
# unrecognised becomes 'X', which the subsequence test below simply fails to match.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def native_chain_sequences(pdb_path: Path) -> dict[str, str]:
    """chain id -> one-letter sequence of its modelled residues, in residue order."""
    out: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            chain, resseq, resname = line[21], line[22:27], line[17:20].strip()
            if (chain, resseq) in seen:      # altlocs
                continue
            seen.add((chain, resseq))
            out.setdefault(chain, []).append(THREE_TO_ONE.get(resname, "X"))
    return {k: "".join(v) for k, v in out.items()}


def subsequence_fraction(full: str, observed: str) -> float:
    """Fraction of `observed` that occurs as an ordered subsequence of `full`.

    The native is the FASTA sequence minus unmodelled residues, so a correct
    pairing scores ~1.0 and a wrong one scores at chance (~0.1 on these).
    """
    if not observed:
        return 0.0
    pos, hits = 0, 0
    for residue in observed:
        found = full.find(residue, pos)
        if found >= 0:
            pos, hits = found + 1, hits + 1
    return hits / len(observed)


def categorise(n_chains: int, n_uniq: int) -> str:
    if n_chains == 2:
        return "homodimer" if n_uniq == 1 else "heterodimer"
    return "homomer" if n_uniq == 1 else "heteromer"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--multimer-dir", required=True, type=Path,
                    help="multimer/ tree produced by scripts/00_fetch_benchmark.py")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    root: Path = args.multimer_dir
    if not root.is_dir():
        raise SystemExit(f"--multimer-dir not found: {root}")
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    entry_dirs = sorted(d for d in root.iterdir() if d.is_dir())

    entries: list[dict] = []
    chain_rows: list[dict] = []
    seq_seen: dict[str, str] = {}        # seq_id -> sequence
    seq_count: Counter = Counter()       # seq_id -> occurrences (chain instances)
    anomalies: dict[str, list[str]] = defaultdict(list)

    for d in entry_dirs:
        eid = d.name
        fasta = d / "seq.fasta"
        native = d / "cleanNative.pdb"

        if not fasta.is_file():
            anomalies["missing_seq_fasta"].append(eid)
            continue
        records = read_fasta(fasta)
        if not records:
            anomalies["empty_seq_fasta"].append(eid)
            continue

        chain_ids, fasta_chains, sids, lengths, sequences_in_order = [], [], [], [], []
        for position, (header, sequence) in enumerate(records):
            cid = assign_chain_id(position)
            fasta_chain = fasta_chain_from_header(header)
            sid = seq_id(sequence)
            seq_seen.setdefault(sid, sequence)
            seq_count[sid] += 1
            chain_ids.append(cid)
            fasta_chains.append(fasta_chain)
            sids.append(sid)
            lengths.append(len(sequence))
            sequences_in_order.append(sequence)
            chain_rows.append({"entry_id": eid, "chain_id": cid,
                               "fasta_chain": fasta_chain,
                               "seq_id": sid, "length": len(sequence)})

        # Verify the positional assignment against the native rather than trusting it.
        if native.is_file():
            native_seqs = native_chain_sequences(native)
            nat = sorted(native_seqs)
            match = set(nat) == set(chain_ids)
            scores = [subsequence_fraction(sequences_in_order[i], native_seqs[cid])
                      for i, cid in enumerate(chain_ids) if cid in native_seqs]
            seq_match_min = min(scores) if scores else 0.0
            if not match:
                anomalies["chain_id_mismatch"].append(
                    f"{eid}: assigned={','.join(chain_ids)} native={','.join(nat)}")
            if seq_match_min < 0.9:
                anomalies["chain_order_suspect"].append(
                    f"{eid}: seq_match_min={seq_match_min:.2f} "
                    f"(fasta labels {','.join(fasta_chains)})")
        else:
            nat, match, seq_match_min = [], False, 0.0
            anomalies["missing_native"].append(eid)

        n_uniq = len(set(sids))
        entries.append({
            "entry_id": eid,
            "n_chains": len(chain_ids),
            "n_uniq_seq": n_uniq,
            "category": categorise(len(chain_ids), n_uniq),
            "total_len": sum(lengths),
            "chain_ids": ",".join(chain_ids),
            "fasta_chains": ",".join(fasta_chains),
            "native_chains": ",".join(nat),
            "chains_match": "yes" if match else "no",
            "seq_match_min": f"{seq_match_min:.3f}",
            "native_pdb": str(native) if native.is_file() else "",
        })

    # --- write the contract --------------------------------------------------
    def write_tsv(path: Path, rows: list[dict], columns: list[str]) -> None:
        with open(path, "w") as fh:
            fh.write("\t".join(columns) + "\n")
            for row in rows:
                fh.write("\t".join(str(row[c]) for c in columns) + "\n")

    write_tsv(out_dir / "entries.tsv", entries,
              ["entry_id", "n_chains", "n_uniq_seq", "category", "total_len",
               "chain_ids", "fasta_chains", "native_chains", "chains_match",
               "seq_match_min", "native_pdb"])

    seq_rows = [{"seq_id": sid, "length": len(seq),
                 "n_occurrences": seq_count[sid], "sequence": seq}
                for sid, seq in sorted(seq_seen.items(), key=lambda kv: -len(kv[1]))]
    write_tsv(out_dir / "sequences.tsv", seq_rows,
              ["seq_id", "length", "n_occurrences", "sequence"])

    write_tsv(out_dir / "chains.tsv", chain_rows,
              ["entry_id", "chain_id", "fasta_chain", "seq_id", "length"])

    by_cat = Counter(e["category"] for e in entries)
    by_nchain = Counter(e["n_chains"] for e in entries)
    summary = {
        "multimer_dir": str(root),
        "n_entries": len(entries),
        "n_unique_sequences": len(seq_seen),
        "n_chain_instances": sum(seq_count.values()),
        "by_category": dict(sorted(by_cat.items())),
        "by_n_chains": {str(k): v for k, v in sorted(by_nchain.items())},
        "total_len": {
            "min": min((e["total_len"] for e in entries), default=0),
            "max": max((e["total_len"] for e in entries), default=0),
        },
        "entries_with_matching_chain_ids": sum(1 for e in entries if e["chains_match"] == "yes"),
        "entries_with_verified_chain_order": sum(1 for e in entries
                                                 if float(e["seq_match_min"]) >= 0.9),
        "anomalies": {k: v for k, v in anomalies.items()},
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # --- report --------------------------------------------------------------
    print(f"Entries:            {summary['n_entries']}")
    print(f"Unique sequences:   {summary['n_unique_sequences']} "
          f"(over {summary['n_chain_instances']} chain instances)")
    print("By category:")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:<12} {n}")
    print("By chain count:")
    for n_chains, n in sorted(by_nchain.items()):
        print(f"  {n_chains:>2} chains   {n}")
    print(f"Chain ids assigned==native: {summary['entries_with_matching_chain_ids']}"
          f"/{summary['n_entries']}")
    print(f"Chain ORDER verified by sequence (seq_match_min >= 0.9): "
          f"{summary['entries_with_verified_chain_order']}/{summary['n_entries']}")
    if anomalies:
        print("\nAnomalies (recorded in dataset_summary.json):")
        for kind, items in anomalies.items():
            print(f"  {kind}: {len(items)}")
            for item in items[:5]:
                print(f"    {item}")
            if len(items) > 5:
                print(f"    ... and {len(items) - 5} more")
    print(f"\nWrote entries.tsv, sequences.tsv, chains.tsv, dataset_summary.json "
          f"to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
