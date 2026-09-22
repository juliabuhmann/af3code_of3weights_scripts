#!/usr/bin/env python3
"""Count PDB entries by release date relative to a model's training cutoff.

Motivation: OpenBind-0 (the `openbind` weights arm) is documented as trained on
"data from the Protein Data Bank (PDB) through June 2025", whereas AlphaFold 3's
released weights use the September 2021 cutoff. This script asks, against the
live RCSB Search API, how much protein-multimer data actually separates the two
cutoffs, and how many multimers postdate the OpenBind cutoff entirely (i.e. are
held out for *both* weight sets).

Two modes:

  counts   PDB-wide entry counts, bucketed by release date and composition.
  entries  Release dates for an explicit list of PDB IDs (e.g. the Benchmark-IV
           targets in results/dockq.tsv), bucketed against the same cutoffs.

Nothing is hardcoded from a previous run: every number printed comes from a live
query, and the cutoff dates are named constants with their source noted below.

Not yet verified against an independent count of the PDB.

Usage:
    python3 scripts/10_pdb_release_stats.py counts
    python3 scripts/10_pdb_release_stats.py entries --tsv results/dockq.tsv
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter

SEARCH_API = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_API = "https://data.rcsb.org/graphql"

# Training cutoffs, as stated by the model authors.
#   AF3   -- AlphaFold 3 paper / released weights.
#   OB0   -- openbind.uk OpenBind-0 blog: "trained on data from the Protein Data
#            Bank (PDB) through June 2025"; the benchmark section gives the exact
#            boundary as "30 June 2025".
CUTOFFS = {
    "af3": "2021-09-30",
    "openbind0": "2025-06-30",
}

# RCSB attribute for the date a structure became public. Training cutoffs are
# stated in terms of release, not deposition, so this is the field to filter on.
RELEASE_DATE = "rcsb_accession_info.initial_release_date"

# Composition of the polymer content of an entry. "Protein (only)" excludes any
# entry containing DNA or RNA, matching the Benchmark-IV constraint that its
# multimers are protein-only.
POLYMER_TYPES = "rcsb_entry_info.selected_polymer_entity_types"

# Number of DISTINCT protein polymer entities -- >= 2 means a heteromer.
N_PROTEIN_ENTITIES = "rcsb_entry_info.polymer_entity_count_protein"

# Number of polymer CHAINS in the deposited model -- >= 2 in a protein-only
# entry means a multimer, counting homomers as well as heteromers.
N_POLYMER_CHAINS = "rcsb_entry_info.deposited_polymer_entity_instance_count"


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as fh:
        return json.load(fh)


def _terminal(attribute: str, operator: str, value) -> dict:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attribute, "operator": operator, "value": value},
    }


def count(nodes: list[dict]) -> int:
    """Number of PDB entries matching the conjunction of `nodes`."""
    query = (
        nodes[0]
        if len(nodes) == 1
        else {"type": "group", "logical_operator": "and", "nodes": nodes}
    )
    payload = {
        "query": query,
        "return_type": "entry",
        "request_options": {"return_counts": True},
    }
    return _post(SEARCH_API, payload)["total_count"]


def released_after(date: str) -> dict:
    return _terminal(RELEASE_DATE, "greater", date)


def released_between(start: str, end: str) -> dict:
    """Entries released after `start` and on/before `end`."""
    return {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            _terminal(RELEASE_DATE, "greater", start),
            _terminal(RELEASE_DATE, "less_or_equal", end),
        ],
    }


def report_counts() -> None:
    protein_only = _terminal(POLYMER_TYPES, "exact_match", "Protein (only)")
    protein_na = _terminal(POLYMER_TYPES, "exact_match", "Protein/NA")
    multimer = _terminal(N_POLYMER_CHAINS, "greater_or_equal", 2)
    heteromer = _terminal(N_PROTEIN_ENTITIES, "greater_or_equal", 2)

    windows = {
        f"released after AF3 cutoff ({CUTOFFS['af3']})": [released_after(CUTOFFS["af3"])],
        f"  ...and on/before OB0 cutoff ({CUTOFFS['openbind0']})": [
            released_between(CUTOFFS["af3"], CUTOFFS["openbind0"])
        ],
        f"released after OB0 cutoff ({CUTOFFS['openbind0']})": [
            released_after(CUTOFFS["openbind0"])
        ],
    }

    slices = {
        "all entries": [],
        "protein-only, >=2 chains (multimer)": [protein_only, multimer],
        "protein-only, >=2 distinct protein entities (heteromer)": [
            protein_only,
            heteromer,
        ],
        "protein/NA, >=2 chains": [protein_na, multimer],
    }

    width = max(len(s) for s in slices)
    for window_label, window_nodes in windows.items():
        print(f"\n{window_label}")
        for slice_label, slice_nodes in slices.items():
            n = count(window_nodes + slice_nodes)
            print(f"  {slice_label:<{width}}  {n:>8,}")


def read_entry_ids(tsv: str, column: str) -> list[str]:
    with open(tsv) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = header.index(column)
        ids = {line.split("\t")[idx].strip().upper() for line in fh if line.strip()}
    return sorted(i for i in ids if i)


def fetch_release_dates(entry_ids: list[str], chunk: int = 100) -> dict[str, str]:
    """Release date per entry, via the RCSB Data API (GraphQL, batched)."""
    dates: dict[str, str] = {}
    for i in range(0, len(entry_ids), chunk):
        batch = entry_ids[i : i + chunk]
        query = """
        query($ids: [String!]!) {
          entries(entry_ids: $ids) {
            rcsb_id
            rcsb_accession_info { initial_release_date }
          }
        }
        """
        data = _post(DATA_API, {"query": query, "variables": {"ids": batch}})
        for entry in data["data"]["entries"] or []:
            date = entry["rcsb_accession_info"]["initial_release_date"]
            dates[entry["rcsb_id"]] = date.split("T")[0]
    return dates


def report_entries(tsv: str, column: str) -> None:
    entry_ids = read_entry_ids(tsv, column)
    dates = fetch_release_dates(entry_ids)
    missing = [e for e in entry_ids if e not in dates]

    buckets: Counter[str] = Counter()
    for entry_id in entry_ids:
        date = dates.get(entry_id)
        if date is None:
            buckets["not found in PDB"] += 1
        elif date <= CUTOFFS["af3"]:
            buckets[f"on/before AF3 cutoff ({CUTOFFS['af3']})"] += 1
        elif date <= CUTOFFS["openbind0"]:
            buckets[
                f"after AF3 cutoff, on/before OB0 cutoff ({CUTOFFS['openbind0']})"
            ] += 1
        else:
            buckets[f"after OB0 cutoff ({CUTOFFS['openbind0']})"] += 1

    print(f"\n{len(entry_ids)} unique entry IDs from {tsv} (column '{column}')")
    width = max(len(k) for k in buckets)
    for label, n in buckets.most_common():
        print(f"  {label:<{width}}  {n:>4}  ({n / len(entry_ids):.1%})")
    if missing:
        print(f"  missing: {', '.join(missing)}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    sub.add_parser("counts", help="PDB-wide counts by release window")

    p_entries = sub.add_parser("entries", help="release dates for a list of PDB IDs")
    p_entries.add_argument("--tsv", default="results/dockq.tsv")
    p_entries.add_argument("--column", default="entry_id")

    args = parser.parse_args()
    if args.mode == "counts":
        report_counts()
    else:
        report_entries(args.tsv, args.column)


if __name__ == "__main__":
    main()
