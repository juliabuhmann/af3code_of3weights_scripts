#!/usr/bin/env python3
"""
Report — and optionally assert — what an OpenFold3 query JSON actually contains.

WHY THIS EXISTS
---------------
OpenFold3 discards templates silently: it can drop every one and still print
`Preprocessing templates: 100%`, exit 0, and write structures. A templates
experiment run that way measures nothing while looking like it succeeded. The
counterpart risk for a templates-OFF arm is milder but real: `--use-templates=false`
paired with a query that does carry `template_cif_paths` is a contradiction worth
failing on rather than silently resolving one way or the other.

So the launcher checks the query against the arm's declared setting BEFORE spending
GPU time, and records the counts in meta.json afterwards.

Prints one line to stdout for the caller to read:

    <n_queries> <n_chains> <n_template_paths> <n_msa_paths> <seeds_csv>

With --require-templates, exits non-zero when the query disagrees:
    false   any template_cif_paths present is an error
    true    zero template_cif_paths is an error

Usage:
    python slurm/of3_query_info.py --query work/of3_input/8r3c/query.json
    python slurm/of3_query_info.py --query work/of3_input/8r3c/query.json \
                                   --require-templates false
"""

import argparse
import json
import sys
from pathlib import Path


def survey(payload: dict) -> tuple[int, int, int, int, list]:
    queries = payload.get("queries", {})
    n_chains = n_templates = n_msas = 0
    for query in queries.values():
        for chain in query.get("chains", []):
            n_chains += len(chain.get("chain_ids") or [])
            n_templates += len(chain.get("template_cif_paths") or [])
            n_msas += len(chain.get("main_msa_file_paths") or [])
            n_msas += len(chain.get("paired_msa_file_paths") or [])
        n_templates += len(query.get("template_cif_paths") or [])
    return len(queries), n_chains, n_templates, n_msas, payload.get("seeds", [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", required=True, type=Path)
    ap.add_argument("--require-templates", choices=["true", "false"], default=None,
                    help="Fail if the query disagrees with this expectation")
    args = ap.parse_args()

    payload = json.loads(args.query.read_text())
    n_queries, n_chains, n_templates, n_msas, seeds = survey(payload)

    if args.require_templates == "false" and n_templates:
        print(f"ERROR: {args.query} carries {n_templates} template_cif_paths but this "
              f"arm declares use_templates=false", file=sys.stderr)
        return 1
    if args.require_templates == "true" and not n_templates:
        print(f"ERROR: {args.query} carries no template_cif_paths but this arm "
              f"declares use_templates=true — OpenFold3 would run without templates "
              f"and report success anyway", file=sys.stderr)
        return 1
    if not n_msas:
        print(f"ERROR: {args.query} lists no MSA files", file=sys.stderr)
        return 1

    seeds_csv = ",".join(str(s) for s in seeds) or "none"
    print(f"{n_queries} {n_chains} {n_templates} {n_msas} {seeds_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
