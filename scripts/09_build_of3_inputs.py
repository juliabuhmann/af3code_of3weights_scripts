#!/usr/bin/env python3
"""
Stage 09 — build OpenFold3 query JSONs from the AF3 input JSONs, MSAs only.

This is what lets the OF3 codebase run on exactly the alignments the AF3 arms used,
so a three-way comparison (AF3 code + AF3 weights, AF3 code + OpenFold3 weights,
OF3 code + OpenFold3 weights) differs in code and weights but not in MSAs. Nothing
is recomputed: the alignments already sit inline in stage 03's output.

TEMPLATES ARE NOT CONVERTED — templates off in both arms
--------------------------------------------------------
The emitted queries carry no `template_cif_paths`, so OF3 must be run with
`--use-templates=false`, and the AF3 arms it is compared against must also run
without templates. Converting AF3's per-chain template mmCIFs into a form OF3 will
actually apply is possible but fiddly (OF3 requires a canonical-sequence field AF3
omits, a flat shared structure directory named in the runner yaml, and
underscore-free filenames — and it drops templates silently when any of these is
wrong). That work is kept out of this repository; see
`pooled_ppi_comparison/scripts/2026sep_of3_template_conversion/`.

The number of templates present in the source JSONs and therefore dropped here is
counted and printed, so a templates-off OF3 arm is a recorded fact rather than an
assumption. slurm/predict_of3.sbatch re-checks the emitted queries against the arm's
`use_templates` setting before it spends GPU time.

Writes, per entry, under --out-dir/<entry>/:

    msa/<entity>/uniref90_hits.a3m   AF3 unpairedMsa, verbatim
    msa/<entity>/uniprot_hits.a3m    AF3 pairedMsa, headers rewritten
    query.json                       OF3 query set

and one array list (--todo, default <out-dir>/of3_todo.txt), ordered LARGEST COMPLEX
FIRST for the same reason stage 03 does it: with an array job the long tail dominates
wall time, so starting the big complexes early shortens time-to-last-result.

FILE NAMING IS LOAD-BEARING
---------------------------
OF3 parses an MSA file only if its stem appears in `max_seq_counts`, and only
concatenates it into the main MSA if the stem is in `aln_order`
(openfold3/projects/of3_all_atom/config/dataset_config_components.py). Both
`uniref90_hits` and `uniprot_hits` are in OF3's defaults, and `uniprot_hits` is
already listed in `msas_to_pair`, so these two names need no runner.yml override.
Renaming them silently drops the alignments.

Both files go into `main_msa_file_paths`; OF3 does its own cross-chain pairing from
the UniProt headers, so `paired_msa_file_paths` is not used.

PAIRED-MSA HEADER REWRITE
-------------------------
OF3's online pairing reads a species id from the 4th of 6 fields obtained by
splitting the header on [|_/:-]. Raw AF3 UniProt headers do not split that way —
they carry description text, isoform suffixes (Q8BGS0-2) and underscores inside
entry names — so non-query headers are rewritten to

    >db|acc|acc|species|start|end

Species comes from OX= in the description when present, else from the suffix of the
UniProt entry name. Ported from
~/src/pooled_ppi_comparison/benchmarks/yeast_real/af3msa_to_of3msa.py.

Caveat: OF3 caps `uniref90_hits` at 10000 rows by default and AF3 applies its own,
different cap, so for deep-MSA chains the two engines do not necessarily see an
identical set of rows.

Usage:
    python scripts/09_build_of3_inputs.py \
        --af3-input-dir <base>/work/af3_input \
        --out-dir       <base>/work/of3_input

    # one or a few entries
    python scripts/09_build_of3_inputs.py \
        --af3-input-dir <base>/work/af3_input \
        --out-dir       <base>/work/of3_input \
        --entries 8r3c 8k44
"""

import argparse
import json
import re
import sys
from pathlib import Path

# AF3 header -> OF3-pairable header. See PAIRED-MSA HEADER REWRITE above.
_HEADER_RE = re.compile(
    r"^(?P<db>[^|]+)\|(?P<acc>[^|]+)\|(?P<entry>[^/\s]+)(?:/(?P<start>\d+)-(?P<end>\d+))?$"
)
_OX_RE = re.compile(r"\bOX=(\d+)")
_ISOFORM_RE = re.compile(r"-\d+$")
_SPECIES_RE = re.compile(r"_([A-Z0-9]+)$")

DEFAULT_SEED = 42


def reformat_header(line: str) -> str:
    """Rewrite one UniProt-style a3m header to >db|acc|acc|species|start|end."""
    body = line.lstrip(">")
    parts = body.split(None, 1)
    token = parts[0]
    description = parts[1] if len(parts) > 1 else ""

    match = _HEADER_RE.match(token)
    if match is None:
        return line                      # unrecognised: pass through untouched

    acc = _ISOFORM_RE.sub("", match.group("acc"))
    ox = _OX_RE.search(description)
    if ox:
        species = ox.group(1)
    else:
        sp = _SPECIES_RE.search(match.group("entry"))
        species = sp.group(1) if sp else ""

    return (f">{match.group('db')}|{acc}|{acc}|{species}"
            f"|{match.group('start') or '1'}|{match.group('end') or '1'}")


def convert_paired_headers(a3m: str) -> tuple[str, int, int]:
    """Rewrite every non-query header. Returns (text, n_rewritten, n_passed)."""
    out, rewritten, passed = [], 0, 0
    first = True
    for line in a3m.splitlines():
        if not line.startswith(">"):
            out.append(line)
            continue
        if first:                        # the query row keeps its own header
            out.append(line)
            first = False
            continue
        new = reformat_header(line)
        out.append(new)
        if new == line:
            passed += 1
        else:
            rewritten += 1
    tail = "\n" if a3m.endswith("\n") else ""
    return "\n".join(out) + tail, rewritten, passed


def count_rows(a3m: str) -> int:
    return a3m.count("\n>") + (1 if a3m.startswith(">") else 0)


def convert(af3_json: Path, out_dir: Path, seed: int) -> dict:
    entry_id = af3_json.stem
    root = out_dir / entry_id

    with open(af3_json) as fh:
        data = json.load(fh)

    non_protein = sorted({k for entity in data.get("sequences", [])
                          for k in entity if k != "protein"})
    if non_protein:
        sys.exit(f"ERROR: {entry_id} has non-protein entities {non_protein}; "
                 "this converter handles protein chains only.")

    chains = []
    stats = {"entry_id": entry_id, "entities": 0, "total_len": 0,
             "templates_dropped": 0, "rewritten": 0, "passed": 0, "rows": []}

    for entity in data["sequences"]:
        protein = entity["protein"]
        ids = protein["id"]
        ids = [ids] if isinstance(ids, str) else list(ids)
        key = ids[0]                     # one directory per entity, by first chain

        # Counted, not silently ignored: this is the evidence that the OF3 arm ran
        # templates-off and how much it gave up relative to a templates-on AF3 arm.
        stats["templates_dropped"] += len(protein.get("templates") or [])
        stats["total_len"] += len(protein["sequence"]) * len(ids)

        unpaired = protein.get("unpairedMsa") or ""
        paired = protein.get("pairedMsa") or ""
        if not unpaired and not paired:
            sys.exit(f"ERROR: {entry_id} entity {key} has no MSA in the AF3 JSON.")

        msa_dir = root / "msa" / key
        msa_dir.mkdir(parents=True, exist_ok=True)

        unpaired_path = msa_dir / "uniref90_hits.a3m"
        paired_path = msa_dir / "uniprot_hits.a3m"
        unpaired_path.write_text(unpaired)
        converted, rewritten, passed = convert_paired_headers(paired)
        paired_path.write_text(converted)

        stats["rewritten"] += rewritten
        stats["passed"] += passed
        stats["rows"].append((key, count_rows(unpaired), count_rows(paired)))

        main_msa = [str(unpaired_path)]
        if paired.strip():
            main_msa.append(str(paired_path))

        chains.append({
            "molecule_type": "protein",
            "chain_ids": ids,
            "sequence": protein["sequence"],
            "main_msa_file_paths": main_msa,
        })
        stats["entities"] += 1

    payload = {"seeds": [seed], "queries": {entry_id: {"chains": chains}}}
    (root / "query.json").write_text(json.dumps(payload, indent=2) + "\n")
    stats["query"] = str(root / "query.json")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--af3-input-dir", required=True, type=Path,
                    help="<base>/work/af3_input, written by stage 03")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Destination, e.g. <base>/work/of3_input")
    ap.add_argument("--entries", nargs="*", default=None,
                    help="Restrict to these entry ids (default: all)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"Model seed, matched to the AF3 arms (default: {DEFAULT_SEED})")
    ap.add_argument("--todo", type=Path, default=None,
                    help="Where to write the array list "
                         "(default: <out-dir>/of3_todo.txt)")
    ap.add_argument("--quiet", action="store_true",
                    help="One line per entry instead of per-entity detail")
    args = ap.parse_args()

    paths = sorted(args.af3_input_dir.glob("*.json"))
    if args.entries:
        wanted = set(args.entries)
        paths = [p for p in paths if p.stem in wanted]
        missing = wanted - {p.stem for p in paths}
        if missing:
            sys.exit(f"ERROR: no input JSON for: {', '.join(sorted(missing))}")
    if not paths:
        sys.exit(f"ERROR: no input JSONs found under {args.af3_input_dir}")

    print(f"Converting {len(paths)} entries -> {args.out_dir}")
    print("Templates are NOT included: run OF3 with --use-templates=false\n")

    converted = []
    total_rewritten = total_passed = total_templates = 0
    for path in paths:
        stats = convert(path, args.out_dir, args.seed)
        converted.append(stats)
        total_rewritten += stats["rewritten"]
        total_passed += stats["passed"]
        total_templates += stats["templates_dropped"]
        if args.quiet:
            print(f"{stats['entry_id']}: {stats['entities']} entities, "
                  f"{stats['total_len']} aa, "
                  f"{stats['templates_dropped']} template(s) dropped")
        else:
            print(f"{stats['entry_id']}: {stats['entities']} entities, "
                  f"{stats['total_len']} aa, "
                  f"{stats['templates_dropped']} template(s) dropped")
            for key, n_unpaired, n_paired in stats["rows"]:
                print(f"  entity {key}: unpaired {n_unpaired} rows, "
                      f"paired {n_paired} rows")

    # Largest complex first — see the docstring.
    converted.sort(key=lambda s: -s["total_len"])
    todo = [s["entry_id"] for s in converted]

    todo_path = args.todo or (args.out_dir / "of3_todo.txt")
    todo_path.parent.mkdir(parents=True, exist_ok=True)
    todo_path.write_text("\n".join(todo) + "\n")

    n_with = sum(1 for s in converted if s["templates_dropped"])
    print(f"\nTemplates: {total_templates} dropped across {n_with} of {len(converted)} "
          f"entries — these queries are templates-off by construction.")
    print(f"\nPaired headers: {total_rewritten} rewritten, {total_passed} left as-is")
    if total_passed:
        print("  NOTE: headers left as-is did not match the expected UniProt form, so "
              "OF3 cannot read a species id from them and will not pair those rows.")
    print(f"Wrote {todo_path} ({len(todo)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
