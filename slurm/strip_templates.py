#!/usr/bin/env python3
"""
Apply an arm's `use_templates` setting to one AlphaFold3 input JSON.

WHY THIS EXISTS
---------------
AlphaFold3 has no inference-time flag for templates. The complete flag list in
run_alphafold.py offers only --max_template_date, which configures the data
pipeline — and these arms run with --norun_data_pipeline, because MSAs and
templates are already inline in the input JSON (stage 03). So the only place a
templates-off AF3 arm can be expressed is the JSON's `templates` field, where an
empty list is AlphaFold3's own "use no templates" value:

    folding_input.py:303-305   templates = None  # Search for templates ...
                               templates = []    # Do not use any templates.

and where, under --norun_data_pipeline, a null or absent `templates` would instead
mean "unfilled" rather than "none".

Rather than fork work/af3_input/ into a second templates-off tree, slurm/predict.sbatch
calls this on the ONE shared input JSON and writes the stripped copy to node-local
scratch for that task. Both arms therefore read the same file for their MSAs, so the
alignments are identical by construction and not merely by intent.

Called with --output it writes the stripped copy and re-reads it to confirm no
template survived, exiting non-zero if any did. Called without, it only counts —
which is how the templates-ON arm records what it actually had.

Prints one line to stdout for the caller to read:

    <n_templates_in> <n_templates_out> <n_entities> <n_entities_with_templates_in>

Usage:
    python slurm/strip_templates.py --input  work/af3_input/8r3c.json
    python slurm/strip_templates.py --input  work/af3_input/8r3c.json \
                                    --output $TMPDIR/8r3c.json
"""

import argparse
import json
import sys
from pathlib import Path


def count_templates(payload: dict) -> tuple[int, int, int]:
    """(n_templates, n_entities, n_entities_with_templates) over protein entities."""
    n_templates = n_entities = n_with = 0
    for entity in payload.get("sequences", []):
        protein = entity.get("protein")
        if protein is None:
            continue
        n_entities += 1
        templates = protein.get("templates") or []
        n_templates += len(templates)
        if templates:
            n_with += 1
    return n_templates, n_entities, n_with


def strip_templates(payload: dict) -> dict:
    """Set `templates` to [] on every protein entity, leaving MSAs untouched."""
    for entity in payload.get("sequences", []):
        protein = entity.get("protein")
        if protein is not None:
            protein["templates"] = []
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path,
                    help="AlphaFold3 input JSON from stage 03 (read only)")
    ap.add_argument("--output", type=Path, default=None,
                    help="Write a templates-stripped copy here. Omit to only count.")
    args = ap.parse_args()

    payload = json.loads(args.input.read_text())
    n_in, n_entities, n_with = count_templates(payload)

    if args.output is None:
        n_out = n_in
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(strip_templates(payload), indent=2) + "\n")
        # Re-read rather than trust the in-memory edit: this file is what inference
        # will consume, and a templates-off arm that silently kept its templates
        # would measure nothing while looking like it worked.
        n_out, _, still_with = count_templates(json.loads(args.output.read_text()))
        if n_out or still_with:
            print(f"ERROR: {n_out} template(s) survived stripping in {args.output}",
                  file=sys.stderr)
            return 1

    print(f"{n_in} {n_out} {n_entities} {n_with}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
