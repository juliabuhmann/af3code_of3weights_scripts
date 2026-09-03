"""
One result table per arm, so adding an arm never rescores the others.

WHY PER-ARM FILES
-----------------
Scoring used to write a single dockq.tsv covering every arm, which meant learning
about one new arm cost a full rescore of all of them — 206 DockQ calls per arm,
growing linearly with the number of arms. The tables here are split instead:

    <base>/results/dockq/<arm>.tsv        one per arm, from scripts/05
    <base>/results/profiling/<arm>.tsv    one per arm, from scripts/04

The file is named by the arm's `name_tag`, which is one-to-one with
predictions/<name_tag>/, so a table can always be traced back to the directory it
describes. The pipeline axes (engine, weights, use_templates) are columns on every
row — see scripts/arm_metadata.py — so the split is a storage layout, not a change
in what is recorded.

There is deliberately NO merged file on disk. Anything that consumes results takes
a file OR a directory (`read_rows`) and concatenates on the fly, so a stale merge can
never disagree with the per-arm tables it came from. scripts/08_publish_results.py is
what flattens them into the single tables committed to the repository.

RESUME
------
`load_scored` lets stage 05 skip entries it has already scored successfully, so
re-running after adding a few entries costs a few DockQ calls rather than all of
them. Rows whose status is not "ok" are NOT treated as done: a timeout or a missing
prediction should be retried once the cause is fixed.
"""

import csv
from pathlib import Path

# Rows are keyed on this within an arm's table.
KEY_COLUMN = "entry_id"


def arm_table_path(out_dir: Path, label: str) -> Path:
    """<out_dir>/<label>.tsv — the table describing one arm."""
    return Path(out_dir) / f"{label}.tsv"


def read_rows(path: Path) -> list[dict]:
    """Rows from one .tsv, or from every .tsv in a directory, concatenated.

    Accepting either is what removes the need for a merged file on disk. Directory
    entries are read in sorted filename order so output is deterministic.
    """
    path = Path(path)
    files = sorted(path.glob("*.tsv")) if path.is_dir() else [path]
    rows: list[dict] = []
    for file in files:
        if not file.is_file():
            continue
        with open(file) as fh:
            rows.extend(csv.DictReader(fh, delimiter="\t"))
    return rows


def union_columns(rows: list[dict], preferred: list[str] | None = None) -> list[str]:
    """Column order for a concatenation: `preferred` first, then any extras seen.

    Arms scored at different times can have different columns (a new axis was added,
    an older arm predates it). Taking the union rather than the first row's keys means
    an older table never silently drops a newer column.
    """
    columns = list(preferred or [])
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def write_rows(path: Path, columns: list[str], rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(c, "")) for c in columns) + "\n")


def load_scored(out_dir: Path, label: str, ok_status: str = "ok") -> dict[str, dict]:
    """{entry_id: row} for entries already recorded with status `ok` for this arm.

    Anything not `ok` is omitted so it gets retried. Returns {} when the arm has no
    table yet, which is what makes a first run and a resumed run the same code path.
    """
    path = arm_table_path(out_dir, label)
    if not path.is_file():
        return {}
    done = {}
    for row in read_rows(path):
        if row.get("status") == ok_status and row.get(KEY_COLUMN):
            done[row[KEY_COLUMN]] = row
    return done
