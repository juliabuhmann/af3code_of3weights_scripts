"""
Where figures and their tables go, what they are called, and how every arm is drawn.

The layout is borrowed from structural_human_interactome: a comparison of arms is a
SWEEP, one arm on its own is an INDIVIDUAL RUN, and both live under a fixed tree with
fixed leaf names so a re-run overwrites rather than accumulates:

    results/<experiment>/individual_runs/<arm>/      one arm alone
    results/<experiment>/sweep/<set>/                the arms of one registered set

`experiment` is what was measured:

    scaling     peak memory and model time against token count, on the pool ladder
                (profiling/plot_vram.py)
    benchmark   DockQ and model time per complex on Benchmark-IV
                (scripts/06_plot_comparison.py)

A set is an entry in configs/figure_sets.json: its arms in legend order, a baseline
the ratio and paired figures divide by, and a title. Registering a set is what makes a
figure reproducible -- the directory name says which comparison it is, and the
manifest.json written beside the figures says which tables it was read from. Passing
arms on the command line still works for a quick look; the directory is then the arms
joined with "_vs_", and the module refuses a name longer than a file system allows
rather than truncating it.

Every arm has ONE colour, in configs/arm_style.json, used in every figure it appears in,
and colour is the only thing that varies: every series is drawn with the same marker and
a solid line, so a figure carries one visual variable rather than three. Two arms that
never share a figure may share a colour (the four kit modes mean the same thing in the
OpenFold3 and the AlphaFold 3 kit, so they wear the same hues); `styles_for` refuses to
draw a set in which two arms collide, so the rule is checked rather than remembered. The
eight hues are a validated categorical palette (colour-vision-deficiency separation and
normal-vision separation both checked pairwise in each set's legend order); do not add a
ninth.

Imported by profiling/plot_vram.py and scripts/06_plot_comparison.py.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SETS_FILE = REPO / "configs" / "figure_sets.json"
STYLE_FILE = REPO / "configs" / "arm_style.json"
RESULTS_ROOT = REPO / "results"

EXPERIMENTS = ("scaling", "benchmark")
# Dropped from every table copied into the repository: they name the compute host, not
# the measurement, and this repository is public.
HOST_COLUMNS = ("node", "slurm_job_id", "model_path")
DPI = 150
# The longest single path component most file systems accept.
NAME_MAX = 255


# ----------------------------------------------------------------------------- roots

def base_dir() -> Path:
    """$AF3OF3_BASE_DIR, the data tree the measurements live in."""
    base = os.environ.get("AF3OF3_BASE_DIR", "")
    if not base:
        sys.exit("ERROR: set AF3OF3_BASE_DIR (the data tree; see README.md)")
    return Path(base)


def individual_dir(experiment: str, arm: str, root: Path = RESULTS_ROOT) -> Path:
    _check_experiment(experiment)
    return root / experiment / "individual_runs" / arm


def sweep_dir(experiment: str, set_name: str | None, arms: list[str],
              root: Path = RESULTS_ROOT) -> Path:
    """<root>/<experiment>/sweep/<set>, or the arms joined with _vs_ for an unregistered
    comparison -- refused, not truncated, when that does not fit in a file name."""
    _check_experiment(experiment)
    if set_name:
        return root / experiment / "sweep" / set_name
    name = "_vs_".join(arms)
    if len(name.encode()) > NAME_MAX:
        sys.exit(f"ERROR: {len(arms)} arms joined make a {len(name)}-character directory "
                 f"name, over the {NAME_MAX} limit. Register the comparison as a set in "
                 f"{SETS_FILE.relative_to(REPO)} and pass --set instead.")
    return root / experiment / "sweep" / name


def _check_experiment(experiment: str) -> None:
    if experiment not in EXPERIMENTS:
        sys.exit(f"ERROR: experiment must be one of {EXPERIMENTS}, not {experiment!r}")


# ------------------------------------------------------------------------------ sets

def load_sets(path: Path = SETS_FILE, experiment: str | None = None) -> dict[str, dict]:
    """{set name: spec} from configs/figure_sets.json, optionally one experiment's."""
    try:
        sets = json.load(open(path))
    except OSError as exc:
        sys.exit(f"ERROR: cannot read {path}: {exc}")
    out = {}
    for name, spec in sets.items():
        if name.startswith("_"):
            continue
        for key in ("experiment", "arms", "title"):
            if key not in spec:
                sys.exit(f"ERROR: set {name!r} in {path} has no `{key}`")
        _check_experiment(spec["experiment"])
        if spec.get("baseline") and spec["baseline"] not in spec["arms"]:
            sys.exit(f"ERROR: set {name!r}: baseline {spec['baseline']!r} is not among its arms")
        if experiment is None or spec["experiment"] == experiment:
            out[name] = spec
    return out


def resolve_selection(args, experiment: str) -> tuple[str | None, list[str], str | None, str | None]:
    """(set name, arms, baseline, title) from --set or --arms [--baseline]."""
    if args.set_name and args.arms:
        sys.exit("ERROR: pass --set or --arms, not both")
    if args.set_name:
        sets = load_sets(args.sets_file, experiment)
        spec = sets.get(args.set_name)
        if spec is None:
            sys.exit(f"ERROR: no {experiment} set {args.set_name!r} in {args.sets_file} "
                     f"(have: {', '.join(sets) or 'none'})")
        baseline = args.baseline or spec.get("baseline")
        return args.set_name, list(spec["arms"]), baseline, spec["title"]
    if not args.arms:
        sys.exit("ERROR: pass --set NAME (see configs/figure_sets.json) or --arms ARM [ARM ...]")
    arms = list(args.arms)
    baseline = args.baseline
    if baseline and baseline not in arms:
        arms = [baseline] + arms
    return None, arms, baseline, None


# ----------------------------------------------------------------------------- style

def apply_style() -> None:
    """One look for every figure in the repository: quiet axes, hairline grid, the data
    as the only loud thing. Font sizes stay large enough for a figure dropped into a
    document at half width."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 13,
        "font.family": "sans-serif",
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 13,
        "axes.labelcolor": "#3b3b38",
        "axes.edgecolor": "#c9c9c4",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.facecolor": "#fcfcfb",
        "figure.facecolor": "white",
        "grid.color": "#e8e8e4",
        "grid.linewidth": 1.0,
        "grid.linestyle": "-",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.color": "#6b6b66",
        "ytick.color": "#6b6b66",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "text.color": "#22221f",
        "legend.fontsize": 11.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "lines.linewidth": 2.0,
        "lines.markersize": 7.5,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": "white",
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "savefig.facecolor": "white",
    })


# Recessive ink for reference lines, thresholds and annotations -- never a series hue.
INK_MUTED = "#9a9a94"
INK_FAINT = "#c9c9c4"
INK_TEXT = "#5a5a55"


def load_styles(path: Path = STYLE_FILE) -> dict[str, dict]:
    try:
        styles = json.load(open(path))
    except OSError as exc:
        sys.exit(f"ERROR: cannot read {path}: {exc}")
    return {k: v for k, v in styles.items() if not k.startswith("_")}


# One marker and one line style for every series: colour is the identity channel, and it
# is the only thing that varies between arms.
MARKER = "o"
LINESTYLE = "-"


def line_width(index: int, total: int) -> float:
    """Series width, tapering from 3.2 to 2.0 down the legend order.

    Not an encoding -- nothing reads it, and the taper is small enough to pass for one
    weight. It exists because some arms measure the same thing and their curves coincide
    exactly (the kit's `off` and `exact` modes; `big` below its reach gate and `big` with
    the gate held open). Drawn at one width the later arm would hide the earlier one
    completely; drawn this way the earlier one shows as a rim around it."""
    if total < 2:
        return 2.4
    return 3.2 - 1.2 * index / (total - 1)


def styles_for(arms: list[str], path: Path = STYLE_FILE) -> dict[str, dict]:
    """{arm: {"color", ...}} for the arms of one figure. Refuses an arm with no entry
    (every arm has a fixed colour, or the promise is empty) and two arms in the same
    figure wearing the same colour."""
    styles = load_styles(path)
    missing = [a for a in arms if a not in styles]
    if missing:
        sys.exit(f"ERROR: no entry in {path.relative_to(REPO)} for {', '.join(missing)}. "
                 f"Add one, picking a hue no other arm in the same set uses.")
    seen: dict[str, str] = {}
    for arm in arms:
        colour = styles[arm]["color"].lower()
        if colour in seen:
            sys.exit(f"ERROR: {arm} and {seen[colour]} share the colour {colour} but are in the "
                     f"same figure. Give one of them a different hue in {path.relative_to(REPO)}.")
        seen[colour] = arm
    return {a: styles[a] for a in arms}


def outside_legend(ax, **kwargs) -> None:
    """Legend to the right of the axes, so it can never cover a data point."""
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, **kwargs)


def save(fig, path: Path, note: str = "") -> Path:
    """bbox_inches='tight' is what keeps a long title or an outside legend from being
    cropped; tight_layout() would reserve space only for artists inside the axes."""
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  {path}{'  ' + note if note else ''}")
    return path


# ---------------------------------------------------------------------------- tables

def write_table(path: Path, rows: list[dict], drop: tuple[str, ...] = HOST_COLUMNS) -> Path:
    """Rows as a TSV with the host columns removed; the column order is the union in
    order of first appearance, so an arm scored before a column existed does not
    truncate the others."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns and key not in drop:
                columns.append(key)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path}  ({len(rows)} rows)")
    return path


def relative_source(path: Path | str | None) -> str | None:
    """A source table's path as the manifest records it: relative to $AF3OF3_BASE_DIR, so a
    published manifest names the file inside the data tree without naming this cluster.
    Same reason HOST_COLUMNS are dropped from every table."""
    if path is None:
        return None
    path = Path(path)
    base = os.environ.get("AF3OF3_BASE_DIR", "")
    if base:
        try:
            return "$AF3OF3_BASE_DIR/" + str(path.relative_to(base))
        except ValueError:
            pass
    return str(path)


def git_commit() -> str:
    """HEAD of this repository, with '+dirty' when the working tree differs."""
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("+dirty" if dirty else "")


def write_manifest(out_dir: Path, *, experiment: str, script: str, set_name: str | None,
                   title: str | None, arms: list[str], baseline: str | None,
                   sources: dict[str, dict], figures: list[Path],
                   missing: list[str] | None = None, extra: dict | None = None) -> Path:
    """What this directory was made from: the arms in legend order, the baseline, each
    arm's source table and row count, the figures written, and the code that wrote
    them. The provenance link the directory name alone cannot carry."""
    manifest = {
        "experiment": experiment,
        "set": set_name,
        "title": title,
        "arms": arms,
        "baseline": baseline,
        "missing_arms": missing or [],
        "sources": sources,
        "figures": sorted(p.name for p in figures),
        "script": script,
        "git_commit": git_commit(),
    }
    if extra:
        manifest.update(extra)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  {path}")
    return path
