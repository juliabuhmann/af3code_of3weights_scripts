"""
Where figures and their tables go, what they are called, and how every arm is drawn.

ONE NAME PER ARM. Every arm config carries a `short` -- af3-ob, of3kit-big, af3-ob-a100 --
and that string is the arm everywhere a person sees it: on the command line, in the legend,
and in the path the figure is written to. A figure and its location therefore always agree,
and no reader has to hold a mapping in their head.

WHERE FIGURES GO. One arm alone, or several compared, each in a directory named after the
arms it holds, with fixed file names inside so a re-run overwrites rather than accumulates:

    results/<experiment>/individual_runs/<short>/
    results/<experiment>/sweep/<short1>_<short2>_<short3>/

`experiment` is what was measured: `scaling` (memory and time against token count, on the
pool ladder, profiling/plot_vram.py) or `benchmark` (DockQ and time per complex on
Benchmark-IV, scripts/06_plot_comparison.py).

A comparison is named by its arms, in the order given, and the FIRST arm is the baseline
every ratio and every paired figure is computed against. There is no registry of named
comparisons: the directory name is the record of what was compared, which is the only thing
a registry was for.

COLOUR IS THE ONLY CHANNEL. Each arm has one colour in configs/arm_style.json, used in every
figure it appears in; every series is drawn with the same marker and a solid line, so a
figure carries one visual variable rather than three. Two arms that never share a figure may
share a colour (the kit modes mean the same thing in the OpenFold3 and the AlphaFold 3 kit,
so they wear the same hues); `styles_for` refuses to draw a figure in which two arms collide.
The eight hues are a validated categorical palette; do not add a ninth.

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
CONFIG_DIR = REPO / "configs"
STYLE_FILE = CONFIG_DIR / "arm_style.json"
RESULTS_ROOT = REPO / "results"

EXPERIMENTS = ("scaling", "benchmark")
# Dropped from every table copied into the repository: they name the compute host, not the
# measurement, and this repository is public.
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


def _check_experiment(experiment: str) -> None:
    if experiment not in EXPERIMENTS:
        sys.exit(f"ERROR: experiment must be one of {EXPERIMENTS}, not {experiment!r}")


def individual_dir(experiment: str, short: str, root: Path = RESULTS_ROOT) -> Path:
    _check_experiment(experiment)
    return root / experiment / "individual_runs" / short


def sweep_dir(experiment: str, shorts: list[str], root: Path = RESULTS_ROOT) -> Path:
    """results/<experiment>/sweep/<short1>_<short2>_..., the arms in the order given."""
    _check_experiment(experiment)
    name = "_".join(shorts)
    if len(name.encode()) > NAME_MAX:
        sys.exit(f"ERROR: {len(shorts)} arms make a {len(name)}-character directory name, "
                 f"over the {NAME_MAX} limit. Compare fewer arms at once.")
    return root / experiment / "sweep" / name


# ------------------------------------------------------------------------------ arms

class Config:
    """One arm config: the short name a person uses, the directory its data is under."""

    def __init__(self, path: Path, data: dict):
        self.path, self.data = path, data
        self.short = data["short"]
        self.name = data["name_tag"]
        self.label = data.get("label") or self.short
        self.engine = data.get("engine", "")

    def __repr__(self) -> str:
        return f"Config({self.short} -> {self.name})"


def load_configs(config_dir: Path = CONFIG_DIR) -> dict[str, Config]:
    """{short: Config} for every arm config, with the invariants a short name must hold.

    A short is what names a directory and labels a legend, so it must be unique and must
    contain neither the separator that joins arms into a comparison nor a path separator --
    otherwise two different comparisons could write to one directory."""
    out: dict[str, Config] = {}
    for path in sorted(config_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            sys.exit(f"ERROR: {path}: {exc}")
        if not isinstance(data, dict) or "short" not in data:
            continue                                  # not an arm config
        if "name_tag" not in data:
            sys.exit(f"ERROR: {path} has a `short` but no `name_tag`")
        short = data["short"]
        if "_" in short or "/" in short or not short:
            sys.exit(f"ERROR: {path}: short {short!r} must be non-empty and contain no "
                     f"'_' (which joins arms into a comparison) and no '/'")
        if short in out:
            sys.exit(f"ERROR: short {short!r} is used by both {out[short].path.name} "
                     f"and {path.name}")
        out[short] = Config(path, data)
    return out


def resolve_arms(shorts: list[str], config_dir: Path = CONFIG_DIR) -> list[Config]:
    """The configs for the short names given, in the order given, or a listing and an exit."""
    configs = load_configs(config_dir)
    if not shorts:
        sys.exit("ERROR: pass --arms SHORT [SHORT ...]. Known arms:\n  "
                 + "\n  ".join(f"{s:20s} {c.label}" for s, c in sorted(configs.items())))
    unknown = [s for s in shorts if s not in configs]
    if unknown:
        sys.exit(f"ERROR: no arm config with short {', '.join(unknown)}. Known arms:\n  "
                 + "\n  ".join(f"{s:20s} {c.label}" for s, c in sorted(configs.items())))
    seen = [s for i, s in enumerate(shorts) if s in shorts[:i]]
    if seen:
        sys.exit(f"ERROR: {', '.join(sorted(set(seen)))} listed more than once")
    return [configs[s] for s in shorts]


# ----------------------------------------------------------------------------- style

def apply_style() -> None:
    """One look for every figure in the repository: quiet axes, hairline grid, the data as
    the only loud thing. Font sizes stay large enough for a figure dropped into a document
    at half width."""
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
# One marker and one line style for every series: colour is the identity channel.
MARKER = "o"
LINESTYLE = "-"


def styles_for(shorts: list[str], path: Path = STYLE_FILE) -> dict[str, dict]:
    """{short: {"color", ...}} for the arms of one figure. Refuses an arm with no entry
    (every arm has a fixed colour, or the promise is empty) and two arms in the same figure
    wearing the same colour."""
    try:
        styles = {k: v for k, v in json.load(open(path)).items() if not k.startswith("_")}
    except OSError as exc:
        sys.exit(f"ERROR: cannot read {path}: {exc}")
    missing = [s for s in shorts if s not in styles]
    if missing:
        sys.exit(f"ERROR: no entry in {path.relative_to(REPO)} for {', '.join(missing)}. "
                 f"Add one, picking a hue no other arm in the same figure uses.")
    seen: dict[str, str] = {}
    for short in shorts:
        colour = styles[short]["color"].lower()
        if colour in seen:
            sys.exit(f"ERROR: {short} and {seen[colour]} share the colour {colour} but are "
                     f"in the same figure. Give one of them a different hue in "
                     f"{path.relative_to(REPO)}.")
        seen[colour] = short
    return {s: styles[s] for s in shorts}


def title(ax, headline: str, subtitle: str | None = None) -> None:
    """A figure's own headline, with what it belongs to as a quieter line beneath.

    The headline names the measurement, so a memory figure and a time figure are told apart
    by their titles and not only by their axes."""
    lines = 1 + (subtitle.count("\n") if subtitle else 0)
    ax.set_title(headline, pad=14 + 16 * lines if subtitle else 12)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, va="bottom", ha="left",
                fontsize=11.5, color=INK_TEXT, linespacing=1.35)


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
    """Rows as a TSV with the host columns removed; the column order is the union in order
    of first appearance, so an arm swept before a column existed does not truncate the
    others."""
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
    published manifest names the file inside the data tree without naming this cluster."""
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


def write_manifest(out_dir: Path, *, experiment: str, script: str, arms: list[Config],
                   baseline: str | None, sources: dict[str, dict], figures: list[Path],
                   extra: dict | None = None) -> Path:
    """What this directory was made from: the arms in legend order with the config each
    short resolved to, the baseline, each arm's source table and row count, the figures
    written, and the code that wrote them."""
    manifest = {
        "experiment": experiment,
        "arms": [{"short": c.short, "config": c.path.name, "name_tag": c.name,
                  "label": c.label} for c in arms],
        "baseline": baseline,
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
