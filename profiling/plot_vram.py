#!/usr/bin/env python3
"""
Peak GPU memory and the three time axes against token count, per arm, from the pool ladder.

Arms are named by their config's `short` (af3-ob, of3kit-big, af3-ob-a100), and a comparison
is written to a directory named after the arms it holds. The FIRST arm listed is the
baseline that ratios and speed-up labels are computed against.

    results/scaling/sweep/<short1>_<short2>_.../      several arms
    results/scaling/individual_runs/<short>/          one arm alone

Each directory holds, with fixed file names:

    vram_vs_tokens.png       peak live GPU memory, with the card's memory and AlphaFold 3's
                             arena as reference lines
    runtime_vs_tokens.png    model time: featurisation + inference
    inference_vs_tokens.png  the inference half alone
    wall_vs_tokens.png       the launcher process start to exit for ONE query, which adds
                             interpreter or container start, weight loading and writing the
                             structures out -- not a throughput number, nothing is pipelined
    ratio_vs_tokens.png      every other arm relative to the baseline (comparisons only)
    metrics.tsv              every plotted row, host columns dropped
    manifest.json            the arms, the baseline, the source tables, the commit

Every figure of a directory is written on each invocation, so a directory can never hold a
figure older than the table beside it.

A curve is drawn only through the sizes that RAN (status ok): where it ends IS where that
arm stopped, and a marker for the failure would say the same thing twice. The boundary and
the reason are printed to stdout and belong in the figure's caption. A peak measured while a
run was dying is the memory it got to, not the memory it needed.

WHICH MEMORY. The default metric is `peak_alloc_gib`: the framework's own live-buffer
high-water (JAX peak_bytes_in_use, PyTorch max_memory_allocated), the same quantity for both
engines. `--metric peak_vram_gib` plots the nvidia-smi whole-device high-water instead,
which for AlphaFold 3 is just its preallocated arena and is flat.

A column a table does not carry is not reconstructed from the ones it does: the figure that
needed it is skipped and names the arms that were missing it. Re-sweep the arm instead.

Usage:
    pixi run plot-scaling --arms af3-ob of3kit-big
    pixi run plot-scaling --arms af3-ob of3kit-big af3-ob-a100 af3kit-big-a100 --speedup-at 5000
    pixi run plot-scaling --arms af3-ob            # one arm, into individual_runs/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import figure_layout as fl  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPERIMENT = "scaling"
SCRIPT = "profiling/plot_vram.py"
TABLE = "vram_sweep.tsv"
X_LABEL = "Tokens (= residues, protein-only pools)"

# The three time axes, drawn the same way so they can be read side by side.
#
#   model_s      featurisation + inference of this query: what the engine costs
#   inference_s  the model on the built features, without the featurisation
#   wall_s       the launcher process start to exit: the same work PLUS interpreter or
#                container start, weight loading and writing the structures out. One query
#                per process, so nothing is pipelined and it is not a throughput number.
TIME_FIGURES = (
    ("model_s", "runtime_vs_tokens.png", "Model time vs complex size", "Model time (s)"),
    ("inference_s", "inference_vs_tokens.png", "Inference time vs complex size",
     "Inference time (s)"),
    ("wall_s", "wall_vs_tokens.png", "Wall-clock time vs complex size, one query per run",
     "Wall time (s)"),
)

# Why an arm stopped, spelled out rather than left as a status code.
STATUS_LABEL = {
    "gpu_oom": "GPU out of memory",
    "cpu_oom": "host out of memory",
    "killed": "killed",
    "failed": "failed",
    "error": "error",
}


def read_tsv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class Arm:
    """One arm's table, under the short name that labels it everywhere."""

    def __init__(self, config: fl.Config, rows: list[dict], style: dict, table: Path):
        self.config, self.rows, self.style, self.table = config, rows, style, table
        self.short = config.short

    @property
    def ok(self) -> list[dict]:
        return [r for r in self.rows if r.get("status") == "ok"]

    @property
    def bad(self) -> list[dict]:
        return [r for r in self.rows if r.get("status") not in ("ok", "")]

    def series(self, key: str) -> dict[int, float]:
        """`key` per token count, over the sizes that ran. A row without the column
        contributes nothing: a missing measurement is missing, never derived."""
        out = {}
        for r in self.ok:
            value = as_float(r.get(key))
            if value is not None:
                out[int(float(r["tokens"]))] = value
        return out

    def first(self, key: str) -> str:
        return next((r.get(key, "") for r in self.rows), "")


def load_arm(sweep_dir: Path, config: fl.Config, style: dict) -> Arm | None:
    """One arm's sweep table, or None with a reason. The data lives under the config's
    `name_tag`, which is what run_sweep.sh named the directory; `short` is only a label."""
    table = sweep_dir / config.name / TABLE
    if not table.is_file():
        print(f"  [WARN] {config.short}: no {table}, skipping")
        return None
    rows = [r for r in read_tsv(table) if as_float(r.get("tokens")) is not None]
    rows.sort(key=lambda r: float(r["tokens"]))
    if not rows:
        print(f"  [WARN] {config.short}: {table} has no rows yet, skipping")
        return None
    return Arm(config, rows, style, table)


# --------------------------------------------------------------------------- figures

def draw_line(ax, arm: Arm, key: str) -> bool:
    pts = sorted(arm.series(key).items())
    if not pts:
        return False
    xs, ys = zip(*pts)
    ax.plot(xs, ys, marker=fl.MARKER, ls=fl.LINESTYLE, color=arm.style["color"],
            label=arm.short)
    return True


def reference_lines(ax, arms: list[Arm]) -> None:
    """The card's memory (dashed) and AlphaFold 3's arena (dotted): the ceilings an arm
    actually stops against."""
    capacities, arenas = set(), set()
    for arm in arms:
        for r in arm.rows:
            total = as_float(r.get("gpu_total_mib"))
            if total:
                capacities.add(round(total / 1024, 2))
                fraction = as_float(r.get("mem_fraction"))
                if fraction:
                    arenas.add((round(total * fraction / 1024, 2), fraction))
    for capacity in sorted(capacities):
        ax.axhline(capacity, ls="--", lw=1.2, color=fl.INK_MUTED, zorder=0)
        ax.text(0.01, capacity, f" GPU memory {capacity:.0f} GiB", va="bottom", ha="left",
                transform=ax.get_yaxis_transform(), fontsize=10.5, color=fl.INK_TEXT)
    for arena, fraction in sorted(arenas):
        ax.axhline(arena, ls=":", lw=1.2, color=fl.INK_FAINT, zorder=0)
        ax.text(0.01, arena, f" AF3 arena ({fraction:g} x card)", va="top", ha="left",
                transform=ax.get_yaxis_transform(), fontsize=10.5, color=fl.INK_MUTED)
    if capacities:
        ax.set_ylim(0, max(capacities) * 1.08)


def plot_vram(arms: list[Arm], metric: str, out: Path, subtitle: str | None) -> Path | None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    # A list, not a generator: any() would short-circuit and leave arms undrawn.
    if not any([draw_line(ax, a, metric) for a in arms]):
        plt.close(fig)
        print(f"  vram_vs_tokens: no '{metric}' values in any arm, skipping")
        return None
    reference_lines(ax, arms)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("Peak live GPU memory (GiB)" if metric == "peak_alloc_gib"
                  else "Peak device memory, nvidia-smi (GiB)")
    fl.title(ax, "Peak live GPU memory vs complex size" if metric == "peak_alloc_gib"
             else "Peak device memory vs complex size", subtitle)
    if len(arms) > 1:
        fl.outside_legend(ax)
    return fl.save(fig, out / "vram_vs_tokens.png")


def annotate_speedup(ax, arms: list[Arm], baseline: Arm | None, at_tokens: int | None,
                     key: str) -> str | None:
    """Label each arm's speed-up over the baseline beside its own marker at one token count.

    A direct label on the point it describes, rather than a number in the legend the reader
    has to carry back to the curve. One size is labelled -- the one the pooled scheme
    actually runs -- because a number on every point goes unread. An arm with no row there
    is not labelled rather than interpolated. Returns the line naming what it is relative
    to, which the caller puts in the subtitle so it is said once."""
    if not at_tokens or baseline is None:
        return None
    base = baseline.series(key).get(at_tokens)
    if base is None:
        print(f"  [WARN] speed-up at {at_tokens} tokens: the baseline {baseline.short} has "
              f"no {key} there; that figure is not annotated")
        return None
    for arm in arms:
        if arm is baseline:
            continue
        value = arm.series(key).get(at_tokens)
        if value is None:
            continue
        # A ring in the surface colour lifts the labelled point off its curve.
        ax.plot([at_tokens], [value], marker=fl.MARKER, ms=11, color=arm.style["color"],
                markeredgecolor="white", markeredgewidth=2.5, zorder=5)
        # "0.61x faster" is not a sentence anyone reads correctly: an arm slower than the
        # baseline says so, with the ratio the way round that makes it a number above one.
        ratio = base / value
        verdict = (f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower")
        # Both times, this arm's first: a ratio alone hides whether it is 5 s against 8 s or
        # 510 s against 835 s, and the second reading is the one that decides anything.
        text = f"{arm.short}: {verdict}  ({value:.0f} s / {base:.0f} s)"
        # Past the middle of the x axis the label would run into the legend, so there it
        # hangs to the left of its marker instead.
        left, right = ax.get_xlim()
        to_left = at_tokens > left + 0.55 * (right - left)
        ax.annotate(text, xy=(at_tokens, value),
                    xytext=(-12 if to_left else 12, -6), textcoords="offset points",
                    va="top", ha="right" if to_left else "left",
                    fontsize=11, color=fl.INK_TEXT)
        print(f"  {arm.short}: {value:.0f} s at {at_tokens:,} tokens, "
              f"{ratio:.2f}x the baseline's {base:.0f} s")
    return f"at {at_tokens:,} tokens, against {baseline.short} ({base:.0f} s)"


def plot_time(arms: list[Arm], out: Path, subtitle: str | None, baseline: Arm | None,
              at_tokens: int | None, key: str, filename: str, headline: str,
              ylabel: str) -> Path | None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    drawn = [draw_line(ax, a, key) for a in arms]
    if not any(drawn):
        plt.close(fig)
        print(f"  {filename}: no arm's table has {key}, skipping. Re-sweep to record it.")
        return None
    against = annotate_speedup(ax, arms, baseline, at_tokens, key)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(ylabel)
    lines = [line for line in (subtitle, against) if line]
    # An arm whose table lacks this column is simply absent from the figure, which would
    # otherwise look like a comparison that only ever had the arms it drew.
    absent = [a.short for a, ok in zip(arms, drawn) if not ok]
    if absent:
        lines.append(f"no {key} in the table of: " + ", ".join(absent))
        print(f"  [WARN] {filename}: no {key} for {', '.join(absent)}")
    fl.title(ax, headline, "\n".join(lines) or None)
    ax.set_ylim(bottom=0)
    if len(arms) > 1:
        fl.outside_legend(ax)
    return fl.save(fig, out / filename)


def plot_ratio(arms: list[Arm], baseline: Arm, metric: str, out: Path,
               subtitle: str | None) -> tuple[Path | None, list[tuple]]:
    """Every other arm relative to the baseline at the token counts both ran. Left: speed-up
    = baseline model_s / arm model_s (>1 faster). Right: peak memory as a fraction of the
    baseline's (<1 less). A token count the baseline did not finish has no ratio, so an arm
    that reaches FURTHER shows nothing there: that reach is a result the absolute figures
    carry, not this one. Dotted verticals mark where an arm stopped before the baseline."""
    base_t, base_m = baseline.series("model_s"), baseline.series(metric)
    if not base_t:
        print(f"  ratio_vs_tokens: baseline {baseline.short} has no model_s, skipping")
        return None, []
    fig, (ax_s, ax_m) = plt.subplots(1, 2, figsize=(16.0, 5.8))
    table = []
    for arm in arms:
        if arm is baseline:
            continue
        t_map, m_map = arm.series("model_s"), arm.series(metric)
        common_t = sorted(set(t_map) & set(base_t))
        common_m = sorted(set(m_map) & set(base_m))
        kw = dict(marker=fl.MARKER, ls=fl.LINESTYLE, color=arm.style["color"], label=arm.short)
        if common_t:
            ax_s.plot(common_t, [base_t[k] / t_map[k] for k in common_t], **kw)
        if common_m:
            ax_m.plot(common_m, [m_map[k] / base_m[k] for k in common_m], **kw)
        for k in sorted(set(common_t) | set(common_m)):
            table.append((arm.short, k,
                          base_t[k] / t_map[k] if k in t_map and k in base_t else None,
                          m_map[k] / base_m[k] if k in m_map and k in base_m else None))
        bad = [int(float(r["tokens"])) for r in arm.bad]
        if bad and min(bad) <= max(base_t):
            for ax in (ax_s, ax_m):
                ax.axvline(min(bad), color=arm.style["color"], ls=":", lw=1.4, alpha=0.8)
    for ax in (ax_s, ax_m):
        ax.axhline(1.0, color=fl.INK_MUTED, lw=1.2, ls="--", zorder=0)
        ax.set_xlabel(X_LABEL)
        ax.set_ylim(bottom=0)
    ax_s.set_ylabel(f"Speed-up over {baseline.short}")
    ax_s.set_title("Model time relative to the baseline  (>1 = faster)")
    ax_m.set_ylabel(f"Peak memory / {baseline.short}")
    ax_m.set_title("Peak live GPU memory relative to the baseline  (<1 = less)")
    fl.outside_legend(ax_m)
    # Above the panel titles, not across them: y > 1 works because every figure is saved
    # with bbox_inches="tight".
    fig.suptitle((subtitle + "\n" if subtitle else "")
                 + f"relative to {baseline.short}; dotted verticals = where an arm stopped",
                 fontsize=11.5, color=fl.INK_TEXT, x=0.0, y=1.05, ha="left", va="bottom")
    return fl.save(fig, out / "ratio_vs_tokens.png"), table


# ------------------------------------------------------------------------ directories

def write_directory(arms: list[Arm], out: Path, *, metric: str, baseline: Arm | None,
                    subtitle: str | None, at_tokens: int | None) -> list[tuple]:
    out.mkdir(parents=True, exist_ok=True)
    ratio_table: list[tuple] = []
    paths = [plot_vram(arms, metric, out, subtitle)]
    paths += [plot_time(arms, out, subtitle, baseline, at_tokens, *spec)
              for spec in TIME_FIGURES]
    if baseline is not None and len(arms) > 1:
        path, ratio_table = plot_ratio(arms, baseline, metric, out, subtitle)
        paths.append(path)
    figures = [p for p in paths if p]
    fl.write_table(out / "metrics.tsv", [r for a in arms for r in a.rows])
    fl.write_manifest(out, experiment=EXPERIMENT, script=SCRIPT,
                      arms=[a.config for a in arms],
                      baseline=baseline.short if baseline else None,
                      sources={a.short: {"table": fl.relative_source(a.table),
                                         "rows": len(a.rows), "ok_rows": len(a.ok)}
                               for a in arms},
                      figures=figures,
                      extra={"metric": metric, "speedup_at_tokens": at_tokens})
    return ratio_table


def print_summary(arms: list[Arm], ratio_table: list[tuple], baseline: Arm | None) -> None:
    """The same numbers as text, so the figure never has to be squinted at."""
    for arm in arms:
        largest = max((int(float(r["tokens"])) for r in arm.ok), default=0)
        print(f"\n{arm.short}: {len(arm.ok)} size(s) ran, largest {largest} tokens")
        for r in arm.rows:
            print(f"  {int(float(r['tokens'])):>6} tokens  {r.get('status', ''):>8}  "
                  f"live {r.get('peak_alloc_gib', '') or '-':>6} GiB  "
                  f"feat {r.get('featurisation_s', '') or '-':>6} s  "
                  f"infer {r.get('inference_s', '') or '-':>7} s  "
                  f"model {r.get('model_s', '') or '-':>7} s  "
                  f"wall {r.get('wall_s', '') or '-':>6} s")
        if arm.bad:
            first = min(arm.bad, key=lambda r: float(r["tokens"]))
            print(f"  boundary: {STATUS_LABEL.get(first.get('status', ''), '?')} "
                  f"at {int(float(first['tokens']))} tokens")
    if ratio_table and baseline is not None:
        print(f"\nrelative to {baseline.short}   (speed-up = baseline model_s / arm model_s; "
              f"memory = arm / baseline)")
        print(f"{'arm':20s} {'tokens':>7s} {'speed-up':>9s} {'memory':>7s}")
        for short, k, s, m in ratio_table:
            print(f"{short:20s} {k:>7d} {('%.2fx' % s) if s else '-':>9s} "
                  f"{('%.2f' % m) if m else '-':>7s}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="*", default=None, metavar="SHORT",
                    help="Arms to draw, by their config's `short`, in legend order. The "
                         "first is the baseline. One arm goes to individual_runs/, several "
                         "to sweep/<short1>_<short2>_...")
    ap.add_argument("--sweep-dir", type=Path, default=None,
                    help=f"Holds one <name_tag>/{TABLE} per arm "
                         f"(default: $AF3OF3_BASE_DIR/profiling)")
    ap.add_argument("--results-root", type=Path, default=fl.RESULTS_ROOT,
                    help="Where results/<experiment>/... is written "
                         "(default: the repository's results/)")
    ap.add_argument("--metric", choices=("peak_alloc_gib", "peak_vram_gib"),
                    default="peak_alloc_gib",
                    help="Which memory column to plot (default: peak_alloc_gib, the "
                         "framework's own live-buffer high-water)")
    ap.add_argument("--speedup-at", type=int, default=None, metavar="TOKENS",
                    help="Label each curve at this token count with its time there and how "
                         "many times the baseline's it is")
    ap.add_argument("--subtitle", default=None,
                    help="A line under each figure's title saying what the comparison is for")
    ap.add_argument("--no-individual", action="store_true",
                    help="Write only the comparison directory, not one per arm")
    args = ap.parse_args()

    configs = fl.resolve_arms(args.arms or [])
    sweep_dir = args.sweep_dir or fl.base_dir() / "profiling"
    styles = fl.styles_for([c.short for c in configs])
    fl.apply_style()

    arms = [a for a in (load_arm(sweep_dir, c, styles[c.short]) for c in configs) if a]
    if not arms:
        sys.exit(f"ERROR: none of {', '.join(c.short for c in configs)} has a table under "
                 f"{sweep_dir} -- has a sweep run?")
    baseline = arms[0] if len(arms) > 1 else None

    ratio_table: list[tuple] = []
    if len(arms) > 1:
        out = fl.sweep_dir(EXPERIMENT, [a.short for a in arms], args.results_root)
        print(f"comparison -> {out}")
        ratio_table = write_directory(arms, out, metric=args.metric, baseline=baseline,
                                      subtitle=args.subtitle, at_tokens=args.speedup_at)
    if not args.no_individual:
        for arm in arms:
            out_i = fl.individual_dir(EXPERIMENT, arm.short, args.results_root)
            print(f"individual -> {out_i}")
            write_directory([arm], out_i, metric=args.metric, baseline=None,
                            subtitle=arm.config.label, at_tokens=None)
    print_summary(arms, ratio_table, baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
