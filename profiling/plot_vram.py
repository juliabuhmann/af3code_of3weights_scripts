#!/usr/bin/env python3
"""
Peak GPU memory and model time against token count, per arm, from the pool ladder.

Reads one table per arm -- <sweep-dir>/<arm>/vram_sweep.tsv, written row by row by
profiling/sweep.sbatch -- and writes, for a registered set (configs/figure_sets.json)
or an ad-hoc list of arms:

    results/scaling/sweep/<set>/
        vram_vs_tokens.png       peak live memory per arm, with the card's memory and
                                 AlphaFold 3's arena as reference lines
        runtime_vs_tokens.png    model time per arm: featurisation + inference
        wall_vs_tokens.png       the launcher process start to exit for one query, which
                                 adds interpreter or container start, weight loading and
                                 writing the structures out
        ratio_vs_tokens.png      every arm relative to the set's baseline at the token
                                 counts both ran: speed-up (>1 faster) and memory
                                 fraction (<1 less); only with a baseline
        metrics.tsv              every plotted row, host columns dropped
        manifest.json            arms in legend order, baseline, source tables, commit

    results/scaling/individual_runs/<arm>/
        vram_vs_tokens.png, runtime_vs_tokens.png, wall_vs_tokens.png, metrics.tsv,
        manifest.json
                                 the same, for each arm on its own

All figures of a directory are written on every invocation, so a directory can never
hold a figure that is older than the table beside it. The layout, the file names and
the per-arm colours come from scripts/figure_layout.py.

A curve is drawn only through the sizes that RAN (status ok): where it ends IS where
that arm stopped, and a marker for the failure would say the same thing twice. The
boundary and the reason are printed to stdout and belong in the figure's caption. A
peak measured while a run was dying is the memory it got to, not the memory it needed.

WHICH MEMORY. The default metric is `peak_alloc_gib`: the framework's own live-buffer
high-water (JAX peak_bytes_in_use, PyTorch max_memory_allocated), the same quantity for
both engines. `--metric peak_vram_gib` plots the nvidia-smi whole-device high-water
instead, which for AF3 is just its preallocated arena and is flat.

Usage:
    pixi run plot-scaling --set of3kit_rtxpro6000
    pixi run plot-scaling --arms af3_openbind_templateoff of3kit_fast_templateoff \
        --baseline af3_openbind_templateoff
"""

from __future__ import annotations

import argparse
import csv
import json
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
TEMPLATE_STATE = {"true": "templates on", "false": "templates off", "": "templates ?"}
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
    """One arm's table, its legend label and its drawing style."""

    def __init__(self, name: str, rows: list[dict], tag: str | None, style: dict):
        self.name, self.rows, self.tag, self.style = name, rows, tag, style
        self.label = tag or name

    @property
    def ok(self) -> list[dict]:
        return [r for r in self.rows if r.get("status") == "ok"]

    @property
    def bad(self) -> list[dict]:
        return [r for r in self.rows if r.get("status") not in ("ok", "")]

    def ok_map(self, key: str) -> dict[int, float]:
        out = {}
        for r in self.ok:
            v = as_float(r.get(key))
            if v is not None:
                out[int(float(r["tokens"]))] = v
        return out

    def first(self, key: str) -> str:
        return next((r.get(key, "") for r in self.rows), "")


def load_arm(sweep_dir: Path, name: str, style: dict) -> Arm | None:
    path = sweep_dir / name / TABLE
    if not path.is_file():
        print(f"  [WARN] {name}: no {path}, skipping")
        return None
    rows = [r for r in read_tsv(path) if as_float(r.get("tokens")) is not None]
    rows.sort(key=lambda r: float(r["tokens"]))
    if not rows:
        print(f"  [WARN] {name}: {path} has no rows yet, skipping")
        return None
    return Arm(name, rows, arm_tag(sweep_dir, name), style)


def arm_tag(sweep_dir: Path, name: str) -> str | None:
    """The arm's short display label: this repository's config if it still has one, else
    the copy run_sweep.sh left beside the table.

    The two are read for different things on purpose. The axes a row is interpreted by
    (engine, weights, use_templates) come from the copy beside the data, so they describe
    what actually ran and cannot drift when the repository's configs move on. `tag` is
    only what a legend says, and the repository owns it: renaming a series then costs a
    re-plot instead of a re-run, and a sweep whose data lives on read-only storage can
    still be relabelled."""
    for path in (fl.REPO / "configs" / f"{name}.json", sweep_dir / name / "config.json"):
        if path.is_file():
            try:
                tag = json.load(open(path)).get("tag")
            except (OSError, json.JSONDecodeError):
                continue
            if tag:
                return tag
    return None


def label_arms(arms: list[Arm]) -> None:
    """Legend labels: the config's tag, minus the weights when every arm runs the same
    ones, plus the template state only when the arms disagree on it, plus the arm name
    only when tags still collide (the two cards of one configuration carry the same tag).
    Two series never wear the same label."""
    stripped = fl.drop_shared_term({a.name: a.label for a in arms},
                                   {a.name: a.first("weights") for a in arms})
    for a in arms:
        a.label = stripped[a.name]
    if len({a.first("use_templates") for a in arms}) > 1:
        for a in arms:
            a.label = f"{a.label} ({TEMPLATE_STATE.get(a.first('use_templates'), 'templates ?')})"
    labels = [a.label for a in arms]
    for a in arms:
        if labels.count(a.label) > 1:
            a.label = f"{a.label} [{a.name}]"


# --------------------------------------------------------------------------- figures

def draw_line(ax, arm: Arm, key: str, label: str | None = None) -> bool:
    pts = sorted(arm.ok_map(key).items())
    if not pts:
        return False
    xs, ys = zip(*pts)
    ax.plot(xs, ys, marker=fl.MARKER, ls=fl.LINESTYLE, color=arm.style["color"],
            label=arm.label if label is None else label)
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


def plot_vram(arms: list[Arm], metric: str, out: Path, title: str | None) -> Path | None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    drawn = [draw_line(ax, a, metric) for a in arms]
    if not any(drawn):
        plt.close(fig)
        print(f"  vram_vs_tokens: no '{metric}' values in any arm, skipping "
              f"(an older table predates the column; re-run the sweep or pass --metric peak_vram_gib)")
        return None
    reference_lines(ax, arms)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("Peak live GPU memory (GiB)" if metric == "peak_alloc_gib"
                  else "Peak device memory, nvidia-smi (GiB)")
    fl.title(ax, "Peak live GPU memory vs complex size" if metric == "peak_alloc_gib"
             else "Peak device memory vs complex size", title)
    if len(arms) > 1:
        fl.outside_legend(ax)
    return fl.save(fig, out / "vram_vs_tokens.png")


def annotate_speedup(ax, arms: list[Arm], baseline: Arm | None, at_tokens: int | None,
                     key: str = "model_s") -> str | None:
    """Write each arm's speed-up over the baseline beside its own marker at one token count.

    A direct label on the point it describes, rather than a number in the legend the reader
    has to carry back to the curve. One size is labelled -- the one the pooled scheme
    actually runs -- because a number on every point goes unread. An arm with no row there
    is not labelled rather than interpolated."""
    if not at_tokens or baseline is None:
        return None
    base = baseline.ok_map(key).get(at_tokens)
    if base is None:
        print(f"  [WARN] speed-up at {at_tokens} tokens: the baseline {baseline.name} has no "
              f"{key} there; that figure is not annotated")
        return None
    for arm in arms:
        if arm is baseline:
            continue
        value = arm.ok_map(key).get(at_tokens)
        if value is None:
            continue
        # A ring in the surface colour lifts the labelled point off its curve.
        ax.plot([at_tokens], [value], marker=fl.MARKER, ms=11, color=arm.style["color"],
                markeredgecolor="white", markeredgewidth=2.5, zorder=5)
        # "0.61x faster" is not a sentence anyone reads correctly: an arm slower than the
        # baseline says so, with the ratio the way round that makes it a number above one.
        ratio = base / value
        verdict = (f"{ratio:.2f}x faster than" if ratio >= 1
                   else f"{1 / ratio:.2f}x slower than")
        # Both times, this arm's first: a ratio alone hides whether it is 5 s against 8 s
        # or 510 s against 835 s, and the second reading is the one that decides anything.
        # WHAT it is relative to is said once, in the subtitle: spelling the baseline out
        # beside every marker repeats it and, with a long arm name, runs off the axes.
        text = f"{verdict}  ({value:.0f} s / {base:.0f} s)"
        # Past the middle of the x axis the label would run into the legend, so there it
        # hangs to the left of its marker instead.
        left, right = ax.get_xlim()
        to_left = at_tokens > left + 0.55 * (right - left)
        ax.annotate(text, xy=(at_tokens, value),
                    xytext=(-12 if to_left else 12, -6), textcoords="offset points",
                    va="top", ha="right" if to_left else "left",
                    fontsize=11, color=fl.INK_TEXT)
        print(f"  {arm.name}: {value:.0f} s at {at_tokens:,} tokens, "
              f"{ratio:.2f}x the baseline's {base:.0f} s")
    return (f"at {at_tokens:,} tokens, against {baseline.label} "
            f"({base:.0f} s)")


# An of3 row whose model time came from OpenFold3's own timer covers its predict loop and
# not the featurisation before it, while every af3 row includes featurisation. Such a row
# is still plotted -- it is the measurement that exists -- but the figure says so, because
# a speed-up read off it flatters the OpenFold3 side.
TIMELINE_SOURCE = "of3 predict timeline"
MIXED_NOTE = ("OpenFold3 times its predict loop only: featurisation is in the AlphaFold 3 "
              "times and not in theirs. Re-sweep for like-for-like numbers.")


def model_sources(arms: list[Arm]) -> dict[str, str]:
    return {a.name: next((r.get("model_source", "") for r in a.ok if r.get("model_source")),
                         "" if any("model_source" in r for r in a.rows) else "(not recorded)")
            for a in arms}


def mixed_definitions(arms: list[Arm]) -> bool:
    """True when an of3 arm's model time leaves featurisation out while an af3 arm's does not."""
    engines = {a.first("engine") for a in arms}
    if "af3" not in engines:
        return False
    sources = model_sources(arms)
    return any(a.first("engine") == "of3" and not sources[a.name].startswith(TIMELINE_SOURCE)
               for a in arms)


# The two time axes, drawn the same way so they can be read side by side.
#
#   model_s  featurisation + inference of this query: what the engine costs.
#   wall_s   the launcher process from start to exit -- the same work PLUS interpreter or
#            container start, weight loading and writing the structures out. One query per
#            process, so it is the cost of a single prediction end to end and NOT a
#            throughput number: nothing overlaps because there is nothing to overlap with.
TIME_FIGURES = (
    ("model_s", "runtime_vs_tokens.png", "Model time vs complex size", "Model time (s)"),
    ("wall_s", "wall_vs_tokens.png", "Wall-clock time vs complex size, one query per run",
     "Wall time (s)"),
)


def plot_time(arms: list[Arm], out: Path, title: str | None, baseline: Arm | None,
              at_tokens: int | None, key: str, filename: str, headline: str,
              ylabel: str) -> Path | None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    # A list, not a generator: any() would short-circuit and leave every arm after the
    # first one undrawn.
    if not any([draw_line(ax, a, key) for a in arms]):
        plt.close(fig)
        print(f"  {filename}: no {key} values, skipping")
        return None
    against = annotate_speedup(ax, arms, baseline, at_tokens, key)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(ylabel)
    lines = [line for line in (title, against) if line]
    # Only the model-time figure can mix definitions: wall_s is one clock for both engines.
    if key == "model_s" and mixed_definitions(arms):
        lines.append(MIXED_NOTE)
        print(f"  [WARN] {MIXED_NOTE}")
    subtitle = "\n".join(lines) or None
    fl.title(ax, headline, subtitle)
    ax.set_ylim(bottom=0)
    if len(arms) > 1:
        fl.outside_legend(ax)
    return fl.save(fig, out / filename)


def plot_ratio(arms: list[Arm], baseline: Arm, metric: str, out: Path,
               title: str | None) -> tuple[Path | None, list[tuple]]:
    """Every other arm relative to the baseline at the token counts both ran. Left:
    speed-up = baseline model_s / arm model_s (>1 faster). Right: peak memory as a
    fraction of the baseline's (<1 less). A token count the baseline did not finish has
    no ratio, so an arm that reaches FURTHER shows nothing there: that reach is a result
    the absolute figures carry, not this one. Dotted verticals mark where an arm stopped
    before the baseline did."""
    base_t, base_m = baseline.ok_map("model_s"), baseline.ok_map(metric)
    if not base_t:
        print(f"  ratio_vs_tokens: baseline {baseline.name} has no ok rows with model_s, skipping")
        return None, []
    fig, (ax_s, ax_m) = plt.subplots(1, 2, figsize=(16.0, 5.8))
    table = []
    for arm in arms:
        if arm is baseline:
            continue
        t_map, m_map = arm.ok_map("model_s"), arm.ok_map(metric)
        common_t = sorted(set(t_map) & set(base_t))
        common_m = sorted(set(m_map) & set(base_m))
        kw = dict(marker=fl.MARKER, ls=fl.LINESTYLE, color=arm.style["color"], label=arm.label)
        if common_t:
            ax_s.plot(common_t, [base_t[k] / t_map[k] for k in common_t], **kw)
        if common_m:
            ax_m.plot(common_m, [m_map[k] / base_m[k] for k in common_m], **kw)
        for k in sorted(set(common_t) | set(common_m)):
            table.append((arm.name, k,
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
    ax_s.set_ylabel(f"Speed-up over {baseline.label}")
    ax_s.set_title("Model time relative to the baseline  (>1 = faster)")
    ax_m.set_ylabel(f"Peak memory / {baseline.label}")
    ax_m.set_title("Peak live GPU memory relative to the baseline  (<1 = less)")
    fl.outside_legend(ax_m)
    # Above the panel titles, not across them: y > 1 works because every figure is saved
    # with bbox_inches="tight".
    fig.suptitle((title + "\n" if title else "")
                 + f"relative to {baseline.name}; dotted verticals = where an arm stopped",
                 fontsize=11.5, color=fl.INK_TEXT, x=0.0, y=1.05, ha="left", va="bottom")
    return fl.save(fig, out / "ratio_vs_tokens.png"), table


def sources_of(arms: list[Arm], sweep_dir: Path) -> dict[str, dict]:
    return {a.name: {"table": fl.relative_source(sweep_dir / a.name / TABLE),
                     "rows": len(a.rows), "ok_rows": len(a.ok)} for a in arms}


def write_directory(arms: list[Arm], out: Path, *, metric: str, baseline: Arm | None,
                    set_name: str | None, title: str | None, sweep_dir: Path,
                    missing: list[str], at_tokens: int | None = None) -> list[tuple]:
    out.mkdir(parents=True, exist_ok=True)
    figures, ratio_table = [], []
    paths = [plot_vram(arms, metric, out, title)]
    paths += [plot_time(arms, out, title, baseline, at_tokens, *spec) for spec in TIME_FIGURES]
    for path in paths:
        if path:
            figures.append(path)
    if baseline is not None and len(arms) > 1:
        path, ratio_table = plot_ratio(arms, baseline, metric, out, title)
        if path:
            figures.append(path)
    fl.write_table(out / "metrics.tsv", [r for a in arms for r in a.rows])
    fl.write_manifest(out, experiment=EXPERIMENT, script=SCRIPT, set_name=set_name, title=title,
                      arms=[a.name for a in arms], baseline=baseline.name if baseline else None,
                      sources=sources_of(arms, sweep_dir), figures=figures, missing=missing,
                      extra={"metric": metric, "speedup_at_tokens": at_tokens,
                             "model_sources": model_sources(arms),
                             "model_time_definitions_mixed": mixed_definitions(arms)})
    return ratio_table


def print_summary(arms: list[Arm], ratio_table: list[tuple], baseline: Arm | None) -> None:
    """The same numbers as text, so the figure never has to be squinted at."""
    for arm in arms:
        largest = max((int(float(r["tokens"])) for r in arm.ok), default=0)
        print(f"\n{arm.name}: {len(arm.ok)} size(s) ran, largest {largest} tokens")
        for r in arm.rows:
            print(f"  {int(float(r['tokens'])):>6} tokens  {r.get('status', ''):>8}  "
                  f"live {r.get('peak_alloc_gib', '') or '-':>6} GiB  "
                  f"device {r.get('peak_vram_gib', '') or '-':>6} GiB  "
                  f"model {r.get('model_s', '') or '-':>7} s")
        if arm.bad:
            first = min(arm.bad, key=lambda r: float(r["tokens"]))
            print(f"  boundary: {STATUS_LABEL.get(first.get('status', ''), '?')} "
                  f"at {int(float(first['tokens']))} tokens")
    if ratio_table and baseline is not None:
        print(f"\nrelative to {baseline.name}   (speed-up = baseline model_s / arm model_s; "
              f"memory = arm / baseline)")
        print(f"{'arm':44s} {'tokens':>7s} {'speed-up':>9s} {'memory':>7s}")
        for name, k, s, m in ratio_table:
            print(f"{name:44s} {k:>7d} {('%.2fx' % s) if s else '-':>9s} {('%.2f' % m) if m else '-':>7s}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="set_name", default=None,
                    help="A scaling set from configs/figure_sets.json: its arms, baseline and title; "
                         "written to results/scaling/sweep/<set>/. The reproducible unit.")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="Arm directories to plot instead, in legend order; written to "
                         "results/scaling/sweep/<arms joined with _vs_>/")
    ap.add_argument("--baseline", default=None,
                    help="Arm the ratio figure divides by (default: the set's `baseline`; "
                         "with --arms, none unless given)")
    ap.add_argument("--sets-file", type=Path, default=fl.SETS_FILE)
    ap.add_argument("--sweep-dir", type=Path, default=None,
                    help=f"Holds one <arm>/{TABLE} per arm (default: $AF3OF3_BASE_DIR/profiling)")
    ap.add_argument("--results-root", type=Path, default=fl.RESULTS_ROOT,
                    help="Where results/<experiment>/... is written (default: the repository's results/)")
    ap.add_argument("--metric", choices=("peak_alloc_gib", "peak_vram_gib"), default="peak_alloc_gib",
                    help="Which memory column to plot (default: peak_alloc_gib, the framework's own "
                         "live-buffer high-water)")
    ap.add_argument("--speedup-at", type=int, default=None, metavar="TOKENS",
                    help="Label each curve on the runtime figure with its model time at this "
                         "token count and how many times the baseline's it is (default: the "
                         "set's `speedup_at`, when it names one)")
    ap.add_argument("--no-individual", action="store_true",
                    help="Write only the sweep directory, not one directory per arm")
    args = ap.parse_args()

    set_name, names, baseline_name, title = fl.resolve_selection(args, EXPERIMENT)
    at_tokens = args.speedup_at or fl.set_key(args, EXPERIMENT, "speedup_at")
    sweep_dir = args.sweep_dir or fl.base_dir() / "profiling"
    styles = fl.styles_for(names)
    fl.apply_style()

    arms, missing = [], []
    for name in names:
        arm = load_arm(sweep_dir, name, styles[name])
        (arms if arm else missing).append(arm or name)
    if not arms:
        sys.exit(f"ERROR: none of {names} has a table under {sweep_dir} -- has a sweep run?")
    label_arms(arms)
    baseline = next((a for a in arms if a.name == baseline_name), None)
    if baseline_name and baseline is None:
        print(f"  [WARN] baseline {baseline_name} has no table; no ratio figure")

    out = fl.sweep_dir(EXPERIMENT, set_name, [a.name for a in arms], args.results_root)
    print(f"sweep -> {out}")
    ratio_table = write_directory(arms, out, metric=args.metric, baseline=baseline, set_name=set_name,
                                  title=title, sweep_dir=sweep_dir, missing=missing,
                                  at_tokens=at_tokens)
    if not args.no_individual:
        for arm in arms:
            out_i = fl.individual_dir(EXPERIMENT, arm.name, args.results_root)
            print(f"individual -> {out_i}")
            write_directory([arm], out_i, metric=args.metric, baseline=None, set_name=None,
                            title=arm.label, sweep_dir=sweep_dir, missing=[])
    print_summary(arms, ratio_table, baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
