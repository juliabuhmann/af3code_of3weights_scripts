#!/usr/bin/env python3
"""
Stage 06 — figures comparing two arms on accuracy and on cost.

Takes the tables the previous stages wrote and produces, in --out-dir:

    dockq_scatter.png        per-entry DockQ, arm A (x) vs arm B (y), y=x dashed,
                             coloured by category. The headline: does swapping the
                             weights move individual complexes, and which way?
    dockq_distribution.png   ECDF + paired-difference histogram over the common set
                             (entries scored in BOTH arms — the only fair comparison).
    runtime_vs_size.png      model time against complex size, one series per arm.
                             Cost as a function of input size — the figure that shows
                             a crossover a single median would hide.
    runtime_paired.png       model time, arm A vs arm B, one point per complex.

Every aggregate is computed over the COMMON SET — entries present in both arms — so a
difference in how many entries each arm completed cannot masquerade as a difference in
quality. The n is reported for every figure: on the panel where there is room, and on
stdout for dockq_scatter.png, which is kept free of a summary box so the points are
the only thing in the plotting area.

LABELS: TAGS, AND ONLY WHAT DIFFERS
-----------------------------------
Each arm carries a short `tag` naming its pipeline — "AF3 code / openbind" — set in
its config and copied into every result row (see scripts/arm_metadata.py). Figures
use that rather than the arm's directory name.

The template state is appended to the tag ONLY when the two arms being plotted
disagree on it. Comparing templates-on against templates-off, it is the whole point
of the figure and has to be visible; comparing two templates-off arms it is shared
context that belongs in the caption, and repeating it on every series just crowds the
legend. An arm whose state is unknown is never assumed — it shows as "templates ?".

Legends sit OUTSIDE the axes so no data point is hidden behind them, and every figure
is saved with bbox_inches="tight" so a long title or an outside legend cannot be
cropped.

Runtime figures default to `model_s` (featurisation + inference), which is the column
comparable ACROSS engines; --runtime-column inference_s restricts them to AF3's
inference timer, which OF3 rows do not have.

Usage:
    pixi run python scripts/06_plot_comparison.py \
        --dockq   <base>/results/dockq \
        --profile <base>/results/profiling \
        --arm-a af3_native --arm-b af3_openbind \
        --out-dir <base>/results/figures
"""

import argparse
import sys
from pathlib import Path

from result_tables import read_rows

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 150

# Large enough to stay legible when a figure is dropped into a document at half
# width, which is how these end up being read.
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})

CATEGORY_COLOURS = {
    "homodimer": "#4C72B0",
    "heterodimer": "#DD8452",
    "homomer": "#55A868",
    "heteromer": "#C44E52",
}
ARM_COLOURS = ("#4C72B0", "#DD8452")
TEMPLATE_STATE = {"true": "templates on", "false": "templates off", "": "templates ?"}


def read_tsv(path: Path) -> list[dict]:
    """One .tsv, or every .tsv in a directory concatenated.

    Stages 04 and 05 write one table per arm, so --dockq and --profile normally point
    at those directories; a single combined file still works unchanged.
    """
    return read_rows(path)


def as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def by_entry(rows: list[dict], run: str, column: str) -> dict[str, float]:
    out = {}
    for row in rows:
        if row.get("run") != run:
            continue
        value = as_float(row.get(column, ""))
        if value is not None:
            out[row["entry_id"]] = value
    return out


def _field(rows: list[dict], arm: str, column: str, default: str = "") -> str:
    """One arm's value for `column`, or a "/"-joined string if the arm disagrees.

    Read off the rows rather than passed in, so a label cannot disagree with the data
    being plotted. A disagreement within one arm is surfaced, not silently reduced.
    """
    values = {r.get(column, "") for r in rows if r.get("run") == arm}
    values.discard("")
    if len(values) == 1:
        return values.pop()
    if len(values) > 1:
        return "/".join(sorted(values))
    return default


def arm_labels(rows: list[dict], arm_a: str, arm_b: str) -> tuple[str, str]:
    """Legend and axis labels for the pair, spelling out only what DIFFERS.

    See "LABELS: TAGS, AND ONLY WHAT DIFFERS" in the module docstring. If the two
    tags collide and the template state does not separate them either, the arm names
    are appended — two series must never carry the same label.
    """
    tags = {arm: _field(rows, arm, "tag", default=arm) or arm for arm in (arm_a, arm_b)}
    states = {arm: _field(rows, arm, "use_templates") for arm in (arm_a, arm_b)}

    labels = dict(tags)
    if states[arm_a] != states[arm_b]:
        for arm in (arm_a, arm_b):
            labels[arm] = f"{tags[arm]} ({TEMPLATE_STATE.get(states[arm], 'templates ?')})"
    if labels[arm_a] == labels[arm_b]:
        labels[arm_a] = f"{labels[arm_a]} [{arm_a}]"
        labels[arm_b] = f"{labels[arm_b]} [{arm_b}]"
    return labels[arm_a], labels[arm_b]


def outside_legend(ax, **kwargs) -> None:
    """Legend to the right of the axes, so it can never cover a data point."""
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
              frameon=False, **kwargs)


def annotate_n(ax, text: str) -> None:
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=12, color="#333333",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9))


def save(fig, path: Path, note: str = "") -> None:
    """bbox_inches='tight' is what guarantees titles and outside legends survive.

    tight_layout() is deliberately NOT used: it reserves space only for artists
    inside the axes, so a legend anchored outside them gets clipped instead.
    """
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  {path.name}{'  ' + note if note else ''}")


def plot_dockq_scatter(dockq_rows, arm_a, arm_b, out_dir) -> None:
    a = by_entry(dockq_rows, arm_a, "dockq")
    b = by_entry(dockq_rows, arm_b, "dockq")
    common = sorted(set(a) & set(b))
    if not common:
        print("  dockq_scatter: no entries scored in both arms, skipping")
        return
    label_a, label_b = arm_labels(dockq_rows, arm_a, arm_b)
    category = {r["entry_id"]: r.get("category", "") for r in dockq_rows}

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for cat in sorted({category.get(e, "") for e in common}):
        pts = [(a[e], b[e]) for e in common if category.get(e, "") == cat]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=34, alpha=0.75, label=f"{cat or 'uncategorised'} (n={len(pts)})",
                   color=CATEGORY_COLOURS.get(cat, "#888888"),
                   edgecolor="white", linewidth=0.5)
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="#999999", zorder=0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(f"DockQ — {label_a}")
    ax.set_ylabel(f"DockQ — {label_b}")
    ax.set_title("Per-complex DockQ")
    outside_legend(ax, title="category")

    # No on-figure summary box here: the per-category legend already carries the ns,
    # and the aggregates belong in the caption. They are printed to stdout instead so
    # the numbers are still available without crowding the panel.
    deltas = np.array([b[e] - a[e] for e in common])
    better = int((deltas > 0.05).sum())
    worse = int((deltas < -0.05).sum())
    save(fig, out_dir / "dockq_scatter.png", f"(n={len(common)})")
    print(f"    n={len(common)}  median Δ={np.median(deltas):+.3f}  "
          f"{better} better / {worse} worse by >0.05")


def plot_dockq_distribution(dockq_rows, arm_a, arm_b, out_dir) -> None:
    a = by_entry(dockq_rows, arm_a, "dockq")
    b = by_entry(dockq_rows, arm_b, "dockq")
    common = sorted(set(a) & set(b))
    if not common:
        print("  dockq_distribution: no common entries, skipping")
        return
    label_a, label_b = arm_labels(dockq_rows, arm_a, arm_b)
    va = np.array([a[e] for e in common])
    vb = np.array([b[e] for e in common])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.6))

    for values, label, colour in ((va, label_a, ARM_COLOURS[0]),
                                  (vb, label_b, ARM_COLOURS[1])):
        xs = np.sort(values)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax1.step(xs, ys, where="post", lw=2.4, color=colour,
                 label=f"{label}\nmedian {np.median(values):.3f}")
    for threshold, name in ((0.23, "acceptable"), (0.49, "medium"), (0.80, "high")):
        ax1.axvline(threshold, ls=":", lw=1, color="#aaaaaa")
        ax1.text(threshold, 0.02, f" {name}", rotation=90, fontsize=11,
                 color="#888888", va="bottom")
    ax1.set_xlabel("DockQ")
    ax1.set_ylabel("cumulative fraction of complexes")
    ax1.set_title(f"DockQ distribution (common set, n={len(common)})")
    ax1.legend(loc="upper left", frameon=False)

    deltas = vb - va
    ax2.hist(deltas, bins=30, color="#7f7f7f", edgecolor="white")
    ax2.axvline(0, ls="--", lw=1.2, color="#333333")
    ax2.set_xlabel(f"Δ DockQ   ({label_b} − {label_a})")
    ax2.set_ylabel("complexes")
    ax2.set_title("Paired difference")
    annotate_n(ax2, f"median {np.median(deltas):+.3f}\n"
                    f"mean {deltas.mean():+.3f}\n"
                    f"n={len(deltas)}")
    save(fig, out_dir / "dockq_distribution.png", f"(n={len(common)})")


def plot_metric_vs_size(profile_rows, arm_a, arm_b, column, ylabel,
                        title, filename, out_dir) -> None:
    size = {}
    for row in profile_rows:
        value = as_float(row.get("total_len", ""))
        if value is not None:
            size[row["entry_id"]] = value
    labels = dict(zip((arm_a, arm_b), arm_labels(profile_rows, arm_a, arm_b)))

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    plotted = False
    for arm, colour in zip((arm_a, arm_b), ARM_COLOURS):
        values = by_entry(profile_rows, arm, column)
        pts = [(size[e], v) for e, v in values.items() if e in size]
        if not pts:
            continue
        xs, ys = zip(*sorted(pts))
        ax.scatter(xs, ys, s=32, alpha=0.75, color=colour, edgecolor="white",
                   linewidth=0.5,
                   label=f"{labels[arm]}\nn={len(pts)}, median {np.median(ys):.0f} s")
        plotted = True
    if not plotted:
        print(f"  {filename}: no data in column '{column}', skipping")
        plt.close(fig)
        return
    ax.set_xlabel("complex size (summed chain length, residues)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    outside_legend(ax)
    ax.grid(alpha=0.25, lw=0.5)
    save(fig, out_dir / filename)


def plot_runtime_paired(profile_rows, arm_a, arm_b, column, out_dir) -> None:
    a = by_entry(profile_rows, arm_a, column)
    b = by_entry(profile_rows, arm_b, column)
    common = sorted(set(a) & set(b))
    if not common:
        print(f"  runtime_paired: no entries timed in both arms "
              f"(column '{column}'), skipping")
        return
    label_a, label_b = arm_labels(profile_rows, arm_a, arm_b)
    xs = np.array([a[e] for e in common])
    ys = np.array([b[e] for e in common])

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(xs, ys, s=34, alpha=0.75, color="#4C72B0",
               edgecolor="white", linewidth=0.5)
    hi = max(xs.max(), ys.max()) * 1.05
    ax.plot([0, hi], [0, hi], ls="--", lw=1.2, color="#999999", zorder=0)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel(f"{column} (s) — {label_a}")
    ax.set_ylabel(f"{column} (s) — {label_b}")
    ax.set_title(f"Model time per complex ({column})")
    ratio = float(np.median(ys / np.where(xs == 0, np.nan, xs)))
    annotate_n(ax, f"n={len(common)}\n"
                   f"median x {np.median(xs):.1f} s\n"
                   f"median y {np.median(ys):.1f} s\n"
                   f"median ratio {ratio:.2f}x")
    save(fig, out_dir / "runtime_paired.png", f"(n={len(common)})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dockq", type=Path,
                    help="scripts/05 output: the per-arm directory "
                         "(<base>/results/dockq) or a single combined .tsv")
    ap.add_argument("--profile", type=Path,
                    help="scripts/04 output: the per-arm directory "
                         "(<base>/results/profiling) or a single combined .tsv")
    ap.add_argument("--arm-a", default="af3_native")
    ap.add_argument("--arm-b", default="af3_openbind")
    ap.add_argument("--runtime-column", default="model_s",
                    choices=["model_s", "inference_s", "wall_s"],
                    help="Which timing column the runtime figures use. model_s "
                         "(default) is featurisation+inference and is the only one "
                         "comparable across engines; inference_s is AF3-only.")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    if not args.dockq and not args.profile:
        raise SystemExit("Nothing to plot: pass --dockq and/or --profile")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Figures -> {args.out_dir}")

    if args.dockq:
        rows = [r for r in read_tsv(args.dockq) if r.get("status") == "ok"]
        print(f"DockQ rows with status=ok: {len(rows)}")
        print("  labels: {} | {}".format(*arm_labels(rows, args.arm_a, args.arm_b)))
        plot_dockq_scatter(rows, args.arm_a, args.arm_b, args.out_dir)
        plot_dockq_distribution(rows, args.arm_a, args.arm_b, args.out_dir)

    if args.profile:
        rows = read_tsv(args.profile)
        print(f"Profiling rows: {len(rows)}")
        print("  labels: {} | {}".format(*arm_labels(rows, args.arm_a, args.arm_b)))
        plot_metric_vs_size(rows, args.arm_a, args.arm_b, args.runtime_column,
                            f"{args.runtime_column} (s)",
                            f"Model time ({args.runtime_column}) vs complex size",
                            "runtime_vs_size.png", args.out_dir)
        plot_runtime_paired(rows, args.arm_a, args.arm_b,
                            args.runtime_column, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
