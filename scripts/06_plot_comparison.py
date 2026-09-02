#!/usr/bin/env python3
"""
Stage 06 — figures comparing the two weight arms on accuracy and on cost.

Takes the two tables the previous stages wrote and produces, in --out-dir:

    dockq_scatter.png        per-entry DockQ, arm A (x) vs arm B (y), y=x dashed,
                             coloured by category. The headline: does swapping the
                             weights move individual complexes, and which way?
    dockq_distribution.png   ECDF + paired-difference histogram over the common set
                             (entries scored in BOTH arms — the only fair comparison).
    runtime_vs_size.png      inference_s against total_len, one series per arm.
                             Model cost as a function of input size.
    runtime_paired.png       inference_s, arm A vs arm B, one point per complex.

Every aggregate is computed over the COMMON SET and the n is printed on the figure,
so a difference in how many entries each arm completed cannot masquerade as a
difference in quality.

Usage:
    pixi run python scripts/06_plot_comparison.py \
        --dockq   <base>/results/dockq.tsv \
        --profile <base>/results/profiling.tsv \
        --arm-a af3_native --arm-b af3_openbind \
        --out-dir <base>/results/figures
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 150
CATEGORY_COLOURS = {
    "homodimer": "#4C72B0",
    "heterodimer": "#DD8452",
    "homomer": "#55A868",
    "heteromer": "#C44E52",
}
ARM_COLOURS = ("#4C72B0", "#DD8452")


def read_tsv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


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


def annotate_n(ax, text: str) -> None:
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color="#444444",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.85))


def plot_dockq_scatter(dockq_rows, arm_a, arm_b, out_dir) -> None:
    a = by_entry(dockq_rows, arm_a, "dockq")
    b = by_entry(dockq_rows, arm_b, "dockq")
    common = sorted(set(a) & set(b))
    if not common:
        print("  dockq_scatter: no entries scored in both arms, skipping")
        return
    category = {r["entry_id"]: r.get("category", "") for r in dockq_rows}

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    for cat in sorted({category.get(e, "") for e in common}):
        pts = [(a[e], b[e]) for e in common if category.get(e, "") == cat]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=26, alpha=0.75, label=f"{cat} (n={len(pts)})",
                   color=CATEGORY_COLOURS.get(cat, "#888888"),
                   edgecolor="white", linewidth=0.4)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#999999", zorder=0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(f"DockQ — {arm_a}")
    ax.set_ylabel(f"DockQ — {arm_b}")
    ax.set_title(f"Per-complex DockQ: {arm_a} vs {arm_b}")
    ax.legend(loc="lower right", fontsize=8, frameon=True)

    deltas = np.array([b[e] - a[e] for e in common])
    better = int((deltas > 0.05).sum())
    worse = int((deltas < -0.05).sum())
    annotate_n(ax, f"common set n={len(common)}\n"
                   f"median Δ = {np.median(deltas):+.3f}\n"
                   f"{better} better / {worse} worse by >0.05")
    fig.tight_layout()
    fig.savefig(out_dir / "dockq_scatter.png", dpi=DPI)
    plt.close(fig)
    print(f"  dockq_scatter.png  (n={len(common)})")


def plot_dockq_distribution(dockq_rows, arm_a, arm_b, out_dir) -> None:
    a = by_entry(dockq_rows, arm_a, "dockq")
    b = by_entry(dockq_rows, arm_b, "dockq")
    common = sorted(set(a) & set(b))
    if not common:
        print("  dockq_distribution: no common entries, skipping")
        return
    va = np.array([a[e] for e in common])
    vb = np.array([b[e] for e in common])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    for values, label, colour in ((va, arm_a, ARM_COLOURS[0]),
                                  (vb, arm_b, ARM_COLOURS[1])):
        xs = np.sort(values)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax1.step(xs, ys, where="post", lw=2, color=colour,
                 label=f"{label}  median {np.median(values):.3f}")
    for threshold, name in ((0.23, "acceptable"), (0.49, "medium"), (0.80, "high")):
        ax1.axvline(threshold, ls=":", lw=1, color="#aaaaaa")
        ax1.text(threshold, 0.02, f" {name}", rotation=90, fontsize=7,
                 color="#888888", va="bottom")
    ax1.set_xlabel("DockQ")
    ax1.set_ylabel("cumulative fraction of complexes")
    ax1.set_title(f"DockQ distribution (common set, n={len(common)})")
    ax1.legend(loc="upper left", fontsize=8)

    deltas = vb - va
    ax2.hist(deltas, bins=30, color="#7f7f7f", edgecolor="white")
    ax2.axvline(0, ls="--", lw=1, color="#333333")
    ax2.set_xlabel(f"ΔDockQ  ({arm_b} − {arm_a})")
    ax2.set_ylabel("complexes")
    ax2.set_title("Paired difference")
    annotate_n(ax2, f"median {np.median(deltas):+.3f}\n"
                    f"mean {deltas.mean():+.3f}\n"
                    f"n={len(deltas)}")
    fig.tight_layout()
    fig.savefig(out_dir / "dockq_distribution.png", dpi=DPI)
    plt.close(fig)
    print(f"  dockq_distribution.png  (n={len(common)})")


def plot_metric_vs_size(profile_rows, arm_a, arm_b, column, ylabel,
                        title, filename, out_dir) -> None:
    size = {}
    for row in profile_rows:
        value = as_float(row.get("total_len", ""))
        if value is not None:
            size[row["entry_id"]] = value

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    plotted = False
    for arm, colour in zip((arm_a, arm_b), ARM_COLOURS):
        values = by_entry(profile_rows, arm, column)
        pts = [(size[e], v) for e, v in values.items() if e in size]
        if not pts:
            continue
        xs, ys = zip(*sorted(pts))
        ax.scatter(xs, ys, s=24, alpha=0.75, color=colour, edgecolor="white",
                   linewidth=0.4, label=f"{arm} (n={len(pts)}, median {np.median(ys):.1f})")
        plotted = True
    if not plotted:
        print(f"  {filename}: no data in column '{column}', skipping")
        plt.close(fig)
        return
    ax.set_xlabel("complex size (summed chain length, residues)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=DPI)
    plt.close(fig)
    print(f"  {filename}")


def plot_runtime_paired(profile_rows, arm_a, arm_b, out_dir) -> None:
    a = by_entry(profile_rows, arm_a, "inference_s")
    b = by_entry(profile_rows, arm_b, "inference_s")
    common = sorted(set(a) & set(b))
    if not common:
        print("  runtime_paired: no entries timed in both arms, skipping")
        return
    xs = np.array([a[e] for e in common])
    ys = np.array([b[e] for e in common])

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(xs, ys, s=26, alpha=0.75, color="#4C72B0",
               edgecolor="white", linewidth=0.4)
    hi = max(xs.max(), ys.max()) * 1.05
    ax.plot([0, hi], [0, hi], ls="--", lw=1, color="#999999", zorder=0)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel(f"inference time (s) — {arm_a}")
    ax.set_ylabel(f"inference time (s) — {arm_b}")
    ax.set_title("Inference time per complex, paired between arms")
    ratio = float(np.median(ys / np.where(xs == 0, np.nan, xs)))
    annotate_n(ax, f"n={len(common)}\n"
                   f"median {arm_a} {np.median(xs):.1f} s\n"
                   f"median {arm_b} {np.median(ys):.1f} s\n"
                   f"median ratio {ratio:.2f}x")
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_paired.png", dpi=DPI)
    plt.close(fig)
    print(f"  runtime_paired.png  (n={len(common)})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dockq", type=Path, help="Output of scripts/05_score_dockq.py")
    ap.add_argument("--profile", type=Path, help="Output of scripts/04_collect_profiling.py")
    ap.add_argument("--arm-a", default="af3_native")
    ap.add_argument("--arm-b", default="af3_openbind")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    if not args.dockq and not args.profile:
        raise SystemExit("Nothing to plot: pass --dockq and/or --profile")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Figures -> {args.out_dir}")

    if args.dockq:
        rows = [r for r in read_tsv(args.dockq) if r.get("status") == "ok"]
        print(f"DockQ rows with status=ok: {len(rows)}")
        plot_dockq_scatter(rows, args.arm_a, args.arm_b, args.out_dir)
        plot_dockq_distribution(rows, args.arm_a, args.arm_b, args.out_dir)

    if args.profile:
        rows = read_tsv(args.profile)
        print(f"Profiling rows: {len(rows)}")
        plot_metric_vs_size(rows, args.arm_a, args.arm_b, "inference_s",
                            "inference time (s)",
                            "Model inference time vs complex size",
                            "runtime_vs_size.png", args.out_dir)
        plot_runtime_paired(rows, args.arm_a, args.arm_b, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
