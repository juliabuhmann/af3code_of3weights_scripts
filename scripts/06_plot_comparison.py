#!/usr/bin/env python3
"""
Stage 06 -- accuracy and cost on Benchmark-IV, for a comparison of arms.

Arms are named by their config's `short` (af3-native-tmpl, af3-ob, of3-ob, zheng), and a
comparison is written to a directory named after the arms it holds. The FIRST arm listed is
the baseline the paired figures are computed against. Run with no arms to list them.

    results/benchmark/sweep/<short1>_<short2>_.../
        dockq_distribution.png   ECDF per arm over the common set, plus the paired
                                 difference against the baseline (two arms)
        dockq_scatter.png        per-entry DockQ, baseline (x) vs the other arm (y),
                                 coloured by category. Two arms only: colour carries the
                                 category there, which is why it is a pairwise figure
        runtime_vs_size.png      model time against complex size, one series per arm --
                                 cost as a function of input size, which a median hides
        runtime_paired.png       model time, baseline vs the other arm. Two arms only
        metrics.tsv              per-arm summary over the common set
        complexes_evaluated.tsv  the entries every arm scored, with each arm's values
        manifest.json            the arms, the baseline, the source tables, the commit

    results/benchmark/individual_runs/<short>/
        dockq_distribution.png, runtime_vs_size.png, metrics.tsv, manifest.json

Every aggregate is computed over the COMMON SET -- entries present in every arm -- so a
difference in how many entries each arm completed cannot masquerade as a difference in
quality. The n is reported on every figure or on stdout beside it.

The result tables key a row by `run`, which is the arm's directory name; it is relabelled to
the short once, on load, so legends, columns and paths all use the one name a person sees.

Runtime figures default to `model_s` (featurisation + inference), the column comparable
ACROSS engines; --runtime-column inference_s restricts them to the inference half.

Legends sit OUTSIDE the axes so no data point is hidden behind them, and every figure is
saved with bbox_inches="tight" so a long title or an outside legend cannot be cropped.

Usage:
    pixi run plot-benchmark --arms af3-ob of3-ob \
        --dockq $AF3OF3_BASE_DIR/results/dockq --profile $AF3OF3_BASE_DIR/results/profiling
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_layout as fl
from result_tables import read_rows

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT = "benchmark"
SCRIPT = "scripts/06_plot_comparison.py"
# The four DockQ categories: a second, independent identity channel inside one figure, so
# they take their own slots from the same validated palette.
CATEGORY_COLOURS = {
    "homodimer": "#2a78d6",
    "heterodimer": "#eb6834",
    "homomer": "#1baf7a",
    "heteromer": "#e34948",
}
DOCKQ_THRESHOLDS = ((0.23, "acceptable"), (0.49, "medium"), (0.80, "high"))


def as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def by_entry(rows: list[dict], arm: str, column: str) -> dict[str, float]:
    out = {}
    for row in rows:
        if row.get("run") != arm:
            continue
        value = as_float(row.get(column, ""))
        if value is not None:
            out[row["entry_id"]] = value
    return out




def common_entries(rows: list[dict], arms: list[str], column: str) -> tuple[list[str], dict]:
    """Entries every arm has a value for, and each arm's values."""
    values = {a: by_entry(rows, a, column) for a in arms}
    common = sorted(set.intersection(*(set(v) for v in values.values()))) if values else []
    return common, values


def annotate(ax, text: str) -> None:
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left", fontsize=11.5,
            color=fl.INK_TEXT,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=fl.INK_FAINT, alpha=0.92))


# --------------------------------------------------------------------------- figures

def plot_dockq_distribution(rows, arms, labels, styles, baseline, out, subtitle) -> Path | None:
    common, values = common_entries(rows, arms, "dockq")
    if not common:
        print("  dockq_distribution: no entries scored in every arm, skipping")
        return None
    paired = baseline is not None and len(arms) == 2
    fig, axes = plt.subplots(1, 2 if paired else 1, figsize=(14.0 if paired else 8.0, 5.6),
                             squeeze=False)
    ax1 = axes[0][0]
    for arm in arms:
        xs = np.sort(np.array([values[arm][e] for e in common]))
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax1.step(xs, ys, where="post", color=styles[arm]["color"],
                 label=f"{labels[arm]}\nmedian {np.median(xs):.3f}")
    for threshold, name in DOCKQ_THRESHOLDS:
        ax1.axvline(threshold, ls=":", lw=1.0, color=fl.INK_FAINT, zorder=0)
        ax1.text(threshold, 0.02, f" {name}", rotation=90, fontsize=10.5, color=fl.INK_MUTED,
                 va="bottom")
    ax1.set_xlabel("DockQ")
    ax1.set_ylabel("cumulative fraction of complexes")
    fl.title(ax1, f"DockQ distribution (common set, n={len(common)})",
             None if paired else subtitle)
    if len(arms) > 1:
        ax1.legend(loc="upper left")

    if paired:
        other = next(a for a in arms if a != baseline)
        deltas = np.array([values[other][e] - values[baseline][e] for e in common])
        ax2 = axes[0][1]
        ax2.hist(deltas, bins=30, color=styles[other]["color"], edgecolor="white", linewidth=1.2)
        ax2.axvline(0, ls="--", lw=1.2, color=fl.INK_MUTED)
        ax2.set_xlabel(f"Δ DockQ   ({labels[other]} − {labels[baseline]})")
        ax2.set_ylabel("complexes")
        ax2.set_title("Paired difference")
        annotate(ax2, f"median {np.median(deltas):+.3f}\nmean {deltas.mean():+.3f}\nn={len(deltas)}")
    if paired and subtitle:
        # Above the panel titles, not across them: y > 1 works because every figure is
        # saved with bbox_inches="tight".
        fig.suptitle(subtitle, fontsize=11.5, color=fl.INK_TEXT,
                     x=0.0, y=1.04, ha="left", va="bottom")
    return fl.save(fig, out / "dockq_distribution.png", f"(n={len(common)})")


def plot_runtime_vs_size(rows, arms, labels, styles, column, out, subtitle) -> Path | None:
    size = {r["entry_id"]: v for r in rows if (v := as_float(r.get("total_len", ""))) is not None}
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    plotted = False
    for arm in arms:
        values = by_entry(rows, arm, column)
        pts = sorted((size[e], v) for e, v in values.items() if e in size)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=34, alpha=0.8, color=styles[arm]["color"], marker=fl.MARKER,
                   edgecolor="white", linewidth=0.8,
                   label=f"{labels[arm]}\nn={len(pts)}, median {np.median(ys):.0f} s")
        plotted = True
    if not plotted:
        plt.close(fig)
        print(f"  runtime_vs_size: no data in column '{column}', skipping")
        return None
    ax.set_xlabel("complex size (summed chain length, residues)")
    ax.set_ylabel(f"{column} (s)")
    fl.title(ax, f"Model time ({column}) vs complex size", subtitle)
    if len(arms) > 1:
        fl.outside_legend(ax)
    return fl.save(fig, out / "runtime_vs_size.png")


def plot_dockq_scatter(rows, arms, labels, baseline, out) -> Path | None:
    """Two arms only: per-entry DockQ, baseline on x. Colour carries the category here,
    which is why this figure exists for a pair and not for a set."""
    other = next(a for a in arms if a != baseline)
    common, values = common_entries(rows, arms, "dockq")
    if not common:
        print("  dockq_scatter: no entries scored in both arms, skipping")
        return None
    category = {r["entry_id"]: r.get("category", "") for r in rows}
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for cat in sorted({category.get(e, "") for e in common}):
        pts = [(values[baseline][e], values[other][e]) for e in common
               if category.get(e, "") == cat]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=34, alpha=0.8, color=CATEGORY_COLOURS.get(cat, fl.INK_MUTED),
                   edgecolor="white", linewidth=0.8, label=f"{cat or 'uncategorised'} (n={len(pts)})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=fl.INK_MUTED, zorder=0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(f"DockQ — {labels[baseline]}")
    ax.set_ylabel(f"DockQ — {labels[other]}")
    ax.set_title("Per-complex DockQ")
    fl.outside_legend(ax, title="category")
    # No summary box: the legend carries the ns and the aggregates belong in the caption.
    deltas = np.array([values[other][e] - values[baseline][e] for e in common])
    path = fl.save(fig, out / "dockq_scatter.png", f"(n={len(common)})")
    print(f"    n={len(common)}  median Δ={np.median(deltas):+.3f}  "
          f"{int((deltas > 0.05).sum())} better / {int((deltas < -0.05).sum())} worse by >0.05")
    return path


def plot_runtime_paired(rows, arms, labels, styles, baseline, column, out) -> Path | None:
    other = next(a for a in arms if a != baseline)
    common, values = common_entries(rows, arms, column)
    if not common:
        print(f"  runtime_paired: no entries timed in both arms (column '{column}'), skipping")
        return None
    xs = np.array([values[baseline][e] for e in common])
    ys = np.array([values[other][e] for e in common])
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(xs, ys, s=34, alpha=0.8, color=styles[other]["color"], marker=fl.MARKER,
               edgecolor="white", linewidth=0.8)
    hi = max(xs.max(), ys.max()) * 1.05
    ax.plot([0, hi], [0, hi], ls="--", lw=1.2, color=fl.INK_MUTED, zorder=0)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel(f"{column} (s) — {labels[baseline]}")
    ax.set_ylabel(f"{column} (s) — {labels[other]}")
    ax.set_title(f"Model time per complex ({column})")
    ratio = float(np.median(ys / np.where(xs == 0, np.nan, xs)))
    annotate(ax, f"n={len(common)}\nmedian x {np.median(xs):.1f} s\nmedian y {np.median(ys):.1f} s\n"
                 f"median ratio {ratio:.2f}x")
    return fl.save(fig, out / "runtime_paired.png", f"(n={len(common)})")


# ----------------------------------------------------------------------------- tables

def metrics_rows(dockq_rows, profile_rows, arms, labels, baseline, column) -> list[dict]:
    """One row per arm: the aggregates over the common set, the denominator stated."""
    dq_common, dq = common_entries(dockq_rows, arms, "dockq") if dockq_rows else ([], {})
    rt_common, rt = common_entries(profile_rows, arms, column) if profile_rows else ([], {})
    out = []
    for arm in arms:
        row = {"run": arm, "label": labels[arm], "is_baseline": str(arm == baseline).lower()}
        if dq_common:
            v = np.array([dq[arm][e] for e in dq_common])
            row.update({"n_common_dockq": len(dq_common), "dockq_median": f"{np.median(v):.4f}",
                        "dockq_mean": f"{v.mean():.4f}",
                        "n_scored": len(by_entry(dockq_rows, arm, "dockq"))})
            for threshold, name in DOCKQ_THRESHOLDS:
                row[f"frac_{name}"] = f"{float((v >= threshold).mean()):.4f}"
            if baseline and arm != baseline:
                d = v - np.array([dq[baseline][e] for e in dq_common])
                row.update({"dockq_delta_median": f"{np.median(d):+.4f}",
                            "n_better_005": int((d > 0.05).sum()),
                            "n_worse_005": int((d < -0.05).sum())})
        if rt_common:
            t = np.array([rt[arm][e] for e in rt_common])
            row.update({"n_common_runtime": len(rt_common), f"{column}_median": f"{np.median(t):.1f}",
                        f"{column}_total": f"{t.sum():.0f}"})
            if baseline and arm != baseline:
                b = np.array([rt[baseline][e] for e in rt_common])
                row["speedup_median"] = f"{float(np.median(b / np.where(t == 0, np.nan, t))):.2f}"
        out.append(row)
    return out


def evaluated_rows(dockq_rows, profile_rows, arms, column) -> list[dict]:
    """The common set itself: one row per entry, each arm's DockQ and model time."""
    dq_common, dq = common_entries(dockq_rows, arms, "dockq") if dockq_rows else ([], {})
    _, rt = common_entries(profile_rows, arms, column) if profile_rows else ([], {})
    meta = {r["entry_id"]: r for r in dockq_rows}
    rows = []
    for entry in dq_common:
        row = {"entry_id": entry, "category": meta.get(entry, {}).get("category", ""),
               "n_chains": meta.get(entry, {}).get("n_chains", ""),
               "total_len": meta.get(entry, {}).get("total_len", "")}
        for arm in arms:
            row[f"dockq__{arm}"] = f"{dq[arm][entry]:.4f}"
            if rt.get(arm, {}).get(entry) is not None:
                row[f"{column}__{arm}"] = f"{rt[arm][entry]:.1f}"
        rows.append(row)
    return rows


def write_directory(dockq_rows, profile_rows, configs, styles, baseline, *, out,
                    column, subtitle, sources, subset) -> None:
    """One directory: its figures, its tables and the manifest saying what it was made from.
    `subset` is recorded there because the three DockQ subsets write to the SAME directory —
    without it, nothing in the output says whether a figure holds the dimers or the whole set."""
    arms = [c.short for c in configs]
    labels = {a: a for a in arms}
    out.mkdir(parents=True, exist_ok=True)
    figures = []
    if dockq_rows:
        figures.append(plot_dockq_distribution(dockq_rows, arms, labels, styles, baseline,
                                               out, subtitle))
        if baseline and len(arms) == 2:
            figures.append(plot_dockq_scatter(dockq_rows, arms, labels, baseline, out))
    if profile_rows:
        figures.append(plot_runtime_vs_size(profile_rows, arms, labels, styles, column, out,
                                            subtitle))
        if baseline and len(arms) == 2:
            figures.append(plot_runtime_paired(profile_rows, arms, labels, styles, baseline,
                                               column, out))
    fl.write_table(out / "metrics.tsv",
                   metrics_rows(dockq_rows, profile_rows, arms, labels, baseline, column))
    if len(arms) > 1 and dockq_rows:
        fl.write_table(out / "complexes_evaluated.tsv",
                       evaluated_rows(dockq_rows, profile_rows, arms, column))
    fl.write_manifest(out, experiment=EXPERIMENT, script=SCRIPT, arms=configs,
                      baseline=baseline, sources=sources,
                      figures=[f for f in figures if f],
                      extra={"runtime_column": column, "dockq_subset": subset})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="*", default=None, metavar="SHORT",
                    help="Arms to compare, by their config's `short`, in legend order. "
                         "The first is the baseline.")
    ap.add_argument("--dockq", type=Path,
                    help="scripts/05 output: the per-arm directory "
                         "(<base>/results/dockq) or a single combined .tsv")
    ap.add_argument("--profile", type=Path,
                    help="scripts/04 output: the per-arm directory "
                         "(<base>/results/profiling) or a single combined .tsv")
    ap.add_argument("--runtime-column", default="model_s",
                    choices=["model_s", "inference_s", "wall_s"],
                    help="Which timing column the runtime figures use. model_s (default) is "
                         "featurisation+inference and is the only one comparable across engines.")
    ap.add_argument("--results-root", type=Path, default=fl.RESULTS_ROOT)
    ap.add_argument("--subset", choices=["dimers", "hetero", "all"], default="dimers",
                    help="Which scored entries enter the DockQ figures and metrics. dimers "
                         "(default): every 2-chain entry, one native interface each. hetero: "
                         "heterodimer + heteromer, the entries whose chains are not all one "
                         "sequence. all: every row in the tables, higher-order oligomers included. "
                         "The filter is applied here as well as in scripts/05, so a table scored "
                         "with a wider subset still draws the narrower figure.")
    ap.add_argument("--subtitle", default=None,
                    help="A line under each figure's title saying what the comparison is for")
    ap.add_argument("--no-individual", action="store_true",
                    help="Write only the comparison directory, not one per arm")
    args = ap.parse_args()

    if not args.dockq and not args.profile:
        sys.exit("ERROR: nothing to plot: pass --dockq and/or --profile")
    configs = fl.resolve_arms(args.arms or [])
    styles = fl.styles_for([c.short for c in configs])
    fl.apply_style()

    # The result tables key a row by `run`, which is the arm's name_tag. Relabel to the
    # short here, once, so everything downstream -- legends, columns, paths -- uses the one
    # name a person sees.
    to_short = {c.name: c.short for c in configs}

    def load(path):
        rows = read_rows(path) if path else []
        for row in rows:
            row["run"] = to_short.get(row.get("run", ""), row.get("run", ""))
        return rows

    dockq_rows = [r for r in load(args.dockq) if r.get("status") == "ok"]
    # The DockQ subset (see --subset): applied to the TABLE rows, so a table scored with a wider
    # subset draws the narrower figure without a rescore. Profiling rows are untouched: a runtime
    # figure is about the engine, not about which interfaces are worth scoring.
    if args.subset != "all":
        before = len(dockq_rows)
        if args.subset == "hetero":
            dockq_rows = [r for r in dockq_rows if r.get("category") in ("heterodimer", "heteromer")]
        else:
            dockq_rows = [r for r in dockq_rows if (r.get("n_chains") or "").strip() == "2"]
        print(f"  DockQ subset '{args.subset}': {len(dockq_rows)} of {before} scored rows")
    profile_rows = load(args.profile)
    present = {r.get("run") for r in dockq_rows} | {r.get("run") for r in profile_rows}
    missing = [c.short for c in configs if c.short not in present]
    if missing:
        print(f"  [WARN] no rows for {', '.join(missing)}; dropped from the figures")
        configs = [c for c in configs if c.short not in missing]
    if not configs:
        sys.exit("ERROR: none of those arms has any rows -- run --stage score / collect-profiling")
    baseline = configs[0].short if len(configs) > 1 else None

    sources = {c.short: {
        "dockq_table": fl.relative_source(Path(args.dockq) / f"{c.name}.tsv") if args.dockq else None,
        "dockq_rows": sum(1 for r in dockq_rows if r.get("run") == c.short),
        "profile_table": fl.relative_source(Path(args.profile) / f"{c.name}.tsv") if args.profile else None,
        "profile_rows": sum(1 for r in profile_rows if r.get("run") == c.short),
    } for c in configs}

    if len(configs) > 1:
        out = fl.sweep_dir(EXPERIMENT, [c.short for c in configs], args.results_root)
        print(f"comparison -> {out}")
        write_directory(dockq_rows, profile_rows, configs, styles, baseline, out=out,
                        column=args.runtime_column, subtitle=args.subtitle, sources=sources,
                        subset=args.subset)
    if not args.no_individual:
        for c in configs:
            out_i = fl.individual_dir(EXPERIMENT, c.short, args.results_root)
            print(f"individual -> {out_i}")
            write_directory(dockq_rows, profile_rows, [c], styles, None, out=out_i,
                            column=args.runtime_column, subtitle=c.label,
                            sources={c.short: sources[c.short]}, subset=args.subset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
