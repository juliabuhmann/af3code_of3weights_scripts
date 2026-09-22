#!/usr/bin/env python3
"""
Stage 06 -- accuracy and cost on Benchmark-IV, for a registered set of arms.

Reads the per-arm tables stages 04 and 05 wrote and writes, for a set from
configs/figure_sets.json (or an ad-hoc list of arms):

    results/benchmark/sweep/<set>/
        dockq_distribution.png   ECDF per arm over the common set, plus the paired
                                 difference against the baseline
        runtime_vs_size.png      model time against complex size, one series per arm --
                                 cost as a function of input size, which a median hides
        dockq_scatter.png        per-entry DockQ, baseline (x) vs the other arm (y),
                                 coloured by category. Two arms only: the headline of a
                                 pairwise comparison, meaningless for three
        runtime_paired.png       model time, baseline vs the other arm, per complex.
                                 Two arms only
        metrics.tsv              per-arm summary over the common set
        complexes_evaluated.tsv  the entries every arm scored, with each arm's values
        manifest.json            arms in legend order, baseline, source tables, commit

    results/benchmark/individual_runs/<arm>/
        dockq_distribution.png, runtime_vs_size.png, metrics.tsv, manifest.json

Every aggregate is computed over the COMMON SET -- entries present in every arm -- so a
difference in how many entries each arm completed cannot masquerade as a difference in
quality. The n is reported on every figure or on stdout beside it.

LABELS AND COLOURS. Each arm carries a short `tag` naming its pipeline ("AF3 code /
openbind weights"), set in its config and copied into every result row (see
scripts/arm_metadata.py); figures use that rather than the directory name. The template
state is appended only when the arms plotted disagree on it -- comparing templates-on
against templates-off it is the whole point of the figure, comparing two templates-off
arms it is shared context that belongs in the caption. Colours come from
configs/arm_style.json: one fixed hue per arm across every figure in the repository.

Runtime figures default to `model_s` (featurisation + inference), the column comparable
ACROSS engines; --runtime-column inference_s restricts them to AlphaFold 3's inference
timer, which OpenFold3 rows do not have.

Usage:
    pixi run plot-benchmark --set engine
    pixi run plot-benchmark --arms af3_native af3_openbind --baseline af3_native
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
TEMPLATE_STATE = {"true": "templates on", "false": "templates off", "": "templates ?"}
# The four DockQ categories: a second, independent identity channel inside one figure,
# so they take their own slots from the same validated palette.
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


def field(rows: list[dict], arm: str, column: str, default: str = "") -> str:
    """One arm's value for `column`, or a "/"-joined string when the arm disagrees with
    itself. Read off the rows rather than passed in, so a label cannot disagree with the
    data being plotted."""
    values = {r.get(column, "") for r in rows if r.get("run") == arm}
    values.discard("")
    if len(values) == 1:
        return values.pop()
    if len(values) > 1:
        return "/".join(sorted(values))
    return default


def arm_labels(rows: list[dict], arms: list[str]) -> dict[str, str]:
    """Legend labels spelling out only what DIFFERS; never two series with one label."""
    labels = {a: field(rows, a, "tag", default=a) or a for a in arms}
    states = {a: field(rows, a, "use_templates") for a in arms}
    if len(set(states.values())) > 1:
        labels = {a: f"{labels[a]} ({TEMPLATE_STATE.get(states[a], 'templates ?')})" for a in arms}
    for arm in arms:
        if list(labels.values()).count(labels[arm]) > 1:
            labels[arm] = f"{labels[arm]} [{arm}]"
    return labels


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

def plot_dockq_distribution(rows, arms, labels, styles, baseline, out, title) -> Path | None:
    common, values = common_entries(rows, arms, "dockq")
    if not common:
        print("  dockq_distribution: no entries scored in every arm, skipping")
        return None
    paired = baseline is not None and len(arms) == 2
    fig, axes = plt.subplots(1, 2 if paired else 1, figsize=(14.0 if paired else 8.0, 5.6),
                             squeeze=False)
    ax1 = axes[0][0]
    for index, arm in enumerate(arms):
        xs = np.sort(np.array([values[arm][e] for e in common]))
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax1.step(xs, ys, where="post", color=styles[arm]["color"],
                 lw=fl.line_width(index, len(arms)),
                 label=f"{labels[arm]}\nmedian {np.median(xs):.3f}")
    for threshold, name in DOCKQ_THRESHOLDS:
        ax1.axvline(threshold, ls=":", lw=1.0, color=fl.INK_FAINT, zorder=0)
        ax1.text(threshold, 0.02, f" {name}", rotation=90, fontsize=10.5, color=fl.INK_MUTED,
                 va="bottom")
    ax1.set_xlabel("DockQ")
    ax1.set_ylabel("cumulative fraction of complexes")
    ax1.set_title(f"DockQ distribution (common set, n={len(common)})")
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
    if title:
        # Above the panel titles, not across them: y > 1 works because every figure is
        # saved with bbox_inches="tight".
        fig.suptitle(title, fontsize=14, fontweight="bold", color="#22221f",
                     x=0.0, y=1.06, ha="left", va="bottom")
    return fl.save(fig, out / "dockq_distribution.png", f"(n={len(common)})")


def plot_runtime_vs_size(rows, arms, labels, styles, column, out, title) -> Path | None:
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
    ax.set_title(title or f"Model time ({column}) vs complex size")
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


def write_directory(dockq_rows, profile_rows, arms, labels, styles, baseline, *, out, column,
                    set_name, title, sources, missing) -> None:
    out.mkdir(parents=True, exist_ok=True)
    figures = []
    if dockq_rows:
        figures.append(plot_dockq_distribution(dockq_rows, arms, labels, styles, baseline, out, title))
        if baseline and len(arms) == 2:
            figures.append(plot_dockq_scatter(dockq_rows, arms, labels, baseline, out))
    if profile_rows:
        figures.append(plot_runtime_vs_size(profile_rows, arms, labels, styles, column, out, title))
        if baseline and len(arms) == 2:
            figures.append(plot_runtime_paired(profile_rows, arms, labels, styles, baseline,
                                               column, out))
    fl.write_table(out / "metrics.tsv",
                   metrics_rows(dockq_rows, profile_rows, arms, labels, baseline, column))
    if len(arms) > 1 and dockq_rows:
        fl.write_table(out / "complexes_evaluated.tsv",
                       evaluated_rows(dockq_rows, profile_rows, arms, column))
    fl.write_manifest(out, experiment=EXPERIMENT, script=SCRIPT, set_name=set_name, title=title,
                      arms=arms, baseline=baseline, sources=sources,
                      figures=[f for f in figures if f], missing=missing,
                      extra={"runtime_column": column})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="set_name", default=None,
                    help="A benchmark set from configs/figure_sets.json: its arms, baseline and "
                         "title; written to results/benchmark/sweep/<set>/")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="Arms to compare instead, in legend order")
    ap.add_argument("--baseline", default=None,
                    help="Arm the paired figures compare against (default: the set's `baseline`)")
    ap.add_argument("--sets-file", type=Path, default=fl.SETS_FILE)
    ap.add_argument("--dockq", type=Path,
                    help="scripts/05 output: the per-arm directory (<base>/results/dockq) "
                         "or a single combined .tsv")
    ap.add_argument("--profile", type=Path,
                    help="scripts/04 output: the per-arm directory (<base>/results/profiling) "
                         "or a single combined .tsv")
    ap.add_argument("--runtime-column", default="model_s",
                    choices=["model_s", "inference_s", "wall_s"],
                    help="Which timing column the runtime figures use. model_s (default) is "
                         "featurisation+inference and is the only one comparable across engines.")
    ap.add_argument("--results-root", type=Path, default=fl.RESULTS_ROOT)
    ap.add_argument("--no-individual", action="store_true",
                    help="Write only the sweep directory, not one directory per arm")
    args = ap.parse_args()

    if not args.dockq and not args.profile:
        sys.exit("ERROR: nothing to plot: pass --dockq and/or --profile")
    set_name, arms, baseline, title = fl.resolve_selection(args, EXPERIMENT)
    styles = fl.styles_for(arms)
    fl.apply_style()

    sources: dict[str, dict] = {}
    dockq_rows = [r for r in read_rows(args.dockq) if r.get("status") == "ok"] if args.dockq else []
    profile_rows = read_rows(args.profile) if args.profile else []
    present = {r.get("run") for r in dockq_rows} | {r.get("run") for r in profile_rows}
    missing = [a for a in arms if a not in present]
    if missing:
        print(f"  [WARN] no rows for {', '.join(missing)}; dropped from the figures")
        arms = [a for a in arms if a not in missing]
    if not arms:
        sys.exit("ERROR: none of the set's arms has any rows -- run --stage score / collect-profiling")
    if baseline in missing:
        print(f"  [WARN] baseline {baseline} has no rows; no paired figures")
        baseline = None
    for arm in arms:
        sources[arm] = {
            "dockq_table": fl.relative_source(Path(args.dockq) / f"{arm}.tsv") if args.dockq else None,
            "dockq_rows": sum(1 for r in dockq_rows if r.get("run") == arm),
            "profile_table": fl.relative_source(Path(args.profile) / f"{arm}.tsv") if args.profile else None,
            "profile_rows": sum(1 for r in profile_rows if r.get("run") == arm),
        }
    labels = arm_labels(dockq_rows or profile_rows, arms)
    print("labels: " + " | ".join(labels[a] for a in arms))

    out = fl.sweep_dir(EXPERIMENT, set_name, arms, args.results_root)
    print(f"sweep -> {out}")
    write_directory(dockq_rows, profile_rows, arms, labels, styles, baseline, out=out,
                    column=args.runtime_column, set_name=set_name, title=title,
                    sources=sources, missing=missing)
    if not args.no_individual:
        for arm in arms:
            out_i = fl.individual_dir(EXPERIMENT, arm, args.results_root)
            print(f"individual -> {out_i}")
            write_directory(dockq_rows, profile_rows, [arm], labels, styles, None, out=out_i,
                            column=args.runtime_column, set_name=None, title=labels[arm],
                            sources={arm: sources[arm]}, missing=[])
    return 0


if __name__ == "__main__":
    sys.exit(main())
