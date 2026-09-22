#!/usr/bin/env python3
"""
Plot peak GPU VRAM against token count for the arms that have been swept.

Reads one table per arm — <sweep-dir>/<arm>/vram_sweep.tsv, written row by row by
profiling/sweep.sbatch — and draws:

    a line per arm through the sizes that RAN (status ok)
    a dashed line at the card's memory, from gpu_total_mib in the table
    a dotted line at AlphaFold3's arena (mem_fraction x card), when an arm has one —
    that, not the card, is the ceiling an AF3 run actually hits

Nothing is drawn for the sizes that failed: where a curve ends IS where that arm
stopped, and a marker plus a legend entry for it said the same thing twice. The
boundary and the reason are printed to stdout and belong in the figure's caption.

WHICH MEMORY. The default metric is `peak_alloc_gib`: the framework's own live-buffer
high-water (JAX peak_bytes_in_use, PyTorch max_memory_allocated), which is the same
quantity for both engines. `--metric peak_vram_gib` plots the nvidia-smi whole-device
high-water instead, which for AF3 is just its preallocated arena and is flat. The two
are never mixed in one line.

Sizes that failed are never drawn as data points: a peak measured while a run was
dying is the memory it got to, not the memory it needed, and joining it to the curve
would suggest the arm ran there.

With --runtime a second panel adds model_s (the engine's own featurisation+inference
timer, as in scripts/04_collect_profiling.py) on the same x axis.

--publish-dir copies the per-arm tables next to the figure, dropping the two columns
that only mean anything on the machine that produced them (`node`, `slurm_job_id`) —
the same pair scripts/08_publish_results.py drops, and for the same reason: this
repository is public.

Usage:
    python profiling/plot_vram.py --sweep-dir <base>/profiling \
        --out profiling/results/vram_vs_tokens.png
"""

import itertools
import json
import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 150

# Same rcParams as scripts/06_plot_comparison.py, so a sweep figure and a benchmark
# figure sit side by side in a document without a font-size mismatch.
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})

# Eight, and cycled below: `zip` with a shorter palette would drop every arm past the last
# colour from the figure without a word, and a five-arm plot (stock + four kit modes) is now
# the common case.
ARM_COLOURS = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C")
# Dropped from the published copy of a table: they name the compute host, not the
# measurement. Same pair as scripts/08_publish_results.py drops.
HOST_COLUMNS = ("node", "slurm_job_id")
TEMPLATE_STATE = {"true": "templates on", "false": "templates off", "": "templates ?"}
# Why an arm stopped, spelled out on the figure rather than left as a status code.
STATUS_LABEL = {
    "gpu_oom": "GPU out of memory",
    "cpu_oom": "host out of memory",
    "failed": "failed",
    "error": "error",
}


def read_tsv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def as_float(value: str | None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def arm_label(rows: list[dict], with_templates: bool) -> str:
    """Short display label: the codebase, and the template state only if it varies.

    The axis this figure is about is the codebase, and every arm plotted so far runs
    templates-off, so naming that in each legend entry repeated context identical
    across the series — it belongs in the caption instead. It is appended only when
    the arms actually disagree, so a mixed figure can still be read.
    """
    tag = next((r.get("arm", "") for r in rows), "")
    engine = next((r.get("engine", "") for r in rows), "")
    templates = next((r.get("use_templates", "") for r in rows), "")
    engine_label = {"af3": "AF3 code", "of3": "OF3 code"}.get(engine, engine or tag)
    if not with_templates:
        return engine_label
    return f"{engine_label} ({TEMPLATE_STATE.get(templates, 'templates ?')})"


def publish_tables(series, args) -> None:
    """Copy each arm's table beside the figure, without the host columns (the same pair
    scripts/08_publish_results.py drops)."""
    if args.no_publish:
        return
    publish_dir = args.publish_dir or args.out.parent
    publish_dir.mkdir(parents=True, exist_ok=True)
    for name, rows, _ in series:
        columns = [c for c in rows[0] if c not in HOST_COLUMNS]
        dest = publish_dir / f"{name}.tsv"
        with open(dest, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {dest}")


def plot_relative(series, args, set_title=None) -> int:
    """Every arm relative to one baseline arm, at the token counts both ran.

    Left: speed-up = baseline model_s / arm model_s -- above 1 the arm is faster than the
    baseline, below 1 slower. Right: peak memory as a fraction of the baseline's -- below 1 the
    arm needs less. Both panels share the x axis (tokens) and carry a line at 1. A token count
    the baseline did not finish (its boundary) has no ratio, so an arm that reaches FURTHER than
    the baseline shows nothing there: that reach is a result the absolute figure carries, not
    this one. Only `status == ok` rows on both sides are used.
    """
    by_name = {name: rows for name, rows, _ in series}
    tags = {name: tag for name, rows, tag in series}
    base_name = args.relative_to
    if base_name not in by_name:
        sys.exit(f"ERROR: --relative-to {base_name}: no such arm among {sorted(by_name)}")

    def ok_map(rows, key):
        out = {}
        for r in rows:
            if r.get("status") != "ok":
                continue
            v = as_float(r.get(key))
            if v is not None:
                out[int(float(r["tokens"]))] = v
        return out

    base_t = ok_map(by_name[base_name], "model_s")
    base_m = ok_map(by_name[base_name], args.metric)
    if not base_t:
        sys.exit(f"ERROR: baseline {base_name} has no ok rows with model_s")
    template_states = {next((r.get("use_templates", "") for r in rows), "") for _, rows, _ in series}
    with_templates = len(template_states) > 1
    base_label = tags.get(base_name) or arm_label(by_name[base_name], with_templates)

    fig, (ax_s, ax_m) = plt.subplots(1, 2, figsize=(18.0, 5.8))
    table = []                                   # (arm, tokens, speedup, mem_ratio) for the printout
    others = [(n, r, tg) for n, r, tg in series if n != base_name]
    for (name, rows, tag), colour in zip(others, itertools.cycle(ARM_COLOURS)):
        label = tag or arm_label(rows, with_templates)
        t_map, m_map = ok_map(rows, "model_s"), ok_map(rows, args.metric)
        common_t = sorted(set(t_map) & set(base_t))
        common_m = sorted(set(m_map) & set(base_m))
        if common_t:
            ys = [base_t[k] / t_map[k] for k in common_t]
            ax_s.plot(common_t, ys, "o-", color=colour, lw=2.2, ms=7, label=label)
        if common_m:
            ax_m.plot(common_m, [m_map[k] / base_m[k] for k in common_m], "s-", color=colour, lw=2.2, ms=7, label=label)
        for k in sorted(set(common_t) | set(common_m)):
            table.append((name, k,
                          base_t[k] / t_map[k] if k in t_map and k in base_t else None,
                          m_map[k] / base_m[k] if k in m_map and k in base_m else None))
        # the arm's own boundary, if it stopped before the baseline did
        bad = [int(float(r["tokens"])) for r in rows if r.get("status") not in ("ok", "")]
        if bad and min(bad) <= max(base_t):
            for ax in (ax_s, ax_m):
                ax.axvline(min(bad), color=colour, ls=":", lw=1.4, alpha=0.8)

    for ax in (ax_s, ax_m):
        ax.axhline(1.0, color="#555555", lw=1.4, ls="--", zorder=0)
        ax.set_xlabel("Tokens (= residues, protein-only pools)")
        ax.grid(True, alpha=0.3)
    ax_s.set_ylabel(f"Speed-up over {base_label}  (baseline time / arm time)")
    ax_s.set_title("Model time relative to the baseline  (>1 = faster)")
    ax_s.set_ylim(bottom=0)
    ax_s.legend(loc="best", frameon=False)
    ax_m.set_ylabel(f"Peak memory / {base_label}")
    ax_m.set_title("Peak live GPU memory relative to the baseline  (<1 = less)")
    ax_m.set_ylim(bottom=0)
    ax_m.legend(loc="best", frameon=False)
    fig.suptitle((set_title + "\n" if set_title else "") +
                 f"relative to {base_name}; dotted verticals = where an arm stopped (its boundary)",
                 fontsize=12)
    fig.tight_layout()

    publish_tables(series, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved {args.out}")

    print(f"\nrelative to {base_name}   (speed-up = baseline model_s / arm model_s; memory = arm / baseline)")
    print(f"{'arm':44s} {'tokens':>7s} {'speed-up':>9s} {'memory':>7s}")
    for name, k, s, m in table:
        print(f"{name:44s} {k:>7d} {('%.2fx' % s) if s else '-':>9s} {('%.2f' % m) if m else '-':>7s}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", type=Path, default=None,
                    help="Holds one <arm>/vram_sweep.tsv per arm "
                         "(default: $AF3OF3_BASE_DIR/profiling)")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="Arm directories to plot, in legend order (default: all)")
    ap.add_argument("--set", dest="set_name", default=None,
                    help="A named figure set from profiling/results_sets.json: its arms, its "
                         "title, and profiling/results/<set>/ as the output directory with fixed "
                         "file names (vram_vs_tokens.png, vram_and_runtime.png, <arm>.tsv, "
                         "manifest.json). The reproducible unit -- prefer it over --arms/--out.")
    ap.add_argument("--sets-file", type=Path,
                    default=Path(__file__).resolve().parent / "results_sets.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="Figure path (default: profiling/results/vram_vs_tokens.png, or the "
                         "set's directory when --set is given)")
    ap.add_argument("--metric", choices=("peak_alloc_gib", "peak_vram_gib"),
                    default="peak_alloc_gib",
                    help="Which memory column to plot (default: peak_alloc_gib, the "
                         "framework's own live-buffer high-water)")
    ap.add_argument("--publish-dir", type=Path, default=None,
                    help="Also copy each arm's table here, without the host columns "
                         "(default: alongside --out)")
    ap.add_argument("--no-publish", action="store_true",
                    help="Do not copy the tables, only write the figure")
    ap.add_argument("--runtime", action="store_true",
                    help="Add a second panel with model_s vs tokens")
    ap.add_argument("--relative-to", metavar="ARM", default=None,
                    help="Instead of absolute curves, plot every other arm RELATIVE to this one at "
                         "each token count the two share: speed-up = ARM's model_s / the arm's "
                         "(>1 faster), and peak memory as a fraction of ARM's (<1 less). ARM is "
                         "the production engine, typically af3_openbind_templateoff[_a100].")
    ap.add_argument("--relative", action="store_true",
                    help="With --set: --relative-to the set's `baseline` from results_sets.json")
    args = ap.parse_args()

    # --set: resolve arms, title and the output directory from results_sets.json
    set_title = None
    if args.set_name:
        try:
            sets = json.load(open(args.sets_file))
        except OSError as exc:
            sys.exit(f"ERROR: cannot read {args.sets_file}: {exc}")
        spec = sets.get(args.set_name)
        if spec is None:
            named = ", ".join(k for k in sets if not k.startswith("_"))
            sys.exit(f"ERROR: no set '{args.set_name}' in {args.sets_file} (have: {named})")
        if args.arms:
            sys.exit("ERROR: pass --set or --arms, not both")
        args.arms = list(spec["arms"])
        set_title = spec.get("title")
        if args.relative and not args.relative_to:
            args.relative_to = spec.get("baseline")
            if not args.relative_to:
                sys.exit(f"ERROR: set '{args.set_name}' names no `baseline` in {args.sets_file}; pass --relative-to ARM")
        set_dir = (args.out.parent if args.out else Path("profiling/results")) / args.set_name
        set_dir.mkdir(parents=True, exist_ok=True)
        if args.relative_to:
            args.out = set_dir / f"relative_to_{args.relative_to}.png"
        else:
            args.out = set_dir / ("vram_and_runtime.png" if args.runtime else "vram_vs_tokens.png")
    elif args.relative and not args.relative_to:
        sys.exit("ERROR: --relative needs --set (whose baseline it uses) or --relative-to ARM")
    elif args.out is None:
        args.out = Path("profiling/results/relative.png" if args.relative_to else "profiling/results/vram_vs_tokens.png")
    if args.relative_to and args.arms and args.relative_to not in args.arms:
        args.arms = [args.relative_to] + list(args.arms)          # the baseline's table is needed even if not asked for

    sweep_dir = args.sweep_dir
    if sweep_dir is None:
        base = os.environ.get("AF3OF3_BASE_DIR", "")
        if not base:
            sys.exit("ERROR: pass --sweep-dir or set AF3OF3_BASE_DIR")
        sweep_dir = Path(base) / "profiling"

    names = args.arms or sorted(p.parent.name
                                for p in sweep_dir.glob("*/vram_sweep.tsv"))
    if not names:
        sys.exit(f"ERROR: no <arm>/vram_sweep.tsv under {sweep_dir}")

    series = []
    for name in names:
        path = sweep_dir / name / "vram_sweep.tsv"
        if not path.is_file():
            sys.exit(f"ERROR: no such table: {path}")
        rows = [r for r in read_tsv(path) if as_float(r.get("tokens")) is not None]
        rows.sort(key=lambda r: float(r["tokens"]))
        if not rows:
            print(f"  [WARN] {path} has no rows yet, skipping")
            continue
        # An arm launched through a kit (configs/of3kit_*.json, `launcher` block) shares its
        # engine with the plain of3 arm, so the engine label alone would not tell them apart;
        # run_sweep.sh copies the arm's config next to its table, and its `tag` names the launcher.
        tag = None
        cfg_path = sweep_dir / name / "config.json"
        if cfg_path.is_file():
            try:
                cfg = json.load(open(cfg_path))
                tag = cfg.get("tag") if cfg.get("launcher") else None
            except (OSError, json.JSONDecodeError):
                tag = None
        series.append((name, rows, tag))
    if not series:
        sys.exit("ERROR: every table is empty — has a sweep run?")

    if args.relative_to:
        return plot_relative(series, args, set_title)

    n_panels = 2 if args.runtime else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(9.0 * n_panels, 5.8),
                             squeeze=False)
    ax_v = axes[0][0]

    # Only spell out the template state when the arms plotted do not agree on it.
    template_states = {next((r.get("use_templates", "") for r in rows), "")
                       for _, rows, _ in series}
    with_templates = len(template_states) > 1

    capacities = set()
    arenas = set()
    for (name, rows, tag), colour in zip(series, itertools.cycle(ARM_COLOURS)):
        ok = [r for r in rows if r.get("status") == "ok"]
        label = tag or arm_label(rows, with_templates)

        xs = [float(r["tokens"]) for r in ok]
        vram = [as_float(r.get(args.metric)) for r in ok]
        if ok and not any(v is not None for v in vram):
            sys.exit(f"ERROR: {name} has no '{args.metric}' values. An older table "
                     f"predates the column — re-run the sweep, or pass "
                     f"--metric peak_vram_gib.")
        pairs = [(x, v) for x, v in zip(xs, vram) if v is not None]
        if pairs:
            ax_v.plot(*zip(*pairs), "o-", color=colour, lw=2.2, ms=7, label=label)

        for r in rows:
            total = as_float(r.get("gpu_total_mib"))
            if total:
                capacities.add(round(total / 1024, 2))
                fraction = as_float(r.get("mem_fraction"))
                if fraction:
                    arenas.add((round(total * fraction / 1024, 2), fraction))

        if args.runtime:
            ax_t = axes[0][1]
            times = [(float(r["tokens"]), as_float(r.get("model_s"))) for r in ok]
            times = [(x, t) for x, t in times if t is not None]
            if times:
                ax_t.plot(*zip(*times), "s-", color=colour, lw=2.2, ms=7, label=label)

    for capacity in sorted(capacities):
        ax_v.axhline(capacity, ls="--", lw=1.4, color="#999999", zorder=0)
        ax_v.text(0.01, capacity, f" GPU memory {capacity:.0f} GiB", va="bottom",
                  ha="left", transform=ax_v.get_yaxis_transform(), fontsize=11,
                  color="#777777")

    for arena, fraction in sorted(arenas):
        ax_v.axhline(arena, ls=":", lw=1.4, color="#bbbbbb", zorder=0)
        ax_v.text(0.01, arena, f" AF3 arena ({fraction:g} x card)", va="top",
                  ha="left", transform=ax_v.get_yaxis_transform(), fontsize=11,
                  color="#999999")

    ax_v.set_xlabel("Tokens (= residues, protein-only pools)")
    ax_v.set_ylabel("Peak live GPU memory (GiB)" if args.metric == "peak_alloc_gib"
                    else "Peak device memory, nvidia-smi (GiB)")
    ax_v.set_title("Peak live GPU memory vs complex size"
                   if args.metric == "peak_alloc_gib"
                   else "Peak device memory vs complex size")
    ax_v.set_ylim(0, max(capacities) * 1.08 if capacities else None)
    ax_v.grid(True, alpha=0.3)
    # Bottom right, which a memory-vs-size curve leaves empty: every series rises to
    # the right, so the corner under the longest curve carries no data. The reference
    # lines and their labels hold the top left.
    ax_v.legend(loc="lower right", frameon=False)

    if args.runtime:
        ax_t = axes[0][1]
        ax_t.set_xlabel("Tokens (= residues, protein-only pools)")
        ax_t.set_ylabel("Model time (s)")
        ax_t.set_title("Featurisation + inference vs complex size")
        ax_t.set_ylim(bottom=0)
        ax_t.grid(True, alpha=0.3)
        ax_t.legend(loc="upper left", frameon=False)

    publish_tables(series, args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved {args.out}")

    # The same numbers as text, so the figure never has to be squinted at.
    for name, rows, _ in series:
        ok = [r for r in rows if r.get("status") == "ok"]
        bad = [r for r in rows if r.get("status") not in ("ok", "")]
        largest = max((int(float(r["tokens"])) for r in ok), default=0)
        print(f"\n{name}: {len(ok)} size(s) ran, largest {largest} tokens")
        for r in rows:
            print(f"  {int(float(r['tokens'])):>6} tokens  "
                  f"{r.get('status',''):>8}  "
                  f"live {r.get('peak_alloc_gib','') or '-':>6} GiB  "
                  f"device {r.get('peak_vram_gib','') or '-':>6} GiB  "
                  f"model {r.get('model_s','') or '-':>7} s")
        if bad:
            first = min(bad, key=lambda r: float(r["tokens"]))
            print(f"  boundary: {STATUS_LABEL.get(first.get('status',''), '?')} "
                  f"at {int(float(first['tokens']))} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
