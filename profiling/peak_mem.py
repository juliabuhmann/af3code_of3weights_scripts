#!/usr/bin/env python3
"""
Run an engine's entry point in-process and record the framework's own peak memory.

WHY THIS EXISTS. nvidia-smi reports whole-device memory, and under AlphaFold3's
production allocator settings (XLA_PYTHON_CLIENT_PREALLOCATE=true,
XLA_CLIENT_MEM_FRACTION=0.95 — what scripts/2026aug_af3_openbind_weights in the
pooled_ppi_comparison repo runs) JAX reserves that fraction of the card up front. The
device then reads ~93 GiB at every token count, which says nothing about how much the
model actually needed. Turning preallocation OFF to make nvidia-smi follow demand is
not a fix: it caps the arena at the default 0.75 fraction AND leaves the allocator
growing in regions, so a single large block (41.6 GiB at 4000 tokens) cannot be placed
and the run OOMs thousands of tokens below where it otherwise would.

So the token-vs-memory curve is taken from the allocators themselves, which report the
same kind of number for both engines:

    af3   jax.Device.memory_stats()["peak_bytes_in_use"]
    of3   torch.cuda.max_memory_allocated()

Both are the high-water mark of LIVE buffers, independent of how much the allocator
reserved from the driver, so the two engines are compared on the same quantity. The
reserved high-water is recorded too (JAX `peak_pool_bytes`/`bytes_reserved`, PyTorch
`max_memory_reserved`), and the sweep keeps the nvidia-smi trace alongside.

The counters are read in a `finally` block, NOT in an atexit hook. atexit runs
handlers last-registered-first, and both frameworks register their own teardown when
they are imported — which is after this wrapper starts — so an atexit hook here would
run after the backend had been shut down and would read zeroes (it did, first time
round). A `finally` block runs while the engine's process state is still intact, and
still runs when the engine calls sys.exit (absl and click both do) or dies with an
out-of-memory traceback. A run killed by a signal writes nothing, and the sweep records
that as a missing value rather than as a zero.

Usage — everything after `--` is passed to the target unchanged:

    python profiling/peak_mem.py --out PATH --engine af3 \
        --script /app/alphafold/run_alphafold.py -- --json_path=... --output_dir=...
    python profiling/peak_mem.py --out PATH --engine of3 \
        --module openfold3.run_openfold -- predict --query-json ...
"""

import json
import runpy
import sys
import time

MIB = 1024 * 1024

# --------------------------------------------------------------- OF3 predict timeline
#
# WHY. OpenFold3's own timer (runtime_s in each seed's timing.json) starts when Lightning
# hands the model a batch and stops when the batch is done. Producing that batch -- parsing
# the MSA, building the template and token features -- happens in the DataLoader BEFORE
# that clock starts, so it is not in the number. AlphaFold3's log reports featurisation and
# inference separately and the sweep adds them, so the two engines' "model time" did not
# mean the same thing, and the difference grows with input size (55 s of it at 5000 tokens).
#
# The fix is measured, not estimated: a Lightning callback of ours, attached to whatever
# Trainer the runner builds, times the gaps the loop spends waiting for data and the time
# it spends inside batches. `compute_s` is the same window OpenFold3 times itself, so a row
# where the two disagree is a sign this instrumentation did not see every batch.
#
# It measures what it can see: with DataLoader workers the fetch overlaps the compute and
# the gap understates the work, so `num_workers` is recorded beside it and anything other
# than 0 makes `featurisation_s` a lower bound. The profiling runner yamls set it to 0.
TIMELINE: dict = {"featurisation_s": 0.0, "compute_s": 0.0, "batches": 0,
                  "num_workers": None, "attached": False}


def install_of3_timeline() -> None:
    """Attach a predict-timeline callback to every Lightning Trainer this process builds.

    Patching the constructor rather than the runner keeps this out of OpenFold3 and out of
    the kit: neither knows it is being timed, and the sweep runs the same code path it
    would have run anyway. A failure to attach is recorded and never raised -- an engine
    that still produces its structures is worth more than an instrumented crash."""
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import Callback
    except Exception as exc:                        # noqa: BLE001 - recorded, not raised
        TIMELINE["error"] = f"pytorch_lightning not importable: {type(exc).__name__}"
        return

    class PredictTimeline(Callback):
        def __init__(self) -> None:
            self.mark = None

        def on_predict_start(self, trainer, pl_module) -> None:
            self.mark = time.perf_counter()
            loaders = getattr(trainer, "predict_dataloaders", None)
            loader = loaders[0] if isinstance(loaders, (list, tuple)) and loaders else loaders
            TIMELINE["num_workers"] = getattr(loader, "num_workers", None)

        def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx,
                                   dataloader_idx=0) -> None:
            now = time.perf_counter()
            if self.mark is not None:               # the loop waiting for the DataLoader
                TIMELINE["featurisation_s"] += now - self.mark
            self.mark = now

        def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                                 dataloader_idx=0) -> None:
            now = time.perf_counter()
            if self.mark is not None:               # the model on this batch
                TIMELINE["compute_s"] += now - self.mark
            self.mark = now
            TIMELINE["batches"] += 1

    original = pl.Trainer.__init__

    def patched(self, *args, **kwargs):
        original(self, *args, **kwargs)
        try:
            self.callbacks.append(PredictTimeline())
            TIMELINE["attached"] = True
        except Exception as exc:                    # noqa: BLE001 - recorded, not raised
            TIMELINE["error"] = f"could not attach the callback: {type(exc).__name__}: {exc}"

    pl.Trainer.__init__ = patched


def timeline_payload() -> dict:
    """The timeline as result fields, or {} when nothing was timed."""
    if not TIMELINE["batches"]:
        return {"of3_timeline_error": TIMELINE.get("error", "no predict batch was timed")}
    return {
        "featurisation_s": round(TIMELINE["featurisation_s"], 3),
        "compute_s": round(TIMELINE["compute_s"], 3),
        "timeline_batches": TIMELINE["batches"],
        "timeline_num_workers": TIMELINE["num_workers"],
        # A lower bound rather than a measurement when the fetch ran in worker processes.
        "featurisation_is_lower_bound": bool(TIMELINE["num_workers"]),
    }


def read_af3_stats() -> dict:
    import jax

    devices = jax.local_devices()
    if not devices:
        return {"error": "no local jax device"}
    stats = devices[0].memory_stats() or {}
    return {k: v for k, v in stats.items() if isinstance(v, (int, float))}


def read_of3_stats() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"error": "cuda not available"}
    return {"max_memory_allocated": int(torch.cuda.max_memory_allocated()),
            "max_memory_reserved": int(torch.cuda.max_memory_reserved())}


# The one field the sweep plots, per engine: the live-buffer high-water.
PEAK_KEY = {"af3": "peak_bytes_in_use", "of3": "max_memory_allocated"}
# The reserved high-water, for the record. JAX's key name has moved between
# jaxlib versions, so the first one present wins.
RESERVED_KEYS = {"af3": ("peak_pool_bytes", "bytes_reserved", "pool_bytes"),
                 "of3": ("max_memory_reserved",)}


def dump(out_path: str, engine: str) -> None:
    try:
        stats = read_af3_stats() if engine == "af3" else read_of3_stats()
    except Exception as exc:                       # noqa: BLE001 — recorded, not raised
        stats = {"error": f"{type(exc).__name__}: {exc}"}

    peak = stats.get(PEAK_KEY[engine])
    reserved = next((stats[k] for k in RESERVED_KEYS[engine] if k in stats), None)
    payload = {
        "engine": engine,
        "peak_alloc_mib": round(peak / MIB, 1) if isinstance(peak, (int, float)) else None,
        "peak_reserved_mib": (round(reserved / MIB, 1)
                              if isinstance(reserved, (int, float)) else None),
        "raw": stats,
    }
    if engine == "of3":
        payload.update(timeline_payload())
    try:
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)
    except OSError as exc:
        print(f"peak_mem.py: could not write {out_path}: {exc}", file=sys.stderr)


def main() -> int:
    args = sys.argv[1:]
    out = engine = script = module = None
    rest: list[str] = []
    while args:
        token = args.pop(0)
        if token == "--":
            rest = args
            break
        if token == "--out":
            out = args.pop(0)
        elif token == "--engine":
            engine = args.pop(0)
        elif token == "--script":
            script = args.pop(0)
        elif token == "--module":
            module = args.pop(0)
        else:
            print(f"peak_mem.py: unknown argument {token}", file=sys.stderr)
            return 2
    if not out or engine not in PEAK_KEY or not (script or module):
        print("usage: peak_mem.py --out PATH --engine af3|of3 "
              "(--script PATH | --module NAME) -- ARGS...", file=sys.stderr)
        return 2

    if engine == "of3":
        install_of3_timeline()

    # The target must see the argv it would have seen if it had been invoked directly.
    sys.argv = [script or module, *rest]
    code = 0
    try:
        if script:
            runpy.run_path(script, run_name="__main__")
        else:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        # The engine's own exit code is what the sweep classifies on, so it is
        # captured and re-raised as this wrapper's exit code.
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        dump(out, engine)
    return code


if __name__ == "__main__":
    sys.exit(main())
