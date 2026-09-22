#!/usr/bin/env python3
"""
Run one prediction under a VRAM trace and write what it cost to results.json.

    python profiling/run_with_trace.py --trace-dir DIR --engine af3|of3 [...] -- CMD ...

Ported from pooled_ppi_comparison/scripts/run_with_trace.py, which is the
version the earlier Protenix/OF3 scaling curves were measured with, and extended to
this repository's two engines. The measurement itself is unchanged: nvidia-smi is
polled once a second and the high-water mark is the reported peak.

WHAT peak_vram_mib IS. `nvidia-smi --query-gpu=memory.used` — WHOLE-DEVICE memory,
not this process's. On a job holding the card that is the number of interest (it
includes the CUDA context and the allocator's cached-but-free blocks, which is what
actually has to fit), but it also counts any other tenant of the same GPU, so a
sweep is only comparable if it ran with the device to itself. gpu_total_mib is
recorded alongside so the ceiling is on the record.

WHAT status IS. Exit code alone is not enough: OpenFold3 exits 0 having failed every
query, and both engines print an OOM before dying, so the log is read too.
  ok        ran, and (for of3, when --summary is given) reported no failed queries
  gpu_oom   an out-of-memory from the device — THE boundary this sweep looks for
  cpu_oom   killed by the host OOM killer, which is not a GPU limit: exit 137, or
            exit -9 when the sweep's cgroup oom_kill counter moved (sweep.sbatch
            upgrades `killed` to this after reading the counter)
  killed    SIGKILL (Popen returncode -9) with no OOM evidence yet -- the cgroup
            check in sweep.sbatch decides; on its own it is neither ok nor a GPU limit
  failed    finished without an OOM but did not produce a result
  error     any other non-zero exit

WHAT peak_alloc_mib IS. The framework's own live-buffer high-water, written by
profiling/peak_mem.py and merged in here from --peak-mem-json. It is the number the
sweep plots: nvidia-smi cannot see inside AlphaFold3's preallocated arena, and this
can. Missing (null) when the run was killed before the counter could be read.

WHAT model_time_s IS. The engine's own timer, the same definition stage 04 uses so
the two are comparable:
  af3   "Featurising data ... took X seconds" + "Running model inference ... took Y"
  of3   runtime_s from each seed's timing.json, summed (--timing-glob)
It excludes interpreter start, container start and parameter loading; wall_time_s
includes them.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time

# AF3's aggregate timers, copied from scripts/04_collect_profiling.py so a sweep row
# and a benchmark row mean the same thing by construction. The seed clause wording
# varies between container versions, so both spellings are accepted.
_SEED_CLAUSE = r"(?:for seeds \([^)]*\)|with \d+ seed\(s\))"
FEATURISE_RE = re.compile(rf"Featurising data {_SEED_CLAUSE} took\s+([\d.]+) seconds")
INFERENCE_RE = re.compile(
    rf"Running model inference and extracting output structures {_SEED_CLAUSE} "
    rf"took\s+([\d.]+) seconds")

# Every out-of-memory spelling the two stacks produce: JAX/XLA raises
# RESOURCE_EXHAUSTED, PyTorch raises torch.OutOfMemoryError / "CUDA out of memory".
OOM_PATTERNS = (
    "CUDA out of memory",
    "OutOfMemoryError",
    "RESOURCE_EXHAUSTED",
    "Out of memory while trying to allocate",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "ran out of memory trying to allocate",
    # OpenFold3's runner catches the device OOM itself and reports it in this form,
    # then exits 0 — without this pattern its boundary reads as a plain failure.
    "OOM for query_id",
)

POLL_INTERVAL_S = 1.0


def nvidia_query(field: str) -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={field}",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def poll_vram(trace_path: str, stop_event: threading.Event) -> None:
    with open(trace_path, "w") as fh:
        while not stop_event.is_set():
            used = nvidia_query("memory.used")
            if used is not None:
                fh.write(f"{int(time.time())} {used}\n")
                fh.flush()
            time.sleep(POLL_INTERVAL_S)


def parse_peak(trace_path: str) -> int:
    peak = 0
    try:
        with open(trace_path) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peak = max(peak, int(parts[1]))
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return peak


def parse_model_time(engine: str, log_path: str, timing_glob: str) -> float | None:
    if engine == "of3":
        values = []
        for path in glob.glob(timing_glob) if timing_glob else []:
            try:
                value = json.load(open(path)).get("runtime_s")
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, (int, float)):
                values.append(float(value))
        return round(sum(values), 3) if values else None
    try:
        with open(log_path, errors="replace") as fh:
            text = fh.read()
    except FileNotFoundError:
        return None
    feat = FEATURISE_RE.search(text)
    infer = INFERENCE_RE.search(text)
    if feat is None and infer is None:
        return None
    return round((float(feat.group(1)) if feat else 0.0)
                 + (float(infer.group(1)) if infer else 0.0), 3)


KIT_PARTIAL_RE = re.compile(r"DONE status=partial .*predictions=(\d+)/\1\b.*levers_short")
KIT_PEAK_RE = re.compile(r"PEAK-NOTE scope=pass status=ok peak_alloc_gib=([\d.]+)(?: kind=(\S+))?")


def parse_kit_peak(log_path: str) -> dict:
    """peak_alloc_mib from the af3_jax kit's PEAK-NOTE line (GiB -> MiB), or {} when absent."""
    try:
        with open(log_path, errors="replace") as fh:
            text = fh.read()
    except FileNotFoundError:
        return {}
    m = KIT_PEAK_RE.search(text)
    if m is None:
        return {}
    return {"peak_alloc_mib": round(float(m.group(1)) * 1024, 1), "peak_alloc_source": "kit PEAK-NOTE"}


def classify(log_path: str, exit_code: int, summary_path: str) -> str:
    text = ""
    try:
        with open(log_path, errors="replace") as fh:
            text = fh.read()
    except FileNotFoundError:
        pass
    # The log is consulted BEFORE the exit code: an OOM that the runner caught and
    # reported still exits 0, and calling that a success would move the boundary.
    if any(pattern in text for pattern in OOM_PATTERNS):
        return "gpu_oom"
    if exit_code == 137:
        return "cpu_oom"
    if exit_code == -9:
        # Popen reports a signal death as -signal; the OOM killer sends SIGKILL, and so
        # does SLURM at a limit. Which one it was is read from the cgroup by the caller.
        return "killed"
    if summary_path and os.path.isfile(summary_path):
        try:
            summary = open(summary_path, errors="replace").read()
        except OSError:
            summary = ""
        if re.search(r"Failed Queries:\s+[1-9]", summary):
            return "failed"
    if exit_code == 3 and KIT_PARTIAL_RE.search(text):
        # Anthropic's af3_jax kit: every prediction written, but a lever of the mode engaged at no
        # call site for this input (its DONE line: status=partial ... levers_short=<names>). The
        # kit signals that with exit 3 by design; for a sweep it is a successful, measured run.
        return "ok"
    if exit_code != 0:
        return "error"
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--trace-dir", required=True,
                    help="Where vram_trace.log, inference.log and results.json go")
    ap.add_argument("--engine", choices=("af3", "of3"), required=True)
    ap.add_argument("--timing-glob", default="",
                    help="of3 only: glob for the per-seed timing.json files")
    ap.add_argument("--summary", default="",
                    help="of3 only: summary.txt to check for failed queries")
    ap.add_argument("--peak-mem-json", default="",
                    help="JSON written by profiling/peak_mem.py, merged into results")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        print("run_with_trace.py: no command given", file=sys.stderr)
        return 1

    os.makedirs(args.trace_dir, exist_ok=True)
    trace_log = os.path.join(args.trace_dir, "vram_trace.log")
    infer_log = os.path.join(args.trace_dir, "inference.log")

    gpu_total = nvidia_query("memory.total")
    stop_event = threading.Event()
    poller = threading.Thread(target=poll_vram, args=(trace_log, stop_event),
                              daemon=True)
    poller.start()

    start = time.time()
    exit_code = 0
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        with open(infer_log, "w") as log_fh:
            for line in proc.stdout:
                sys.stdout.write(line)
                log_fh.write(line)
        proc.wait()
        exit_code = proc.returncode
    except Exception as exc:                       # noqa: BLE001 — recorded, not raised
        print(f"run_with_trace.py: failed to run command: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        stop_event.set()
        poller.join(timeout=3)

    status = classify(infer_log, exit_code, args.summary)
    peak_mem = {}
    if args.peak_mem_json and os.path.isfile(args.peak_mem_json):
        try:
            peak_mem = json.load(open(args.peak_mem_json))
        except (OSError, json.JSONDecodeError):
            peak_mem = {}
    if peak_mem.get("peak_alloc_mib") is None:
        # Anthropic's af3_jax kit runs the model in a child process peak_mem.py cannot wrap, and
        # prints the same counter itself (jax peak_bytes_in_use, its own sitecustomize hook):
        #   [af3-jax-opt] PEAK-NOTE scope=pass status=ok peak_alloc_gib=3.97 kind=jax_peak_bytes_in_use ...
        peak_mem = {**peak_mem, **parse_kit_peak(infer_log)}
    results = {
        "status": status,
        "peak_alloc_mib": peak_mem.get("peak_alloc_mib"),
        # "peak_mem.py" (the wrapper read the framework's counter) or "kit PEAK-NOTE" (the kit's own line)
        "peak_alloc_source": peak_mem.get("peak_alloc_source", "peak_mem.py" if peak_mem.get("peak_alloc_mib") is not None else None),
        "peak_reserved_mib": peak_mem.get("peak_reserved_mib"),
        "peak_vram_mib": parse_peak(trace_log),
        "gpu_total_mib": gpu_total,
        "wall_time_s": int(time.time() - start),
        # Only for a run that produced a result: a failed run still prints timers.
        "model_time_s": (parse_model_time(args.engine, infer_log, args.timing_glob)
                         if status == "ok" else None),
        "exit_code": exit_code,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    with open(os.path.join(args.trace_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[trace] {json.dumps(results)}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
