#!/usr/bin/env python3
"""
Stage 00 — download the Zheng & Peng Benchmark-IV multimer set from Zenodo.

Source of truth
---------------
Zheng, Peng et al. (2025), "A Comprehensive Benchmarking of AlphaFold3 for
Predicting Biomacromolecules and Their Interactions", Briefings in Bioinformatics
(bbaf616). The paper's Data Availability section points at

    Zenodo record 15502855   (DOI 10.5281/zenodo.15502855)

whose `multimer.tar.gz` (~1.2 GB) IS the Benchmark-IV structure set: one directory
per PDB entry containing

    seq.fasta         the query sequences the authors folded, one record per chain,
                      headers of the form >{pdb}-assembly1_{chain}
    cleanNative.pdb   the ground-truth structure DockQ is scored against
    alphaFold2/, alphaFold3/, ...   the authors' own predictions (not used here)

Supplementary Table S4 is the human-readable index of the same 206 entries. We use
the archive rather than the table so that nothing about the dataset is transcribed
by hand — the entry list, the chain composition and the sequences are all derived
from files this script downloads.

This script only fetches and unpacks. Turning the archive into the dataset contract
(entries.tsv / sequences.tsv / native/) is stage 01.

Usage:
    python scripts/00_fetch_benchmark.py --out-dir <base>/benchmark_iv
    python scripts/00_fetch_benchmark.py --out-dir <base>/benchmark_iv --archive /path/to/multimer.tar.gz
"""

import argparse
import hashlib
import os
import sys
import tarfile
from pathlib import Path

# On ETH Euler the conda/pixi certifi bundle does not include the site proxy cert,
# so HTTPS fails with CERTIFICATE_VERIFY_FAILED. Point at the system CA bundle.
_SYSTEM_CA = "/etc/ssl/certs/ca-certificates.crt"
if os.path.exists(_SYSTEM_CA):
    os.environ.setdefault("SSL_CERT_FILE", _SYSTEM_CA)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _SYSTEM_CA)

import requests

ZENODO_RECORD = "15502855"
ZENODO_API = "https://zenodo.org/api/records/{record}"
ARCHIVE_NAME = "multimer.tar.gz"

# What a complete extraction looks like. Checked after unpacking so a truncated
# download or a partial extract is an error here rather than a confusing gap in
# stage 01. The count comes from the paper (206 multimers) and is verified against
# the archive's own contents, not assumed.
EXPECTED_PER_ENTRY = ("seq.fasta", "cleanNative.pdb")


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def resolve_archive_url(record: str) -> tuple[str, str, int]:
    """Return (download_url, expected_md5, size_bytes) for multimer.tar.gz."""
    resp = requests.get(ZENODO_API.format(record=record), timeout=60)
    resp.raise_for_status()
    meta = resp.json()
    for entry in meta.get("files", []):
        if entry.get("key") == ARCHIVE_NAME:
            checksum = entry.get("checksum", "")
            # Zenodo reports "md5:<hex>"
            md5 = checksum.split(":", 1)[1] if ":" in checksum else checksum
            return entry["links"]["self"], md5, int(entry.get("size", 0))
    available = ", ".join(f.get("key", "?") for f in meta.get("files", []))
    raise SystemExit(
        f"{ARCHIVE_NAME} not found in Zenodo record {record}. Files present: {available}"
    )


def download(url: str, dest: Path, size_hint: int = 0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = 0
    with requests.get(url, stream=True, timeout=(30, 300)) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in resp.iter_content(chunk_size=1 << 20):
                fh.write(block)
                done += len(block)
                if size_hint:
                    pct = 100.0 * done / size_hint
                    print(f"\r  {done / 1e9:.2f} / {size_hint / 1e9:.2f} GB ({pct:.1f}%)",
                          end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    tmp.rename(dest)


def extract(archive: Path, out_dir: Path) -> Path:
    """Unpack multimer/ into out_dir; return the multimer/ directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        # `filter="data"` (Python 3.12+) refuses absolute paths, ".." escapes,
        # symlinks and device nodes — the archive is third-party, so extract it
        # under the safe filter rather than trusting its member names.
        tar.extractall(path=out_dir, filter="data")
    root = out_dir / "multimer"
    if not root.is_dir():
        raise SystemExit(f"expected {root} after extraction; archive layout changed?")
    return root


def verify(root: Path) -> dict:
    """Report per-entry completeness. Returns a summary dict; does not raise."""
    entries = sorted(d.name for d in root.iterdir() if d.is_dir())
    incomplete = {}
    for eid in entries:
        missing = [f for f in EXPECTED_PER_ENTRY if not (root / eid / f).is_file()]
        if missing:
            incomplete[eid] = missing
    return {"n_entries": len(entries), "incomplete": incomplete}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Where to place multimer.tar.gz and the extracted multimer/ tree")
    ap.add_argument("--archive", type=Path,
                    help="Use this local multimer.tar.gz instead of downloading "
                         "(still verified and extracted)")
    ap.add_argument("--record", default=ZENODO_RECORD,
                    help=f"Zenodo record id (default {ZENODO_RECORD})")
    ap.add_argument("--skip-checksum", action="store_true",
                    help="Skip the md5 check (it reads the whole 1.2 GB archive)")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    archive = args.archive or (out_dir / ARCHIVE_NAME)

    if args.archive:
        if not archive.is_file():
            raise SystemExit(f"--archive not found: {archive}")
        print(f"Using local archive {archive}")
        expected_md5 = None
    elif archive.is_file():
        print(f"Archive already present, not re-downloading: {archive}")
        _, expected_md5, _ = resolve_archive_url(args.record)
    else:
        url, expected_md5, size = resolve_archive_url(args.record)
        print(f"Zenodo record {args.record}: {ARCHIVE_NAME} ({size / 1e9:.2f} GB)")
        print(f"  {url}")
        download(url, archive, size)

    if expected_md5 and not args.skip_checksum:
        print("Verifying md5 ...", end=" ", flush=True)
        got = md5sum(archive)
        if got != expected_md5:
            raise SystemExit(f"\nmd5 mismatch: got {got}, Zenodo says {expected_md5}")
        print("ok")

    print(f"Extracting into {out_dir} ...")
    root = extract(archive, out_dir)

    summary = verify(root)
    print(f"Entries extracted: {summary['n_entries']}")
    if summary["incomplete"]:
        print(f"WARNING: {len(summary['incomplete'])} entries are missing expected files:")
        for eid, missing in sorted(summary["incomplete"].items())[:10]:
            print(f"  {eid}: missing {', '.join(missing)}")
        if len(summary["incomplete"]) > 10:
            print(f"  ... and {len(summary['incomplete']) - 10} more")
    print(f"\nBenchmark tree: {root}")
    print("Next: scripts/01_build_dataset.py --multimer-dir "
          f"{root} --out-dir <dataset dir>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
