"""Download GLM-5.3-Flash-BF16 from Hugging Face at the pinned revision and verify every file.

    python -m models.demos.blackhole.glm53_flash.tools.download_checkpoint --out <dir>/GLM-5.3-Flash-BF16

then point GLM53_CHECKPOINT at that directory.

The revision's listing (path, size, git blob id, LFS SHA-256 per file) comes from the Hub API and is saved as
``<out>/.download/files.json``.  ``huggingface_hub`` fetches the files into ``<out>`` (resumable).  Every file is then
hashed and compared with the listing: the SHA-256 for LFS files, the git blob SHA-1 for the others.  A file that fails
is deleted so the next run fetches it again.  ``<out>/.download/verified.json`` records each verified file's size and
mtime, so a later run hashes only what changed.  ``--verify-only`` fetches nothing.  Needs ``huggingface_hub`` only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from models.demos.blackhole.glm53_flash.checkpoint import HF_REPO, PINNED_REVISION


def say(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}", flush=True)


def fetch_listing(revision: str) -> list[dict]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(HF_REPO, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise SystemExit(f"the Hub resolved {revision} to {info.sha}")
    return [
        {
            "path": s.rfilename,
            "size": s.size,
            "blob_id": s.blob_id,
            "sha256": s.lfs.sha256 if s.lfs else None,
        }
        for s in info.siblings
    ]


def digest(path: Path, entry: dict) -> str:
    """The file's digest in the listing's terms: SHA-256 for an LFS file, the git blob SHA-1 otherwise."""
    if entry["sha256"]:
        h = hashlib.sha256()
    else:
        h = hashlib.sha1()
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(out: Path, files: list[dict], threads: int) -> list[str]:
    """Hash every file not already verified at its current size and mtime; return the paths that fail."""
    record_path = out / ".download" / "verified.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}

    def check(entry: dict) -> tuple[str, str | None]:
        path = out / entry["path"]
        if not path.exists():
            return entry["path"], "missing"
        stat = path.stat()
        if stat.st_size != entry["size"]:
            return entry["path"], f"size {stat.st_size} != {entry['size']}"
        seen = record.get(entry["path"])
        expected = entry["sha256"] or entry["blob_id"]
        if seen and seen == [stat.st_size, stat.st_mtime_ns, expected]:
            return entry["path"], None
        got = digest(path, entry)
        if got != expected:
            return entry["path"], f"digest {got} != {expected}"
        record[entry["path"]] = [stat.st_size, stat.st_mtime_ns, expected]
        return entry["path"], None

    failed = []
    t0 = time.time()
    with ThreadPoolExecutor(threads) as pool:
        for n, (name, problem) in enumerate(pool.map(check, files), 1):
            if problem:
                failed.append(name)
                say(f"FAIL {name}: {problem}")
            if n % 10 == 0 or n == len(files):
                say(f"verified {n - len(failed)}/{len(files)} ({time.time() - t0:.0f}s)")
    record_path.write_text(json.dumps(record, indent=1, sort_keys=True))
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument("--workers", type=int, default=16, help="parallel downloads")
    parser.add_argument("--hash-threads", type=int, default=16)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    out = args.out.resolve()
    (out / ".download").mkdir(parents=True, exist_ok=True)
    files = fetch_listing(args.revision)
    total = sum(f["size"] for f in files)
    (out / ".download" / "files.json").write_text(
        json.dumps({"repo": HF_REPO, "revision": args.revision, "files": files}, indent=1)
    )
    say(f"{HF_REPO}@{args.revision}: {len(files)} files, {total:,} bytes -> {out}")

    if not args.verify_only:
        from huggingface_hub import snapshot_download

        t0 = time.time()
        snapshot_download(HF_REPO, revision=args.revision, local_dir=out, max_workers=args.workers)
        say(f"fetched in {time.time() - t0:.0f}s")

    failed = verify(out, files, args.hash_threads)
    if failed:
        if not args.verify_only:
            for name in failed:
                (out / name).unlink(missing_ok=True)
            say(f"{len(failed)} files failed and were deleted; run again to fetch them")
        return 1
    say(f"all {len(files)} files verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
