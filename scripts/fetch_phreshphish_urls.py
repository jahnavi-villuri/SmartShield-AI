"""Download ONLY the URL-level columns of PhreshPhish (no HTML).

Run:  python -m scripts.fetch_phreshphish_urls [--limit N]

PhreshPhish (CC BY 4.0, "for anti-phishing research") has 77 parquet files of about 500 MB each,
mostly the HTML of phishing pages. We read the columns url, label, target (impersonated brand),
date and lang using HTTP range requests, so only a few megabytes per file are transferred.
The html column is never requested. URLs are stored as plain text and are never visited.

Robust to a flaky connection: every finished file is saved in `parts/`, a failed file is retried
with a pause, and re-running the script skips the files that are already done.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow.parquet as pq
import requests

from config import settings
from src.data.range_file import RangeFile

LISTING = "https://datasets-server.huggingface.co/parquet?dataset=phreshphish/phreshphish"
OUT_DIR = settings.DATA_RAW / "phishing_phreshphish"
PARTS_DIR = OUT_DIR / "parts"
COLUMNS = ["sha256", "url", "label", "target", "date", "lang"]
ATTEMPTS = 6


def part_path(entry: dict):
    return PARTS_DIR / f"{entry['split']}_{entry['filename']}"


def read_one(entry: dict) -> tuple[int, int, bool]:
    """Fetch one file's columns (with retries) and save them. Returns (requests, bytes, skipped)."""
    target = part_path(entry)
    if target.exists():
        return 0, 0, True
    last_error = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            remote = RangeFile(entry["url"])
            frame = pq.ParquetFile(remote).read(columns=COLUMNS).to_pandas()
            frame["split"] = entry["split"]
            frame.to_parquet(target, index=False)
            return remote.requests_made, remote.bytes_fetched, False
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            time.sleep(2 * attempt)
    raise RuntimeError(
        f"{entry['split']}/{entry['filename']} failed after {ATTEMPTS} attempts"
    ) from last_error


def main(limit: int | None) -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    files = requests.get(LISTING, timeout=120).json()["parquet_files"]
    if limit:
        files = files[:limit]
    print(f"{len(files)} parquet files; reading columns {COLUMNS} only", flush=True)
    started = time.perf_counter()
    total_bytes = skipped = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(read_one, entry) for entry in files]
        for done, future in enumerate(as_completed(futures), start=1):
            _, n_bytes, was_skipped = future.result()
            total_bytes += n_bytes
            skipped += was_skipped
            if done % 5 == 0 or done == len(files):
                print(
                    f"  {done}/{len(files)} files | {total_bytes / 1e6:.0f} MB fetched this run | "
                    f"{skipped} already done | {time.perf_counter() - started:.0f}s",
                    flush=True,
                )
    data = pd.concat([pd.read_parquet(part_path(e)) for e in files], ignore_index=True)
    for split, part in data.groupby("split"):
        part.drop(columns=["split"]).to_parquet(OUT_DIR / f"urls_{split}.parquet", index=False)
        print(f"saved {split}: {len(part):,} rows", flush=True)
    (OUT_DIR / "fetch_summary.json").write_text(
        json.dumps({"files": len(files), "rows": len(data)}), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only the first N files (testing)")
    main(parser.parse_args().limit)
