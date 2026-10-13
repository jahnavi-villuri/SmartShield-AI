"""Measure what it costs to run each detector: load time, per-item latency and file size.

    python -m scripts.benchmark_models

Uses the hand-written challenge files in tests/data as inputs (no private or dataset text) and
writes reports/metrics/performance.csv. Latency is for one item at a time on this machine; run it
again on the deployment target before quoting numbers.
"""

import statistics
import time
from pathlib import Path

import pandas as pd

from config import settings
from src.detection.email_detector import load_email_detector
from src.detection.phishing_detector import load_phishing_detector
from src.detection.spam_detector import load_sms_detector
from src.detection.toxicity_detector import load_toxicity_detector

DATA = settings.PROJECT_ROOT / "tests" / "data"
OUT = settings.REPORTS_DIR / "metrics" / "performance.csv"
REPEATS = 20


def _inputs() -> dict[str, list[tuple]]:
    sms = pd.read_csv(DATA / "sms_challenge.csv", keep_default_na=False)["text"].tolist()
    mail = pd.read_csv(DATA / "email_challenge.csv", keep_default_na=False)
    comments = pd.read_csv(DATA / "toxicity_probe.csv", keep_default_na=False)["text"].tolist()
    urls = pd.read_csv(DATA / "url_challenge.csv", keep_default_na=False)["url"].tolist()
    return {
        "SMS": [(t,) for t in sms],
        "Email": list(zip(mail["subject"], mail["body"], strict=True)),
        "Comments": [(t,) for t in comments],
        "URL": [(u,) for u in urls],
    }


def _time_calls(predict, items: list[tuple]) -> list[float]:
    for item in items[:3]:  # warm-up
        predict(*item)
    samples = []
    for _ in range(REPEATS):
        for item in items:
            start = time.perf_counter()
            predict(*item)
            samples.append((time.perf_counter() - start) * 1000)
    return samples


def benchmark() -> pd.DataFrame:
    loaders = {
        "SMS": (load_sms_detector, "sms_spam"),
        "Email": (load_email_detector, "email_spam"),
        "Comments": (load_toxicity_detector, "toxicity"),
        "URL": (load_phishing_detector, "phishing"),
    }
    inputs = _inputs()
    rows = []
    for name, (loader, folder) in loaders.items():
        start = time.perf_counter()
        detector = loader()
        load_ms = (time.perf_counter() - start) * 1000
        samples = _time_calls(detector.predict, inputs[name])
        samples.sort()
        size_mb = (
            sum(f.stat().st_size for f in Path(settings.MODELS_DIR / folder).glob("*.joblib")) / 1e6
        )
        rows.append(
            {
                "detector": name,
                "model_files_mb": round(size_mb, 2),
                "load_seconds": round(load_ms / 1000, 2),
                "median_ms": round(statistics.median(samples), 2),
                "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 2),
                "items_per_second": round(1000 / statistics.mean(samples), 1),
                "samples": len(samples),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    table = benchmark()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(table.to_string(index=False))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
