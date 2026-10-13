"""One command to see (or run) the whole pipeline in the right order.

    python -m scripts.rebuild_all                  # print the plan only (safe default)
    python -m scripts.rebuild_all --run            # run every step (hours; needs the raw data)
    python -m scripts.rebuild_all --run --only sms # run one module's steps
    python -m scripts.rebuild_all --check          # only check that raw data and tools are present

Raw datasets are downloaded by hand (licences differ; see README section 5). A step that fails
stops the run: later steps depend on earlier outputs.
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass

from config import settings


@dataclass(frozen=True)
class Step:
    module: str
    command: tuple[str, ...]  # arguments after `python -m`
    minutes: int
    needs: tuple[str, ...] = ()  # folders under data/ that must exist and be non-empty


STEPS = [
    Step(
        "sms",
        ("scripts.prepare_sms",),
        3,
        ("raw/sms_uci", "raw/sms_mendeley", "raw/sms_scamshield"),
    ),
    Step("sms", ("src.training.train_sms",), 15),
    Step("sms", ("src.training.train_sms_type",), 1, ("labels",)),
    Step("email", ("scripts.prepare_email",), 10, ("raw/email_enron", "raw/email_untroubled")),
    Step("email", ("src.training.train_email",), 20),
    Step("comments", ("scripts.prepare_toxicity",), 10, ("raw/toxicity_jigsaw",)),
    Step("comments", ("src.training.train_toxicity", "--augment"), 15),
    Step("url", ("scripts.fetch_phreshphish_urls",), 2),
    Step("url", ("scripts.prepare_phishing",), 25, ("raw/phishing_phiusiil",)),
    Step("url", ("scripts.split_phiusiil",), 2),
    Step("url", ("src.training.train_phishing", "--with-phiusiil"), 18),
    Step("all", ("scripts.benchmark_models",), 1),
    Step("all", ("scripts.make_model_cards",), 1),
]


def select(only: str | None) -> list[Step]:
    if only is None:
        return list(STEPS)
    chosen = [s for s in STEPS if s.module == only]
    if not chosen:
        raise SystemExit(
            f"unknown module {only!r}; choose from {sorted({s.module for s in STEPS})}"
        )
    return chosen


def missing_inputs(steps: list[Step]) -> list[str]:
    missing = []
    for step in steps:
        for need in step.needs:
            folder = settings.DATA_DIR / need
            if not folder.exists() or not any(folder.iterdir()):
                missing.append(f"{need} (needed by {' '.join(step.command)})")
    return sorted(set(missing))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--run", action="store_true", help="execute the steps")
    parser.add_argument("--check", action="store_true", help="only check inputs")
    parser.add_argument("--only", choices=sorted({s.module for s in STEPS}))
    args = parser.parse_args()

    steps = select(args.only)
    total = sum(s.minutes for s in steps)
    for i, step in enumerate(steps, 1):
        print(
            f"{i:2d}. [{step.module:8s}] python -m {' '.join(step.command)}  (~{step.minutes} min)"
        )
    print(f"estimated total: about {total} minutes")

    missing = missing_inputs(steps)
    if missing:
        print("\nMissing inputs (download them first):")
        for line in missing:
            print("  -", line)
    if args.check or not args.run:
        return 1 if missing and args.check else 0
    if missing:
        print("\nRefusing to run with missing inputs.")
        return 1
    for step in steps:
        print(f"\n=== python -m {' '.join(step.command)} ===", flush=True)
        if subprocess.run([sys.executable, "-m", *step.command]).returncode != 0:
            print("step failed; stopping.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
