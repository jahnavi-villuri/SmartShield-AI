"""Honest report of how the current SMS models do on the owner's private messages.

Run:  python -m scripts.evaluate_my_sms

Messages are split by a hash of their text (src/data/private_sms.py). Only the "test" half is a
fair test of the trained model; the "train" half was repeated inside the training data, so good
results there prove nothing. Both halves are printed, clearly separated.

The expectations are the three categories the owner uses:
  legitimate -> should be "Not spam"
  promotional -> should be "Promotional" (marketing is NOT spam, but is not an ordinary message)
  scam -> should be "Scam"
A "Scam" label on a legitimate or promotional message is the worst mistake, so it is counted
separately.
"""

import pandas as pd

from src.data.private_sms import load_private_sms
from src.detection.spam_detector import load_sms_detector

EXPECTED_LABEL = {"legitimate": "Not spam", "promotional": "Promotional", "scam": "Scam"}


def main() -> None:
    private = load_private_sms()
    if private is None or private.empty:
        raise SystemExit("No private messages found (data/private/my_sms/messages.csv).")
    detector = load_sms_detector()
    results = []
    for _, row in private.iterrows():
        result = detector.predict(row["text"], explain=False)
        results.append((result.label, result.score, result.scam_score))
    private["got"] = [r[0] for r in results]
    private["unwanted_score"] = [r[1] for r in results]
    private["scam_score"] = [r[2] for r in results]
    private["expected"] = private["category"].map(EXPECTED_LABEL)
    private["correct"] = private["got"] == private["expected"]
    private["false_scam"] = (private["got"] == "Scam") & (private["category"] != "scam")

    for half, title in (
        ("test", "HELD-OUT half (fair test)"),
        ("train", "TRAINING half (seen, not a test)"),
    ):
        part = private[private["split"] == half]
        if part.empty:
            continue
        legit = part[part["category"] == "legitimate"]
        flagged = int((legit["got"] != "Not spam").sum())
        print(f"\n== {title}: {len(part)} messages ==")
        print(f"exactly right: {int(part['correct'].sum())}/{len(part)}")
        print(f"legitimate messages flagged as promotional or scam: {flagged}/{len(legit)}")
        print(f"false 'Scam' labels: {int(part['false_scam'].sum())}/{len(part)}")
        for _, row in part.iterrows():
            mark = "ok " if row["correct"] else ("SCAM-ALARM" if row["false_scam"] else "miss")
            snippet = " ".join(row["text"].split())[:60]
            print(
                f"  [{mark:10s}] expected={row['category']:11s} got={row['got']:11s} "
                f"score={row['unwanted_score']:.2f}  {snippet}"
            )
    pd.set_option("display.width", 200)


if __name__ == "__main__":
    main()
