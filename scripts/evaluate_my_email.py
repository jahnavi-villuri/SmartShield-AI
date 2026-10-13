"""Honest report of how the current email model does on the owner's private emails.

Run:  python -m scripts.evaluate_my_email

Emails are split by a hash of subject + body (src/data/private_email.py). Only the "test" half is a
fair test: the "train" half was repeated inside the training data, so good results there prove
nothing. A legitimate email flagged as spam is the costly mistake and is counted separately.
"""

from src.data.private_email import load_private_email
from src.detection.email_detector import load_email_detector


def main() -> None:
    private = load_private_email()
    if private is None or private.empty:
        raise SystemExit(
            "No private emails yet. Add some to data/private/my_email/emails.csv "
            "(see the README.txt in that folder)."
        )
    detector = load_email_detector()
    results = [
        detector.predict(r["subject"], r["body"], explain=False) for _, r in private.iterrows()
    ]
    private["flagged"] = [r.is_spam for r in results]
    private["score"] = [r.score for r in results]
    for half, title in (
        ("test", "HELD-OUT half (fair test)"),
        ("train", "TRAINING half (seen, not a test)"),
    ):
        part = private[private["split"] == half]
        if part.empty:
            continue
        legit = part[part["label"] == 0]
        spam = part[part["label"] == 1]
        print(f"\n== {title}: {len(part)} emails ==")
        print(
            f"legitimate or promotional flagged as spam: {int(legit['flagged'].sum())}/{len(legit)}"
        )
        if len(spam):
            print(f"spam caught: {int(spam['flagged'].sum())}/{len(spam)}")
        for _, row in part.iterrows():
            wrong = row["flagged"] != bool(row["label"])
            mark = "WRONG" if wrong else "ok   "
            print(
                f"  [{mark}] expected={row['category']:11s} flagged={str(row['flagged']):5s} "
                f"score={row['score']:.2f}  {row['subject'][:60]!r}"
            )


if __name__ == "__main__":
    main()
