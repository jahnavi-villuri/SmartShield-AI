"""Reusable dataset inspection (the "10 questions" from the Phase 2 README)."""

import pandas as pd


def inspect_text_frame(
    df: pd.DataFrame, name: str, text_col: str = "text", label_col: str = "label"
) -> dict:
    """Return plain facts about a labeled text dataset. Nothing is modified."""
    text = df[text_col]
    lengths = text.str.len()
    class_counts = df[label_col].value_counts().sort_index()
    return {
        "name": name,
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_per_column": {c: int(n) for c, n in df.isna().sum().items()},
        "empty_text": int((text.str.strip() == "").sum()),
        "class_counts": {int(k): int(v) for k, v in class_counts.items()},
        "bad_class_share": round(float((df[label_col] == 1).mean()), 4),
        "length_min": int(lengths.min()),
        "length_median": float(lengths.median()),
        "length_max": int(lengths.max()),
        "replacement_char_rows": int(text.str.contains("�", regex=False).sum()),
        "html_entity_rows": int(text.str.contains(r"&(?:lt|gt|amp|quot);", regex=True).sum()),
        "exact_duplicate_rows": int(df.duplicated([text_col, label_col]).sum()),
    }


def print_inspection(facts: dict, sample: pd.DataFrame | None = None, n: int = 3) -> None:
    """Print the facts, plus a few random rows per class so you can READ real examples."""
    print(f"\n=== {facts['name']} ===")
    for key, value in facts.items():
        if key != "name":
            print(f"{key:>24}: {value}")
    if sample is not None:
        for label, group in sample.groupby("label"):
            print(f"\n-- {n} random examples with label={label}")
            for text in group["text"].sample(min(n, len(group)), random_state=0):
                print("   *", text[:120])
