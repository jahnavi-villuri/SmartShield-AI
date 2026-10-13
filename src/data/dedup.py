"""Duplicate handling for labeled text data.

Levels covered here (see docs/phases/phase-02-readme.md):
    1/2  exact + normalized duplicates   -> keep one copy
    3    conflicting duplicates          -> drop every copy (label is unreliable)
    5    cross-dataset duplicates        -> same function, applied after concatenating datasets
Level 4 (near-duplicate templates) is NOT deleted; it is handled by group splitting.
"""

import pandas as pd


def deduplicate(
    df: pd.DataFrame, key_col: str = "key", label_col: str = "label"
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return (deduplicated frame, counts of what was removed).

    Order of operations matters:
      1. find keys that appear with more than one label  -> remove ALL their rows
      2. among the rest keep the first row per key
    The `source` column of a kept row lists every dataset the message appeared in,
    e.g. "mendeley+uci".
    """
    rows_in = len(df)

    labels_per_key = df.groupby(key_col)[label_col].nunique()
    conflicting_keys = labels_per_key[labels_per_key > 1].index
    is_conflict = df[key_col].isin(conflicting_keys)
    conflict_rows = int(is_conflict.sum())
    remaining = df[~is_conflict]

    if "source" in remaining.columns:
        all_sources = remaining.groupby(key_col)["source"].agg(lambda s: "+".join(sorted(set(s))))
    else:
        all_sources = None

    kept = remaining.drop_duplicates(subset=key_col, keep="first").copy()
    if all_sources is not None:
        kept["source"] = kept[key_col].map(all_sources)

    counts = {
        "rows_in": rows_in,
        "conflicting_keys": int(len(conflicting_keys)),
        "conflicting_rows_removed": conflict_rows,
        "duplicate_rows_removed": int(len(remaining) - len(kept)),
        "rows_out": int(len(kept)),
    }
    return kept.reset_index(drop=True), counts
