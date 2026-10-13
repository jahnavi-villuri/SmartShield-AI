"""Train / validation / test splitting.

We use a *grouped, stratified* split:
  - stratified: every split keeps (about) the same spam ratio
  - grouped:    rows that share a group (here: near-duplicate templates) always land in the SAME
                split, so a spam template seen in training can't reappear in the test set
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def grouped_stratified_split(
    df: pd.DataFrame,
    label_col: str,
    group_col: str,
    seed: int,
    n_folds: int = 20,
    val_folds: int = 3,
    test_folds: int = 3,
) -> pd.Series:
    """Assign each row to "train", "val" or "test" (default about 70 / 15 / 15).

    Idea: cut the data into `n_folds` equal, stratified, group-respecting chunks, then hand
    `test_folds` chunks to test, `val_folds` to validation and the rest to training.
    """
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of_row = np.full(len(df), -1)
    for fold, (_, held_out) in enumerate(splitter.split(df, df[label_col], groups=df[group_col])):
        fold_of_row[held_out] = fold

    split = np.where(
        fold_of_row < test_folds,
        "test",
        np.where(fold_of_row < test_folds + val_folds, "val", "train"),
    )
    return pd.Series(split, index=df.index, name="split")


def check_no_overlap(splits: dict[str, pd.DataFrame], column: str) -> dict[str, int]:
    """Count values of `column` shared by each pair of splits. All counts must be 0."""
    names = list(splits)
    overlaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlaps[f"{a} & {b}"] = len(set(splits[a][column]) & set(splits[b][column]))
    return overlaps
