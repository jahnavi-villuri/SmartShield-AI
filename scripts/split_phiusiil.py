"""Round 2: split the cleaned PhiUSIIL URLs into train / validation (grouped by domain).

Run (after `python -m scripts.prepare_phishing`):
    python -m scripts.split_phiusiil

Why PhiUSIIL joins training in Round 2: its legitimate side is bare homepages and PhreshPhish's
legitimate side is deep pages. Each alone teaches a structural shortcut ("no path means phishing"
or "a path means phishing"). Mixed, no URL shape identifies the class, so the model has to look at
the domain and address itself. PhishBench, CompPhish, OpenPhish and PhreshPhish's newer test split
stay completely outside training, so they measure generalisation across sources.

PhiUSIIL labels were already flipped to the project convention (1 = phishing) by prepare_phishing.
"""

import pandas as pd

from config import settings
from src.data.splitting import check_no_overlap, grouped_stratified_split

SOURCE = settings.DATA_EXTERNAL_TEST / "phishing" / "phiusiil.csv"
OUT_DIR = settings.DATA_PROCESSED / "phishing"
COLUMNS = ["url", "label", "group", "registered_domain", "hosting", "key"]


def main() -> None:
    frame = pd.read_csv(SOURCE, keep_default_na=False)
    phresh = {
        name: pd.read_csv(OUT_DIR / f"{name}.csv", keep_default_na=False)
        for name in ("train", "val")
    }

    # A domain group must live in ONE split across ALL sources. PhreshPhish was split first, so a
    # PhiUSIIL URL whose group already appears there follows it; only groups new to PhreshPhish are
    # split here. (Splitting each source on its own put 38% of validation rows in groups that
    # training also contained, e.g. wikipedia.org in both.)
    in_train = frame["group"].isin(set(phresh["train"]["group"]))
    in_val = frame["group"].isin(set(phresh["val"]["group"]))
    assert not (in_train & in_val).any()
    frame["split"] = ""
    frame.loc[in_train, "split"] = "train"
    frame.loc[in_val, "split"] = "val"
    new_groups = frame[~(in_train | in_val)]
    decided = grouped_stratified_split(
        new_groups, label_col="label", group_col="group", seed=settings.RANDOM_SEED,
        n_folds=20, val_folds=3, test_folds=0,
    )  # fmt: skip
    frame.loc[new_groups.index, "split"] = decided
    followed = int((in_train | in_val).sum())
    print(f"followed PhreshPhish: {followed:,} rows; split fresh: {len(new_groups):,} rows")

    parts = {
        name: frame[frame["split"] == name].reset_index(drop=True) for name in ("train", "val")
    }
    everything = {
        "train": pd.concat([phresh["train"], parts["train"]]),
        "val": pd.concat([phresh["val"], parts["val"]]),
    }
    assert not check_no_overlap(everything, "key")["train & val"]
    assert not check_no_overlap(everything, "group")["train & val"]
    for name, part in parts.items():
        part[COLUMNS].to_csv(OUT_DIR / f"phiusiil_{name}.csv", index=False, encoding="utf-8")
        print(f"phiusiil_{name}: {len(part):,} rows, phishing share {part['label'].mean():.3f}")


if __name__ == "__main__":
    main()
