"""MovieLens 100K: positives, leave-last-out split and ID mapping."""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RATINGS_PATH = REPO_ROOT / "data" / "ml-100k" / "u.data"
COLUMNS = ["user", "item", "rating", "timestamp"]


def load_ratings(path=RATINGS_PATH):
    return pd.read_csv(path, sep="\t", names=COLUMNS)


def filter_positives(ratings):
    """Keep ratings >= 4, one row per (user, item), and users with at least 2 positives."""
    positives = ratings[ratings["rating"] >= 4]
    positives = positives.sort_values(["user", "timestamp", "item"])
    positives = positives.drop_duplicates(["user", "item"], keep="last")
    counts = positives.groupby("user")["item"].transform("size")
    return positives[counts >= 2].reset_index(drop=True)


def leave_last_out(positives):
    """Each user's latest positive goes to test. Timestamp ties break on the item ID."""
    ordered = positives.sort_values(["user", "timestamp", "item"]).reset_index(drop=True)
    is_test = ~ordered.duplicated("user", keep="last")
    return ordered[~is_test].reset_index(drop=True), ordered[is_test].reset_index(drop=True)


def encode_ids(train, test):
    """Map user and item IDs to 0-based indices shared by both splits."""
    both = pd.concat([train, test])
    user_ids = {u: i for i, u in enumerate(sorted(both["user"].unique()))}
    item_ids = {v: i for i, v in enumerate(sorted(both["item"].unique()))}
    encoded = []
    for frame in (train, test):
        frame = frame.copy()
        frame["user"] = frame["user"].map(user_ids)
        frame["item"] = frame["item"].map(item_ids)
        encoded.append(frame)
    return encoded[0], encoded[1], len(user_ids), len(item_ids)


def load_split(path=RATINGS_PATH):
    train, test = leave_last_out(filter_positives(load_ratings(path)))
    return encode_ids(train, test)
