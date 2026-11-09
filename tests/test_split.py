import pandas as pd
import pytest

from src.recommender.data import filter_positives, leave_last_out


@pytest.fixture
def positives():
    ratings = pd.DataFrame(
        [
            (1, 10, 5, 100),
            (1, 11, 4, 300),
            (1, 12, 5, 200),
            (1, 13, 2, 900),  # latest for user 1, but not a positive
            (2, 10, 4, 50),
            (2, 14, 5, 40),
            (3, 11, 5, 10),   # only one positive, so user 3 is dropped
            (4, 10, 4, 70),
            (4, 11, 4, 20),
            (4, 10, 5, 90),   # repeat of (4, 10)
            (4, 12, 4, 80),
        ],
        columns=["user", "item", "rating", "timestamp"],
    )
    return filter_positives(ratings)


def test_no_pair_in_both_train_and_test(positives):
    train, test = leave_last_out(positives)
    train_pairs = set(zip(train["user"], train["item"]))
    test_pairs = set(zip(test["user"], test["item"]))
    assert not train_pairs & test_pairs


def test_each_user_has_exactly_one_test_row(positives):
    _, test = leave_last_out(positives)
    assert sorted(test["user"]) == [1, 2, 4]
    assert test["user"].is_unique


def test_test_row_is_the_latest_positive(positives):
    _, test = leave_last_out(positives)
    latest = positives.groupby("user")["timestamp"].max()
    for row in test.itertuples():
        assert row.timestamp == latest[row.user]
    assert dict(zip(test["user"], test["item"])) == {1: 11, 2: 10, 4: 10}
