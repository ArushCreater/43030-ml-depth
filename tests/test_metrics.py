import pytest

from src.recommender.metrics import ndcg_at_k, precision_at_k, recall_at_k


def test_precision_and_recall_two_hits():
    recs = [1, 2, 3, 4, 5]
    assert precision_at_k(recs, {2, 5}, 5) == pytest.approx(0.4)
    assert recall_at_k(recs, {2, 5}, 5) == pytest.approx(1.0)


def test_recall_with_a_relevant_item_missing():
    assert recall_at_k([1, 2, 3, 4, 5], {2, 9}, 5) == pytest.approx(0.5)


def test_ndcg_first_position():
    assert ndcg_at_k([1, 2, 3], {1}, 3) == pytest.approx(1.0)


def test_ndcg_third_position():
    assert ndcg_at_k([1, 2, 3], {3}, 3) == pytest.approx(0.5)


def test_ndcg_two_relevant():
    assert ndcg_at_k([1, 2, 3], {1, 3}, 3) == pytest.approx(0.9197, abs=1e-4)


def test_relevant_item_outside_top_k_scores_zero():
    recs = [1, 2, 3, 4, 5]
    assert precision_at_k(recs, {5}, 3) == 0.0
    assert recall_at_k(recs, {5}, 3) == 0.0
    assert ndcg_at_k(recs, {5}, 3) == 0.0
