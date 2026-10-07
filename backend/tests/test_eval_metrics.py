"""Metric logic with SYNTHETIC labels (test data only, not an evaluation)."""

import pytest

from mediaindex.eval_metrics import aggregate, hit_at_k, ndcg_at_k, recall_at_k


def test_hit_handles_unknowns():
    ranked = ["a", "b", "c", "d", "e", "f"]
    assert hit_at_k(ranked, {"c": 2}, 5) == 1
    assert hit_at_k(ranked, {"a": 0, "b": 0}, 5) is None  # c,d,e unknown -> undecidable
    assert hit_at_k(ranked, {x: 0 for x in "abcde"}, 5) == 0
    assert hit_at_k(ranked, {"a": 1}, 1) == 1 and hit_at_k(ranked, {"a": 1}, 1, threshold=2) == 0


def test_recall_and_ndcg_need_complete_pools():
    ranked = ["a", "b", "c", "d"]
    pool = ["a", "b", "c", "d"]
    labels = {"a": 0, "b": 2, "c": 0, "d": 1}
    assert recall_at_k(ranked, labels, pool, 2) == pytest.approx(0.5)
    assert recall_at_k(ranked, {"a": 0}, pool, 2) is None
    assert recall_at_k(ranked, {x: 0 for x in pool}, pool, 2) is None
    perfect = ndcg_at_k(["b", "d", "a", "c"], labels, pool, 4)
    assert perfect == pytest.approx(1.0)
    assert 0 < ndcg_at_k(ranked, labels, pool, 4) < 1
    assert ndcg_at_k(ranked, {"a": 1}, pool, 4) is None


def test_aggregate_reports_coverage():
    assert aggregate([1, 0, None, 1]) == {"value": pytest.approx(0.6667), "n": 3, "excluded": 1}
    assert aggregate([None, None]) == {"value": None, "n": 0, "excluded": 2}
