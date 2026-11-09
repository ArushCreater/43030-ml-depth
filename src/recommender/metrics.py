"""Top-k ranking metrics."""
import math


def _hits(recs, relevant, k):
    return sum(1 for item in recs[:k] if item in relevant)


def precision_at_k(recs, relevant, k):
    return _hits(recs, relevant, k) / k


def recall_at_k(recs, relevant, k):
    if not relevant:
        return 0.0
    return _hits(recs, relevant, k) / len(relevant)


def ndcg_at_k(recs, relevant, k):
    dcg = sum(1 / math.log2(rank + 1)
              for rank, item in enumerate(recs[:k], start=1) if item in relevant)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg else 0.0
