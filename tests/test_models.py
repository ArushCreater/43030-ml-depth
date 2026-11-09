import pandas as pd
import torch

from src.classifier.model import MLP
from src.recommender.baseline import PopularityBaseline
from src.recommender.model import MF


def test_mlp_output_shape():
    assert MLP(10)(torch.randn(4, 10)).shape == (4, 1)


def test_mf_score_shape():
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([0, 1, 2])
    assert MF(5, 7)(users, items).shape == (3,)


def test_popularity_never_recommends_seen_items():
    train = pd.DataFrame({"user": [0, 1, 2, 0, 1, 2], "item": [0, 0, 0, 1, 1, 2]})
    model = PopularityBaseline(n_items=5).fit(train)
    seen = {0, 2}
    recs = model.recommend(seen, k=3)
    assert recs == [1, 3, 4]
    assert not seen & set(recs)
