"""Popularity baseline: the most-liked items the user has not seen."""
import numpy as np


class PopularityBaseline:
    def __init__(self, n_items):
        self.n_items = n_items
        self.ranking = np.arange(n_items)

    def fit(self, train):
        counts = np.bincount(train["item"].to_numpy(), minlength=self.n_items)
        # Most positives first; equal counts break on the item index.
        self.ranking = np.lexsort((np.arange(self.n_items), -counts))
        return self

    def recommend(self, seen, k=10):
        return [int(item) for item in self.ranking if item not in seen][:k]
