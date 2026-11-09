"""Matrix factorisation with an item bias, trained with BPR."""
import numpy as np
import torch
from torch import nn
from torch.nn.functional import softplus

SEED = 42
LR = 5e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 1024
EPOCHS = 20


class MF(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users, items):
        dot = (self.user_embedding(users) * self.item_embedding(items)).sum(dim=1)
        return dot + self.item_bias(items).squeeze(1)


def bpr_loss(positive_scores, negative_scores):
    # softplus(-x) is -log(sigmoid(x)), written so it stays numerically stable.
    return softplus(negative_scores - positive_scores).mean()


def sample_negatives(users, train_items, n_items, rng):
    """One item per user that the user has no interaction with in train."""
    negatives = rng.integers(n_items, size=len(users))
    for i, user in enumerate(users):
        while negatives[i] in train_items[user]:
            negatives[i] = rng.integers(n_items)
    return negatives


def train_mf(train, n_users, n_items, dim=64, epochs=EPOCHS, seed=SEED):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_items = train.groupby("user")["item"].apply(set).to_dict()
    users = train["user"].to_numpy()
    items = train["item"].to_numpy()

    model = MF(n_users, n_items, dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(1, epochs + 1):
        model.train()
        negatives = sample_negatives(users, train_items, n_items, rng)
        order = rng.permutation(len(users))
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            batch_users = torch.as_tensor(users[batch])
            positive = model(batch_users, torch.as_tensor(items[batch]))
            negative = model(batch_users, torch.as_tensor(negatives[batch]))
            loss = bpr_loss(positive, negative)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        print(f"epoch {epoch:2d}  BPR loss {total_loss / len(order):.4f}")
    return model
