"""Train MF, evaluate it against popularity, and write the results table."""
import argparse

import numpy as np
import torch

from src.recommender.baseline import PopularityBaseline
from src.recommender.data import REPO_ROOT, load_split
from src.recommender.metrics import ndcg_at_k, precision_at_k, recall_at_k
from src.recommender.model import EPOCHS, SEED, train_mf

RESULTS_PATH = REPO_ROOT / "results" / "recommender_results.md"
K = 10


def recommend_mf(model, user, seen, n_items, k=K):
    model.eval()
    with torch.no_grad():
        scores = model(torch.full((n_items,), user), torch.arange(n_items))
    scores[list(seen)] = -np.inf
    return scores.topk(k).indices.tolist()


def average_metrics(recommend, test, train_items, k=K):
    """recommend(user, seen) gives one user's ranked list; returns mean P, R and NDCG."""
    scores = []
    for user, item in zip(test["user"], test["item"]):
        recs = recommend(int(user), train_items[user])
        relevant = {int(item)}
        scores.append((precision_at_k(recs, relevant, k), recall_at_k(recs, relevant, k),
                       ndcg_at_k(recs, relevant, k)))
    return tuple(float(x) for x in np.mean(scores, axis=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    train, test, n_users, n_items = load_split()
    train_items = train.groupby("user")["item"].apply(set).to_dict()

    popularity = PopularityBaseline(n_items).fit(train)
    model = train_mf(train, n_users, n_items, dim=args.dim, epochs=args.epochs)

    pop = average_metrics(lambda u, seen: popularity.recommend(seen, K), test, train_items)
    mf = average_metrics(lambda u, seen: recommend_mf(model, u, seen, n_items), test, train_items)

    rows = [("Popularity", pop), (f"MF (dim {args.dim}, {args.epochs} epochs)", mf)]
    table = "\n".join(f"| {name} | {p:.4f} | {r:.4f} | {n:.4f} |" for name, (p, r, n) in rows)
    verdict = "beats" if mf[2] > pop[2] else "does not beat"
    report = (
        "# Recommender results\n\n"
        f"MovieLens 100K, leave-last-out. Test users: **{len(test)}**. "
        f"Seed {SEED}.\n\n"
        "| Model | P@10 | R@10 | NDCG@10 |\n"
        "|---|---|---|---|\n"
        f"{table}\n\n"
        f"MF {verdict} popularity on NDCG@10.\n"
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
