# Recommender results

MovieLens 100K, leave-last-out. Test users: **942**. Seed 42.

| Model | P@10 | R@10 | NDCG@10 |
|---|---|---|---|
| Popularity | 0.0058 | 0.0584 | 0.0289 |
| MF (dim 64, 20 epochs) | 0.0092 | 0.0924 | 0.0428 |

MF beats popularity on NDCG@10.
