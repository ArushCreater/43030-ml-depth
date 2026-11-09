# Classifier results

Stratified 80/20 split, random_state 42: 28,000 train rows, 7,000 validation rows.

| Model | Validation AUC |
|---|---|
| LightGBM | 0.8612 |
| MLP | 0.8407 |

The MLP ran for 22 epochs and kept the weights with the best validation AUC.
