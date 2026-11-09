"""Train the MLP and compare it with LightGBM on the same validation split."""
import copy
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn

from src.classifier.data import CSV_PATH, RANDOM_STATE, REPO_ROOT, build_preprocessor, load_split
from src.classifier.lightgbm_baseline import PREDICTIONS_PATH
from src.classifier.model import MLP

RESULTS_DIR = REPO_ROOT / "results"
LR = 1e-3
BATCH_SIZE = 512
MAX_EPOCHS = 50
PATIENCE = 5


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X)).squeeze(1).numpy()


def train_mlp(X_train, y_train, X_val, y_val):
    """Returns the best model by validation AUC and the number of epochs run."""
    set_seed()
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    model = MLP(X_train.shape[1])
    positives = y_train.sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=(len(y_train) - positives) / positives)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_auc, best_state, stale_epochs, epoch = -1.0, None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        order = torch.randperm(len(X_train))
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(X_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        auc = roc_auc_score(y_val, predict_proba(model, X_val))
        print(f"epoch {epoch:2d}  loss {total_loss / len(order):.4f}  val AUC {auc:.4f}")
        if auc > best_auc:
            best_auc, best_state, stale_epochs = auc, copy.deepcopy(model.state_dict()), 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"early stop: no improvement for {PATIENCE} epochs")
                break

    model.load_state_dict(best_state)
    return model, epoch


def plot_roc(y_val, lgbm_prob, lgbm_auc, mlp_prob, mlp_auc, path):
    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=surface)
    ax.set_facecolor(surface)
    ax.plot([0, 1], [0, 1], color=muted, linewidth=1, linestyle=(0, (4, 4)), label="Chance")
    fpr, tpr, _ = roc_curve(y_val, lgbm_prob)
    ax.plot(fpr, tpr, color="#2a78d6", linewidth=2, label=f"LightGBM (AUC {lgbm_auc:.4f})")
    fpr, tpr, _ = roc_curve(y_val, mlp_prob)
    ax.plot(fpr, tpr, color="#eb6834", linewidth=2, label=f"MLP (AUC {mlp_auc:.4f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.005), aspect="equal")
    ax.set_xlabel("False positive rate", color=muted)
    ax.set_ylabel("True positive rate", color=muted)
    ax.set_title("ROC curves on the validation split", color=ink, loc="left")
    ax.grid(color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=muted, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    legend = ax.legend(loc="lower right", frameon=False)
    for text in legend.get_texts():
        text.set_color(ink)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=surface)
    plt.close(fig)


def main(csv_path=CSV_PATH, predictions_path=PREDICTIONS_PATH, results_dir=RESULTS_DIR):
    if not predictions_path.exists():
        raise SystemExit("Run python -m src.classifier.lightgbm_baseline first")
    X_train, X_val, y_train, y_val = load_split(csv_path)
    lgbm = np.load(predictions_path)
    if not np.array_equal(lgbm["index"], y_val.index.to_numpy()):
        raise SystemExit("LightGBM predictions were made on a different split; re-run the baseline")

    prep = build_preprocessor(scale=True)
    X_train_t = prep.fit_transform(X_train)
    X_val_t = prep.transform(X_val)
    y_val = y_val.to_numpy()

    model, epochs = train_mlp(X_train_t, y_train.to_numpy(), X_val_t, y_val)
    mlp_prob = predict_proba(model, torch.tensor(X_val_t, dtype=torch.float32))
    mlp_auc = roc_auc_score(y_val, mlp_prob)
    lgbm_auc = float(lgbm["auc"])

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "classifier_results.md").write_text(
        "# Classifier results\n\n"
        f"Stratified 80/20 split, random_state {RANDOM_STATE}: {len(y_train):,} train rows, "
        f"{len(y_val):,} validation rows.\n\n"
        "| Model | Validation AUC |\n"
        "|---|---|\n"
        f"| LightGBM | {lgbm_auc:.4f} |\n"
        f"| MLP | {mlp_auc:.4f} |\n\n"
        f"The MLP ran for {epochs} epochs and kept the weights with the best validation AUC.\n",
        encoding="utf-8",
    )
    plot_roc(y_val, lgbm["y_prob"], lgbm_auc, mlp_prob, mlp_auc, results_dir / "roc_curve.png")
    print(f"LightGBM AUC {lgbm_auc:.4f} | MLP AUC {mlp_auc:.4f}")


if __name__ == "__main__":
    main()
