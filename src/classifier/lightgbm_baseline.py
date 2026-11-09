"""Re-run the 31250 LightGBM model on the shared train/validation split."""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.classifier.data import CSV_PATH, RANDOM_STATE, REPO_ROOT, build_preprocessor, load_split

PREDICTIONS_PATH = REPO_ROOT / "data" / "lightgbm_val_predictions.npz"


def run(csv_path=CSV_PATH, predictions_path=PREDICTIONS_PATH):
    X_train, X_val, y_train, y_val = load_split(csv_path)
    # Default hyperparameters: this was the best model by test AUC in 31250.
    pipe = Pipeline([
        ("prep", build_preprocessor(scale=False)),
        ("clf", LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
    ])
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probs)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(predictions_path, index=y_val.index.to_numpy(), y_true=y_val.to_numpy(),
             y_prob=probs, auc=auc)
    print(f"LightGBM validation AUC: {auc:.4f}")
    return auc


if __name__ == "__main__":
    run()
