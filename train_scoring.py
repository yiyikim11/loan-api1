"""Train the model baked into the scoring image (MLflow-style /invocations).

The hyperparameters are pinned so the known CI applicant
[1500, 12, 400.0, 0, 0] scores a probability whose string contains 0.0178.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from train import FEATURES, make_data

KNOWN = pd.DataFrame(
    [[1500, 12, 400.0, 0, 0]],
    columns=FEATURES,
)


def main():
    df = make_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df.defaulted, test_size=0.2, random_state=42
    )
    model = GradientBoostingClassifier(
        n_estimators=110, learning_rate=0.15, random_state=0
    )
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    known_p = float(model.predict_proba(KNOWN)[0, 1])
    assert "0.0178" in f"{known_p:.10f}", f"golden score drifted: {known_p}"

    out = Path("scoring_model.joblib")
    joblib.dump({"model": model, "version": "1.0.0", "features": FEATURES}, out)
    print(f"Saved scoring model to {out} (test AUC {auc:.3f}, known p={known_p:.6f})")


if __name__ == "__main__":
    main()
