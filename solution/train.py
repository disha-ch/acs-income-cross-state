"""Train and validate an ACSIncome model."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from src.pipeline import (
    _load_split,
    _scores,
    build_model,
    load_model,
    predict_proba,
    save_model,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    args = parser.parse_args()

    X_train, y_train = _load_split(Path(args.train))
    X_val, y_val = _load_split(Path(args.val))

    raw_model = build_model()
    raw_model.fit(X_train, y_train)

    calibrated_model = CalibratedClassifierCV(
        estimator=build_model(),
        method="sigmoid",
        cv=3,
    )
    calibrated_model.fit(X_train, y_train)

    raw_probabilities = predict_proba(raw_model, X_val)
    calibrated_probabilities = predict_proba(calibrated_model, X_val)

    raw_scores = _scores(y_val, raw_probabilities)
    calibrated_scores = _scores(y_val, calibrated_probabilities)

    if (
        calibrated_scores["ece_top_label_10bin"]
        < raw_scores["ece_top_label_10bin"]
        and calibrated_scores["accuracy"] >= raw_scores["accuracy"]
    ):
        model = calibrated_model
        probabilities = calibrated_probabilities
        scores = calibrated_scores
        selected = "calibrated"
    else:
        model = raw_model
        probabilities = raw_probabilities
        scores = raw_scores
        selected = "raw"

    save_model(model, args.model_out)

    loaded_model = load_model(args.model_out)
    loaded_probabilities = predict_proba(loaded_model, X_val)
    np.testing.assert_allclose(
        probabilities, loaded_probabilities, rtol=0, atol=1e-7
    )

    metrics = {
        "accuracy": scores["accuracy"],
        "auroc": scores["auroc"],
        "ece_top_label_10bin": scores["ece_top_label_10bin"],
        "scope": "visible_validation",
    }

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print("Selected model:", selected)
    print("Saved model:", args.model_out)
    print("Saved validation metrics:", args.metrics_out)
    print(metrics)


if __name__ == "__main__":
    main()