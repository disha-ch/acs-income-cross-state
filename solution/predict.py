"""Load a trained ACSIncome model and generate prediction probabilities.

Reads input features from an NPZ file, checks feature order, and saves
positive-class probabilities to a NumPy file without retraining.
"""
import argparse
import numpy as np
from scripts.package_data import main
from src.pipeline import FEATURES, load_model, predict_proba

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)    # NPZ with X and feature_names
    parser.add_argument("--output", required=True)   # NPY probability vector
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as data:
        X = data["X"]
        names = data["feature_names"].astype(str).tolist()

    if names != FEATURES:
        raise ValueError("Input feature order does not match")

    model = load_model(args.model)
    np.save(args.output, predict_proba(model, X))


if __name__ == "__main__":
    main()