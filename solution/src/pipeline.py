"""Reference pipeline for ACS Income: Cross-State, Post-2020 Shift With Worst-State and Calibration Bars.

This module contains the oracle solution. solve.sh copies it into /app/src/
and run.py calls run_full() to produce all required artifacts.

The agent is expected to produce code that generates the same artifacts
(as defined in instruction.md). checks.py validates those artifacts.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any
import time
import threading

import joblib
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(os.environ.get(
    "OTTER_APP_DIR",
    "/app" if Path("/app").is_dir() else ROOT / "workdir",
))
ARTIFACTS = APP_DIR / "artifacts"
MODELS = APP_DIR / "models"
DEVICE = os.environ.get("OTTER_DEVICE", "cuda")

FEATURES = [
    "AGEP", "COW", "SCHL", "MAR", "OCCP",
    "POBP", "RELP", "WKHP", "SEX", "RAC1P",
]
CATEGORICAL = ["COW", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]
EXPECTED_ROWS = {"train": 434_610, "val": 98_925}
EXPECTED_STATES = {"train": {"CA", "TX", "NY"}, "val": {"FL"}}

def _resolve_data_path(path: str | Path | None = None) -> Path:
    """Find the training archive."""
    paths = [
        Path(path) if path else None,
        Path(os.environ["OTTER_DATA_PATH"])
        if os.environ.get("OTTER_DATA_PATH") else None,
        APP_DIR / "data" / "train.npz",
        ROOT / "environment" / "task_inputs" / "train.npz",
    ]
    for item in paths:
        if item is not None and item.is_file():
            return item
    raise FileNotFoundError("Could not find train.npz")


def _resolve_val_path() -> Path:
    """Find the Florida validation archive."""
    paths = [
        Path(os.environ["OTTER_VAL_PATH"])
        if os.environ.get("OTTER_VAL_PATH") else None,
        APP_DIR / "data" / "val.npz",
        ROOT / "environment" / "task_inputs" / "val.npz",
    ]
    for item in paths:
        if item is not None and item.is_file():
            return item
    raise FileNotFoundError("Could not find val.npz")


def _load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an archive and validate features and labels."""
    with np.load(path, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]
        names = data["feature_names"].astype(str).tolist()

    if X.ndim != 2 or X.shape[1] != len(FEATURES):
        raise ValueError(f"Expected 10 features in {path}")
    if names != FEATURES:
        raise ValueError(f"Wrong feature names or order in {path}")
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"X and y do not align in {path}")
    if not np.isin(y, [0, 1, False, True]).all():
        raise ValueError(f"Labels must be binary in {path}")
    if not np.isfinite(X).all():
        raise ValueError(f"X contains invalid values in {path}")

    return X.astype(np.float32), y.astype(np.int8)

# -------------------------------------------------------------------------
# Milestone 1: Data loading and preprocessing
# -------------------------------------------------------------------------

def load_and_preprocess(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Load raw data, preprocess, and write artifacts.

    Args:
        data_path: Path to input data. Falls back to OTTER_DATA_PATH env var.
        output_dir: Where to write artifacts. Falls back to APP_DIR/artifacts.
        seed: Random seed for reproducibility.

    Returns:
        Dict summarizing what was produced, e.g.:
        {"n_samples": 500, "splits": {"train": 350, "valid": 75, "test": 75}}
    """
    del seed

    X_train, y_train = _load_split(_resolve_data_path(data_path))
    X_val, y_val = _load_split(_resolve_val_path())

    out = Path(output_dir) if output_dir else ARTIFACTS
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "processed_data.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    return {
        "train_rows": len(y_train),
        "validation_rows": len(y_val),
        "feature_names": FEATURES,
    }

def build_model(seed: int = 42) -> Pipeline:
    """One-hot encode nominal features and retain ordered SCHL values."""
    category_columns = [FEATURES.index(name) for name in CATEGORICAL]
    numeric_columns = [
        i for i, name in enumerate(FEATURES) if name not in CATEGORICAL
    ]

    preprocess = ColumnTransformer([
        (
            "categories",
            OneHotEncoder(handle_unknown="ignore", dtype=np.float32),
            category_columns,
        ),
        ("numeric", "passthrough", numeric_columns),
    ])

    classifier = xgb.XGBClassifier(
        tree_method="hist",
        device=DEVICE,
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=8,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def compute_ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Calculate equal-width top-label ECE."""
    y, p = np.asarray(y), np.asarray(p)

    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bins must be a positive integer")
    if y.ndim != 1 or p.ndim != 1 or len(y) == 0 or len(y) != len(p):
        raise ValueError("y and p must be non-empty aligned vectors")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")

    predicted = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1 - p)
    correct = (predicted == y).astype(float)
    bin_ids = np.minimum((confidence * bins).astype(int), bins - 1)

    ece = 0.0
    for i in range(bins):
        rows = bin_ids == i
        if rows.any():
            ece += rows.mean() * abs(
                correct[rows].mean() - confidence[rows].mean()
            )
    return float(ece)


def _scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Calculate validation metrics at a fixed 0.5 threshold."""
    return {
        "auroc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "ece_top_label_10bin": compute_ece(y, p),
    }


def predict_proba(
    model,
    X: np.ndarray,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """Return one validated positive-class probability per row."""
    X = np.asarray(X)
    if X.ndim != 2 or len(X) == 0 or X.shape[1] != len(FEATURES):
        raise ValueError("X must be a non-empty matrix with 10 columns")
    if feature_names is not None and feature_names != FEATURES:
        raise ValueError("Feature names or order differ from training")
    if not np.isfinite(X).all():
        raise ValueError("X contains invalid values")

    p = np.asarray(model.predict_proba(X)[:, 1])
    if p.shape != (len(X),) or not np.isfinite(p).all():
        raise ValueError("Model returned invalid probabilities")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be in [0, 1]")
    return p


def save_model(model, path: str | Path) -> Path:
    """Save a fitted model and its feature order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, path)
    return path


def load_model(path: str | Path):
    """Load a model and check its feature order."""
    saved = joblib.load(path)
    if saved["features"] != FEATURES:
        raise ValueError("Saved model uses a different feature order")
    return saved["model"]


def _fitted_xgb_models(model):
    """Find fitted XGBoost models inside pipelines and calibrators."""
    if isinstance(model, CalibratedClassifierCV):
        for fold in model.calibrated_classifiers_:
            yield from _fitted_xgb_models(fold.estimator)
    elif isinstance(model, Pipeline):
        for step in model.named_steps.values():
            yield from _fitted_xgb_models(step)
    elif isinstance(model, xgb.XGBClassifier):
        yield model


def _device_values(obj):
    """Find device settings in an XGBoost configuration."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() == "device" and isinstance(value, str):
                yield value
            yield from _device_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _device_values(value)


def _watch_gpu(stop: threading.Event, process_seen: threading.Event) -> None:
    """Look for this Python process in nvidia-smi while models fit."""
    while not stop.is_set():
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            pids = {line.strip() for line in result.stdout.splitlines()}
            if str(os.getpid()) in pids:
                process_seen.set()
        stop.wait(0.25)


def _check_gpu(*models, process_seen: bool) -> dict[str, Any]:
    """Check CUDA support, visible hardware, process use, and fitted boosters."""
    if DEVICE != "cuda":
        return {"verified": False, "device": DEVICE}

    if not xgb.build_info().get("USE_CUDA"):
        raise RuntimeError("XGBoost was built without CUDA support")

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    gpu_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not gpu_names:
        raise RuntimeError("No NVIDIA GPU is visible")
    if not process_seen:
        raise RuntimeError("Training process was not seen using the GPU")

    expected_gpu = os.environ.get("OTTER_EXPECTED_GPU")
    if expected_gpu and not any(
        expected_gpu.lower() in name.lower() for name in gpu_names
    ):
        raise RuntimeError(f"Expected {expected_gpu}; found {gpu_names}")

    count = 0
    for model in models:
        for estimator in _fitted_xgb_models(model):
            config = json.loads(estimator.get_booster().save_config())
            if not any(value.startswith("cuda") for value in _device_values(config)):
                raise RuntimeError("A fitted booster did not record CUDA")
            count += 1

    if count == 0:
        raise RuntimeError("No fitted XGBoost models were checked")

    return {
        "verified": True,
        "gpu_names": gpu_names,
        "boosters_checked": count,
        "training_process_seen_by_nvidia_smi": True,
    }


# -------------------------------------------------------------------------
# Milestone 2: Model training and evaluation
# -------------------------------------------------------------------------

def train_and_evaluate(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Train model, evaluate, and write artifacts.

    Args:
        data_path: Path to input data.
        output_dir: Where to write artifacts and model checkpoints.
        seed: Random seed for reproducibility.

    Returns:
        Dict with metrics, e.g.:
        {"auroc": 0.85, "accuracy": 0.82, "f1": 0.80}
    """
    out = Path(output_dir) if output_dir else ARTIFACTS
    processed = out / "processed_data.npz"

    if not processed.is_file():
        load_and_preprocess(data_path, out, seed)

    with np.load(processed, allow_pickle=False) as data:
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]
        names = (
            data["feature_names"].astype(str).tolist()
            if "feature_names" in data
            else FEATURES
        )

    raw_model = build_model(seed)
    calibrated_model = CalibratedClassifierCV(
        estimator=build_model(seed),
        method="sigmoid",
        cv=3,
        ensemble=True,
    )

    stop = threading.Event()
    process_seen = threading.Event()
    watcher = None

    if DEVICE == "cuda":
        watcher = threading.Thread(
            target=_watch_gpu,
            args=(stop, process_seen),
            daemon=True,
        )
        watcher.start()

    start = time.perf_counter()
    try:
        raw_model.fit(X_train, y_train)
        calibrated_model.fit(X_train, y_train)
    finally:
        training_seconds = time.perf_counter() - start
        if watcher is not None:
            stop.set()
            watcher.join()

    raw_p = predict_proba(raw_model, X_val)
    calibrated_p = predict_proba(calibrated_model, X_val)

    raw_scores = _scores(y_val, raw_p)
    calibrated_scores = _scores(y_val, calibrated_p)

    if (
        calibrated_scores["ece_top_label_10bin"]
        < raw_scores["ece_top_label_10bin"]
        and calibrated_scores["accuracy"] >= raw_scores["accuracy"]
    ):
        selected_model = calibrated_model
        selected_p = calibrated_p
        selected_scores = calibrated_scores
        selected_name = "calibrated"
    else:
        selected_model = raw_model
        selected_p = raw_p
        selected_scores = raw_scores
        selected_name = "raw"

    gpu = _check_gpu(
        raw_model,
        calibrated_model,
        process_seen=process_seen.is_set(),
    )

    model_dir = MODELS if output_dir is None else out.parent / "models"
    model_paths = {
        "raw": model_dir / "acs_income_raw.joblib",
        "calibrated": model_dir / "acs_income_calibrated.joblib",
        "selected": model_dir / "trained_model.joblib",
    }

    for name, model, expected_p in [
        ("raw", raw_model, raw_p),
        ("calibrated", calibrated_model, calibrated_p),
        ("selected", selected_model, selected_p),
    ]:
        save_model(model, model_paths[name])
        restored_model = load_model(model_paths[name])
        restored_p = predict_proba(restored_model, X_val)
        np.testing.assert_allclose(
            expected_p, restored_p, rtol=0, atol=1e-7
        )

    metrics = {
        "accuracy": selected_scores["accuracy"],
        "auroc": selected_scores["auroc"],
        "ece_top_label_10bin": selected_scores["ece_top_label_10bin"],
        "scope": "visible_validation",
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out / "validation_diagnostics.json").write_text(
        json.dumps(
            {
                "raw": raw_scores,
                "calibrated": calibrated_scores,
                "selected_model": selected_name,
                "training_seconds": training_seconds,
                "gpu": gpu,
                "checkpoint_check": "validation probabilities match after reload",
            },
            indent=2,
        )
        + "\n"
    )

    return {
        "metrics": metrics,
        "validation_scores": {
            "raw": raw_scores,
            "calibrated": calibrated_scores,
        },
        "selected_model": selected_name,
        "selected_checkpoint": str(model_paths["selected"]),
        "training_seconds": training_seconds,
        "gpu": gpu,
    }


# -------------------------------------------------------------------------
# Full end-to-end pipeline
# -------------------------------------------------------------------------

def run_full(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the complete pipeline: load -> preprocess -> train -> evaluate -> report.

    When called with no arguments (from run.py), uses env vars and defaults.

    Returns:
        Dict with final results and metrics.
    """
    out = Path(output_dir) if output_dir else ARTIFACTS
    data = load_and_preprocess(data_path, out, seed)
    results = train_and_evaluate(data_path, out, seed)
    selected_name = results["selected_model"]

    claims = {
        "training_split": "Visible CA/TX/NY 2018 data",
        "validation_split": "Visible Florida 2018 data",
        "metrics_scope": "Validation only",
        "hidden_labels_loaded": False,
        "selected_model": selected_name,
        "selection_rule": (
            "Choose calibrated when validation top-label ECE is lower "
            "and validation accuracy is at least as high as raw; otherwise choose raw."
        ),
        "training_seconds_include_calibration": results["training_seconds"],
        "gpu_check": results["gpu"],
        "selected_checkpoint": results["selected_checkpoint"],
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "claims.json").write_text(json.dumps(claims, indent=2) + "\n")

    REPORTS.mkdir(parents=True, exist_ok=True)
    raw = results["validation_scores"]["raw"]
    calibrated = results["validation_scores"]["calibrated"]
    report = f"""# ACSIncome validation report

Training rows: {data['train_rows']:,}
Florida validation rows: {data['validation_rows']:,}

The selected model is the {selected_name} model. Selection used validation data only:
calibrated was chosen when its top-label ECE was lower and its accuracy was at
least as high as raw; otherwise raw was chosen.

Raw validation AUROC: {raw['auroc']:.4f}
Raw validation accuracy: {raw['accuracy']:.4f}
Raw validation top-label ECE: {raw['ece_top_label_10bin']:.4f}

Calibrated validation AUROC: {calibrated['auroc']:.4f}
Calibrated validation accuracy: {calibrated['accuracy']:.4f}
Calibrated validation top-label ECE: {calibrated['ece_top_label_10bin']:.4f}

Total fit time, including calibration: {results['training_seconds']:.1f} seconds.
GPU check: {results['gpu']}

These are validation results. The verifier must calculate hidden metrics
independently from verifier-owned labels.
"""
    (REPORTS / "report.md").write_text(report)

    final = {"data": data, "results": results}
    (ARTIFACTS / "final_results.json").write_text(
        json.dumps(final, indent=2) + "\n"
    )
    return final
