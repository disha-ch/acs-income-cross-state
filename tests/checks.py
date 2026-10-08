"""Task-specific verification checks for [YOUR TASK NAME].

This is the ONLY scoring file you edit. Replace the placeholder checks
below with your task's actual logic. rubric_score.py loads this file and
calls evaluate(context) exactly once.

Scoring is computed by the two standardized harness scripts (do not
edit them), both invoked by tests/test.sh:
  rubric_score.py: Rubric Score — weighted sum of your criteria (0 to 1,
                   diagnostic), plus per-milestone scores
  pass_fail.py:    Binary Pass/Fail Grade — 1 if all critical milestones
                   pass AND all primary metrics meet their human-derived
                   thresholds, else 0

Criterion weights and each milestone's critical flag are defined in
milestones_and_rubrics.json — do NOT set them here. checks.py only
determines pass/fail per criterion and returns metric values.

Primary metric thresholds are handled by pass_fail.py using
human_scores.json. Return the agent's actual metric values under "metrics"
in your evaluate() response — pass_fail.py compares them against
pass_criteria in human_scores.json and generates the final pass/fail
milestone automatically from those keys. Do NOT implement threshold
comparisons here, and do NOT write checks for the final pass/fail
milestone — just read and return the metric values.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

APP_DIR = Path(os.environ.get("OTTER_APP_DIR", "/app"))
HIDDEN_DIR = Path(__file__).resolve().parent / "hidden_data"
FEATURES = [
    "AGEP", "COW", "SCHL", "MAR", "OCCP",
    "POBP", "RELP", "WKHP", "SEX", "RAC1P",
]


def _load_hidden_data():
    """Load and validate the verifier-only evaluation archives."""
    yearly = {}
    for year in (2018, 2021):
        path = HIDDEN_DIR / f"hidden_{year}.npz"
        if not path.is_file():
            raise AssertionError(f"Missing hidden archive for {year}")

        with np.load(path, allow_pickle=False) as data:
            X = data["X"]
            y = data["y"]
            state_ids = data["state_ids"].astype(str)
            names = data["feature_names"].astype(str).tolist()

        if X.ndim != 2 or X.shape[1] != len(FEATURES):
            raise AssertionError(f"Hidden {year} features have the wrong shape")
        if names != FEATURES:
            raise AssertionError(f"Hidden {year} feature order is incorrect")
        if y.ndim != 1 or state_ids.ndim != 1:
            raise AssertionError(f"Hidden {year} labels or states have wrong shape")
        if len(X) == 0 or len(X) != len(y) or len(X) != len(state_ids):
            raise AssertionError(f"Hidden {year} arrays are not row-aligned")
        if not np.isin(y, [0, 1, False, True]).all():
            raise AssertionError(f"Hidden {year} labels are not binary")
        if not np.isfinite(X).all():
            raise AssertionError(f"Hidden {year} features contain invalid values")

        yearly[year] = (X, y.astype(np.int8), state_ids, names)

    if set(yearly[2018][2].tolist()) != set(yearly[2021][2].tolist()):
        raise AssertionError("Hidden years do not cover the same states")
    if len(set(yearly[2018][2].tolist())) < 2:
        raise AssertionError("Worst-state scoring requires multiple states")

    return yearly


def _top_label_ece(y, probabilities, bins=10):
    """Calculate equal-width top-label ECE with confidence 1 in the last bin."""
    y = np.asarray(y)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= 0.5).astype(np.int8)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    correct = (predicted == y).astype(float)
    bin_ids = np.minimum((confidence * bins).astype(int), bins - 1)

    ece = 0.0
    for bin_id in range(bins):
        rows = bin_ids == bin_id
        if rows.any():
            ece += rows.mean() * abs(correct[rows].mean() - confidence[rows].mean())
    return float(ece)


def _hidden_metrics():
    """Run the agent predictor without passing hidden labels or state IDs."""
    yearly = _load_hidden_data()
    model_path = app_path("models", "trained_model.joblib")
    predict_path = app_path("predict.py")
    if not model_path.is_file():
        raise AssertionError(f"Missing selected checkpoint: {model_path}")
    if not predict_path.is_file():
        raise AssertionError(f"Missing inference program: {predict_path}")

    probabilities_by_year = {}
    with tempfile.TemporaryDirectory(prefix="acs_hidden_eval_") as temp_dir:
        temp_dir = Path(temp_dir).resolve()
        for year in (2018, 2021):
            X, _, _, names = yearly[year]
            input_path = temp_dir / f"prediction_{year}.npz"
            output_path = temp_dir / f"predictions_{year}.npy"

            np.savez_compressed(
                input_path,
                X=X,
                feature_names=np.asarray(names),
            )
            with np.load(input_path, allow_pickle=False) as prediction_input:
                if set(prediction_input.files) != {"X", "feature_names"}:
                    raise AssertionError("Prediction input contains extra fields")

            subprocess.run(
                [
                    sys.executable,
                    str(predict_path.resolve()),
                    "--model", str(model_path.resolve()),
                    "--input", str(input_path),
                    "--output", str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=900,
                cwd=temp_dir,
            )

            if not output_path.is_file():
                raise AssertionError(f"Predictor did not create {year} output")
            probabilities = np.load(output_path, allow_pickle=False)
            if probabilities.shape != (len(X),):
                raise AssertionError(f"{year} predictions have the wrong shape")
            if not np.isfinite(probabilities).all():
                raise AssertionError(f"{year} predictions are non-finite")
            if ((probabilities < 0) | (probabilities > 1)).any():
                raise AssertionError(f"{year} predictions are outside [0, 1]")
            probabilities_by_year[year] = probabilities

    yearly_scores = {}
    for year in (2018, 2021):
        _, y, state_ids, _ = yearly[year]
        probabilities = probabilities_by_year[year]
        predicted = (probabilities >= 0.5).astype(np.int8)
        state_accuracies = [
            float((predicted[state_ids == state] == y[state_ids == state]).mean())
            for state in np.unique(state_ids)
        ]
        yearly_scores[year] = {
            "accuracy": float((predicted == y).mean()),
            "worst_state_accuracy": min(state_accuracies),
            "ece_top_label_10bin": _top_label_ece(y, probabilities),
        }

    X = np.concatenate([yearly[year][0] for year in (2018, 2021)])
    y = np.concatenate([yearly[year][1] for year in (2018, 2021)])
    state_ids = np.concatenate([yearly[year][2] for year in (2018, 2021)])
    probabilities = np.concatenate(
        [probabilities_by_year[year] for year in (2018, 2021)]
    )
    predicted = (probabilities >= 0.5).astype(np.int8)
    state_accuracies = [
        float((predicted[state_ids == state] == y[state_ids == state]).mean())
        for state in np.unique(state_ids)
    ]

    metrics = {
        "hidden_accuracy": float((predicted == y).mean()),
        "worst_state_accuracy": min(state_accuracies),
        "hidden_ece_top_label_10bin": _top_label_ece(y, probabilities),
    }
    return metrics, yearly_scores


def app_path(*parts: str) -> Path:
    """Resolve a path relative to the agent's working directory."""
    return APP_DIR.joinpath(*parts)


def load_json(path: Path) -> dict:
    """Load and validate a JSON file."""
    if not path.exists():
        raise AssertionError(f"Missing JSON file: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"Invalid JSON in {path}: line {exc.lineno}, col {exc.colno}: "
            f"{exc.msg}"
        ) from exc


def require_keys(obj: dict, keys: set[str], label: str) -> None:
    """Assert that a dict contains all required keys."""
    if not isinstance(obj, dict):
        raise AssertionError(
            f"{label} must be a dict; got {type(obj).__name__}"
        )
    missing = sorted(keys - set(obj))
    if missing:
        raise AssertionError(
            f"{label} missing required keys {missing}; "
            f"found: {sorted(obj)}"
        )


# ---------------------------------------------------------------------------
# Milestone 1: [YOUR FIRST MILESTONE NAME]
#
# Example: "Data Acquisition and Preprocessing"
# ---------------------------------------------------------------------------

def check_data_deliverables() -> str:
    """Check that required data-stage files exist."""
    for path in (app_path("train.py"), app_path("predict.py")):
        assert path.exists(), f"Missing: {path}"
    return "Training and inference programs exist"


def check_data_schema() -> str:
    """Validate structure of data artifacts."""
    metadata = load_json(app_path("data", "feature_metadata.json"))
    assert metadata.get("features") == FEATURES, "Feature metadata order is incorrect"
    yearly = _load_hidden_data()
    assert yearly[2018][3] == yearly[2021][3] == FEATURES
    return "Feature names and order match the required contract"


# ---------------------------------------------------------------------------
# Milestone 2: [YOUR SECOND MILESTONE NAME]
#
# Example: "Model Training and Evaluation"
# ---------------------------------------------------------------------------

def check_model_deliverables() -> str:
    """Check that required model-stage files exist."""
    path = app_path("models", "trained_model.joblib")
    assert path.is_file(), f"Missing selected model: {path}"
    return "Selected model checkpoint exists"


def check_metrics_schema() -> str:
    """Validate metrics structure and value ranges."""
    metrics = load_json(app_path("artifacts", "metrics.json"))
    expected = {"accuracy", "auroc", "ece_top_label_10bin", "scope"}
    assert set(metrics) == expected, f"metrics.json keys must be {sorted(expected)}"
    assert metrics["scope"] == "visible_validation", "Metrics scope is incorrect"

    for key in ("accuracy", "auroc", "ece_top_label_10bin"):
        value = metrics[key]
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"{key} must be a number"
        )
        assert np.isfinite(value) and 0.0 <= value <= 1.0, (
            f"{key} must be finite and in [0, 1]"
        )
    return "Visible validation metric schema is valid"


def check_metrics_thresholds() -> str:
    """Sanity-check that metrics are above a trivial baseline.

    NOTE: This is NOT the primary metric gate. The harness (pass_fail.py)
    separately compares the agent's metrics against the human-calibrated
    thresholds in human_scores.json pass_criteria.

    This check is for catching obviously broken results — e.g. a model
    that performs at or below random chance. Set these thresholds well
    below the human bar. Example: if human AUROC is 0.85 and pass_criteria
    is 0.80, set this sanity check to 0.55 (just above random).
    """
    metrics = load_json(app_path("artifacts", "metrics.json"))
    assert metrics["auroc"] > 0.5, "Validation AUROC must beat random ranking"
    assert metrics["accuracy"] > 0.5, "Validation accuracy is below a basic baseline"
    return "Visible validation metrics pass basic sanity checks"


# ---------------------------------------------------------------------------
# Milestone 3: Deliverables and Report Quality
#
# NOTE: This is still an intermediate (deterministic) milestone. The true
# final "pass/fail" milestone compares primary metrics against human
# thresholds and is generated automatically by pass_fail.py from
# human_scores.json pass_criteria — do not implement it here and do not
# declare it in milestones_and_rubrics.json.
# ---------------------------------------------------------------------------

def check_final_deliverables() -> str:
    """Check that final output files exist."""
    for path in (
        app_path("artifacts", "claims.json"),
        app_path("reports", "report.md"),
    ):
        assert path.exists(), f"Missing: {path}"
    return "Claims and report exist"


def check_report_content() -> str:
    """Verify report discusses required topics with sufficient depth."""
    report = app_path("reports", "report.md").read_text().lower()
    assert len(report.split()) >= 80, "Report is too short to explain the work"
    assert any(word in report for word in ("xgboost", "model", "training")), (
        "Report should describe the method"
    )
    assert any(word in report for word in ("validation", "accuracy", "auroc")), (
        "Report should discuss visible results"
    )
    assert any(word in report for word in ("limitation", "hidden", "shift")), (
        "Report should discuss limitations"
    )
    return "Report covers method, visible results, and limitations"


# ---------------------------------------------------------------------------
# Criterion wrapper — do not edit this function
#
# Weights and milestone critical flags come from
# milestones_and_rubrics.json. This wrapper only captures id,
# milestone_id, and pass/fail + detail.
# ---------------------------------------------------------------------------

def _criterion(id: str, fn, milestone_id: str = "final") -> dict[str, Any]:
    """Run a check function, catch exceptions, return a criterion dict.

    id must match a criterion id in milestones_and_rubrics.json, where
    its weight is defined. milestone_id must match a milestone id in
    that same file.
    """
    try:
        detail = fn()
        return {
            "id": id, "passed": True,
            "detail": detail or "passed", "milestone_id": milestone_id,
        }
    except Exception:
        return {
            "id": id, "passed": False,
            "detail": traceback.format_exc(limit=6),
            "milestone_id": milestone_id,
        }


# ---------------------------------------------------------------------------
# Entry point — called by rubric_score.py
# ---------------------------------------------------------------------------

def evaluate(context: dict) -> dict:
    """Run all checks and return criteria plus verifier-computed hidden metrics.

    The harness computes the scores from these criteria:
      rubric_score.py: per-milestone scores and the overall Rubric Score
      pass_fail.py:    the Binary Pass/Fail Grade (critical milestones +
                       primary metrics vs human thresholds)

    Weights and milestone critical flags come from
    milestones_and_rubrics.json.
    Primary metric thresholds are evaluated by pass_fail.py automatically.
    Return {"criteria": [...], "metrics": {...}}.
    """
    criteria = [
        # --- Milestone 1: [YOUR FIRST MILESTONE] ---
        _criterion("data_deliverables",
                   check_data_deliverables,
                   milestone_id="data-preparation"),
        _criterion("data_schema",
                   check_data_schema,
                   milestone_id="data-preparation"),

        # --- Milestone 2: [YOUR SECOND MILESTONE] ---
        _criterion("model_deliverables",
                   check_model_deliverables,
                   milestone_id="model-evaluation"),
        _criterion("metrics_schema",
                   check_metrics_schema,
                   milestone_id="model-evaluation"),
        _criterion("metrics_thresholds",
                   check_metrics_thresholds,
                   milestone_id="model-evaluation"),

        # --- Milestone 3: Deliverables and Report ---
        _criterion("final_deliverables",
                   check_final_deliverables,
                   milestone_id="deliverables-and-report"),
        _criterion("report_content",
                   check_report_content,
                   milestone_id="deliverables-and-report"),

    ]

    # Keep individual checks available even if hidden inference fails.
    model_check = next(item for item in criteria if item["id"] == "model_deliverables")
    try:
        metrics, yearly_scores = _hidden_metrics()
        model_check["detail"] += (
            "; 2018: "
            f"accuracy={yearly_scores[2018]['accuracy']:.4f}, "
            f"worst-state={yearly_scores[2018]['worst_state_accuracy']:.4f}, "
            f"ECE={yearly_scores[2018]['ece_top_label_10bin']:.4f}; "
            "2021: "
            f"accuracy={yearly_scores[2021]['accuracy']:.4f}, "
            f"worst-state={yearly_scores[2021]['worst_state_accuracy']:.4f}, "
            f"ECE={yearly_scores[2021]['ece_top_label_10bin']:.4f}"
        )
    except Exception as exc:
        metrics = {}
        model_check["passed"] = False
        model_check["detail"] = (
            f"Hidden inference failed: {type(exc).__name__}: {exc}"
        )

    return {"criteria": criteria, "metrics": metrics}
