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
import traceback
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

APP_DIR = Path(os.environ.get("OTTER_APP_DIR", "/app"))


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
    # EDIT: Delete this raise once you've implemented the check. Every
    # placeholder fails loudly so an unedited check can never pass silently.
    raise NotImplementedError("EDIT checks.py: implement check_data_deliverables")
    for path in [
        # EDIT: List the files instruction.md tells the agent to produce.
        # app_path("src/data_loader.py"),
        # app_path("artifacts/dataset_manifest.json"),
        # app_path("artifacts/processed_data.npz"),
    ]:
        assert path.exists(), f"Missing: {path}"
    return "data deliverables exist"


def check_data_schema() -> str:
    """Validate structure of data artifacts."""
    # EDIT: Delete this raise once you've implemented the check.
    raise NotImplementedError("EDIT checks.py: implement check_data_schema")
    # EDIT: Load the agent's output and check structure/values.
    # manifest = load_json(app_path("artifacts/dataset_manifest.json"))
    # require_keys(
    #     manifest,
    #     {"source", "splits", "n_samples", "preprocessing_steps"},
    #     "dataset_manifest.json",
    # )
    # assert sum(manifest["splits"].values()) == manifest["n_samples"], (
    #     f"splits must sum to n_samples"
    # )
    return "data schema valid"


# ---------------------------------------------------------------------------
# Milestone 2: [YOUR SECOND MILESTONE NAME]
#
# Example: "Model Training and Evaluation"
# ---------------------------------------------------------------------------

def check_model_deliverables() -> str:
    """Check that required model-stage files exist."""
    # EDIT: Delete this raise once you've implemented the check.
    raise NotImplementedError("EDIT checks.py: implement check_model_deliverables")
    for path in [
        # EDIT: List the model-stage files.
        # app_path("artifacts/metrics.json"),
        # app_path("artifacts/predictions.csv"),
        # app_path("models/trained_model.pkl"),
    ]:
        assert path.exists(), f"Missing: {path}"
    return "model deliverables exist"


def check_metrics_schema() -> str:
    """Validate metrics structure and value ranges."""
    # EDIT: Delete this raise once you've implemented the check.
    raise NotImplementedError("EDIT checks.py: implement check_metrics_schema")
    # EDIT: Load and validate the agent's metrics.
    # metrics = load_json(app_path("artifacts/metrics.json"))
    # require_keys(metrics, {"auroc", "accuracy", "f1"}, "metrics.json")
    # assert 0.0 <= metrics["auroc"] <= 1.0, (
    #     f"auroc must be in [0,1]; got {metrics['auroc']}"
    # )
    return "metrics schema valid"


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
    # EDIT: Delete this raise once you've implemented the check.
    raise NotImplementedError("EDIT checks.py: implement check_metrics_thresholds")
    # EDIT: Set sanity-check baselines well below human performance.
    # metrics = load_json(app_path("artifacts/metrics.json"))
    # assert metrics["auroc"] >= 0.55, (
    #     f"AUROC below random chance; got {metrics['auroc']}"
    # )
    return "metrics above baseline"


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
    # EDIT: Delete this raise once you've implemented the check.
    raise NotImplementedError("EDIT checks.py: implement check_final_deliverables")
    for path in [
        # EDIT: List final deliverables.
        # app_path("artifacts/final_results.json"),
        # app_path("reports/report.md"),
    ]:
        assert path.exists(), f"Missing: {path}"
    return "final deliverables exist"


def check_report_content() -> str:
    """Verify report discusses required topics with sufficient depth."""
    # EDIT: Delete this raise once you've implemented the check.
    raise NotImplementedError("EDIT checks.py: implement check_report_content")
    # EDIT: Check the agent's report.
    # report = app_path("reports/report.md").read_text().lower()
    # required = ["baseline", "method", "results", "limitations"]
    # missing = [t for t in required if t not in report]
    # assert not missing, f"Report missing required terms: {missing}"
    # assert len(report.split()) >= 200, (
    #     f"Report must be >= 200 words; got {len(report.split())}"
    # )
    return "report content valid"


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
    """Run all checks and return criteria + metrics.

    The harness computes the scores from these criteria:
      rubric_score.py: per-milestone scores and the overall Rubric Score
      pass_fail.py:    the Binary Pass/Fail Grade (critical milestones +
                       primary metrics vs human thresholds)

    Weights and milestone critical flags come from
    milestones_and_rubrics.json.
    Primary metric thresholds are evaluated by pass_fail.py automatically.
    Return {"criteria": [...], "metrics": {...}}.
    """
    # EDIT: Read the agent's metrics from its output artifacts.
    # The keys here must match all the keys in human_scores.json pass_criteria.
    # Example: if pass_criteria has "auroc", return {"auroc": <value>}.
    #
    # metrics = load_json(app_path("artifacts/metrics.json"))
    metrics = {}  # EDIT: Replace with actual metric values.

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

    return {"criteria": criteria, "metrics": metrics}
