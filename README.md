# ACSIncome: Cross-State and Temporal Generalization

An end-to-end machine learning project investigating how well an income classifier generalizes across U.S. states and across the post-2020 period.

The project uses the **ACSIncome benchmark**, derived from the U.S. Census American Community Survey (ACS) Public Use Microdata Sample (PUMS), and follows the broader research motivation of evaluating machine learning systems under real-world distribution shift.

## 1. Project Overview

Traditional machine learning evaluation often assumes that training and testing data follow similar distributions. In production, this assumption may fail when a model is deployed in a different geographic region or evaluated on data collected in a different year.

This project investigates three questions:

1. **Geographic generalization:** How well does a model trained on selected states perform on previously unseen states?
2. **Temporal generalization:** How does predictive performance change when evaluating 2018-trained models on 2021 data?
3. **Reliability and calibration:** Does the model maintain acceptable accuracy across states, and do its predicted probabilities reflect actual correctness?

The emphasis is on reproducible training, calibration, worst-group evaluation and separating model selection from final held-out testing.

## 2. Dataset and Experimental Design

**Dataset:** ACSIncome, based on ACS PUMS and the Folktables benchmark.

The current experimental split is:

| Split | Year | States | Purpose |
|---|---|---|---|
| Training | 2018 | California, Texas, New York | Model fitting |
| Validation | 2018 | Florida | Model selection and calibration diagnostics |
| Geographic evaluation | 2018 | Mississippi, West Virginia, New Mexico, Arkansas, Louisiana, Montana | Cross-state generalization |
| Temporal evaluation | 2021 | Same six held-out states | Post-2020 generalization |

The classifier predicts the binary income target defined by ACSIncome.

### Input features

The model uses ten ACS features:

| Feature | Description |
|---|---|
| AGEP | Age |
| COW | Class of worker |
| SCHL | Educational attainment |
| MAR | Marital status |
| OCCP | Occupation |
| POBP | Place of birth |
| RELP | Household relationship |
| WKHP | Usual hours worked per week |
| SEX | Sex |
| RAC1P | Race |

The pipeline maintains a consistent feature order across the dataset splits and inference interface.

**Data availability:** Downloaded ACS microdata and packaged NPZ datasets are not committed to this repository. Local dataset preparation is supported through the research and packaging scripts.

## 3. Modeling Approach

The implementation uses a GPU-capable XGBoost classifier with preprocessing appropriate for mixed numerical and categorical ACS variables.

Two alternatives are evaluated:

- **Raw classifier:** XGBoost probability predictions without additional calibration.
- **Calibrated classifier:** XGBoost with sigmoid probability calibration using cross-validation.

The selection policy uses only visible validation data.

The calibrated version is selected if its top-label expected calibration error improves without reducing validation accuracy. Otherwise, the raw classifier is retained.

The selected model is saved as a reloadable Joblib checkpoint.

## 4. Evaluation Metrics

### Accuracy

The proportion of observations with correct binary classifications, using a positive-class probability threshold of 0.5.

### AUROC

The area under the receiver operating characteristic curve, measuring the model's ability to rank positive examples above negative examples.

### Worst-State Accuracy

The minimum classification accuracy across the evaluated states.

This captures geographic performance disparities that aggregate accuracy can hide.

### Expected Calibration Error (ECE)

Top-label ECE is measured with ten equal-width confidence bins.

For binary classification, confidence is the probability assigned to the predicted class. The metric measures the average discrepancy between confidence and observed correctness.

Lower ECE indicates better calibration.

### Cross-Year Comparison

Hidden evaluation can be broken down by 2018 and 2021 to analyze geographic and temporal differences separately.

An observed performance difference does not, by itself, establish the causal source or type of distribution shift.

## 5. Repository Structure

- **`instruction.md`** — Problem requirements and execution contract.
- **`task.toml`** — Runtime and task metadata.
- **`environment/`** — CUDA-enabled Docker environment, dependencies and local packaged-input structure.
- **`solution/train.py`** — Standalone offline training entry point.
- **`solution/predict.py`** — Model inference command-line interface.
- **`solution/run.py`** — End-to-end experiment launcher.
- **`solution/src/pipeline.py`** — Preprocessing, model training, calibration, evaluation and artifact generation.
- **`solution/solve.sh`** — Reproducible experiment execution script.
- **`tests/`** — Artifact validation and held-out evaluation framework.
- **`scripts/`** — ACS dataset packaging and preparation utilities.
- **`exploration/`** — Research notes, exploratory analysis and experimentation.

## 6. Setup

### Requirements

- Python 3.12
- Dependencies specified in `environment/requirements.txt`
- NVIDIA GPU with a compatible CUDA runtime for GPU training
- Local packaged ACSIncome datasets

The Docker environment is configured for CUDA-based execution. A macOS machine without an NVIDIA GPU can perform syntax and interface checks but cannot validate the CUDA training path.

### Installation

Create a Python virtual environment and install the required dependencies from:

`environment/requirements.txt`

The project also includes a Dockerfile at:

`environment/Dockerfile`

### Training

Run the standalone training interface with four arguments:

`python solution/train.py --train environment/task_inputs/train.npz --val environment/task_inputs/val.npz --model-out models/trained_model.joblib --metrics-out artifacts/metrics.json`

The model and artifact directories must be available or created by the training implementation.

### Inference

Run the inference interface with:

`python solution/predict.py --model models/trained_model.joblib --input prediction_input.npz --output predictions.npy`

The input NPZ must contain `X` and `feature_names`. The output is a one-dimensional NPY array with a positive-class probability for each row.

## 7. Generated Artifacts

The pipeline is designed to generate:

| Artifact | Purpose |
|---|---|
| `models/trained_model.joblib` | Selected reloadable classifier |
| `artifacts/metrics.json` | Visible validation accuracy, AUROC and ECE |
| `artifacts/claims.json` | Model selection and evaluation scope |
| `artifacts/validation_diagnostics.json` | Comparison of raw and calibrated models |
| `reports/report.md` | Methods, results and limitations |

The validation metrics are explicitly labeled as validation results. Held-out geographic and temporal evaluation must be computed independently.

## 8. Reproducibility and Data Integrity

The project separates training, validation and held-out evaluation data.

- Validation data is used for model selection.
- Held-out labels are not used for model fitting or selection.
- Training and inference expose explicit command-line interfaces.
- Model checkpoints are reloaded to check prediction consistency.
- Feature schemas are validated at inference time.
- Generated datasets, models and raw downloads are excluded from Git version control.

**Current validation status:** Python syntax checks and CLI argument validation have been performed. Successful end-to-end GPU execution and complete held-out benchmark results are not yet established.

## 9. Limitations

- Results from the selected states may not generalize to all U.S. states.
- ACS features and coding conventions may change across years.
- Demographic and socioeconomic features raise fairness and representativeness considerations.
- Calibration measured on validation data may not transfer to unseen geographic or temporal distributions.
- Public-use microdata requires careful handling of record-linkage and privacy risks.
- Geographic or temporal performance differences alone do not prove a specific conditional-shift mechanism.

## 10. References and Attribution

- **Folktables:** https://github.com/socialfoundations/folktables
- **WhyShift:** https://github.com/namkoong-lab/whyshift
- **ACS PUMS:** https://www.census.gov/programs-surveys/acs/microdata.html
- **Folktables paper:** *Retiring Adult: New Datasets for Fair Machine Learning*, arXiv:2108.04884

ACS PUMS is U.S. Census public-use data. Folktables and WhyShift are separate software projects with their own licensing terms.

---

**Project status:** Active development and experimental validation.