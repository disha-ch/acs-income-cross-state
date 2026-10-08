# ACSIncome cross-state and temporal generalization

Build an offline binary classifier using the packaged ACSIncome data. Train with `/app/data/train.npz` and use `/app/data/val.npz` for model selection and visible diagnostics. Do not download Census data or access hidden evaluation data during training or model selection. Preserve the input feature order and label contract.

Provide these command-line interfaces:

- `python /app/train.py --train /app/data/train.npz --val /app/data/val.npz --model-out /app/models/trained_model.joblib --metrics-out /app/artifacts/metrics.json`
- `python /app/predict.py --model /app/models/trained_model.joblib --input /app/data/prediction_input.npz --output /app/artifacts/predictions.npy`

The verifier may supply different absolute paths for prediction input and output. Training must run offline, use no more than one GPU, and save a reloadable selected model at `/app/models/trained_model.joblib`. Inference must read an NPZ containing `X` and `feature_names`, validate the feature contract, and write one finite positive-class probability per row to the requested NPY file.

Write `/app/artifacts/metrics.json` with exactly three numerical metrics—`accuracy`, `auroc`, and `ece_top_label_10bin`—plus the string field `"scope": "visible_validation"`. Also write `/app/artifacts/claims.json` with the selected model and the scope of its claims, and `/app/reports/report.md` describing the method, visible validation results, and limitations.

The verifier independently computes hidden accuracy, worst-state accuracy (the minimum accuracy across hidden states), and hidden top-label ECE from verifier-owned labels and state membership. For ECE, predict positive when probability is at least 0.5; confidence is the probability assigned to the predicted class. Use ten equal-width confidence bins, placing confidence 1.0 in the final bin. Lower ECE indicates better calibration. Hidden data and labels are unavailable to the agent.