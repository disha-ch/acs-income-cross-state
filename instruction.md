<!--
EC TEMPLATE NOTE — DELETE THIS ENTIRE COMMENT BEFORE DELIVERY.

Replace the placeholder below with your task's real instruction prompt
(written like a realistic human prompt to a coding agent).

REQUIREMENT: The instructions MUST tell the agent to write its final
primary metric values to /app/artifacts/metrics.json, with keys exactly
matching the pass_criteria keys in tests/human_scores.json
(e.g. {"auroc": 0.87}). This is the standardized path every Otter task
uses — checks.py reads it and pass_fail.py grades it against the
human-derived thresholds. Example line to include in your prompt:

  "Write your final evaluation metrics to /app/artifacts/metrics.json
   as a JSON object, e.g. {\"auroc\": <your test-set AUROC>}."

The agent never sees this file's comments — but only because you deleted
them. Do not ship EC-facing notes in instruction.md.
-->

Write "hello world" to `/app/artifacts/output.txt`
