# Derived result summaries

Generated from the archived result JSONs by `python scripts/summarize_results.py`.
- `all_runs_metrics.csv`: one row per protocol/run/evaluation split; 72 original split rows + 36 weighted validation rows = 108 rows.
- `group_summary.csv`: per-graph mean ± sample SD across three seeds; original validation/test and weighted validation remain separate (36 rows).
- `verification_report.json`: generated consistency checks; file-level verification is not external source authentication.

Accuracy and balanced accuracy are percent. Macro-F1 and recall are fractions. Stored per-run metrics are rounded, so reconstructed confusion-matrix values can differ at rounding precision. No frames or graph realizations should be treated as independent route replications.
