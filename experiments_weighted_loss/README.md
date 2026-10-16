# WEIGHTED-LOSS V8 — 36 main runs

Shuffled sampling plus inverse-frequency weighted cross-entropy. `validation_result.json` contains validation results only. Seeds 42,43,44 for Bio, CNN, five rewired and five random graph realizations. Test evaluation was disabled.

Six pilot runs (Bio/CNN) plus 30 graph controls complete this protocol. `_control_submission/` contains historical ledgers, not fresh submission instructions. Do not resubmit or remove artifact locks blindly. Fresh labelled summary: `../results/group_summary.csv` (`protocol=weighted_loss`). Historical local comparison CSVs may predate the final six runs.
