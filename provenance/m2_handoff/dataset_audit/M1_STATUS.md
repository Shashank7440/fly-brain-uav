# M1: incomplete — actual manifests unavailable
No train/val/test pairs.csv files or pose data are present in this workspace.
V7 reads prev_frame, curr_frame, label_id from split-specific pairs.csv.
Run audit_pairs.py on the HPC dataset and return its JSON output.

Older make_dataset_v7.py is a six-class builder producing image folders, not
this exact three-class pairs.csv dataset. It cannot certify V7 provenance.
That older builder labels world-NED north/east displacement as forward/right;
world-frame displacement is not camera/body-relative steering unless heading
and coordinate transforms justify it. Pose/image alignment also needs review.
These are risks to investigate, NOT a finding that the current labels are wrong.
Need the actual builder, command/config, split manifests and label derivation.
