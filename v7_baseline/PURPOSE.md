# v7_baseline/ — frozen V7 reference files (do NOT retrain or modify)

These are the user's V7 route-split experiment files, included unmodified so that:
1. The V8 retina preprocessing (models/wireflynet.py `Retina`) can be verified against
   the exact V7 source (`flynet_v7_routesplit.py`, TinyFlyNet.forward) — see
   tests/test_suite.py section 10.
2. The dataset interface (MidAirDataset, make_loader, flow caches, manifest hash
   checks) stays byte-compatible with V7.
3. M1 path-audit tooling (audit_pairs.py, expected_*.json) travels with the package.

V7 results (frozen): TinyFlyNet 98.164% acc / 66.942% bal-acc / 0.6506 macro-F1;
CNN 0.5767; SNN 0.5842. Never re-run training from this folder.
