# Analysis plan (after M5 runs)

1. **Primary comparison** (Phase 7.1): acc, balanced acc, macro-F1, per-class P/R/F1,
   confusion, mean±SD across seeds for: bio vs rewired:0-4 vs random:0-4 vs cnn_matched
   (+ frozen V7 numbers). Paired seed-level differences; bootstrap CIs accounting for
   route/trajectory dependence (do NOT treat nearby frames as independent).
2. **Robustness** (Phase 7.2): A) V7-compatible representation perturbations (photometric
   on clean flow) for continuity; B) camera-level perturbations: corrupt source RGB, recompute
   flow, rebuild inputs; shared severities/noise realizations across models, severity-specific
   reporting. Script to be written when models exist (torchvision-style corruptions).
3. **Efficiency** (Phase 7.4): params, model-only + end-to-end latency (flow included),
   peak memory, batch-1, repeated timing; document op-count convention for the recurrent core
   (T x E multiply-adds). No energy claims without power measurement.
4. **Ablations** (Phase 8): real-vs-rewired (PRIMARY — identical degree sequences, so
   fan-in-normalized init is matched within 10%; random-sparse is secondary: its init
   mean differs +23% after sqrt(in-degree) normalization, see graphs_manifest.json
   init_fanin_stats), fan-in-normalization sensitivity (`--no-fanin-norm` raw-init arm
   for random:k), synapse-vs-random init (config `init: random`), entry-subset vs
   pooled injection (needs a model variant — add only after the primary result),
   fewer/more recurrent steps (config `steps`), without flow (channel variant — defer).
5. All scripts write into analysis/ with the graphs_manifest + config hashes recorded.
