# V8 novelty positioning (Phase 10) — updated 2026-10-08

## The claim we CANNOT make
"First connectome-constrained network" and "first connectome-based navigation model" are
both taken. The literature (as of Oct 2026):

| Work | What it did | What it did NOT do |
|---|---|---|
| Lappalainen et al., Nature 2024 (FlyNet/DMN) | Connectome-constrained deep mechanistic network over 64 optic-lobe cell types; predicts neural activity to natural stimuli; ablated which constraints matter | Behavior/actions; matched rewired-graph controls |
| FLYNN (Wang & Chen, arXiv:2607.00025, Jun–Jul 2026) | Full FAFB v783 connectome as an RNN; vision-based navigation in MuJoCo; robust to OOD + sensory loss | Degree-preserving rewired controls; real (non-sim) flight imagery; action prediction on a real dataset |
| Garcia-Duran et al., NeurIPS 2026 poster | FlyNet (Lappalainen 2024) on a looming collision task vs CNN/RAFT/**unconstrained CTRNN**; found connectome constraints give **cost, not robustness** vs CTRNN | Degree-preserving rewiring; real drone data; motion-direction/steering prediction |
| FlyGNN / FlyGM (NeurIPS 2025 / arXiv:2602.17997) | Whole-brain connectome as graph/locomotion controller | Visual action prediction; matched topology controls |
| Pospisil et al. 2024 (effectome) | Connectome+anatomy model for active visual head control | Not a task-trained network benchmark |

## The claim we CAN make (if the experiment supports it)
V8's contribution is a **controlled causal test of wiring**, not the existence of a
connectome-constrained network:

1. **Real vs matched controls**: degree-preserving rewired graphs (exact in/out degree
   sequences, 5 realizations) and matched random sparse graphs (5 realizations) — the
   question "does *specific* biological wiring matter beyond degree structure?" has not
   been answered for action prediction, as far as our search shows.
2. **A selected, documented visual subconnectome** (G1, LPLC2-centered, pre-declared
   selection rule, FlyWire r783) rather than a whole-brain dump — interpretable scope.
3. **Real drone imagery** (MidAir, route-grouped split — the audited V7 protocol) with
   camera-level perturbations, instead of MuJoCo rendering.
4. Honest uncertainty: multi-seed, small LEFT-support caveats carried over from V7.

A NEGATIVE result (real wiring does not beat rewired/random controls) is publishable and
would corroborate the Garcia-Duran et al. finding in a different task/modality. Do not
change the claim after seeing test results.

## Positioning sentence (draft)
"Unlike prior connectome-constrained models evaluated against hand-crafted baselines
[FLYNN] or single unconstrained RNN controls [Garcia-Duran et al.], we test whether a
selected LPLC2-centered visual subconnectome outperforms degree-preserving rewired and
random matched controls on real drone motion-direction prediction."

## Naming
"FlyNet" is taken (Lappalainen et al.), as is "FLYNN". Working name for V8:
**WireFlyNet** (provisional — confirm availability before camera-ready).
