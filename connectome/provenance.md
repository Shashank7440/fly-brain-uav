# Connectome provenance — V8 candidate circuit G1

## Biological source
- **Repository:** murthylab/visual-system-parts-list — https://github.com/murthylab/visual-system-parts-list
- **Associated manuscript:** https://www.biorxiv.org/content/10.1101/2023.10.12.562119v3
- **FlyWire release:** 783 (root IDs are strings; never round-trip through float/spreadsheet precision)
- **Exported by:** `extract_circuit.py` in the M2 handoff (selection rule: all 210 LPLC2 seeds,
  both hemispheres; direct partners on >=5-synapse edges; induced edges >=5 among selected nodes).
  The M2 export contains 5,750 neurons / 33,071 directed edges and passed validation:
  no duplicates, no missing endpoints, no self-loops, single weakly-connected component.
- **Validation run:** 2026-10-08 on the M2 export (see M2 handoff README).

## G1 derivation (this directory)
- Derived by `derive_g1.py` from the M2 export; rule pre-declared before any V8 training:
  LPLC2 seeds (210) + their DIRECT downstream partners (445); induced edges among them
  (6,470). Upstream columnar fan-in excluded — the source export is dominated by LPLC2's
  5,312 upstream partners, and the engineered input interface lands at LPLC2, the entry stage.
- Hashes of `g1_neurons.csv` / `g1_edges.csv` are in `selection_g1.json`.

## What this is NOT
- Not a complete visual pathway; visual-system repository coverage only, not whole-brain.
- The >=5-synapse threshold is a declared exploratory filter, not a biological law.
  Threshold sensitivity must be checked before strong claims (needs the raw source files).
- Synapse counts are anatomical counts, not measured physiological strengths.
- No neurotransmitter/sign annotations are included in this export; sign-constrained
  weight variants are deferred until annotations and assumptions can be justified.

## Engineered interfaces (assumptions, not biology)
- Entry subset: the 210 LPLC2 cells receive the projected sensory-encoder features.
- Output subset: the 445 downstream partners feed the readout.
- Rate-based dynamics, T update steps, and per-edge learnable weights are modelling choices.

## Citation obligations
Consult the source repository README/license and FlyWire citation guidelines
(https://flywire.ai/guidelines) before publishing. Cite the Murthy lab parts-list
manuscript, FlyWire, and the release-783 dataset.
