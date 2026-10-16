# Biological and dataset provenance

## Internal biological chain — verified
1. Recovered original M2 handoff: `provenance/m2_handoff/`.
2. M2 export: 5,750 unique neurons; 33,071 unique directed edges; 210 locally annotated LPLC2 seeds (108 left / 102 right); no duplicate pairs, self-loops, missing endpoints, or isolated nodes; one weakly connected component.
3. G1 rule: seeds plus direct postsynaptic partners on connections with >=5 synapses; induced thresholded edges.
4. Reconstructed G1: 556 nodes, 6,470 directed connections, 210 entry nodes, 445 output nodes, 99-node overlap.
5. G1 neuron annotation rows, edge synapse counts, and node set match M2 derivation. G1 file hashes match `connectome/selection_g1.json` and reconstruction matches `graphs/biological/g1_bio.npz`.

IDs must be read as **strings**, never floating-point values. CSV byte order may vary on rerunning `derive_g1.py` because it iterates over a set; compare node/edge content and annotations, not only output byte hashes.

## Source-to-M2 boundary
`provenance/m2_handoff/extract_circuit.py` reads these original repository inputs from its sibling `raw/` folder:
- `neuron_table.csv`
- `synapse_table.csv.gz`
- `synapse_table_compact_ids_map.csv.gz`

These raw tables are not included. Expected raw hashes and release 783 are recorded in the handoff's `connectome/selection.json`. Repository commit recorded there: `0d8574d46627ce7fadd968a3c5d602e837325373`. The script queries main-branch commit metadata at execution time; this is not proof the downloads were pinned to that commit. Source-to-M2 extraction is documented but has not been independently rerun from the raw tables in this repository preparation.

Source links recorded in the handoff (external contents/terms not independently verified here):
- https://github.com/murthylab/visual-system-parts-list
- https://www.biorxiv.org/content/10.1101/2023.10.12.562119v3
- https://flywire.ai/guidelines

The handoff README is not the upstream license. Check upstream licenses, dataset release terms, and citation obligations before reuse/publication. Do not infer whole-brain coverage, physiological weight strengths, transmitter signs, or complete visual-pathway reconstruction.

## Controls
All graph variants share node IDs/order, entry/output indices, N=556 and E=6,470. Rewired graphs preserve each node's exact in/out-degree; random graphs do not. Both permute raw synapse-derived initialization values across edges. Subsequent destination fan-in normalization changes final initialization distributions; even rewired controls do not perfectly isolate topology from anatomical weight assignment. Fixed-edge-count random graphs should not be described as independent-edge G(n,p) draws.

## Dataset verification scope
The included M1 audit reports path/file-identity checks. It cannot establish content independence across renamed/climate copies, body/camera-relative label semantics, or image/pose synchronization. The author states that label generation and alignment were verified separately; that is **author-verified**, not independently established by this audit. Dataset images, poses, actual split manifests, and the exact three-class generation pipeline have not been supplied in this repository package.

`provenance/m2_handoff/dataset_audit/M1_STATUS.md` records an earlier audit status; it does not override the later author verification or constitute a finding that current labels are wrong.
