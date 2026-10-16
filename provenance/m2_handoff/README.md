# V8 M1/M2 handoff
## M1: blocked on actual cluster manifests
Run on the HPC from the extracted folder:
python dataset_audit/audit_pairs.py \
 --root /Data4/mi_25026207/drone/data/flyuav_midair_3class \
 --out m1_audit.json
Upload m1_audit.json and train/val/test/pairs.csv, plus the actual three-class
builder and generation command. The tool only reads files; no training occurs.
Path checks cannot certify matching content under different names, world/body
coordinate semantics, synchronization, or repeated trajectories across climates.
See dataset_audit/M1_STATUS.md.

## M2: candidate exported, not final model circuit
connectome/neurons.csv and edges.csv contain release-783 root IDs and actual
source connections, not a generated or type-averaged graph.
Selection: all 210 LPLC2 seed cells, both hemispheres; include direct partners
on >=5-synapse edges; retain induced edges >=5 among selected nodes.
Candidate: 5,750 neurons and 33,071 directed edges. These include partners of
LPLC2 and edges between those partners; not all edges are LPLC2 connections.
Metadata, validation, hashes and commit are in selection.json.

This is a visual-system repository export. Whole-brain partner coverage and
neurotransmitter annotations have not been independently certified. No final
input/output subset or neuron dynamics have been selected; no V8 training ran.
The five-synapse threshold is a declared exploratory filter, not a biological law.
Inspect sensitivity to threshold and full whole-brain edge coverage before M3.
Root IDs are strings: do not convert them through float or spreadsheet precision.

Source: https://github.com/murthylab/visual-system-parts-list
Associated manuscript: https://www.biorxiv.org/content/10.1101/2023.10.12.562119v3
Version and terms: consult source README/license and FlyWire citation guidelines.
extract_circuit.py reproduces extraction after downloading the raw CSV/GZ files
listed in selection.json from the repository's data/ directory. Large raw source
files are not included in this handoff ZIP.
