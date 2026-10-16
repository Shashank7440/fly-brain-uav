"""Read-only research-artifact checks; no model imports or checkpoint loading."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
failures = []
def check(condition, label):
    if not condition:
        failures.append(label)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))
meta = json.loads((ROOT / 'connectome/selection_g1.json').read_text())
for name in ['g1_neurons', 'g1_edges']:
    check(sha(ROOT / ('connectome/' + name + '.csv')) == meta[name + '_sha256'], name + ' hash')
base = ROOT / 'provenance/m2_handoff/connectome'
mn = read_csv(base / 'neurons.csv')
me = read_csv(base / 'edges.csv')
gn = read_csv(ROOT / 'connectome/g1_neurons.csv')
ge = read_csv(ROOT / 'connectome/g1_edges.csv')
seeds = {r['root_id'] for r in mn if r['is_seed'].lower() == 'true'}
down = {r['post_id'] for r in me if r['pre_id'] in seeds and int(r['synapses']) >= 5}
nodes = seeds | down
expected = {(r['pre_id'], r['post_id']): int(r['synapses']) for r in me if r['pre_id'] in nodes and r['post_id'] in nodes and int(r['synapses']) >= 5}
actual = {(r['pre_id'], r['post_id']): int(r['synapses']) for r in ge}
check({r['root_id'] for r in gn} == nodes, 'M2 to G1 node reconstruction')
check(actual == expected, 'M2 to G1 directed edges and synapse counts')
check({r['root_id']: r for r in gn} == {r['root_id']: r for r in mn if r['root_id'] in nodes}, 'G1 annotations match M2')
check((len(nodes), len(actual), len(seeds), len(down)) == (556, 6470, 210, 445), 'graph counts')
check(len(actual) == len(ge) and all(a != b for a,b in actual), 'no duplicate pairs or self-loops')
max_error = dict(macro_f1=0.0, balanced_accuracy=0.0, accuracy=0.0)
counts = {}
for protocol, folder, filename in [('original', 'experiments', 'result.json'), ('weighted_loss', 'experiments_weighted_loss', 'validation_result.json')]:
    paths = sorted((ROOT / folder).glob('*/' + filename))
    counts[protocol] = len(paths)
    check(len(paths) == 36, protocol + ' has 36 results')
    groups = defaultdict(list)
    for p in paths:
        d = json.loads(p.read_text())
        check(d['run'] == p.parent.name, str(p) + ' run name')
        groups[(d['model'], d['graph'])].append(d['seed'])
        if protocol == 'weighted_loss':
            m = d['run_metadata']
            check(m['eval_test'] is False and 'test' not in d, d['run'] + ' validation only')
            check(m['script_sha256'] == sha(ROOT / 'models/wireflynet.py'), d['run'] + ' script hash')
            check(m['config_sha256'] == sha(ROOT / 'configs/wireflynet_g1_weighted_loss.json'), d['run'] + ' config hash')
        for split in ['val', 'test']:
            if split not in d:
                continue
            v = d[split]; cm = v['confusion']
            support = [sum(r) for r in cm]
            pred = [sum(r[i] for r in cm) for i in range(3)]
            recalls = [cm[i][i] / support[i] for i in range(3)]
            f1 = [2 * cm[i][i] / (support[i] + pred[i]) if support[i] + pred[i] else 0 for i in range(3)]
            computed = dict(macro_f1=sum(f1)/3, balanced_accuracy=100*sum(recalls)/3, accuracy=100*sum(cm[i][i] for i in range(3))/sum(support))
            for metric, value in computed.items():
                error = abs(value - v[metric]); max_error[metric] = max(max_error[metric], error)
                check(error <= (0.000051 if metric == 'macro_f1' else 0.00051), d['run'] + '/' + split + '/' + metric + ' rounding')
    check(len(groups) == 12 and all(sorted(s) == [42,43,44] for s in groups.values()), protocol + ' seed groups')
report = dict(status='PASS' if not failures else 'FAIL', result_file_counts=counts, g1_counts=dict(nodes=len(nodes), directed_edges=len(actual), entry=len(seeds), output=len(down), overlap=len(seeds & down)), maximum_metric_rounding_errors=max_error, failures=failures, scope='Internal supplied-artifact consistency only. Raw-source extraction, external identities/license, dataset correctness, execution history and independent generalization are not certified. No checkpoints loaded.')
(ROOT / 'results/verification_report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
raise SystemExit(bool(failures))
