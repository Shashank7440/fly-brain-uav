"""Summarize archived JSON metrics without loading models or changing runs."""
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = []
for protocol, folder, filename in [('original', 'experiments', 'result.json'), ('weighted_loss', 'experiments_weighted_loss', 'validation_result.json')]:
    files = sorted((ROOT / folder).glob('*/' + filename))
    assert len(files) == 36, (protocol, len(files))
    groups = defaultdict(list)
    for path in files:
        d = json.loads(path.read_text())
        assert d['run'] == path.parent.name
        groups[(d['model'], d['graph'])].append(d['seed'])
        if protocol == 'weighted_loss':
            assert 'test' not in d and d['run_metadata']['eval_test'] is False
        for split in ['val', 'test']:
            if split not in d:
                continue
            v = d[split]
            row = dict(protocol=protocol, split=split, run=d['run'], model=d['model'], graph=d['graph'], seed=d['seed'], params=d['params'], best_epoch=d.get('best_epoch', ''))
            row.update({k: v[k] for k in ['macro_f1', 'balanced_accuracy', 'accuracy']})
            row.update({k.lower() + '_recall': v['per_class'][k]['recall'] for k in ['FORWARD', 'LEFT', 'RIGHT']})
            rows.append(row)
    assert len(groups) == 12 and all(sorted(s) == [42, 43, 44] for s in groups.values())
groups = defaultdict(list)
for row in rows:
    groups[tuple(row[k] for k in ['protocol', 'split', 'model', 'graph'])].append(row)
summary = []
for key, runs in sorted(groups.items()):
    row = dict(zip(['protocol', 'split', 'model', 'graph'], key), n=len(runs))
    for metric in ['macro_f1', 'balanced_accuracy', 'accuracy', 'forward_recall', 'left_recall', 'right_recall']:
        values = [r[metric] for r in runs]
        row[metric + '_mean'] = st.mean(values)
        row[metric + '_sd'] = st.stdev(values)
    summary.append(row)
for name, records in [('all_runs_metrics.csv', rows), ('group_summary.csv', summary)]:
    with (ROOT / 'results' / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
for protocol, split in sorted(set((r['protocol'], r['split']) for r in summary)):
    print('\n' + protocol.upper() + ' / ' + split.upper() + ' (mean +/- sample SD)')
    for r in sorted([r for r in summary if (r['protocol'], r['split']) == (protocol, split)], key=lambda r: -r['macro_f1_mean']):
        print(f"{r['model']:12} {r['graph']:10} n={r['n']} F1={r['macro_f1_mean']:.4f} +/- {r['macro_f1_sd']:.4f} BA={r['balanced_accuracy_mean']:.3f}%")
print(f'\nSaved {len(rows)} protocol/split records and {len(summary)} groups; historical JSONs unchanged.')
