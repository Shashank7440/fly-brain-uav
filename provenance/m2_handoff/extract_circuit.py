from pathlib import Path
import pandas as pd,json,hashlib,urllib.request
from collections import Counter
P=Path(__file__).parent; raw=P/'raw'; out=P/'connectome'
n=pd.read_csv(raw/'neuron_table.csv',dtype={'cell ID':str})
mp=pd.read_csv(raw/'synapse_table_compact_ids_map.csv.gz',dtype={'full cell ID (root ID)':str})
m=dict(zip(mp['compact cell ID'],mp['full cell ID (root ID)']))
seeds=set(n.loc[n['resolved type'].eq('LPLC2'),'cell ID']); compact={k for k,v in m.items() if v in seeds}
f=raw/'synapse_table.csv.gz'; all_rows=0; selected=[]
for c in pd.read_csv(f,chunksize=500000):
 all_rows+=len(c)
 c.columns=['pre','post','synapses']
 c=c[(c.pre.isin(compact)|c.post.isin(compact)) & (c.synapses>=5)]
 selected.append(c)
e=pd.concat(selected,ignore_index=True)
e=e.groupby(['pre','post'],as_index=False).synapses.sum() # threshold was per source row; document
# source inspection: verify duplicate pairs before aggregating independently below
nodes=set(e.pre)|set(e.post)|compact
ind=[]; duplicates=0
for c in pd.read_csv(f,chunksize=500000):
 c.columns=['pre','post','synapses']
 c=c[c.pre.isin(nodes)&c.post.isin(nodes)]
 ind.append(c)
a=pd.concat(ind,ignore_index=True)
duplicates=int(a.duplicated(['pre','post']).sum())
assert duplicates==0,'Need aggregate-before-threshold for duplicate pair records'
a=a[a.synapses>=5].copy()
a['pre_id']=a.pre.map(m);a['post_id']=a.post.map(m)
a[['pre_id','post_id','synapses']].to_csv(out/'edges.csv',index=False)
ids={m[k] for k in nodes}; q=n[n['cell ID'].isin(ids)].copy()
missing=ids-set(q['cell ID'])
if missing:
 q=pd.concat([q,pd.DataFrame({'cell ID':sorted(missing),'resolved type':['UNANNOTATED']*len(missing)})],ignore_index=True)
q=q.rename(columns={'cell ID':'root_id','resolved type':'cell_type'})
q['is_seed']=q.root_id.isin(seeds);q.to_csv(out/'neurons.csv',index=False)
typed=dict(zip(q.root_id,q.cell_type.fillna('UNANNOTATED')))
incoming=Counter();outgoing=Counter()
for r in a.itertuples():
 if r.post_id in seeds: incoming[typed.get(r.pre_id,'UNANNOTATED')]+=int(r.synapses)
 if r.pre_id in seeds: outgoing[typed.get(r.post_id,'UNANNOTATED')]+=int(r.synapses)
summary=dict(seed_cells=len(seeds),left_seeds=int((n[n['cell ID'].isin(seeds)]['side (cell body)']=='left').sum()),right_seeds=int((n[n['cell ID'].isin(seeds)]['side (cell body)']=='right').sum()),candidate_nodes=len(ids),candidate_edges=len(a),self_edges=int((a.pre==a.post).sum()),missing_annotations=len(missing),source_edge_rows=all_rows,source_selected_pair_duplicates=duplicates,top_input_types=incoming.most_common(12),top_output_types=outgoing.most_common(12))
meta=dict(status='CANDIDATE_NOT_FINAL_TRAINING_GRAPH',release=783,source='https://github.com/murthylab/visual-system-parts-list',selection='All annotated LPLC2 cells (both hemispheres), their direct pre/post partners on edges >=5 synapses, then induced graph on this node set using edges >=5. Seed cells retained even if isolated.',scope_warning='Visual-system repository edge-table scope; not independently certified as full whole-brain partner coverage. No shortest path to descending neurons or giant fiber claimed.',id_handling='Root IDs retained as strings, never float',hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in raw.iterdir() if x.is_file()},summary=summary)
try:meta['repository_commit']=json.load(urllib.request.urlopen('https://api.github.com/repos/murthylab/visual-system-parts-list/commits/main'))['sha']
except Exception:pass
(out/'selection.json').write_text(json.dumps(meta,indent=2));print(json.dumps(summary,indent=2))
