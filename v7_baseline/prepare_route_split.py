#!/usr/bin/env python3
"""Group identical retained motion sequences before splitting. Never overwrite data."""
import argparse,csv,hashlib,json,random,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',default='/Data4/mi_25026207/drone/data/flyuav_midair_3class');p.add_argument('--output',default='/Data4/mi_25026207/drone/data/flyuav_midair_3class_route_split');p.add_argument('--images',default='/Data4/mi_25026207/drone/data/flyuav_midair_3class/MidAir_HPC/Kite_training');p.add_argument('--seed',type=int,default=42);p.add_argument('--skip-image-check',action='store_true');a=p.parse_args()
source,output=Path(a.source),Path(a.output)
if output.exists():raise SystemExit(f'Refusing to overwrite {output}. Use a NEW output directory.')
keys=['frame_index','label_id','dx_world','dy_world','dz_world','dx_body','dy_body','dz_body'];records=collections.defaultdict(list);hashes={};fields=None
for sp in ['train','val','test']:
 f=source/sp/'pairs.csv';hashes[sp]=hashlib.sha256(f.read_bytes()).hexdigest()
 with f.open(newline='') as h:
  r=csv.DictReader(h)
  if fields is None: fields=r.fieldnames
  if r.fieldnames!=fields: raise ValueError('Manifest schemas differ')
  if not set(keys+['climate','trajectory','prev_frame','curr_frame'])<=set(fields): raise ValueError('Missing provenance columns')
  for row in r:records[(row['climate'],row['trajectory'])].append(row)
# Canonical numeric signatures use exact hexadecimal float representations.
def payload(rows):
 result=[]
 for row in sorted(rows,key=lambda x:int(x['frame_index'])):
  result.append([int(row['frame_index']),int(row['label_id'])]+[float(row[k]).hex() for k in keys[2:]])
 return json.dumps(result,separators=(',',':')).encode()
groups=collections.defaultdict(list)
for key,rows in records.items():groups[hashlib.sha256(payload(rows)).hexdigest()].append(key)
# Conservative naming correspondence: members of a group must share route suffix.
# Same suffix appearing in different signatures means exact grouping is insufficient.
suffixes=collections.defaultdict(set)
for sig,members in groups.items():
 for cl,tr in members:
  try:suffix=int(tr.rsplit('_',1)[-1])%1000
  except ValueError:raise ValueError(f'Cannot parse trajectory {tr}')
  suffixes[suffix].add(sig)
if any(len(v)>1 for v in suffixes.values()):raise SystemExit('Route naming / exact-motion grouping disagreement. Inspect source poses before proceeding.')
gids=sorted(groups);random.Random(a.seed).shuffle(gids)
if len(gids)<6:raise SystemExit('Too few independent groups for this splitter')
ntrain=round(.64*len(gids));nval=max(1,round(.19*len(gids)))
assignment={g:sp for sp,gs in [('train',gids[:ntrain]),('val',gids[ntrain:ntrain+nval]),('test',gids[ntrain+nval:])] for g in gs}
new={s:[] for s in ['train','val','test']};members=[]
for sig,ks in groups.items():
 for cl,tr in sorted(ks):
  rs=records[(cl,tr)];new[assignment[sig]].extend(rs)
  members.append(dict(group_id=sig,split=assignment[sig],climate=cl,trajectory=tr,rows=len(rs)))
seenframes={};report={}
for sp,rows in new.items():
 rows.sort(key=lambda r:(r['climate'],r['trajectory'],int(r['frame_index'])))
 counts=collections.Counter(r['label_id'] for r in rows)
 if set(counts)!={'0','1','2'}:raise SystemExit(f'{sp} lacks a class. Do not tune using test performance; review a documented split policy.')
 frames={r[k] for r in rows for k in ['prev_frame','curr_frame']}
 if not a.skip_image_check:
  for name in frames:
   if not (Path(a.images)/name).is_file():raise FileNotFoundError(Path(a.images)/name)
 seenframes[sp]=frames
 report[sp]=dict(rows=len(rows),class_counts=dict(counts),route_groups=sum(assignment[g]==sp for g in assignment),recordings=sum(m['split']==sp for m in members))
import itertools
for x,y in itertools.combinations(new,2):
 if seenframes[x]&seenframes[y]:raise ValueError('Shared image paths remain')
output.mkdir(parents=True)
for sp,rows in new.items():
 (output/sp).mkdir()
 with (output/sp/'pairs.csv').open('w',newline='') as h:
  w=csv.DictWriter(h,fields);w.writeheader();w.writerows(rows)
 report[sp]['manifest_sha256']=hashlib.sha256((output/sp/'pairs.csv').read_bytes()).hexdigest()
meta=dict(seed=a.seed,source=str(source),image_root=a.images,original_manifest_hashes=hashes,grouping='Exact complete retained label/world/body displacement sequence, corroborated by route-number suffix correspondence; not complete raw pose identity.',labels_changed=False,threshold_applied=False,groups=members,counts=report,limitations=['Quaternion and capture-time conventions not independently validated','Task labels are recorded motion directions, not verified safe avoidance actions','Only identical retained routes plus checked naming correspondences grouped; raw pose validation recommended'])
(output/'split_metadata.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(report,indent=2));print('Prepared NEW split. Original data unchanged. No flow caches copied.')
