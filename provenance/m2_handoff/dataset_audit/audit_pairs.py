#!/usr/bin/env python3
import argparse,csv,json,hashlib,re,itertools
from pathlib import Path,PurePosixPath
from collections import Counter
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',default='m1_audit.json');a=p.parse_args()
R={};sets={};trajs={};pairs={}
def norm(s): return str(PurePosixPath(s.replace('\\','/')))
def traj(s):
 parts=PurePosixPath(s).parts
 for i,x in enumerate(parts):
  if re.fullmatch(r'trajectory_\d+',x): return '/'.join(parts[:i+1])
 return None
for split in ['train','val','test']:
 f=Path(a.root)/split/'pairs.csv'; frames=set();ts=set();ps=set();counts=Counter();bad=[];unresolved=0;dup=0;cross=0;n=0
 with f.open(newline='') as h:
  reader=csv.DictReader(h)
  assert {'prev_frame','curr_frame','label_id'}<=set(reader.fieldnames or []),f
  for row in reader:
   n+=1; x,y=norm(row['prev_frame']),norm(row['curr_frame']);t1,t2=traj(x),traj(y)
   if not t1 or not t2: unresolved+=1
   if t1 and t2 and t1!=t2: cross+=1
   ts.update(t for t in (t1,t2) if t);frames.update((x,y))
   if (x,y) in ps:dup+=1
   ps.add((x,y));counts[row['label_id']]+=1
   if row['label_id'] not in ['0','1','2']:bad.append(n)
 R[split]=dict(rows=n,class_counts=dict(counts),unique_frames=len(frames),trajectory_count=len(ts),unresolved_trajectory_rows=unresolved,cross_trajectory_pairs=cross,duplicate_pairs=dup,invalid_label_rows=bad[:20],manifest_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
 sets[split]=frames;trajs[split]=ts;pairs[split]=ps
R['cross_split']={}
for x,y in itertools.combinations(sets,2):
 R['cross_split'][x+'__'+y]=dict(shared_frames=len(sets[x]&sets[y]),shared_pairs=len(pairs[x]&pairs[y]),shared_trajectories=len(trajs[x]&trajs[y]),example_shared_trajectories=sorted(trajs[x]&trajs[y])[:10])
R['scope']='Path-based audit only. Does not certify content duplicates, trajectory identity across climate copies, body-frame label validity, pose/image time alignment, or label generation.'
R['status']='FAIL' if any(v[k]>0 for v in R['cross_split'].values() for k in ['shared_frames','shared_pairs','shared_trajectories']) else ('INCOMPLETE' if any(R[s]['unresolved_trajectory_rows'] for s in sets) else 'PATH_CHECKS_PASS_REQUIRES_PROVENANCE_REVIEW')
Path(a.out).write_text(json.dumps(R,indent=2));print(json.dumps(R,indent=2))
