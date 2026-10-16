#!/usr/bin/env python3
"""Fresh index-aligned flow cache with hash binding; never substitute zeros on errors."""
import argparse,csv,json,hashlib,os
from pathlib import Path
import numpy as np,cv2
from PIL import Image
cv2.setNumThreads(1)
p=argparse.ArgumentParser();p.add_argument('--root',default='/Data4/mi_25026207/drone/data/flyuav_midair_3class_route_split');p.add_argument('--images',default='/Data4/mi_25026207/drone/data/flyuav_midair_3class/MidAir_HPC/Kite_training');a=p.parse_args()
params=dict(pyr_scale=.5,levels=3,winsize=15,iterations=3,poly_n=5,poly_sigma=1.2,flags=0)
for sp in ['train','val','test']:
 root=Path(a.root)/sp;mf=root/'pairs.csv';sig=hashlib.sha256(mf.read_bytes()).hexdigest();cache=root/'flow';marker=cache/'cache_metadata.json'
 meta=dict(manifest_sha256=sig,image_root=str(Path(a.images).resolve()),size=128,farneback=params,dtype='float16')
 if cache.exists():
  if not marker.exists() or json.loads(marker.read_text())!=meta:raise SystemExit(f'Unbound or stale cache: {cache}. Refusing reuse.')
 else:
  cache.mkdir();marker.write_text(json.dumps(meta,indent=2))
 with mf.open(newline='') as h:rows=list(csv.DictReader(h))
 for i,r in enumerate(rows):
  out=cache/f'{i:06d}.npz'
  if out.exists():
   with np.load(out) as v:
    if any(v[k].shape!=(128,128) or not np.isfinite(v[k]).all() for k in ['flow_x','flow_y']):raise ValueError(f'Invalid {out}')
   continue
  def gray(name):
   with Image.open(Path(a.images)/name) as im:return np.asarray(im.convert('L').resize((128,128),Image.Resampling.BILINEAR),dtype=np.uint8)
  fl=cv2.calcOpticalFlowFarneback(gray(r['prev_frame']),gray(r['curr_frame']),None,**params)
  if not np.isfinite(fl).all():raise ValueError(f'Nonfinite flow at {sp}/{i}')
  tmp=cache/f'{i:06d}.tmp.npz';np.savez_compressed(tmp,flow_x=fl[:,:,0].astype(np.float16),flow_y=fl[:,:,1].astype(np.float16));os.replace(tmp,out)
  if (i+1)%1000==0:print(f'{sp}: {i+1}/{len(rows)}',flush=True)
 print(f'{sp}: cache ready for {len(rows)} pairs',flush=True)
