import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
from ..datasets import read_manifest, validate_splits


def prepare(root, output):
    root, output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    splits={k:read_manifest(root/(k+'.csv')) for k in ('train','val','test')}
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    test_hashes={digest(r[k]) for r in splits['test'] for k in ('input','target')}
    omitted=[]
    clean_val=[]
    for row in splits['val']:
        if any(digest(row[k]) in test_hashes for k in ('input','target')):
            omitted.append({'split':'val','id':row['id'],'reason':'exact file content overlaps test'})
        else:clean_val.append(row)
    splits['val']=clean_val
    heldout=test_hashes|{digest(r[k]) for r in clean_val for k in ('input','target')}
    records=[]
    for row in splits['train']:
        if any(digest(row[k]) in heldout for k in ('input','target')):
            omitted.append({'split':'train','id':row['id'],'reason':'exact file content overlaps val/test'})
        else:records.append(row)
    parents=list(range(len(records)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(i,j):parents[find(i)]=find(j)
    hashes={}; dhashes=[];statistics={}
    for i,row in enumerate(records):
        for key in ('input','target'):
            h=digest(row[key])
            if h in heldout:raise ValueError('Exact content crosses train and val/test; repair split before training')
            if h in hashes:union(i,hashes[h])
            hashes[h]=i
        with Image.open(row['input']) as im:
            rgb=np.asarray(im.convert('RGB').resize((32,32)),dtype=np.float32)/255
            thumb=np.asarray(im.convert('L').resize((9,8)))
        y=rgb@np.array([.2126,.7152,.0722])
        statistics[row['id']]=np.concatenate([rgb.mean((0,1)),rgb.std((0,1)),
            [y.mean(),y.std(),np.abs(np.diff(y,axis=1)).mean(),np.abs(np.diff(y,axis=0)).mean()]]).tolist()
        bits=(thumb[:,1:]>thumb[:,:-1]).flatten()
        dh=int(''.join('1' if v else '0' for v in bits),2)
        # Conservative near-duplicate grouping, not a complete scene identity audit.
        for j,other in enumerate(dhashes):
            if bin(dh^other).count('1')<=4:union(i,j)
        dhashes.append(dh)
    groups={}
    for i,row in enumerate(records):groups.setdefault(find(i),[]).append(row)
    items=list(groups.values());random.Random(42).shuffle(items);items.sort(key=len,reverse=True)
    names=['train','reference','calibration','audit_candidates','audit_references']
    ratio=[.8,.1,.05,.025,.025];allocated={k:[] for k in names}
    for group in items:
        name=min(names,key=lambda k:len(allocated[k])/(len(records)*ratio[names.index(k)]))
        allocated[name].extend(group)
    allocated.update(val=splits['val'],test=splits['test'])
    validate_splits(allocated)
    for name,rows in allocated.items():
        with (output/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['id','input','target']);writer.writeheader();writer.writerows(rows)
    info={'counts':{k:len(v) for k,v in allocated.items()},'groups':len(groups),
          'grouping':'input/target SHA256 equality or input dHash hamming<=4 connected components',
          'scene_independence_verified':False,'teacher_saw_reference_data':False,'omitted_cross_split_duplicates':omitted,
          'statistics':statistics,'group_ids':{r['id']:str(find(i)) for i,r in enumerate(records)}}
    (output/'metadata.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info['counts']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a.root,a.output)
