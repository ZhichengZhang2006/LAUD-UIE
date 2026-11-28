"""Fixed 256x256 data protocol and deterministic epoch-level augmentation."""
from __future__ import annotations
import csv,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
from torch.utils.data import Dataset,Sampler

ALIASES={'EUVO-d':'EUVP-d','CH60':'C60','UFO':'UFO-120'}
PAIRED={'UIEB','LSUI','UFO-120','EUVP-s','EUVP-d'}
NR_SOURCE={'C60':'UIEB','U45':'UIEB','EUVP':'EUVP-s'}


def canonical(name:str)->str:return ALIASES.get(name,name)


def sha256(path)->str:
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_manifest(path,paired=True,domain='')->list:
    path=Path(path).resolve()
    with path.open(encoding='utf-8-sig',newline='') as stream:source=list(csv.DictReader(stream))
    rows=[]
    for entry in source:
        if not entry.get('id') or not entry.get('input') or paired and not entry.get('target'):
            raise ValueError(f'{path}: expected id,input,target (target absent only for NR)')
        item={'id':entry['id'],'group':entry.get('group') or '', 'domain':canonical(domain)}
        for key in ('input','target'):
            item[key]=str((path.parent/entry[key]).resolve()) if entry.get(key) else None
            if item[key] and not Path(item[key]).is_file():raise FileNotFoundError(item[key])
        if not paired and item['target']:
            raise ValueError('NR test manifest should not supply reference labels')
        rows.append(item)
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Empty or duplicate-ID manifest')
    return rows


def check_splits(splits:dict,content_hash:bool=True):
    """Compare all roles once. Does not certify absence of semantic near-duplicates."""
    owners={};cache={}
    for role,rows in splits.items():
        for row in rows:
            tokens=[('id',row['domain'],row['id'])]
            if row.get('group'):tokens.append(('group',row['domain'],row['group']))
            for field in ('input','target'):
                if row.get(field):
                    path=row[field];tokens.append(('path',path))
                    if content_hash:
                        if path not in cache:cache[path]=sha256(path)
                        tokens.append(('bytes',cache[path]))
            for token in tokens:
                if token in owners and owners[token]!=role:
                    raise ValueError(f'Split overlap: {owners[token]} vs {role}: {token}')
                owners[token]=role
    return hashlib.sha256(json.dumps(sorted(cache.items())).encode()).hexdigest()


def load_rgb(path)->torch.Tensor:
    with Image.open(path) as image:
        array=np.asarray(image.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)).copy()
    return torch.from_numpy(array).permute(2,0,1).float()/255


class Images256(Dataset):
    def __init__(self,rows:list,augment=False,seed=42,allow_scaled_pair=False):
        self.rows,self.augment,self.seed,self.allow_scaled_pair=rows,augment,seed,allow_scaled_pair
    def __len__(self):return len(self.rows)
    def __getitem__(self,key):
        index,epoch=key if isinstance(key,tuple) else (key,0)
        row=self.rows[index];fields=['input']+(['target'] if row.get('target') else [])
        images=[]
        for field in fields:
            with Image.open(row[field]) as image:images.append(image.convert('RGB'))
        if len(images)==2 and images[0].size!=images[1].size:
            ratios=[im.width/im.height for im in images]
            if not self.allow_scaled_pair or abs(ratios[0]-ratios[1])>1e-4:
                raise ValueError(f'Unaligned native dimensions: {row["id"]}')
        images=[im.resize((256,256),Image.Resampling.BILINEAR) for im in images]
        if self.augment:
            rng=random.Random(self.seed+epoch*1000003+index*9176)
            if rng.random()<0.5:images=[ImageOps.mirror(im) for im in images]
            if rng.random()<0.5:images=[ImageOps.flip(im) for im in images]
        result={'id':row['id']}
        for field,im in zip(fields,images):
            result[field]=torch.from_numpy(np.asarray(im).transpose(2,0,1).copy()).float()/255
        return result


class EpochBatches(Sampler):
    """Every training image appears once per epoch, no replacement or hidden limit."""
    def __init__(self,size:int,batch:int,seed:int):
        self.size,self.batch,self.seed,self.epoch=size,batch,seed,1
    def set_epoch(self,epoch:int):self.epoch=epoch
    def __iter__(self):
        ids=list(range(self.size));random.Random(self.seed+self.epoch*1000003).shuffle(ids)
        for start in range(0,self.size,self.batch):
            yield [(idx,self.epoch) for idx in ids[start:start+self.batch]]
    def __len__(self):return (self.size+self.batch-1)//self.batch


def enforce_test_route(train_domain:str,test_domain:str):
    train_domain,test_domain=canonical(train_domain),canonical(test_domain)
    needed=test_domain if test_domain in PAIRED else NR_SOURCE.get(test_domain)
    if needed is None or train_domain!=needed:
        raise ValueError(f'{test_domain} requires {needed} checkpoint, not {train_domain}')
