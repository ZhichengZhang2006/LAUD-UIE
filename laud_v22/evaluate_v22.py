"""Eight locked benchmark tracks. C60/U45 share UIEB; EUVP uses EUVP-s."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from data_v22 import (canonical,PAIRED,NR_SOURCE,read_manifest,Images256,sha256,enforce_test_route)
from model_v22 import build_model
from metrics_v22 import ProjectQuality
from train_v22 import atomic_json


def read_suite(path):
    path=Path(path).resolve();suite=json.loads(path.read_text(encoding='utf-8'))
    if set(suite['tests'])!=PAIRED|set(NR_SOURCE):
        raise ValueError('Suite needs UIEB,LSUI,UFO-120,EUVP-s,EUVP-d,C60,U45,EUVP')
    if set(suite['checkpoints'])!=PAIRED:raise ValueError('Provide exactly five domain checkpoints')
    for group in ('tests','checkpoints'):
        suite[group]={key:str((path.parent/value).resolve()) for key,value in suite[group].items()}
    suite['metric_repo']=str((path.parent/suite['metric_repo']).resolve())
    return suite


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',required=True);p.add_argument('--output',required=True)
    p.add_argument('--only',nargs='+',choices=sorted(PAIRED|set(NR_SOURCE)))
    p.add_argument('--save-images',action='store_true');p.add_argument('--scan-batch',type=int,choices=(1,2,4),default=1)
    args=p.parse_args()
    if not torch.cuda.is_available():p.error('CUDA is required for the project Mamba backbone')
    suite=read_suite(args.suite);out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    names=args.only or ['UIEB','LSUI','UFO-120','EUVP-s','EUVP-d','C60','U45','EUVP']
    quality=ProjectQuality(suite['metric_repo'],'cuda',need_nr=any(k in NR_SOURCE for k in names))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    results={};loaded=None;model=None;state=None
    for name in names:
        source=name if name in PAIRED else NR_SOURCE[name]
        checkpoint=Path(suite['checkpoints'][source])
        if loaded!=checkpoint:
            if model is not None:del model;torch.cuda.empty_cache()
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if state.get('status')!='selected' or state['config'].get('engineering_only'):
                raise ValueError('Use the selected best.pt from a completed formal run')
            if not (checkpoint.parent/'COMPLETE').exists():raise ValueError('Training is not complete')
            if state['config']['metric_protocol']!=quality.meta:
                # NR metadata adds weights; paired evaluator metadata is the common subset.
                common={k:quality.meta.get(k) for k in state['config']['metric_protocol']}
                if common!=state['config']['metric_protocol']:
                    raise ValueError('Validation/test metric protocol changed; do not silently mix scores')
            model=build_model(state['model_config']).cuda().eval().requires_grad_(False)
            model.load_state_dict(state['weights'],strict=True)
            model.set_inference_scan_batch(args.scan_batch);loaded=checkpoint
        enforce_test_route(state['domain'],name)
        manifest=Path(suite['tests'][name]);paired=name in PAIRED
        if paired and sha256(manifest)!=state['config']['manifest_sha256']['test']:
            raise ValueError(f'{name}: test manifest differs from frozen training protocol')
        if not paired:
            protected={canonical(x['name']):x for x in state['config'].get('extra_heldout',[])}
            if name not in protected or sha256(manifest)!=protected[name]['sha256']:
                raise ValueError(f'{name}: NR test inventory was not registered at training time')
        rows=read_manifest(manifest,paired,name)
        loader=DataLoader(Images256(rows,allow_scaled_pair=name=='UFO-120'),batch_size=1,num_workers=2)
        folder=out/name;folder.mkdir();records=[]
        for index,batch in enumerate(loader):
            image=model(batch['input'].cuda()).float().clamp(0,1)
            target=batch['target'].cuda() if paired else None
            values=quality.score(image,target);records.append({'id':batch['id'][0],**values})
            if args.save_images:
                array=image[0].permute(1,2,0).cpu().numpy()
                np.save(folder/f'{index:05d}.npy',array,allow_pickle=False)
                Image.fromarray(np.rint(array*255).clip(0,255).astype(np.uint8)).save(folder/f'{index:05d}.png')
        means={key:float(np.mean([r[key] for r in records])) for key in records[0] if key!='id'}
        report={'dataset':name,'training_domain':state['domain'],'n':len(records),'metrics':means,
                'checkpoint_sha256':sha256(checkpoint),'selected_epoch':state['epoch'],
                'manifest_sha256':sha256(manifest),'precision':'FP32','size':[256,256],
                'scan_batch':args.scan_batch,'metric_protocol':quality.meta}
        atomic_json(records,folder/'per_image.json');atomic_json(report,folder/'summary.json')
        results[name]=report;print(name,means,flush=True)
    atomic_json(results,out/'summary.json')


if __name__=='__main__':main()
