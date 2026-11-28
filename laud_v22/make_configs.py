"""Generate explicit, editable recipes; no image lists or weights are fabricated."""
from __future__ import annotations
import argparse,json
from pathlib import Path

BUDGETS={'UIEB':500,'LSUI':400,'UFO-120':500,'EUVP-s':450,'EUVP-d':350}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--runs-root',required=True)
    p.add_argument('--metric-repo',required=True);p.add_argument('--output',default='configs')
    p.add_argument('--init-map',help='JSON mapping domains to explicit initialization dictionaries')
    args=p.parse_args();folder=Path(args.output).resolve();folder.mkdir(parents=True,exist_ok=True)
    data=Path(args.data_root).resolve();runs=Path(args.runs_root).resolve()
    initial=json.loads(Path(args.init_map).read_text()) if args.init_map else {}
    suite={'metric_repo':str(Path(args.metric_repo).resolve()),'tests':{},'checkpoints':{}}
    for domain,epochs in BUDGETS.items():
        extra=['C60','U45'] if domain=='UIEB' else ['EUVP'] if domain=='EUVP-s' else []
        init=initial.get(domain,{'kind':'scratch','source_note':'No pretrained weights'})
        pretrained=init['kind']!='scratch'
        cfg={'domain':domain,'arm':'ABC','size':256,'epochs':epochs,'seed':42,
             'train':str(data/domain/'train.csv'),'val':str(data/domain/'val.csv'),
             'test':str(data/domain/'test.csv'),'output':str(runs/domain/'ABC_s42'),
             'metric_repo':suite['metric_repo'],'initialization':init,
             'pretraining_manifests':[],
             'extra_heldout':[{'name':name,'manifest':str(data/name/'test.csv'),'paired':False} for name in extra],
             'backbone_lr':2e-5 if pretrained else 2e-4,'head_lr':2e-4,
             'micro_batch':4,'accumulation':2,'workers':4,'amp':True,'eval_every':5,
             'psnr_tolerance':0.15,'model':{'tile':16,'alpha_init':0.05},
             'loss':{'composition_weight':0.03,'geometry_weight':0.01,'lab_weight':0.02,
                     'lpips_weight':0.0,'mse_weight':0.0,'aux_start_epoch':20,'aux_ramp_epochs':30},
             'engineering_only':False}
        (folder/f'{domain}.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf-8')
        suite['tests'][domain]=cfg['test'];suite['checkpoints'][domain]=str(runs/domain/'ABC_s42'/'best.pt')
    for name in ('C60','U45','EUVP'):suite['tests'][name]=str(data/name/'test.csv')
    (folder/'suite.json').write_text(json.dumps(suite,indent=2),encoding='utf-8')
    print('Generated 5 domain recipes and suite.json in',folder)
    print('Inspect manifests and initialization provenance BEFORE training; scratch is the safe default.')


if __name__=='__main__':main()
