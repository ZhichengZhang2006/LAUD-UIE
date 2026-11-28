"""Single-A100 epoch training. Core modules are tested separately on CPU."""
from __future__ import annotations
import argparse,copy,json,math,os,random,shutil,time
from dataclasses import asdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from model_v22 import build_model,ModelConfig
from losses_v22 import LossConfig,ReferenceLoss,training_loss
from data_v22 import (canonical,PAIRED,read_manifest,check_splits,Images256,
                      EpochBatches,sha256)
from metrics_v22 import ProjectQuality,select_epoch

ARMS={'M0':('free',False,False),'AFF':('affine',False,False),
      'A':('soft',False,False),'AB':('soft',True,False),
      'AC':('soft',False,True),'ABC':('soft',True,True),
      'A-output':('soft',False,False),'ABC-fullgrad':('soft',True,True)}


def atomic_torch(state,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(state,tmp);os.replace(tmp,path)


def atomic_json(obj,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def rng_state():
    return {'torch':torch.get_rng_state(),'python':random.getstate(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    torch.set_rng_state(state['torch']);random.setstate(state['python'])
    if state['cuda'] is not None:torch.cuda.set_rng_state_all(state['cuda'])


class EMA:
    def __init__(self,model,decay=0.999):
        self.model=copy.deepcopy(model).eval().requires_grad_(False);self.decay=decay;self.updates=0
    @torch.no_grad()
    def update(self,model):
        self.updates+=1;decay=min(self.decay,(1+self.updates)/(10+self.updates))
        source=dict(model.named_parameters())
        for name,p in self.model.named_parameters():p.lerp_(source[name].detach(),1-decay)
        source=dict(model.named_buffers())
        for name,b in self.model.named_buffers():b.copy_(source[name].detach())
        self.model.eval()
    def state_dict(self):return {'model':self.model.state_dict(),'decay':self.decay,'updates':self.updates}
    def load_state_dict(self,state):
        self.model.load_state_dict(state['model'],strict=True)
        self.decay,self.updates=state['decay'],state['updates']
        self.model.eval().requires_grad_(False)


def set_lr(optimizer,progress,total,warmup=5):
    if progress<warmup:factor=max(1e-3,progress/max(1,warmup))
    else:
        t=min(1.0,(progress-warmup)/max(1,total-warmup))
        factor=0.05+0.95*(1+math.cos(math.pi*t))/2
    for group in optimizer.param_groups:group['lr']=group['base_lr']*factor


@torch.inference_mode()
def validate(model,loader,quality):
    model.eval();rows=[]
    # Validation/scoring FP32 for every arm, regardless of training AMP.
    for data in loader:
        x,y=data['input'].cuda(),data['target'].cuda()
        rows.append(quality.score(model(x).float(),y))
    return {key:sum(row[key] for row in rows)/len(rows) for key in rows[0]}


def resolve_config(config_path,args):
    path=Path(config_path).resolve();cfg=json.loads(path.read_text(encoding='utf-8'))
    for key in ('arm','epochs','seed','output'):
        if getattr(args,key,None) is not None:cfg[key]=getattr(args,key)
    cfg['domain']=canonical(cfg['domain'])
    if cfg['domain'] not in PAIRED:raise ValueError('Train only a paired-domain specialist')
    cfg.setdefault('model',{});cfg.setdefault('loss',{})
    mode,coc,geo=ARMS[cfg['arm']]
    cfg['model']['mode']=mode
    cfg['loss'].update(use_composition=coc,use_geometry=geo,
                      output_consistency=cfg['arm']=='A-output',
                      isolate_photo=cfg['arm']!='ABC-fullgrad')
    cfg['model']=asdict(ModelConfig(**cfg['model']));cfg['loss']=asdict(LossConfig(**cfg['loss']))
    if cfg.get('size',256)!=256:raise ValueError('This protocol fixes ALL input images at 256x256')
    for key in ('train','val','test','output','metric_repo'):
        cfg[key]=str((path.parent/cfg[key]).resolve())
    for entry in cfg.get('extra_heldout',[]):
        entry['manifest']=str((path.parent/entry['manifest']).resolve())
        entry['sha256']=sha256(entry['manifest'])
    for entry in cfg.get('pretraining_manifests',[]):
        entry['manifest']=str((path.parent/entry['manifest']).resolve())
        entry['sha256']=sha256(entry['manifest'])
    if cfg['initialization']['kind']!='scratch':
        cfg['initialization']['path']=str((path.parent/cfg['initialization']['path']).resolve())
    cfg.setdefault('eval_every',5);cfg.setdefault('micro_batch',4);cfg.setdefault('accumulation',2)
    cfg.setdefault('workers',4);cfg.setdefault('amp',True);cfg.setdefault('psnr_tolerance',0.15)
    cfg.setdefault('engineering_only',False)
    if cfg['epochs']<1 or min(cfg['micro_batch'],cfg['accumulation'],cfg['eval_every'])<1:
        raise ValueError('Invalid training budget')
    cfg['source_sha256']={name:sha256(Path(__file__).parent/name) for name in
        ('model_v22.py','losses_v22.py','data_v22.py','metrics_v22.py','train_v22.py')}
    cfg['manifest_sha256']={key:sha256(cfg[key]) for key in ('train','val','test')}
    if cfg['initialization']['kind']!='scratch':
        cfg['initialization']['sha256']=sha256(cfg['initialization']['path'])
    return cfg


def initialize(cfg):
    init=cfg['initialization']
    if init['kind'] in ('v22','v22_free'):
        state=torch.load(init['path'],map_location='cpu',weights_only=True)
        if init['kind']=='v22':
            if state['model_config']!=cfg['model']:raise ValueError('New-phase model configuration differs')
            model=build_model(cfg['model']);model.load_state_dict(state['weights'],strict=True)
        else:
            if state['model_config']['mode']!='free':raise ValueError('v22_free requires a Free checkpoint')
            from model_v22 import LAUD22
            old=build_model(state['model_config']);old.load_state_dict(state['weights'],strict=True)
            model=LAUD22.from_free(old,ModelConfig(**cfg['model']))
        return model
    return build_model(cfg['model'],init)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--epochs',type=int)
    parser.add_argument('--seed',type=int);parser.add_argument('--output')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not torch.cuda.is_available():parser.error('A CUDA GPU is required; use tests_v22.py for CPU tests')
    cfg=resolve_config(args.config,args);out=Path(cfg['output'])
    train=read_manifest(cfg['train'],True,cfg['domain']);val=read_manifest(cfg['val'],True,cfg['domain'])
    test=read_manifest(cfg['test'],True,cfg['domain'])
    prior=[]
    for entry in cfg.get('pretraining_manifests',[]):
        prior.extend(read_manifest(entry['manifest'],True,entry['domain']))
    splits={'train':train+prior,'val':val,'test':test}
    for entry in cfg.get('extra_heldout',[]):
        splits['heldout:'+entry['name']]=read_manifest(entry['manifest'],entry.get('paired',False),entry['name'])
    cfg['registered_files_sha256']=check_splits(splits,content_hash=True)
    # Checks current/prior registered files, not unseen training history of an opaque weight file.
    random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=initialize(cfg).cuda();ema=EMA(model)
    optimizer=torch.optim.AdamW([
        {'params':model.backbone.parameters(),'lr':cfg['backbone_lr'],'base_lr':cfg['backbone_lr']},
        {'params':model.head.parameters(),'lr':cfg['head_lr'],'base_lr':cfg['head_lr']}],
        weight_decay=1e-4,betas=(0.9,0.999))
    loss_cfg=LossConfig(**cfg['loss'])
    reference=ReferenceLoss(loss_cfg.lab_weight,loss_cfg.mse_weight,loss_cfg.lpips_weight).cuda()
    quality=ProjectQuality(cfg['metric_repo'],device='cuda')
    cfg['metric_protocol']=quality.meta
    start=1;history=[]
    if args.resume:
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        if state['config']!=cfg:raise ValueError('Resume requires identical configuration/data/source/metrics')
        model.load_state_dict(state['model'],strict=True);ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        start=state['epoch']+1;history=state['history']
        train_log=state['train_log']
    else:
        out.mkdir(parents=True,exist_ok=False);(out/'candidates').mkdir()
        train_log=[];atomic_json(cfg,out/'config.json')
        # Construction of extra heads must not change augmentation random streams.
        random.seed(cfg['seed']+100);torch.manual_seed(cfg['seed']+100);torch.cuda.manual_seed_all(cfg['seed']+100)
    atomic_json(train_log,out/'train_history.json');atomic_json(history,out/'validation_history.json')
    scaled=cfg['domain']=='UFO-120'
    sampler=EpochBatches(len(train),cfg['micro_batch'],cfg['seed'])
    train_loader=DataLoader(Images256(train,True,cfg['seed'],scaled),batch_sampler=sampler,
        num_workers=cfg['workers'],pin_memory=True,persistent_workers=cfg['workers']>0,
        generator=torch.Generator().manual_seed(cfg['seed']+1000))
    val_loader=DataLoader(Images256(val,False,allow_scaled_pair=scaled),batch_size=1,
        num_workers=cfg['workers'],pin_memory=True,
        generator=torch.Generator().manual_seed(cfg['seed']+2000))
    print(f"{cfg['domain']} {cfg['arm']} starting epoch {start}/{cfg['epochs']} "
          f"with {len(train)} training pairs and {len(val)} validation pairs",flush=True)
    for epoch in range(start,cfg['epochs']+1):
        sampler.set_epoch(epoch);model.train();totals={};seen=0;micro_done=0
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter()
        iterator=iter(train_loader)
        while True:
            batches=[]
            for _ in range(cfg['accumulation']):
                try:batches.append(next(iterator))
                except StopIteration:break
            if not batches:break
            group_n=sum(len(b['input']) for b in batches)
            set_lr(optimizer,(epoch-1)+micro_done/max(1,len(train_loader)),cfg['epochs'])
            optimizer.zero_grad(set_to_none=True)
            for batch in batches:
                x,y=batch['input'].cuda(non_blocking=True),batch['target'].cuda(non_blocking=True)
                loss,stats=training_loss(model,x,y,epoch,loss_cfg,reference,cfg['amp'])
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss; update aborted')
                # Correct scaling also for a partial final accumulation group.
                (loss*(len(x)/group_n)).backward()
                for key,value in stats.items():totals[key]=totals.get(key,0.0)+float(value.detach())*len(x)
                seen+=len(x);micro_done+=1
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step();ema.update(model)
            if micro_done == cfg['accumulation'] or micro_done % (50*cfg['accumulation']) == 0:
                print(f'epoch {epoch}: {seen}/{len(train)} pairs, '
                      f'{time.perf_counter()-begin:.1f}s elapsed',flush=True)
        torch.cuda.synchronize()
        row={'epoch':epoch,**{k:v/seen for k,v in totals.items()},
             'train_seconds':time.perf_counter()-begin,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        train_log.append(row)
        if epoch%cfg['eval_every']==0 or epoch==cfg['epochs']:
            scores=validate(ema.model,val_loader,quality)
            name=f'candidates/epoch_{epoch:04d}.pt'
            candidate={'format':'LAUD22','status':'candidate','domain':cfg['domain'],
                'model_config':cfg['model'],'weights':ema.model.state_dict(),'epoch':epoch,'val':scores,
                'config':cfg,'counts':{'train':len(train),'val':len(val),'test':len(test)}}
            atomic_torch(candidate,out/name)
            history.append({'epoch':epoch,'val':scores,'path':name})
        state={'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
               'epoch':epoch,'config':cfg,'rng':rng_state(),'history':history,'train_log':train_log}
        atomic_torch(state,out/'last.pt')
        atomic_json(history,out/'validation_history.json');atomic_json(train_log,out/'train_history.json')
        print(json.dumps(row),flush=True)
    # Exclude early warm-up candidates from claiming that active modules improved quality.
    eligible=[r for r in history if r['epoch']>=loss_cfg.aux_start_epoch+loss_cfg.aux_ramp_epochs]
    if not eligible:
        if not cfg['engineering_only']:raise ValueError('No fully activated candidate; increase epoch budget')
        eligible=history
    chosen=select_epoch(eligible,cfg['psnr_tolerance'])
    selected=torch.load(out/chosen['path'],map_location='cpu',weights_only=True)
    selected['status']='engineering' if cfg['engineering_only'] else 'selected'
    selected['selection']={'rule':'fixed validation score within PSNR tolerance',
                           'tolerance':cfg['psnr_tolerance'],'record':chosen}
    atomic_torch(selected,out/'best.pt')
    best_p=max(eligible,key=lambda r:r['val']['PSNR'])
    shutil.copyfile(out/best_p['path'],out/'best_psnr_secondary.pt')
    atomic_json(selected['selection'],out/'selection.json')
    (out/'COMPLETE').write_text(str(cfg['epochs']),encoding='utf-8')


if __name__=='__main__':main()
