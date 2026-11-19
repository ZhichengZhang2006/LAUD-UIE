"""Pilot CF training; old teacher saw reference data, explicitly recorded in provenance."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from ..datasets import read_manifest, validate_splits, PairedDataset
from ..models import Student, KnowledgeViews
from ..models.teacher import build_teacher
from ..models.degradation import image_statistics
from ..models.calibrated_actions import (action_context, CalibratedActionPredictor,
    UtilityBuffer, lower_bound, choose_actions, ProbeScheduler)
from ..trainers.counterfactual import ShadowAdamEvaluator, ACTIONS
from ..trainers.engine import validate
from ..losses import ReconstructionLoss
from ..runtime import seed_all, append_json, file_hash


class ReferenceSampler:
    def __init__(self, rows, crop_size, seed):
        self.dataset = PairedDataset(rows, crop_size, False)
        self.rng = random.Random(seed)
        # Precomputed input-only 10D thumbnail statistics; no GT defines neighborhoods.
        self.stats = np.array([r['statistics'] for r in rows])
        self.std = self.stats.std(0).clip(.001)

    def get(self, index=None):
        outer = random.getstate()
        try:
            random.setstate(self.rng.getstate())
            batch = self.dataset[self.rng.randrange(len(self.dataset)) if index is None else index]
            self.rng.setstate(random.getstate())
        finally:
            random.setstate(outer)
        return batch

    def references(self, x):
        stats = image_statistics(x).detach().cpu().numpy()[0]
        nearest = np.argsort(((self.stats-stats)/self.std)**2 @ np.ones(10))[:min(32,len(self.dataset))].tolist()
        local = self.rng.sample(nearest, min(2,len(nearest)))
        available = [i for i in range(len(self.dataset)) if i not in local]
        global_ids = self.rng.sample(available, min(2,len(available)))
        if len(local) != 2 or len(global_ids) != 2:
            raise ValueError('At least four reference images required')
        result = []
        for ids in (local, global_ids):
            batches = [self.get(i) for i in ids]
            result.append((torch.stack([b['input'] for b in batches]).to(x.device),
                           torch.stack([b['target'] for b in batches]).to(x.device)))
        return result, [self.dataset.rows[i]['id'] for i in local+global_ids]


def train(config, output):
    seed_all(config['seed'])
    device = torch.device(config.get('device','cuda'))
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    (out/'config.yaml').write_text(yaml.safe_dump(config))
    rows = {k:read_manifest(v) for k,v in config['data'].items() if v}
    validate_splits(rows)
    metadata = json.loads(Path(config['split_metadata']).read_text())
    stats = metadata['statistics']
    for name in ('reference','calibration'):
        for row in rows[name]: row['statistics'] = stats[row['id']]
    (out/'provenance.json').write_text(json.dumps({
        'manifest_sha256':{k:file_hash(v) for k,v in config['data'].items() if v},
        'teacher_sha256':file_hash(config['teacher']['checkpoint']),
        'teacher_saw_reference_data':True,'scene_independence_verified':False,
        'status':'pilot','torch':torch.__version__, 'split_metadata_sha256':file_hash(config['split_metadata'])},indent=2))
    generator = torch.Generator().manual_seed(config['seed'])
    loader = DataLoader(PairedDataset(rows['train'],config['crop_size'],True), batch_size=config['batch_size'],
                        shuffle=True, num_workers=0, generator=generator)
    val = DataLoader(PairedDataset(rows['val']),batch_size=1)
    model = Student(config['student']['width']).to(device)
    teacher = build_teacher(config['teacher'],device)
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])
    predictor = CalibratedActionPredictor(model.feature_dim).to(device)
    pred_optimizer = torch.optim.Adam(predictor.parameters(),lr=.001)
    evaluator = ShadowAdamEvaluator(config['kd_weight'], config['ssim_weight'])
    views, rec_loss = KnowledgeViews().to(device), ReconstructionLoss(config['ssim_weight']).to(device)
    sampler = ReferenceSampler(rows['reference'],config['crop_size'],config['seed']+100)
    calibration_candidates = ReferenceSampler(rows['calibration'],config['crop_size'],config['seed']+200)
    buffer, calibration = UtilityBuffer(), UtilityBuffer(64)
    scheduler = ProbeScheduler(config['experiment']=='CF-active',config.get('probe_interval',100))
    risk = config['experiment'] != 'CF-mean'
    step, best, scale = 0, -float('inf'), None
    rng = random.Random(config['seed']+300)
    actions = torch.cat([torch.zeros(1,3),ACTIONS]).to(device)
    for epoch in range(config['epochs']):
        started = time.perf_counter(); model.train()
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
        totals = {'rec':0.,'kd':0.,'selected':0.,'probe_seconds':0.,'probes':0}
        count=0
        for batch in loader:
            x,t = batch['input'].to(device),batch['target'].to(device)
            with torch.no_grad():
                state_output, features = model(x,True)
                context = action_context(x,state_output,features,epoch/config['epochs'],config['lr'])
                predicted = predictor(context)
                lower = lower_bound(predictor,predicted,calibration,scale,step) if risk and scale is not None else None
                uncertain = lower is None or bool(((predicted[...,0] < 0)&(predicted[...,2] > 0)).any())
            if scheduler.due(step,uncertain):
                probe_started = time.perf_counter()
                is_calibration = scheduler.count % 5 == 0
                if is_calibration:
                    cb = calibration_candidates.get()
                    px,pt=x.clone(),t.clone()
                    candidate, candidate_id = rng.randrange(len(x)),cb['id']
                    px[candidate]=cb['input'].to(device);pt[candidate]=cb['target'].to(device)
                    with torch.no_grad():
                        po,pf = model(px,True)
                        pc = action_context(px,po,pf,epoch/config['epochs'],config['lr'])[candidate:candidate+1]
                else:
                    px,pt = x,t
                    candidate = rng.randrange(len(x)); candidate_id=batch['id'][candidate]
                    pc = context[candidate:candidate+1]
                references,ref_ids = sampler.references(px[candidate:candidate+1])
                with torch.no_grad(): teacher_output=teacher(px)
                gains,base = evaluator.measure(model,optimizer,px,pt,teacher_output,references,candidate)
                if scale is None: scale=base.mean(0).clamp_min(1e-5)
                target = gains/scale
                with torch.no_grad(): before=predictor(pc)
                (calibration if is_calibration else buffer).add(pc,gains,step)
                if not is_calibration:
                    samples = buffer.recent(step)
                    chosen = rng.sample(samples,min(16,len(samples)))
                    cx=torch.cat([a for a,_,_ in chosen]).to(device)
                    cy=torch.stack([b for _,b,_ in chosen]).to(device)/scale
                    pred_optimizer.zero_grad(set_to_none=True)
                    if risk:
                        loss=predictor.loss(predictor(cx),cy)
                    else:
                        loss=(predictor(cx)[...,1]-cy).square().mean()
                    loss.backward();pred_optimizer.step()
                if x.is_cuda: torch.cuda.synchronize()
                seconds=time.perf_counter()-probe_started
                totals['probe_seconds']+=seconds;totals['probes']+=1
                append_json(out/'utility.jsonl',{'step':step,'candidate_id':candidate_id,'reference_ids':ref_ids,
                    'calibration':is_calibration,'gain':gains.cpu().tolist(),
                    'prediction_before_update':before[0,...,1].cpu().tolist(),
                    'mse_before_update':(before[0,...,1]-target).square().mean().item(),
                    'probe_seconds':seconds})
            selected=choose_actions(predicted,lower,risk,warmup=epoch==0)
            prediction=model(x)
            rec=rec_loss(prediction,t)
            with torch.no_grad(): taught=teacher(x)
            kd=(views(prediction,taught)*actions[selected]).sum(1).mean()
            loss=rec+config['kd_weight']*kd
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            n=len(x);count+=n
            totals['rec']+=rec.item()*n;totals['kd']+=kd.item()*n
            totals['selected']+=(selected>0).sum().item()
            append_json(out/'steps.jsonl',{'step':step,'ids':batch['id'],'actions':selected.cpu().tolist()})
            step+=1
        scores=validate(model,val,device)
        elapsed=time.perf_counter()-started
        row={'epoch':epoch,'step':step,'val':scores,'rec':totals['rec']/count,'kd':totals['kd']/count,
             'adoption_rate':totals['selected']/count,'epoch_seconds':elapsed,
             'probe_seconds':totals['probe_seconds'],'probes':totals['probes'],
             'peak_memory_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None}
        append_json(out/'epochs.jsonl',row);print(json.dumps(row),flush=True)
        state={'student':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,'epoch':epoch,'step':step,
               'predictor':predictor.state_dict(),'predictor_optimizer':pred_optimizer.state_dict(),
               'utility_buffer':buffer.samples,'calibration_buffer':calibration.samples,'scale':scale,
               'scheduler':vars(scheduler),'python_rng':random.getstate(),'sampler_rng':sampler.rng.getstate(),
               'calibration_rng':calibration_candidates.rng.getstate(),'probe_rng':rng.getstate(),
               'torch_rng':torch.random.get_rng_state(),'loader_rng':generator.get_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        torch.save(state,out/'last.pt')
        if scores['psnr']>best:best=scores['psnr'];torch.save(state,out/'best.pt')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();config=yaml.safe_load(Path(args.config).read_text())
    try: train(config,args.output)
    except RuntimeError as error:
        if 'out of memory' in str(error).lower():
            out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
            (out/'STOPPED_OOM.json').write_text(json.dumps({'error':str(error),'retry':False}))
            raise SystemExit(42)
        raise


if __name__=='__main__':main()
