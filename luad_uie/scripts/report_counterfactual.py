"""Write an evidence-only running report, including a fresh held-out utility audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ..datasets import read_manifest
from ..models import Student,KnowledgeViews
from ..models.teacher import build_teacher
from ..models.calibrated_actions import action_context,CalibratedActionPredictor,UtilityBuffer,lower_bound,choose_actions
from ..models.gradient_router import HeadGradientRouter
from ..trainers.counterfactual import ShadowAdamEvaluator
from .train_counterfactual import ReferenceSampler
from ..runtime import seed_all


def audit(run, count=32):
    state=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    cfg=state['config'];seed_all(142);device=torch.device(cfg['device'])
    model=Student(cfg['student']['width']).to(device);model.load_state_dict(state['student']);model.train()
    optimizer=torch.optim.Adam(model.parameters());optimizer.load_state_dict(state['optimizer'])
    teacher=build_teacher(cfg['teacher'],device)
    predictor=CalibratedActionPredictor(model.feature_dim).to(device);predictor.load_state_dict(state['predictor']);predictor.eval()
    calibration=UtilityBuffer(64);calibration.samples=state['calibration_buffer']
    scale=state['scale'].to(device)
    stats=json.loads(Path(cfg['split_metadata']).read_text())['statistics']
    def sampler(name,seed):
        rows=read_manifest(cfg['data'][name])
        for row in rows:row['statistics']=stats[row['id']]
        return ReferenceSampler(rows,cfg['crop_size'],seed)
    candidates=sampler('audit_candidates',142);refs=sampler('audit_references',242)
    evaluator=ShadowAdamEvaluator(cfg['kd_weight'],cfg['ssim_weight']);records=[]
    for i in range(min(count,len(candidates.dataset))):
        items=[candidates.get(i)]+[candidates.get() for _ in range(cfg['batch_size']-1)]
        x=torch.stack([b['input'] for b in items]).to(device);t=torch.stack([b['target'] for b in items]).to(device)
        with torch.no_grad():
            p,f=model(x,True);context=action_context(x,p,f,(state['epoch']+1)/cfg['epochs'],cfg['lr'])[:1]
            predicted=predictor(context)
            lower=lower_bound(predictor,predicted,calibration,scale,state['step'])
            action=choose_actions(predicted,lower,risk=cfg['experiment']!='CF-mean')[0].item()
            taught=teacher(x)
        reference,ids=refs.references(x[:1])
        gain,_=evaluator.measure(model,optimizer,x,t,taught,reference,0)
        # Baseline proxy: current same-image head cosine, on the identical candidate.
        p,f=model(x[:1],True);rec=evaluator.response.rec(p,t[:1]);views=KnowledgeViews()(p,taught[:1])
        _,cosine=HeadGradientRouter().to(device)(p,f['spatial'],rec,views,0)
        records.append({'id':items[0]['id'],'references':ids,'gain':gain.cpu().tolist(),
                        'prediction':predicted[0,...,1].cpu().tolist(),'cosine':cosine[0].cpu().tolist(),
                        'action':action,'selected_gain':float(gain[action-1,:,0].mean()) if action else 0.})
    predicted=np.array([r['prediction'] for r in records]);truth=np.array([r['gain'] for r in records])/scale.cpu().numpy()
    baseline=np.array([r['cosine'] for r in records]);single=truth[:,:3,:,0].mean(2)
    corr=lambda a,b:float(np.corrcoef(a.ravel(),b.ravel())[0,1]) if a.std()>0 and b.std()>0 else None
    selected=[r for r in records if r['action']]
    result={'samples':len(records),'status':'pilot_holdout_audit',
            'prediction_gain_pearson':corr(predicted[...,0],truth[...,0]),
            'cosine_gain_pearson_single_views':corr(baseline,single),
            'prediction_single_view_gain_pearson':corr(predicted[:,:3,:,0].mean(2),single),
            'normalized_mse':float(((predicted-truth)**2).mean()),'adoption_rate':len(selected)/len(records),
            'selected_negative_gain_fraction':sum(r['selected_gain']<0 for r in selected)/len(selected) if selected else None,
            'per_candidate':records,'limitations':['single seed','old teacher saw original train','short-horizon utility']}
    (run/'holdout_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='per_candidate'}),flush=True)


def report(runs):
    names=['CF-mean','E0','E1-detail','E2','E0-extra','CF-periodic','CF-active'];results={}
    for name in names:
        path=runs/(name+'_test')/'metrics.json'
        if path.exists():results[name]=json.loads(path.read_text())
    summary={'completed':list(results),'planned':names,'comparisons':{}}
    if 'E0' in results:
        baseline={r['id']:r['psnr'] for r in results['E0']['per_image']}
        for name,d in results.items():
            delta=np.array([r['psnr']-baseline[r['id']] for r in d['per_image']])
            rng=np.random.default_rng(42);ci=np.quantile(rng.choice(delta,(2000,len(delta))).mean(1),[.025,.975])
            summary['comparisons'][name]={'delta_psnr_E0':float(delta.mean()),'degraded_fraction':float((delta<0).mean()),'paired_image_ci95':ci.tolist()}
    (runs/'results_summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# CF pilot实验报告（自动更新）','','这是单种子工程试验；旧教师见过原训练参考图，未证明场景独立；不得据此宣称创新有效或SOTA。',
           '','| 方法 | PSNR | SSIM | 相对E0下降比例 |','|---|---:|---:|---:|']
    for name,d in results.items():
        s=d['summary'];fraction=summary['comparisons'].get(name,{}).get('degraded_fraction')
        lines.append(f"| {name} | {s['psnr']:.5f} | {s['ssim']:.6f} | {fraction if fraction is not None else '待匹配E0'} |")
    lines.extend(['','尚未完成：'+', '.join(n for n in names if n not in results),
                  '','效用审计详见各run/holdout_audit.json；时间、采用率和显存见epochs.jsonl。',
                  '','固定探测采用相同影子分支数量预算，实际时间必须另行比较。E0-extra额外访问参考与校准GT，不是严格相同计算量对照。',
                  '','多种子、外部测试、教师独立重训及既有工作对照尚未完成。'])
    (runs/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--audit');p.add_argument('--count',type=int,default=32)
    a=p.parse_args();runs=Path(a.runs)
    if a.audit:audit(runs/a.audit,a.count)
    report(runs)
