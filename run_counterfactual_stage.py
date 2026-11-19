"""Dedicated school-gpu2 pilot; any error/OOM stops the queue without retry."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml


def main():
    stage=Path(__file__).resolve().parent;os.chdir(stage)
    runs=stage/'runs';runs.mkdir(exist_ok=True)
    lock=(runs/'counterfactual.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=Path('/home/user/luad_uie_prototype_20261031')
    data=stage/'data/CF_LSUI'
    status={'status':'running','task':'prepare','automatic_retry':False}
    def save(): (runs/'counterfactual_status.json').write_text(json.dumps(status,indent=2))
    def run(args):
        status['command']=args;save();subprocess.run([sys.executable,'-u']+args,check=True)
    try:
        run(['-m','luad_uie.scripts.prepare_counterfactual','--root',str(root/'data/LSUI'),'--output',str(data)])
        cfg={'seed':42,'device':'cuda','epochs':100,'batch_size':4,'crop_size':256,'lr':.0002,
             'ssim_weight':.2,'kd_weight':.1,'student':{'width':24},'clip':False,
             'teacher':{'mode':'checkpoint','width':48,'checkpoint':str(root/'runs/LSUI_teacher_seed42/best.pt')},
             'data':{k:str(data/(k+'.csv')) for k in ['train','reference','calibration','audit_candidates','audit_references','val','test']},
             'split_metadata':str(data/'metadata.json'),'probe_interval':100}
        for name in ['CF-mean','E0','E1-detail','E2','E0-extra','CF-periodic','CF-active']:
            status['task']=name
            c={**cfg,'data':dict(cfg['data'])}
            cf=name.startswith('CF-')
            c['experiment']=name if cf else ('E1' if name=='E1-detail' else 'E0' if name.startswith('E0') else 'E2')
            if not cf:
                c['data']={k:v for k,v in c['data'].items() if k in ('train','val','test')}
                if name.startswith('E0'):c['teacher']={'mode':'none'};c['kd_weight']=0
                if name=='E1-detail':c['view']='detail'
                if name=='E0-extra':
                    extra=data/'train_extra.csv'
                    import csv
                    with extra.open('w',newline='') as f:
                        w=csv.writer(f);w.writerow(['id','input','target'])
                        for source in ('train','reference','calibration'):
                            with (data/(source+'.csv')).open() as g:
                                reader=csv.reader(g);next(reader);w.writerows(reader)
                    c['data']['train']=str(extra)
            path=stage/'configs'/('CFpilot_'+name+'.yaml');path.write_text(yaml.safe_dump(c))
            module='luad_uie.scripts.train_counterfactual' if cf else 'luad_uie.scripts.safe_train'
            run(['-m',module,'--config',str(path),'--output',str(runs/name)])
            # Direct E0 comparisons for CF-mean are added once the matched E0 is available.
            run(['-m','luad_uie.scripts.evaluate','--checkpoint',str(runs/name/'best.pt'),
                 '--manifest',str(data/'test.csv'),'--output',str(runs/(name+'_test'))])
            args=['-m','luad_uie.scripts.report_counterfactual','--runs',str(runs)]
            if cf:args+=['--audit',name]
            run(args)
        status['status']='completed'
    except BaseException as error:
        status['status']='stopped';status['error']=repr(error);raise
    finally:save()


if __name__=='__main__':main()
