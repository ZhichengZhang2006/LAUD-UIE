import csv
import json
import numpy as np
from PIL import Image
from luad_uie.models.student import Student
from luad_uie.scripts.train_counterfactual import train
from luad_uie.scripts.evaluate import evaluate
from luad_uie.scripts.report_counterfactual import audit
import torch


def test_counterfactual_training_calibration_and_student_only_evaluation(tmp_path):
    torch.set_num_threads(1);data={};stats={}
    for name in ('train','reference','calibration','val','test','audit_candidates','audit_references'):
        path=tmp_path/(name+'.csv');data[name]=str(path)
        with path.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['id','input','target'])
            for i in range(8 if name=='train' else 4):
                image_id=name+str(i);stats[image_id]=[.1]*10
                image=np.random.default_rng(i).integers(0,255,(12,12,3),dtype=np.uint8)
                Image.fromarray(image).save(tmp_path/(image_id+'.png'))
                w.writerow([image_id,image_id+'.png',image_id+'.png'])
    meta=tmp_path/'meta.json';meta.write_text(json.dumps({'statistics':stats}))
    ckpt=tmp_path/'teacher.pt';torch.save(Student(2).state_dict(),ckpt)
    cfg={'seed':42,'device':'cpu','student':{'width':2},'teacher':{'mode':'checkpoint','width':2,'checkpoint':str(ckpt)},
         'data':data,'split_metadata':str(meta),'epochs':2,'batch_size':2,'crop_size':12,'lr':.0002,
         'ssim_weight':.2,'kd_weight':.1,'probe_interval':1,'experiment':'CF-periodic'}
    train(cfg,tmp_path/'run')
    state=torch.load(tmp_path/'run/last.pt',weights_only=True)
    assert state['step']==8 and len(state['calibration_buffer'])==1
    assert len(state['utility_buffer'])==7
    assert state['scale'].min()>0
    result=evaluate(tmp_path/'run/best.pt',data['test'],tmp_path/'eval','cpu')
    assert result['summary']['student_only'] and result['summary']['images']==4
    audit(tmp_path/'run',count=2)
    assert json.loads((tmp_path/'run/holdout_audit.json').read_text())['samples']==2
