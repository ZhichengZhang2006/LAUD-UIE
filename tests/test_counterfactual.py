import copy
import torch
from luad_uie.models.student import Student
from luad_uie.losses import ReconstructionLoss
from luad_uie.trainers.counterfactual import ShadowAdamEvaluator, ACTIONS
from luad_uie.models.calibrated_actions import (CalibratedActionPredictor,choose_actions,
                                               ProbeScheduler,UtilityBuffer,lower_bound)


def test_shadow_matches_real_adam_with_history_without_mutating_model_or_optimizer():
    torch.set_num_threads(1);torch.manual_seed(10)
    m=Student(2);o=torch.optim.Adam(m.parameters(),lr=.001,betas=(.8,.95),weight_decay=.01,amsgrad=True)
    x,t=torch.rand(2,3,12,12),torch.rand(2,3,12,12)
    rec=ReconstructionLoss();o.zero_grad();rec(m(x),t).backward();o.step()
    real=copy.deepcopy(m);ro=torch.optim.Adam(real.parameters());ro.load_state_dict(copy.deepcopy(o.state_dict()))
    before=copy.deepcopy(m.state_dict());opt_before=copy.deepcopy(o.state_dict());rng=torch.random.get_rng_state().clone()
    teacher=t*.9;e=ShadowAdamEvaluator()
    virtual=e.step_copy(m,o,x,t,teacher,ACTIONS[3],1)
    ro.zero_grad();pred=real(x)
    loss=rec(pred,t)+.1*(e.response.views(pred[1:2],teacher[1:2])*ACTIONS[3]).sum()/2
    loss.backward();ro.step()
    for key,p in virtual.state_dict().items():torch.testing.assert_close(p,real.state_dict()[key],rtol=0,atol=0)
    for key,p in before.items():torch.testing.assert_close(p,m.state_dict()[key],rtol=0,atol=0)
    for key,value in opt_before['state'].items():
        for name,v in value.items():torch.testing.assert_close(v,o.state_dict()['state'][key][name],rtol=0,atol=0)
    assert torch.equal(rng,torch.random.get_rng_state())


def test_zero_kd_gains_and_predictor_isolation_and_budget():
    torch.manual_seed(5);m=Student(2);o=torch.optim.Adam(m.parameters())
    x=torch.rand(2,3,12,12);t=torch.rand_like(x)
    gains,_=ShadowAdamEvaluator(kd_weight=0).measure(m,o,x,t,t,[(x,t),(x,t)])
    torch.testing.assert_close(gains,torch.zeros_like(gains),rtol=0,atol=0)
    c=CalibratedActionPredictor(2);z=torch.rand(2,46,requires_grad=True)
    p=c(z);assert (p[...,0]<=p[...,1]).all() and (p[...,1]<=p[...,2]).all()
    c.loss(p,torch.zeros_like(p[...,0])).backward();assert z.grad is None
    assert choose_actions(p,risk=True).sum()==0
    bad=-torch.ones_like(p[...,0]);assert choose_actions(p,bad,True).sum()==0
    for active in (False,True):
        schedule=ProbeScheduler(active,100)
        for step in range(2000):schedule.due(step,True)
        assert schedule.count<=20
    b=UtilityBuffer();assert lower_bound(c,p,b,torch.ones(4),0) is None
