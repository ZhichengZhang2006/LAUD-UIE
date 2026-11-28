"""CPU correctness tests. Tiny backbone is ONLY a test double, not a UIE model."""
import csv,json,random,tempfile,unittest
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from model_v22 import ModelConfig,LAUD22,local_project,tiles,matvec
from losses_v22 import (LossConfig,ReferenceLoss,Medium,D4,sample_medium,composition_loss,
                        training_loss,geometry_image,correspondence_loss,normalized_lab,ssim)
from data_v22 import (Images256,EpochBatches,read_manifest,check_splits,enforce_test_route,canonical)
from train_v22 import EMA,atomic_torch,rng_state,restore_rng
from metrics_v22 import select_epoch


class OldHead(nn.Module):
    def __init__(self):
        super().__init__();self.centered=False
        self.spatial=nn.Conv2d(8,3,1,bias=False)
        self.scene=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(8,8,1),nn.SiLU(),nn.Conv2d(8,3,1))
        nn.init.normal_(self.spatial.weight,0,0.025)
        nn.init.normal_(self.scene[-1].weight,0,0.025);nn.init.zeros_(self.scene[-1].bias)
    def forward(self,x,f):return (x+self.scene(f)+self.spatial(f)).clamp(0,1)


class TinyFree(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.SiLU(),
            nn.Conv2d(8,8,3,padding=1),nn.SiLU());self.head=OldHead()
    def forward(self,x):return self.head(x,self.backbone(x))


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17);random.seed(17)
        self.x=0.1+0.8*torch.rand(2,3,32,32);self.y=0.95*self.x+0.025
    def model(self,mode='soft'):
        return LAUD22.from_free(TinyFree(),ModelConfig(mode=mode,channels=8,tile=8))

    def test_projection_is_idempotent(self):
        r=torch.randn_like(self.x);p,_=local_project(self.x,r,8);pp,_=local_project(self.x,p,8)
        torch.testing.assert_close(p,pp,rtol=2e-4,atol=2e-5)

    def test_projection_removes_local_affine_component(self):
        r=2*self.x+0.3
        p,_=local_project(self.x,r,8)
        torch.testing.assert_close(p,r,rtol=2e-4,atol=3e-5)

    def test_projection_orthogonality_of_hard_complement(self):
        r=torch.randn_like(self.x);p,_=local_project(self.x,r,8)
        basis=torch.cat((tiles(self.x,8),torch.ones_like(tiles(self.x,8)[:,:,:1])),2)
        dot=basis.transpose(1,2)@tiles(r-p,8)/64
        self.assertLess(float(dot.abs().max()),1e-4)

    def test_soft_projection_not_claimed_hard_orthogonal(self):
        r=torch.randn_like(self.x);p,_=local_project(self.x,r,8)
        soft=r-0.25*p;ps,_=local_project(self.x,soft,8)
        torch.testing.assert_close(ps,0.75*p,rtol=3e-4,atol=3e-5)
        self.assertGreater(float(ps.abs().mean()),1e-3)

    def test_projection_constant_input_and_backward(self):
        x=torch.ones_like(self.x)*0.4;r=torch.randn_like(x,requires_grad=True)
        p,rank=local_project(x,r,8)
        self.assertEqual(float(rank),1.0)
        expected=tiles(r,8).mean(1,keepdim=True).expand(-1,64,-1)
        torch.testing.assert_close(tiles(p,8),expected,rtol=1e-5,atol=1e-6)
        p.square().mean().backward();self.assertTrue(torch.isfinite(r.grad).all())

    def test_projection_rank_deficient_colors(self):
        channel=self.x[:,:1];x=channel.repeat(1,3,1,1);r=torch.randn_like(x)
        p,rank=local_project(x,r,8)
        self.assertEqual(float(rank),2.0);self.assertTrue(torch.isfinite(p).all())

    def test_projection_subspace_affine_invariance(self):
        medium=sample_medium(self.x);r=torch.randn_like(self.x)
        a,_=local_project(self.x,r,8);b,_=local_project(medium.apply(self.x),r,8)
        torch.testing.assert_close(a,b,rtol=3e-4,atol=3e-5)

    def test_all_head_modes_preserve_initial_free(self):
        old=TinyFree().eval()
        for mode in ('free','affine','soft'):
            m=LAUD22.from_free(old,ModelConfig(mode=mode,channels=8,tile=8)).eval()
            torch.testing.assert_close(m(self.x),old(self.x),rtol=0,atol=2e-7)

    def test_gate_bounds_and_gradient(self):
        m=self.model();m.set_gate_scale(1)
        out=m(self.x,aux=True);self.assertTrue(((out['alpha']>0)&(out['alpha']<1)).all())
        out['image'].square().mean().backward()
        self.assertIsNotNone(m.head.gate.weight.grad)
        self.assertTrue(torch.isfinite(m.head.gate.weight.grad).all())

    def test_head_fp32_under_autocast(self):
        m=self.model()
        with torch.autocast('cpu',dtype=torch.bfloat16):out=m(self.x,aux=True)
        self.assertEqual(out['image'].dtype,torch.float32)

    def test_added_head_parameter_counts(self):
        from model_v22 import PhotometricHead
        count=lambda mode:sum(p.numel() for p in PhotometricHead(ModelConfig(mode=mode)).parameters())
        self.assertEqual(count('affine')-count('free'),2652)
        self.assertEqual(count('soft')-count('free'),2685)

    def test_medium_range_and_composition(self):
        first,second=sample_medium(self.x),sample_medium(self.x)
        torch.testing.assert_close(second.apply(first.apply(self.x)),second.after(first).apply(self.x))
        y=first.apply(self.x);self.assertTrue(((y>=0)&(y<=1)).all())

    def test_coc_exact_oracle_and_column_order(self):
        d=sample_medium(self.x);b,_,h,w=self.x.shape
        a0=torch.rand(b,3,3,h,w)*0.2+torch.eye(3).view(1,3,3,1,1)
        b0=torch.rand_like(self.x)*0.05;r=torch.rand_like(self.x)*0.05
        a1=a0/d.a[:,None];b1=b0-matvec(a1,d.b.expand_as(self.x))
        old={'A':a0,'b':b0,'r':r};new={'A':a1,'b':b1,'r':r}
        self.assertLess(float(composition_loss(old,new,d)),1e-7)
        torch.testing.assert_close(matvec(a0,self.x)+b0+r,
                                   matvec(a1,d.apply(self.x))+b1+r,atol=1e-6,rtol=1e-5)

    def test_d4_inverse_all_shapes(self):
        for turns in range(4):
            for flip in (False,True):
                t=D4(turns,flip);a=torch.rand(2,3,3,32,40)
                self.assertTrue(torch.equal(a,t.inverse(t.apply(a))))

    def test_geometry_no_direct_photometric_grad(self):
        m=self.model();m.set_gate_scale(1);out=m(self.x,aux=True)
        g=geometry_image(out,True);torch.testing.assert_close(g,out['image'])
        g.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in m.head.scene.parameters()))
        self.assertTrue(all(p.grad is None for p in m.head.field.parameters()))
        self.assertIsNotNone(m.head.spatial.weight.grad)
        self.assertIsNotNone(m.head.gate.weight.grad)

    def test_correspondence_has_finite_grad(self):
        p=(self.y+0.04*torch.randn_like(self.y)).clamp(0,1).requires_grad_()
        q=self.y.clone().requires_grad_()
        loss,meta=correspondence_loss(p,q,self.y,torch.ones_like(self.y[:,:1]),anchors=8)
        loss.backward();self.assertGreater(float(meta['geo_support']),0)
        self.assertGreater(float(p.grad.abs().sum()),0);self.assertTrue(torch.isfinite(p.grad).all())

    def test_flat_reference_returns_zero(self):
        p=torch.full_like(self.x,0.5,requires_grad=True)
        loss,_=correspondence_loss(p,p,p.detach(),torch.ones_like(p[:,:1]),anchors=8)
        self.assertEqual(float(loss.detach()),0);loss.backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_zero_visibility_returns_zero(self):
        p=self.x.clone().requires_grad_()
        loss,_=correspondence_loss(p,self.y,self.y,torch.zeros_like(p[:,:1]),anchors=8)
        self.assertEqual(float(loss.detach()),0);loss.backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_full_objective_updates_model_and_ema(self):
        m=self.model();ema=EMA(m);opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
        cfg=LossConfig(aux_start_epoch=0,aux_ramp_epochs=1,anchors=8)
        ref=ReferenceLoss();before=m.head.field[-1].weight.detach().clone()
        loss,stats=training_loss(m,self.x,self.y,2,cfg,ref,amp=False)
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True)
        opt.step();ema.update(m)
        self.assertIn('coc',stats);self.assertIn('picd',stats)
        self.assertFalse(torch.equal(before,m.head.field[-1].weight))
        self.assertEqual(ema.updates,1)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in ema.model.parameters()))

    def test_base_affine_and_soft_backward(self):
        for mode in ('free','affine','soft'):
            m=self.model(mode);cfg=LossConfig(use_composition=False,use_geometry=False)
            loss,_=training_loss(m,self.x,self.y,2,cfg,ReferenceLoss(),False)
            self.assertTrue(torch.isfinite(loss));loss.backward()

    def test_geometry_preserves_shared_augmentation_rng(self):
        m=self.model();ref=ReferenceLoss();state=rng_state()
        training_loss(m,self.x,self.y,50,LossConfig(use_geometry=False),ref,False)
        expected=torch.rand(8)
        restore_rng(state)
        training_loss(m,self.x,self.y,50,LossConfig(use_geometry=True),ref,False)
        torch.testing.assert_close(torch.rand(8),expected,rtol=0,atol=0)

    def test_lab_against_skimage(self):
        from skimage.color import rgb2lab
        expected=rgb2lab(self.x[0].permute(1,2,0).numpy())/np.array([100,128,128])
        actual=normalized_lab(self.x)[0].permute(1,2,0).numpy()
        np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-4)

    def test_identical_ssim(self):
        torch.testing.assert_close(ssim(self.x,self.x),torch.ones(2),atol=1e-6,rtol=1e-6)

    def test_rng_checkpoint_roundtrip(self):
        m=self.model();ema=EMA(m);opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
        m(self.x).mean().backward();opt.step();ema.update(m)
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'last.pt'
            atomic_torch({'model':m.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),
                          'rng':rng_state()},file)
            expected=(torch.rand(5),random.random())
            saved=torch.load(file,weights_only=True,map_location='cpu');restore_rng(saved['rng'])
            torch.testing.assert_close(torch.rand(5),expected[0]);self.assertEqual(random.random(),expected[1])
            fresh=self.model();fresh.load_state_dict(saved['model'],strict=True)
            for a,b in zip(fresh.state_dict().values(),m.state_dict().values()):self.assertTrue(torch.equal(a,b))

    def test_epoch_sampler_visits_each_once(self):
        s=EpochBatches(25,4,42);s.set_epoch(5);first=list(s)
        self.assertEqual(sorted(i for batch in first for i,e in batch),list(range(25)))
        s2=EpochBatches(25,4,42);s2.set_epoch(5);self.assertEqual(first,list(s2))
        s2.set_epoch(6);self.assertNotEqual(first,list(s2))

    def test_dataset_scaled_ufo_and_leak_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            Image.fromarray(np.random.default_rng(1).integers(0,256,(24,32,3),dtype=np.uint8)).save(root/'x.png')
            Image.open(root/'x.png').resize((64,48)).save(root/'y.png')
            path=root/'train.csv';path.write_text('id,input,target\na,x.png,y.png\n')
            rows=read_manifest(path,True,'UFO-120')
            with self.assertRaises(ValueError):Images256(rows)[0]
            item=Images256(rows,allow_scaled_pair=True)[0]
            self.assertEqual(tuple(item['input'].shape),(3,256,256))
            with self.assertRaises(ValueError):check_splits({'train':rows,'test':rows})

    def test_checkpoint_routes(self):
        for train,test in [('UIEB','C60'),('UIEB','U45'),('EUVP-s','EUVP'),('UFO-120','UFO-120')]:
            enforce_test_route(train,test)
        with self.assertRaises(ValueError):enforce_test_route('LSUI','C60')
        with self.assertRaises(ValueError):enforce_test_route('EUVP-d','EUVP')
        self.assertEqual(canonical('EUVO-d'),'EUVP-d')

    def test_validation_selection_uses_psnr_constraint(self):
        def row(epoch,p,l):return {'epoch':epoch,'val':{'PSNR':p,'SSIM':0.9,'LPIPS':l,'E00':5,'UCIQE':0.4}}
        history=[row(100,28,0.15),row(105,27.95,0.12),row(110,27.5,0.01)]
        self.assertEqual(select_epoch(history,0.15)['epoch'],105)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
