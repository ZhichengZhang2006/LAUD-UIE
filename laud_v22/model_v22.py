"""LAUD-UIE 2.2. RGB BCHW in [0,1]; real runs use 256x256.
A=adaptive soft local photometric projection, not hard orthogonality.
"""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass, asdict
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def tiles(x: Tensor, k: int) -> Tensor:
    b,c,h,w=x.shape
    if h%k or w%k:
        raise ValueError('Height and width must be divisible by projection tile')
    return x.reshape(b,c,h//k,k,w//k,k).permute(0,2,4,3,5,1).reshape(-1,k*k,c)


def untile(x: Tensor, shape: tuple, k: int) -> Tensor:
    b,c,h,w=shape
    return x.reshape(b,h//k,w//k,k,k,c).permute(0,5,1,3,2,4).reshape(shape)


def local_project(image: Tensor, residual: Tensor, tile: int=16,
                  rtol: float=1e-5) -> tuple[Tensor, Tensor]:
    """P_X R via 4x4 Gram pseudoinverses; never forms an N-by-N projector.
    Basis is detached. Backprop is linear in residual; eigensolver is not in
    the autograd graph. Finite precision uses a rank cutoff, not exact algebra.
    """
    if image.shape != residual.shape or image.shape[1]!=3:
        raise ValueError('Expected matching RGB tensors')
    with torch.autocast(device_type=image.device.type, enabled=False):
        with torch.no_grad():
            x=tiles(image.detach().float(),tile)
            centered=x-x.mean(1,keepdim=True)
            scale=centered.square().mean(1,keepdim=True).sqrt().clamp_min(1e-4)
            phi=torch.cat((centered/scale,torch.ones_like(x[:,:,:1])),dim=2)
            n=phi.shape[1]
            gram=phi.transpose(1,2)@phi/n
            eig,vec=torch.linalg.eigh(gram)
            keep=eig>rtol*eig.amax(-1,keepdim=True).clamp_min(1e-8)
            inv=torch.where(keep,eig.clamp_min(1e-8).reciprocal(),torch.zeros_like(eig))
            inverse=(vec*inv[:,None,:])@vec.transpose(1,2)
        r=tiles(residual.float(),tile)
        coef=inverse@(phi.transpose(1,2)@r/n)
        projection=untile(phi@coef,residual.shape,tile)
        return projection,keep.sum(-1).float().mean()


def matvec(a:Tensor,x:Tensor)->Tensor:
    return torch.einsum('bijhw,bjhw->bihw',a.float(),x.float())


def total_variation(x:Tensor)->Tensor:
    zero=x.sum()*0
    dy=(x[...,1:,:]-x[...,:-1,:]).abs().mean() if x.shape[-2]>1 else zero
    dx=(x[...,:,1:]-x[...,:,:-1]).abs().mean() if x.shape[-1]>1 else zero
    return dx+dy


@dataclass
class ModelConfig:
    mode: str='soft'  # free | affine | soft
    channels: int=32
    hidden: int=48
    tile: int=16
    field_stride: int=8
    alpha_init: float=0.05
    fixed_alpha: float=-1.0  # -1=learned; 0/0.25/1 are explicit controls
    projection_rtol: float=1e-5


class PhotometricHead(nn.Module):
    def __init__(self,cfg:ModelConfig):
        super().__init__();self.cfg=cfg
        if cfg.mode not in ('free','affine','soft') or not 0<cfg.alpha_init<1:
            raise ValueError('Invalid head configuration')
        if not (cfg.fixed_alpha==-1 or 0<=cfg.fixed_alpha<=1):
            raise ValueError('fixed_alpha must be -1 or in [0,1]')
        self.spatial=nn.Conv2d(cfg.channels,3,1,bias=False)
        self.scene=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(cfg.channels,8,1),
                                 nn.SiLU(),nn.Conv2d(8,3,1))
        self.field=None
        if cfg.mode!='free':
            self.field=nn.Sequential(nn.Conv2d(cfg.channels,cfg.hidden,1),nn.SiLU(),
                nn.Conv2d(cfg.hidden,cfg.hidden,3,padding=1,groups=cfg.hidden),nn.SiLU(),
                nn.Conv2d(cfg.hidden,12,1))
        self.gate=nn.Conv2d(cfg.channels,1,1) if cfg.mode=='soft' else None
        self.register_buffer('gate_scale',torch.tensor(0.0))
        nn.init.zeros_(self.spatial.weight)
        nn.init.zeros_(self.scene[-1].weight);nn.init.zeros_(self.scene[-1].bias)
        if self.field is not None:
            nn.init.zeros_(self.field[-1].weight);nn.init.zeros_(self.field[-1].bias)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias,math.log(cfg.alpha_init/(1-cfg.alpha_init)))

    @torch.no_grad()
    def copy_free(self,head:nn.Module):
        if getattr(head,'centered',False):
            raise ValueError('Initialization must come from a Free head')
        self.spatial.load_state_dict(head.spatial.state_dict(),strict=True)
        self.scene.load_state_dict(head.scene.state_dict(),strict=True)

    def forward(self,x:Tensor,features:Tensor)->dict:
        with torch.autocast(device_type=x.device.type,enabled=False):
            x,f=x.float(),features.float();b,_,h,w=x.shape
            cfg=self.cfg
            eye=torch.eye(3,device=x.device).view(1,3,3,1,1)
            a=eye.expand(b,3,3,h,w)
            bias=self.scene(f).expand(b,3,h,w)
            raw_residual=self.spatial(f)
            tv=raw_residual.sum()*0
            if self.field is not None:
                size=(math.ceil(h/cfg.field_stride),math.ceil(w/cfg.field_stride))
                low=self.field(F.adaptive_avg_pool2d(f,size))
                z=F.interpolate(low,(h,w),mode='bilinear',align_corners=False)
                d=torch.exp(0.9*z[:,:3].tanh());off=0.1*z[:,3:9].tanh()
                a=torch.stack((torch.stack((d[:,0],off[:,0],off[:,1]),1),
                    torch.stack((off[:,2],d[:,1],off[:,3]),1),
                    torch.stack((off[:,4],off[:,5],d[:,2]),1)),1)
                bias=bias+0.25*z[:,9:12].tanh();tv=total_variation(low)
            alpha=x.new_zeros((b,1,h,w));rank=x.new_zeros(())
            projection=torch.zeros_like(raw_residual)
            if self.gate is not None:
                if h%cfg.tile or w%cfg.tile:
                    raise ValueError('Image size must be divisible by local tile')
                logits=self.gate(F.avg_pool2d(f,cfg.tile,cfg.tile))
                local_alpha=logits.sigmoid() if cfg.fixed_alpha<0 else torch.full_like(logits,cfg.fixed_alpha)
                alpha=local_alpha.repeat_interleave(cfg.tile,-2).repeat_interleave(cfg.tile,-1)
                projection,rank=local_project(x,raw_residual,cfg.tile,cfg.projection_rtol)
            residual=raw_residual-self.gate_scale*alpha*projection
            photo=matvec(a,x)+bias
            raw=photo+residual
            return {'image':raw.clamp(0,1),'raw':raw,'photo':photo,'A':a,'b':bias,
                    'r':residual,'r_raw':raw_residual,'projection':projection,
                    'alpha':alpha,'rank':rank,'field_tv':tv}


class LAUD22(nn.Module):
    def __init__(self,backbone:nn.Module,cfg:ModelConfig):
        super().__init__();self.backbone=backbone;self.head=PhotometricHead(cfg);self.cfg=cfg

    @classmethod
    def from_free(cls,old:nn.Module,cfg:ModelConfig):
        model=cls(copy.deepcopy(old.backbone),cfg);model.head.copy_free(old.head);return model

    def forward(self,x:Tensor,aux:bool=False):
        out=self.head(x,self.backbone(x));return out if aux else out['image']

    @torch.no_grad()
    def set_gate_scale(self,value:float):
        if not 0<=value<=1:raise ValueError('Gate scale outside [0,1]')
        self.head.gate_scale.fill_(value)

    def set_inference_scan_batch(self,size:int):
        if size not in (1,2,4):raise ValueError('Invalid scan batch')
        from luad_uie.models.mambacue_backbone.encoder import FourDirectionMamba
        for layer in self.backbone.modules():
            if isinstance(layer,FourDirectionMamba):layer.inference_scan_batch=size


def build_model(cfg:dict,initialization:dict|None=None)->LAUD22:
    """The original project must be importable; no replacement backbone is hidden."""
    from luad_uie.models.backbone_fusion import BackboneFusion
    original=BackboneFusion(centered=False)
    init=initialization or {'kind':'scratch'}
    if init['kind']!='scratch':
        state=torch.load(init['path'],map_location='cpu',weights_only=True)
        if init['kind']=='free':
            if state.get('arm','free')!='free':raise ValueError('Expected Free checkpoint')
            original.load_state_dict(state['student'],strict=True)
        elif init['kind']=='backbone':original.backbone.load_state_dict(state,strict=True)
        else:raise ValueError('Initialization kind must be scratch/free/backbone')
    return LAUD22.from_free(original,ModelConfig(**cfg))
