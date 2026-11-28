"""Training-only consistency and correspondence. No virtual optimization steps."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor,nn
from torch.nn import functional as F
from model_v22 import matvec


def robust(z:Tensor)->Tensor:
    return torch.sqrt(z.float().square()+1e-6)-0.001


def weighted_mean(z:Tensor,w:Tensor)->Tensor:
    w=torch.broadcast_to(w.detach().float(),z.shape)
    return (z.float()*w).sum()/w.sum().clamp_min(1.0)


def ssim(x:Tensor,y:Tensor)->Tensor:
    x,y=x.float(),y.float();size=min(11,x.shape[-2],x.shape[-1]);size-=1-size%2
    t=torch.arange(size,device=x.device,dtype=x.dtype)-(size-1)/2
    g=torch.exp(-t.square()/(2*1.5**2));g=g/g.sum()
    k=(g[:,None]*g[None,:]).expand(x.shape[1],1,size,size)
    conv=lambda q:F.conv2d(q,k,groups=x.shape[1])
    mx,my=conv(x),conv(y)
    vx,vy=(conv(x*x)-mx*mx).clamp_min(0),(conv(y*y)-my*my).clamp_min(0)
    cov=conv(x*y)-mx*my
    return (((2*mx*my+0.01**2)*(2*cov+0.03**2))/
        ((mx*mx+my*my+0.01**2)*(vx+vy+0.03**2))).mean((1,2,3))


def normalized_lab(x:Tensor)->Tensor:
    """Differentiable sRGB->D65 Lab, scaled by [100,128,128]; not DeltaE00."""
    x=x.float().clamp(0,1)
    linear=torch.where(x>0.04045,((x+0.055)/1.055).pow(2.4),x/12.92)
    m=x.new_tensor([[0.4124564,0.3575761,0.1804375],
                    [0.2126729,0.7151522,0.0721750],
                    [0.0193339,0.1191920,0.9503041]])
    xyz=torch.einsum('ij,bjhw->bihw',m,linear)/x.new_tensor([0.95047,1.0,1.08883])[None,:,None,None]
    delta=6/29
    f=torch.where(xyz>delta**3,xyz.clamp_min(delta**3).pow(1/3),xyz/(3*delta**2)+4/29)
    return torch.stack(((116*f[:,1]-16)/100,500*(f[:,0]-f[:,1])/128,
                        200*(f[:,1]-f[:,2])/128),1)


class ReferenceLoss(nn.Module):
    def __init__(self,lab_weight:float=0.02,mse_weight:float=0.0,lpips_weight:float=0.0):
        super().__init__();self.lab_weight=lab_weight;self.mse_weight=mse_weight;self.lpips_weight=lpips_weight
        self.perceptual=None
        if lpips_weight:
            import lpips
            self.perceptual=lpips.LPIPS(net='vgg',version='0.1').eval().requires_grad_(False)

    def forward(self,x:Tensor,y:Tensor,perceptual_scale:float=1.0)->Tensor:
        with torch.autocast(device_type=x.device.type,enabled=False):
            x,y=x.float(),y.float()
            loss=torch.sqrt((x-y).square()+1e-6).mean()+0.2*(1-ssim(x,y)).mean()
            loss=loss+self.lab_weight*(normalized_lab(x)-normalized_lab(y)).abs().mean()
            loss=loss+self.mse_weight*(x-y).square().mean()
            if self.perceptual is not None and perceptual_scale>0:
                self.perceptual.eval()
                loss=loss+self.lpips_weight*perceptual_scale*self.perceptual(x,y,normalize=True).mean()
            return loss


@dataclass
class Medium:
    a:Tensor
    b:Tensor
    def apply(self,x:Tensor)->Tensor:return self.a*x.float()+self.b
    def after(self,first:'Medium')->'Medium':return Medium(self.a*first.a,self.a*first.b+self.b)


@torch.no_grad()
def sample_medium(x:Tensor,severity:float=1.0)->Medium:
    """Per-image/channel affine perturbation. No clipping/noise/blur.
    Global coefficients preserve each local affine column space in exact math.
    """
    if not 0<=severity<=1.5:raise ValueError('Invalid severity')
    b=x.shape[0];beta=severity*(0.02+0.38*torch.rand(b,3,1,1,device=x.device))
    a=torch.exp(-beta);background=0.1+0.75*torch.rand(b,3,1,1,device=x.device)
    return Medium(a,(1-a)*background)


@dataclass(frozen=True)
class D4:
    turns:int=0
    flip:bool=False
    @staticmethod
    def sample()->'D4':return D4(int(torch.randint(4,()).item()),bool(torch.randint(2,()).item()))
    def apply(self,x:Tensor)->Tensor:
        y=torch.rot90(x,self.turns,(-2,-1));return y.flip(-1).contiguous() if self.flip else y.contiguous()
    def inverse(self,x:Tensor)->Tensor:
        y=x.flip(-1) if self.flip else x;return torch.rot90(y,-self.turns,(-2,-1)).contiguous()
    def inverse_aux(self,out:dict)->dict:
        return {k:self.inverse(v) if v.ndim>=4 else v for k,v in out.items()}


def composition_loss(anchor:dict,changed:dict,medium:Medium)->Tensor:
    # Right multiplication scales INPUT columns, not output rows.
    a_composed=changed['A']*medium.a[:,None]
    b_composed=matvec(changed['A'],medium.b.expand_as(changed['b']))+changed['b']
    return (robust(a_composed-anchor['A'].detach()).mean()+
            0.5*robust(b_composed-anchor['b'].detach()).mean()+
            0.1*robust(changed['r']-anchor['r'].detach()).mean())


def geometry_image(out:dict,isolate:bool=True)->Tensor:
    return (out['photo'].detach()+out['r']).clamp(0,1) if isolate else out['image']


def descriptor(x:Tensor)->tuple[Tensor,Tensor]:
    gray=(x.float()*x.new_tensor([0.299,0.587,0.114])[None,:,None,None]).sum(1,keepdim=True)
    all_features=[]
    for image in (gray,F.avg_pool2d(gray,3,1,1)):
        p=F.unfold(image,5,padding=2);p=p-p.mean(1,keepdim=True)
        all_features.append(F.normalize(p,dim=1,eps=1e-6))
    features=F.normalize(torch.cat(all_features,1),dim=1,eps=1e-6)
    var=(F.avg_pool2d(gray.square(),5,1,2)-F.avg_pool2d(gray,5,1,2).square()).clamp_min(0)
    return features,var.sqrt()


def correspondence_loss(s0:Tensor,s1:Tensor,target:Tensor,visibility:Tensor,
                        anchors:int=128,temperature:float=0.1)->tuple[Tensor,dict]:
    """s1 is ALREADY geometrically aligned to s0. All teacher evidence detached."""
    if s0.shape!=s1.shape or s0.shape!=target.shape:raise ValueError('Unaligned tensors')
    b,_,h,w=s0.shape
    factor=min(1.0,128/max(h,w));size=(round(h*factor),round(w*factor))
    if min(size)<=20:raise ValueError('Descriptor size must exceed 20')
    resize=lambda z:F.interpolate(z.float(),size,mode='bilinear',align_corners=False,antialias=True)
    s0,s1,target=resize(s0),resize(s1),resize(target.detach())
    h,w=size;vis=resize(visibility.detach().expand(b,1,*visibility.shape[-2:])).flatten(1)
    d0,_=descriptor(s0);d1,_=descriptor(s1)
    with torch.no_grad():
        dt,texture=descriptor(target)
        rings=[(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1) if dy or dx]
        offsets=torch.tensor([(0,0)]+[(dy*r,dx*r) for r in (4,8) for dy,dx in rings],device=s0.device)
        yy=torch.randint(10,h-10,(b,anchors),device=s0.device)
        xx=torch.randint(10,w-10,(b,anchors),device=s0.device)
        centers=yy*w+xx
        candidates=(yy[...,None]+offsets[:,0])*w+xx[...,None]+offsets[:,1]
    def gather(d,ids):
        return d.gather(2,ids.reshape(b,1,-1).expand(-1,d.shape[1],-1)).reshape(b,d.shape[1],*ids.shape[1:])
    with torch.no_grad():
        sim=(gather(dt,centers)[...,None]*gather(dt,candidates)).sum(1)
        teacher_prob=(sim/temperature).softmax(-1)
        margin_t=(sim[...,0]-sim[...,1:].amax(-1)).clamp_min(0)
        tex=texture.flatten(1).gather(1,centers)
        v=vis.gather(1,candidates.flatten(1)).reshape_as(candidates).amin(-1)
        weights=((tex-0.005)/0.025).clamp(0,1)*(margin_t/0.15).clamp(0,1)*v
    def directed(a,c):
        cosine=(gather(a,centers)[...,None]*gather(c,candidates)).sum(1)
        logp=(cosine/temperature).log_softmax(-1)
        kl=(teacher_prob*(teacher_prob.clamp_min(1e-8).log()-logp)).sum(-1)
        margin=cosine[...,0]-cosine[...,1:].amax(-1)
        return weighted_mean(kl+0.5*F.relu(0.8*margin_t-margin),weights)
    return 0.5*(directed(d0,d1)+directed(d1,d0)),{
        'geo_support':(weights>0).float().mean().detach()}


@dataclass
class LossConfig:
    use_composition:bool=True
    use_geometry:bool=True
    output_consistency:bool=False
    isolate_photo:bool=True
    composition_weight:float=0.03
    geometry_weight:float=0.01
    output_weight:float=0.03
    lab_weight:float=0.02
    mse_weight:float=0.0
    lpips_weight:float=0.0
    tv_weight:float=0.001
    range_weight:float=0.01
    aux_start_epoch:int=20
    aux_ramp_epochs:int=30
    lpips_start_epoch:int=100
    severity:float=1.0
    anchors:int=128


def forward_aux(model,x,amp):
    with torch.autocast(device_type=x.device.type,dtype=torch.bfloat16,enabled=amp and x.is_cuda):
        return model(x,aux=True)


def training_loss(model,x,y,epoch,cfg:LossConfig,reference:ReferenceLoss,amp:bool=True):
    f=max(0.0,min(1.0,(epoch-cfg.aux_start_epoch)/max(1,cfg.aux_ramp_epochs)))
    model.set_gate_scale(f)
    # Every arm gets exactly the same two-view augmentation and supervision.
    medium=sample_medium(x,cfg.severity);transform=D4.sample()
    o0=forward_aux(model,x,amp)
    o1=transform.inverse_aux(forward_aux(model,transform.apply(medium.apply(x)),amp))
    lp=float(epoch>=cfg.lpips_start_epoch)
    rec=0.5*(reference(o0['image'],y,lp)+reference(o1['image'],y,lp))
    tv=0.5*(o0['field_tv']+o1['field_tv'])
    rang=lambda out:(F.relu(-out['raw'])+F.relu(out['raw']-1)).mean()
    loss=rec+cfg.tv_weight*tv+0.5*cfg.range_weight*(rang(o0)+rang(o1))
    stats={'rec':rec.detach(),'alpha_mean':o0['alpha'].mean().detach(),
           'rank':o0['rank'].detach(),'aux_scale':x.new_tensor(f)}
    if cfg.use_composition and f:
        if model.cfg.mode=='free':raise ValueError('COC requires an affine/soft head')
        op=composition_loss(o0,o1,medium);loss=loss+f*cfg.composition_weight*op;stats['coc']=op.detach()
    if cfg.output_consistency and f:
        oc=robust(o1['image']-o0['image'].detach()).mean()
        loss=loss+f*cfg.output_weight*oc;stats['output_consistency']=oc.detach()
    if cfg.use_geometry and f:
        visibility=medium.a.amin(1,keepdim=True).square().expand(-1,1,*x.shape[-2:])
        # Auxiliary anchor draws must not shift the next batch's shared medium
        # or D4 augmentation stream when comparing C-enabled and C-free arms.
        devices=[x.device.index] if x.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            geo,detail=correspondence_loss(geometry_image(o0,cfg.isolate_photo),
                geometry_image(o1,cfg.isolate_photo),y,visibility,cfg.anchors)
        loss=loss+f*cfg.geometry_weight*geo;stats['picd']=geo.detach();stats.update(detail)
    stats['loss']=loss.detach();return loss,stats
