"""Adapter to the EXACT metric interface already used in the uploaded project.
No guessed pyiqa UIQM/UCIQE names; no silent replacement of historical metrics.
Requires server-side MambaCue utils and pretrained LPIPS/URanker resources.
"""
from __future__ import annotations
import contextlib,importlib,os,sys,hashlib
from pathlib import Path
import numpy as np
import torch
from skimage.color import rgb2lab,deltaE_ciede2000
from data_v22 import sha256


@contextlib.contextmanager
def working_directory(path):
    old=Path.cwd();os.chdir(path)
    try:yield
    finally:os.chdir(old)


class ProjectQuality:
    def __init__(self,repo,device='cuda',need_nr=False):
        self.repo=Path(repo).resolve();self.device=torch.device(device)
        required=['utils/metrics.py','utils/niqe_utils.py']
        for name in required:
            if not (self.repo/name).is_file():
                raise FileNotFoundError(f'Missing original metric implementation: {self.repo/name}')
        # Avoid accidentally importing an unrelated package named utils.
        if 'utils' in sys.modules:
            origin=getattr(sys.modules['utils'],'__file__',None)
            if origin and self.repo not in Path(origin).resolve().parents:
                raise RuntimeError('Another utils package is loaded; run metrics in a fresh process')
        sys.path.insert(0,str(self.repo))
        self.metrics=importlib.import_module('utils.metrics')
        import lpips
        self.lpips=lpips.LPIPS(net='vgg',version='0.1').to(self.device).eval().requires_grad_(False)
        self.ranker=None
        self.meta={'backend':'original MambaCue utils.metrics',
                   'source_sha256':{n:sha256(self.repo/n) for n in required},
                   'lpips_net':'vgg','lpips_version':'0.1','lpips_normalize':True,
                   'preprocessing':'PIL RGB bilinear 256x256; float32 [0,1]; no output quantization',
                   'torch':str(torch.__version__)}
        h=hashlib.sha256()
        for name,value in sorted(self.lpips.state_dict().items()):
            h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.meta['lpips_state_sha256']=h.hexdigest()
        if need_nr:
            from utils.uranker.uranker_utils import build_model,get_option
            yaml=self.repo/'utils/uranker/URanker.yaml';ckpt=self.repo/'utils/uranker/URanker_ckpt.pth'
            if not yaml.is_file() or not ckpt.is_file():raise FileNotFoundError('URanker YAML/weights missing')
            with working_directory(self.repo):
                options=get_option(str(yaml))['model']
                options.update(cuda=False,resume_ckpt_path=str(ckpt))
                self.ranker=build_model(options).cpu().eval().requires_grad_(False)
            self.meta['uranker_sha256']=sha256(ckpt)
            self.meta['uranker_yaml_sha256']=sha256(yaml)

    @torch.inference_mode()
    def score(self,prediction:torch.Tensor,target:torch.Tensor|None=None)->dict:
        if prediction.shape!=(1,3,256,256):raise ValueError('Score exactly one 256x256 RGB image')
        p=prediction.float().clamp(0,1)
        pn=p.permute(0,2,3,1).cpu().numpy()
        with working_directory(self.repo):
            result={'UCIQE':float(self.metrics.getUCIQE(pn))}
            if target is not None:
                if target.shape!=p.shape:raise ValueError('Reference dimensions differ')
                y=target.float();yn=y.permute(0,2,3,1).cpu().numpy()
                result.update(PSNR=float(self.metrics.getPSNR(pn,yn,1.0)),
                              SSIM=float(self.metrics.getSSIM(pn,yn,1.0)),
                              LPIPS=float(self.lpips(p.to(self.device),y.to(self.device),normalize=True).mean()),
                              E00=float(deltaE_ciede2000(rgb2lab(pn[0]),rgb2lab(yn[0])).mean()))
            else:
                if self.ranker is None:raise RuntimeError('Initialize with need_nr=True')
                result.update(UIQM=float(self.metrics.getUIQM(pn)),
                              Entropy=float(self.metrics.getEntropy(pn)),
                              NIQE=float(self.metrics.getNIQE(pn)),
                              URanker=float(self.metrics.getURanker(pn,self.ranker)))
        if not all(np.isfinite(v) for v in result.values()):
            raise FloatingPointError(f'Non-finite score (do not silently skip this image): {result}')
        return result


def balanced_score(values:dict)->float:
    """Fixed validation-only utility; additive reference constants cancel."""
    return (values['PSNR']+50*values['SSIM']-12.5*values['LPIPS']
            -0.15*values['E00']+5*values['UCIQE'])


def select_epoch(history:list,psnr_tolerance:float=0.15)->dict:
    if not history:raise ValueError('No validation candidates')
    best=max(row['val']['PSNR'] for row in history)
    feasible=[row for row in history if row['val']['PSNR']>=best-psnr_tolerance]
    # Single checkpoint for ALL metrics and its designated NR test datasets.
    return max(feasible,key=lambda r:(balanced_score(r['val']),r['val']['PSNR'],-r['epoch']))
