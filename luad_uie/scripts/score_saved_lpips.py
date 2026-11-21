"""Evaluate full-precision saved predictions with LPIPS v0.1 VGG on the same inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import lpips


@torch.inference_mode()
def evaluate(folder, manifest):
    provenance = json.loads((folder.parent/'provenance.json').read_text())
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != provenance['manifest_sha256']:
        raise ValueError('Reference manifest hash mismatch')
    rows = json.loads(manifest.read_text())
    if json.loads((folder/'summary.json').read_text())['count'] != len(rows):
        raise ValueError('Incomplete model predictions')
    ids = [json.loads(line)['id'] for line in (folder/'metrics.jsonl').read_text().splitlines()]
    if ids != [row['name'] for row in rows]:
        raise ValueError('Prediction IDs differ')
    metric = lpips.LPIPS(net='vgg', version='0.1').cuda().eval()
    scores = []
    for i,row in enumerate(rows):
        prediction = torch.from_numpy(np.load(folder/(str(i)+'.npy'))).permute(2,0,1)[None].cuda()
        with Image.open(row['target']) as image:
            rgb = image.convert('RGB').resize((256,256),Image.BILINEAR)
            target = torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1)[None].cuda().float()/255
        value = metric(prediction, target, normalize=True).item()
        scores.append({'id':row['name'],'lpips_vgg':value})
    result = {'count':len(scores),'LPIPS':np.mean([r['lpips_vgg'] for r in scores]),'records':scores}
    (folder/'lpips.json').write_text(json.dumps(result,allow_nan=False))
    print('LPIPS',result['LPIPS'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    parser.add_argument('manifest',type=Path)
    args = parser.parse_args()
    evaluate(args.folder,args.manifest)
