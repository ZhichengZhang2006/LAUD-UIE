import csv
import json
import numpy as np
from PIL import Image
from luad_uie.scripts.prepare_counterfactual import prepare


def test_cross_split_content_removed_and_original_manifests_preserved(tmp_path):
    for i in range(5):
        image=np.random.default_rng(i).integers(0,255,(16,16,3),dtype=np.uint8)
        Image.fromarray(image).save(tmp_path/(str(i)+'.png'))
    # Train0 repeats val3; val4 repeats test4, but IDs and file paths differ.
    for name,images in {'train':[0,1,3],'val':[3,4],'test':[4]}.items():
        with (tmp_path/(name+'.csv')).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['id','input','target'])
            for i in images:w.writerow([name+str(i),str(i)+'.png',str(i)+'.png'])
    original=(tmp_path/'train.csv').read_bytes()
    prepare(tmp_path,tmp_path/'clean')
    metadata=json.loads((tmp_path/'clean/metadata.json').read_text())
    removed={(r['split'],r['id']) for r in metadata['omitted_cross_split_duplicates']}
    assert ('train','train3') in removed and ('val','val4') in removed
    assert metadata['counts']['val']==1 and metadata['counts']['test']==1
    assert (tmp_path/'train.csv').read_bytes()==original
