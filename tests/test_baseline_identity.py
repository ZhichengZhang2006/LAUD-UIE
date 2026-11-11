import pytest
from luad_uie.scripts.evaluate import validate_baseline
from luad_uie.runtime import file_hash


def test_teacher_is_rejected_as_student_baseline(tmp_path):
    manifest = tmp_path/'train.csv'
    manifest.write_text('id,input,target\n')
    config = {'student': {'width': 24}, 'seed': 42, 'epochs': 100,
              'batch_size': 4, 'crop_size': 256, 'lr': .0002,
              'data': {'train': str(manifest), 'val': str(manifest)}}
    base = {'experiment': 'E0', 'summary': {'manifest_sha256': 'testhash'},
            'training_config': config,
            'training_manifest_sha256': {k: file_hash(manifest) for k in ('train', 'val')}}
    validate_baseline(base, config, 'testhash')
    base['training_config'] = {**config, 'student': {'width': 48}}
    with pytest.raises(ValueError, match='student'):
        validate_baseline(base, config, 'testhash')
    del base['training_config']
    with pytest.raises(ValueError, match='Legacy'):
        validate_baseline(base, config, 'testhash')
