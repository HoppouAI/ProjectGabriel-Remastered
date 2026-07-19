"""merge the extended smpl+h npz bodies with mano hand pca data into the
SMPLH_*.pkl files smplx expects. mirrors smplx tools/merge_smplh_mano.py.

inputs (download with the mano.is.tue.mpg.de account, see README):
  DART/smplh_extract/<gender>/model.npz   from smplh.tar.xz (Extended SMPL+H)
  DART/mano_v1_2/models/MANO_LEFT.pkl     from mano_v1_2.zip
  DART/mano_v1_2/models/MANO_RIGHT.pkl

output:
  DART/data/smplx_lockedhead_20230207/models_lockedhead/smplh/SMPLH_<GENDER>.pkl

needs chumpy to unpickle the mano files:
  ..\\bin\\uv.exe pip install --python .venv\\Scripts\\python.exe --no-deps chumpy
"""
import os
import pickle
import sys

import numpy as np

# chumpy needs the removed numpy aliases at import time
for _name, _val in [('bool', bool), ('int', int), ('float', float),
                    ('complex', complex), ('object', object),
                    ('unicode', str), ('str', str)]:
    if not hasattr(np, _name):
        setattr(np, _name, _val)

import chumpy  # noqa: F401, E402

DART = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DART')
OUT_DIR = os.path.join(DART, 'data', 'smplx_lockedhead_20230207', 'models_lockedhead', 'smplh')


def unchump(v):
    if hasattr(v, 'r'):
        return np.array(v.r)
    return v


def load_mano(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return {k: unchump(v) for k, v in data.items()}


def main():
    left = load_mano(os.path.join(DART, 'mano_v1_2', 'models', 'MANO_LEFT.pkl'))
    right = load_mano(os.path.join(DART, 'mano_v1_2', 'models', 'MANO_RIGHT.pkl'))
    os.makedirs(OUT_DIR, exist_ok=True)

    for gender in ['male', 'female', 'neutral']:
        npz_path = os.path.join(DART, 'smplh_extract', gender, 'model.npz')
        if not os.path.exists(npz_path):
            print(f'skipping {gender}, {npz_path} missing')
            continue
        body = {k: v for k, v in np.load(npz_path, allow_pickle=True).items()}
        merged = dict(body)
        merged['hands_componentsl'] = left['hands_components']
        merged['hands_componentsr'] = right['hands_components']
        merged['hands_meanl'] = left['hands_mean']
        merged['hands_meanr'] = right['hands_mean']
        merged['hands_coeffsl'] = left['hands_coeffs']
        merged['hands_coeffsr'] = right['hands_coeffs']
        out = os.path.join(OUT_DIR, f'SMPLH_{gender.upper()}.pkl')
        with open(out, 'wb') as f:
            pickle.dump(merged, f)
        print(f'{out}: {os.path.getsize(out) / 1e6:.1f} MB')


if __name__ == '__main__':
    sys.exit(main())
