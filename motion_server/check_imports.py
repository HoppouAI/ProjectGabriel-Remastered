import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / 'shims'))
sys.path.insert(0, str(HERE / 'DART'))

import os
os.chdir(HERE / 'DART')

# numpy>=1.24 removed these aliases, DART's vendored humanml code still uses them
import numpy as np
for _alias, _t in [('float', float), ('int', int), ('bool', bool), ('object', object), ('str', str)]:
    if not hasattr(np, _alias):
        setattr(np, _alias, _t)

print('shim check...')
import torch
from pytorch3d import transforms
aa = torch.tensor([[0.3, -0.5, 0.7]])
m = transforms.axis_angle_to_matrix(aa)
back = transforms.matrix_to_axis_angle(m)
print('roundtrip err:', (aa - back).abs().max().item())
d6 = transforms.matrix_to_rotation_6d(m)
m2 = transforms.rotation_6d_to_matrix(d6)
print('6d roundtrip err:', (m - m2).abs().max().item())

print('DART imports...')
from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from model.mld_vae import AutoMldVae
print('models ok')
from diffusion import gaussian_diffusion as gd
print('diffusion ok')
from utils.misc_util import encode_text
print('misc_util ok')
from utils.smpl_utils import PrimitiveUtility
print('smpl_utils ok (body models found!)')
from data_loaders.humanml.data.dataset import SinglePrimitiveDataset
print('dataset ok')
from mld.train_mvae import Args as MVAEArgs
from mld.train_mld import DenoiserArgs, MLDArgs, create_gaussian_diffusion
print('train arg dataclasses ok')
print('ALL IMPORTS PASS')
