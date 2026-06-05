import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB, eval_sh
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, tone_mapping
from utils.general_utils import strip_symmetric, build_scaling_rotation, get_minimum_axis


# H * W * 3
class EnvironmentMap:
    def __init__(self):
        self.environment_map = torch.zeros()
#sphere harmonic represent indirect light
class SHEnvironmentMap(nn.Module):
    def __init__(self, sh_degree = 2):
        super().__init__()
        self.sh_degree = sh_degree
        self.features_dc = torch.empty(0)
        self.features_rest = torch.empty(0)

    def setup(self, dc_coff):
        features = torch.zeros((3, (self.sh_degree + 1) ** 2)).float().cuda()
        features[:3, 0] = dc_coff
        self.features_dc = nn.Parameter(features[:,0:1].transpose(0, 1).contiguous().requires_grad_(True))
        self.features_rest = nn.Parameter(features[:,1:].transpose(0, 1).contiguous().requires_grad_(True))


    @property
    def get_features(self):
        feature_dc = self.features_dc
        feature_rest = self.features_rest
        return torch.cat((feature_dc, feature_rest), dim=0)

    def sample(self, dirs):
        sh = self.get_features.transpose(0, 1).view(-1, 3, (self.sh_degree + 1) ** 2)
        return eval_sh(self.sh_degree, sh , dirs)

class EnvironmentModel(nn.Module):
    def __init__(self):
        self._direct_light = None
        self._indirect_light = None

    def setup(self):
        super().__init__()
        self._direct_light = nn.Parameter(torch.tensor([0.0, 1.0, 0.0], device="cuda").requires_grad_(True))
        self._indirect_light = SHEnvironmentMap()
        self._indirect_light.setup(torch.tensor([3.141593, 2.094395, 0.785398]))
