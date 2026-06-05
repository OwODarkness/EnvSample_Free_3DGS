import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
from utils.general_utils import positional_encoding


class NeuralSpecular(nn.Module):
    def __init__(self):
        super(NeuralSpecular, self).__init__()
        
        self.filter_net = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softplus()
        )
        
        self.brdf_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )

    def forward(self, F, view,  reflect, NdotV, roughness):
        if NdotV.dim() == 1:
            NdotV = NdotV.unsqueeze(0)
        if roughness.dim() == 1:
            roughness = roughness.unsqueeze(0)
        
        
        reflect_flat = reflect.permute(1, 0)
        Li_flat = reflect_flat.clone()
        roughness_flat = roughness.permute(1, 0)
        
        filter_input = torch.cat([reflect_flat, Li_flat, roughness_flat], dim=1)
        filter_flat = self.filter_net(filter_input)
        filter_out = filter_flat.permute(1, 0)
        
        NdotV_flat = NdotV.permute(1, 0)
        
        brdf_input = torch.cat([NdotV_flat, roughness_flat], dim=1)
        brdf_flat = self.brdf_net(brdf_input)
        
        scale = brdf_flat[:, 0:1]
        bias = brdf_flat[:, 1:2]
        
        F_flat = F.permute(1, 0)
        brdf_out_flat = F_flat * scale + bias
        brdf_out = brdf_out_flat.permute(1, 0)
        
        return filter_out, brdf_out
