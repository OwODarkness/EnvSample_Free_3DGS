#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, get_minimum_axis_with_scale
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, get_minimum_axis
from scene.environment_model import EnvironmentModel
from scene.neural_specular import NeuralSpecular

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.normal_activation = torch.nn.functional.normalize
        self.material_activation = torch.sigmoid


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)

        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.setup_functions()

        # pbr params
        self._normal = torch.empty(0)
        self._default_roughness = 0.3
        self._default_metallic = 0.7
        self._default_ao = 0.9
        self._default_albedo = 0.5
        self._albedo = torch.empty(0)
        self._roughness = torch.empty(0)
        self._metallic = torch.empty(0)
        self._ao = torch.empty(0)
        self.env = None
        self.nerual_specular = None


    def capture(self)  :
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_albedo(self):
        return self.material_activation(self._albedo)

    @property
    def get_roughness(self):
        return self.material_activation(self._roughness)

    @property
    def get_metallic(self):
        return self.material_activation(self._metallic)

    @property
    def get_ao(self):
        return self.material_activation(self._ao)

    @property
    def get_normal(self):
        return self.normal_activation(self._normal)


    def get_view_adjust_normal(self, view_dir):
        opacity = self.get_opacity
        opacity_mean, opacity_var = torch.mean(opacity), torch.var(opacity)
        bound_r = opacity_mean + 3 * opacity_var
        opacity_alpha =  torch.exp(-torch.abs(bound_r - opacity))
        min_axis, min_scale = get_minimum_axis_with_scale(self._rotation, self._scaling)
        alpha = torch.clamp(0.5 * (opacity_alpha + (1.0 - min_scale)), 0.3, 0.7)

        dot_product = torch.sum(min_axis * view_dir, axis=1)  # Shape: (N,)
        flip_mask = (dot_product < 0)
        adjusted_normal = torch.where(flip_mask.unsqueeze(1), -min_axis, min_axis)

        normal_axis = (1.0 - alpha) *  adjusted_normal + alpha * self._normal
        return self.normal_activation(normal_axis)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_minimum_axis(self):
        return get_minimum_axis(self._rotation, self._scaling)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    # generate 3d gauss based on sfm point
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):

        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # setup the pbr param
        normals = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
        albedos = inverse_sigmoid(self._default_albedo * torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda"))
        roughness = inverse_sigmoid(self._default_roughness *  torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        metallic = inverse_sigmoid(self._default_metallic * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        ao = inverse_sigmoid(self._default_ao * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._normal = nn.Parameter(normals.requires_grad_(True))
        self._albedo = nn.Parameter(albedos.requires_grad_(True))
        self._roughness = nn.Parameter(roughness.requires_grad_(True))
        self._metallic = nn.Parameter(metallic.requires_grad_(True))
        self._ao = nn.Parameter(ao.requires_grad_(True))


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.env = EnvironmentModel()
        self.env.setup()
        self.nerual_specular = NeuralSpecular().to("cuda")


        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            # pbr
            {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
            {'params': [self._albedo], 'lr': training_args.albedo_lr, "name": "albedo"},
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
            {'params': [self._metallic], 'lr': training_args.metallic_lr, "name": "metallic"},
            {'params': [self._ao], 'lr': training_args.ao_lr, "name": "ao"},
            # envir
            {'params': [self.env._direct_light], 'lr': training_args.radir_lr, "name": "radir"},
            {'params': [self.env._indirect_light.features_dc], 'lr' : training_args.env_feature_lr, "name": "env_f_dc" },
            {'params': [self.env._indirect_light.features_rest], 'lr' : training_args.env_feature_lr / 20.0, "name": "env_f_rest" },
            # specular
            {'params': list(self.nerual_specular.filter_net.parameters()), 'lr': training_args.specular_lr, "name": "filter_net"},
            {'params': list(self.nerual_specular.brdf_net.parameters()), 'lr': training_args.specular_lr, "name": "brdf_net"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'ax', 'ay', 'az']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('roughness')
        l.append('metallic')
        l.append('ao')
        return l

    def construct_list_of_env_attributes(self):
        l = []
        for i in range(self.env._indirect_light.features_dc.shape[0] * self.env._indirect_light.features_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.env._indirect_light.features_rest.shape[0] * self.env._indirect_light.features_rest.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.extend(['dx', 'dy', 'dz'])
        
        return l


    def construct_filter_attributes(self):
        attri_names = []

        layer_idx = 0
        for name, param in self.nerual_specular.filter_net.named_parameters():
            if 'weight' in name:
                for i in range(param.shape[0]):
                    for j in range(param.shape[1]):
                        attri_names.append(f"filter_layer{layer_idx}_weight_{i}_{j}")
            elif 'bias' in name:
                for i in range(param.shape[0]):
                    attri_names.append(f"filter_layer{layer_idx}_bias_{i}")
            layer_idx += 1

        return attri_names

    def construct_brdf_attributes(self):
        attri_names = []
        layer_idx = 0
        for name, param in self.nerual_specular.brdf_net.named_parameters():
            if 'weight' in name:
                for i in range(param.shape[0]):
                    for j in range(param.shape[1]):
                        attri_names.append(f"brdf_layer{layer_idx}_weight_{i}_{j}")
            elif 'bias' in name:
                for i in range(param.shape[0]):
                    attri_names.append(f"brdf_layer{layer_idx}_bias_{i}")
            layer_idx += 1
        return attri_names



    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = self._normal.detach().cpu().numpy()


        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        albedos = self._albedo.detach().cpu().numpy()
        roughness = self._roughness.detach().cpu().numpy()
        metallic = self._metallic.detach().cpu().numpy()
        ao = self._ao.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals,albedos,  f_dc, f_rest, opacities, scale, rotation,  roughness, metallic, ao), axis=1)
        elements[:] = list(map(tuple, attributes))

        #environment params

        env_attri_names = self.construct_list_of_env_attributes()
        env_dtype = [(attribute, 'f4') for attribute in env_attri_names]

        env_f_dc = self.env._indirect_light.features_dc.detach().transpose(0, 1).flatten().contiguous().cpu().numpy()
        env_f_rest = self.env._indirect_light.features_rest.detach().transpose(0, 1).flatten().contiguous().cpu().numpy()
        direct_light_np = self.env._direct_light.detach().cpu().numpy().flatten()
        
        env_attributes = np.concatenate((env_f_dc, env_f_rest, direct_light_np))
        env_elements = np.empty(1, dtype=env_dtype)
        for i, name in enumerate(env_attri_names):
            env_elements[name] = env_attributes[i]


        #specular params
        filter_attri_names = self.construct_filter_attributes()
        filter_dtype = [(attribute, 'f4') for attribute in filter_attri_names]

        filter_params = []
        for param in self.nerual_specular.filter_net.parameters():
            filter_params.append(param.detach().flatten().contiguous().cpu().numpy())
        filter_attributes = np.concatenate(filter_params)

        filter_elements = np.empty(1, dtype=filter_dtype)
        for i, name in enumerate(filter_attri_names):
            filter_elements[name] = filter_attributes[i]


        #brdf
        brdf_attri_names = self.construct_brdf_attributes()
        brdf_dtype = [(attribute, 'f4') for attribute in brdf_attri_names]

        brdf_params = []
        for param in self.nerual_specular.brdf_net.parameters():
            brdf_params.append(param.detach().flatten().contiguous().cpu().numpy())
        brdf_attributes = np.concatenate(brdf_params)

        brdf_elements = np.empty(1, dtype=brdf_dtype)
        for i, name in enumerate(brdf_attri_names):
            brdf_elements[name] = brdf_attributes[i]


        vertex_el = PlyElement.describe(elements, 'vertex')
        env_el = PlyElement.describe(env_elements, 'environment')
        filter_el = PlyElement.describe(filter_elements, 'filter_net')
        brdf_el = PlyElement.describe(brdf_elements, 'brdf_net')

        PlyData([vertex_el, env_el, filter_el, brdf_el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # construct gaussians from point_cloud.ply
    def load_ply(self, path):
        self.env = EnvironmentModel()
        self.env.setup()
        self.nerual_specular = NeuralSpecular().to("cuda")
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)

        normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                           np.asarray(plydata.elements[0]["ny"]),
                           np.asarray(plydata.elements[0]["nz"])), axis=1)
        

        albedo = np.stack((np.asarray(plydata.elements[0]["ax"]),
                           np.asarray(plydata.elements[0]["ay"]),
                           np.asarray(plydata.elements[0]["az"])), axis=1)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]
        metallic = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis]
        ao = np.asarray(plydata.elements[0]["ao"])[..., np.newaxis]

        #load environment data
        env_element = None
        for element in plydata.elements:
            if element.name == 'environment':
                env_element = element
                break
        if env_element is None:
            raise ValueError("No 'environment' element found in PLY file")
        env_data = env_element.data[0]
        # Extract features in order

        num_dc_features = 3
        num_rest_features = 3 * ((1 + self.env._indirect_light.sh_degree) ** 2 - 1 )
        f_dc_values = [env_data[f'f_dc_{i}'] for i in range(num_dc_features)]
        f_rest_values = [env_data[f'f_rest_{i}'] for i in range(num_rest_features)]
        env_f_dc = np.array(f_dc_values, dtype=np.float32)
        env_f_rest = np.array(f_rest_values, dtype=np.float32)
        env_features_dc = env_f_dc.reshape(3, 1)
        env_features_rest = env_f_rest.reshape(3, -1)
        dx, dy, dz = env_data["dx"], env_data["dx"], env_data["dz"]
        print(dx, dy, dz)

        #load neural specular
        filter_data = plydata['filter_net'].data
        filter_attributes = np.concatenate([filter_data[name] for name in filter_data.dtype.names])
        filter_idx = 0
        for param in self.nerual_specular.filter_net.parameters():
            param_size = param.numel()
            param_data = filter_attributes[filter_idx:filter_idx + param_size]
            param_data = torch.from_numpy(param_data).reshape(param.shape)
            param.data.copy_(param_data)
            filter_idx += param_size

        brdf_data = plydata['brdf_net'].data
        brdf_attributes = np.concatenate([brdf_data[name] for name in brdf_data.dtype.names])
        brdf_idx = 0
        for param in self.nerual_specular.brdf_net.parameters():
            param_size = param.numel()
            param_data = brdf_attributes[brdf_idx:brdf_idx + param_size]
            param_data = torch.from_numpy(param_data).reshape(param.shape)
            param.data.copy_(param_data)
            brdf_idx += param_size

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

        self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
        self._albedo = nn.Parameter(torch.tensor(albedo, dtype=torch.float, device="cuda").requires_grad_(True))
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
        self._metallic = nn.Parameter(torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True))
        self._ao = nn.Parameter(torch.tensor(ao, dtype=torch.float, device="cuda").requires_grad_(True))

        #env
        self.env._indirect_light.features_dc = nn.Parameter(torch.tensor(env_features_dc, dtype = torch.float, device="cuda").transpose(0, 1).contiguous().requires_grad_(True))
        self.env._indirect_light.features_rest = nn.Parameter(torch.tensor(env_features_rest, dtype = torch.float, device="cuda").transpose(0, 1).contiguous().requires_grad_(True))
        self.env._direct_light =nn.Parameter(torch.tensor([dx, dy, dz], dtype = torch.float, device="cuda")).requires_grad_(True)



    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        ignore_tensors = {"env_f_dc", "env_f_rest", "radir", "filter_net", "brdf_net"}
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ignore_tensors:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._normal = optimizable_tensors["normal"]
        self._albedo = optimizable_tensors["albedo"]
        self._roughness = optimizable_tensors["roughness"]
        self._metallic = optimizable_tensors["metallic"]
        self._ao = optimizable_tensors["ao"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        ignore_tensors = {"env_f_dc", "env_f_rest", "radir","filter_net", "brdf_net"}
        for group in self.optimizer.param_groups:
            if group["name"] in ignore_tensors:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_normal, new_albedo, new_roughness, new_metallic, new_ao):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "normal" : new_normal,
        "albedo" : new_albedo,
             "roughness" : new_roughness,
             "metallic" : new_metallic,
             "ao" : new_ao
             }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._normal = optimizable_tensors["normal"]
        self._albedo = optimizable_tensors["albedo"]


        self._roughness = optimizable_tensors["roughness"]
        self._metallic = optimizable_tensors["metallic"]
        self._ao = optimizable_tensors["ao"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_normal = self._normal[selected_pts_mask].repeat(N,1)
        new_albedo = self._albedo[selected_pts_mask].repeat(N,1)


        new_roughness = self._roughness[selected_pts_mask].repeat(N,1)
        new_metallic = self._metallic[selected_pts_mask].repeat(N,1)
        new_ao = self._ao[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_normal,new_albedo,  new_roughness, new_metallic, new_ao)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_albedo = self._albedo[selected_pts_mask]

        new_roughness = self._roughness[selected_pts_mask]
        new_metallic = self._metallic[selected_pts_mask]
        new_ao = self._ao[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_normal,new_albedo, new_roughness, new_metallic, new_ao)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
