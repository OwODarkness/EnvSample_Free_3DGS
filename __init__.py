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
import math
import diff_gaussian_rasterization
import diff_gaussian_rasterization1
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import reflect,  fresnel_schlick, tone_mapping, ggx_normal_distribution, geometry_smith, rgb_to_gray
from utils.loss_utils import correlation_aware_loss
from utils.sh_utils import eval_sh
from scene.environment_model import EnvironmentModel 
from scene.neural_specular import NeuralSpecular
from utils.general_utils import get_minimum_axis_with_scale, get_minimum_axis

def generate_sh_degree_mipmap(max_mip_level, roughness, imgs):
    mip_level = max_mip_level * roughness
    mip_low = floor(mip_level)
    mip_high = ceil(mip_level)
    alpha = mip_level - mip_low
    return torch.lerp(imgs[mip_low], imgs[mip_high], alpha)

def pbr_shading(g_buffer, env, neural_specular, debug=False):
    mask = g_buffer["mask"]
    albedo = g_buffer["albedo"]
    
    fg_mask = mask[0]
    
    if not fg_mask.any():
        bg = torch.ones_like(albedo)
        return {"render": bg}
    
    albedo_fg = albedo[:, fg_mask]
    roughness_fg = g_buffer["material"][0][fg_mask]
    metallic_fg = g_buffer["material"][1][fg_mask]
    ao_fg = g_buffer["material"][2][fg_mask]
    
    normal = g_buffer["normal"] * 2.0 - 1.0
    normal_fg = normal[:, fg_mask]
    normal_fg = normal_fg / normal_fg.norm(dim=0, keepdim=True)
    
    view = g_buffer["view"] * 2.0 - 1.0
    view_fg = view[:, fg_mask]
    view_fg = view_fg / (view_fg.norm(dim=0, keepdim=True) + 1e-6)
    
    reflect_dir_fg = reflect(-view_fg, normal_fg)
    

    L = reflect_dir_fg
    half_fg = view_fg + L
    half_fg = half_fg / (half_fg.norm(dim=0, keepdim=True) + 1e-6)
    
    
    H, W = normal.shape[1], normal.shape[2]
    
    dielectric_f0 = torch.tensor(0.04, device=albedo.device)
    F0_fg = dielectric_f0 * (1.0 - metallic_fg) + albedo_fg * metallic_fg
    
    normal_flat = normal_fg.permute(1, 0)
    irradiance_fg = env._indirect_light.sample(normal_flat).permute(1, 0)
    
    diffuse_color_fg = irradiance_fg * albedo_fg / math.pi
    NdotV_fg = torch.clamp(torch.sum(normal_fg * view_fg, dim=0), 0.0, 1.0)
    
    F_fg = fresnel_schlick(NdotV_fg, F0_fg)
    kd_fg = (1.0 - F_fg) * (1.0 - metallic_fg.unsqueeze(0))
    
    diffuse_fg = kd_fg * diffuse_color_fg * ao_fg.unsqueeze(0)
    
    prefilter_fg, brdf_fg = neural_specular(
        F_fg, view_fg, reflect_dir_fg, NdotV_fg, roughness_fg
    )
    
    if brdf_fg.dim() == 1:
        brdf_fg = brdf_fg.unsqueeze(0).expand(3, -1)
    specular_fg = prefilter_fg * brdf_fg
    
    indirect_fg = diffuse_fg + specular_fg

    radiance_fg = g_buffer["direct"][:, fg_mask]

    NdotH_fg = torch.clamp(torch.sum(normal_fg * half_fg, dim=0), 0.0, 1.0)
    NdotL_fg = torch.clamp(torch.sum(normal_fg * L, dim=0), 0.0, 1.0)
    
    NDF_fg = ggx_normal_distribution(NdotH_fg, roughness_fg)
    G_fg = geometry_smith(NdotV_fg, NdotL_fg, roughness_fg)
    
    denom = 4.0 * NdotV_fg * NdotL_fg + 1e-6
    direct_specular_fg = (NDF_fg * G_fg * F_fg) / denom.unsqueeze(0)
    direct_diffuse_fg = kd_fg * albedo_fg  / math.pi 
    
    direct_fg = (direct_specular_fg + direct_diffuse_fg) * radiance_fg * NdotL_fg.unsqueeze(0)
    
    final_fg = indirect_fg + direct_fg
    
    bg_color = torch.ones(3, H, W, device=albedo.device)
    final_color = bg_color.clone()
    final_color[:, fg_mask] = final_fg
    
    def reconstruct(data_fg):
        full = bg_color.clone() 
        full[:, fg_mask] = data_fg
        return full
    
    out = {
        "render": final_color,
        "direct": reconstruct(direct_fg),
        "diffuse": reconstruct(diffuse_fg),
        "brdf": reconstruct(brdf_fg),
        "prefilter": reconstruct(prefilter_fg),
    }
    
    if debug:
        out.update({
            "indirect": reconstruct(indirect_fg),
        })
    
    return out

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, mask_eps = 0.01, scaling_modifier = 1.0, override_color = None, mode="train", debug = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = diff_gaussian_rasterization.GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug = False
    )

    rasterizer = diff_gaussian_rasterization.GaussianRasterizer(raster_settings=raster_settings)

    raster_settings1 = diff_gaussian_rasterization1.GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.tensor([1.0]*12, dtype=torch.float32, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug = False
    )
    rasterizer1 = diff_gaussian_rasterization1.GaussianRasterizer(raster_settings=raster_settings1)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    dir_pp = viewpoint_camera.camera_center.repeat(pc.get_xyz.shape[0], 1) - pc.get_xyz
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            sh2rgb =  eval_sh(pc.active_sh_degree, shs_view, -dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # treat rendered_image as albedo map


    opacity_mean, opacity_var = torch.mean(opacity), torch.var(opacity)
    bound_r = opacity_mean + 3 * opacity_var
    opacity_alpha = torch.exp(-torch.abs(bound_r - opacity))
    del opacity_mean, opacity_var
    min_axis= get_minimum_axis(rotations, scales)
    max_scale = scales.max(dim=1, keepdim=True)[0]
    min_scale = scales.min(dim=1, keepdim=True)[0]
    scale_weight = 1.0 -  min_scale / (max_scale + 1e-6)
    del max_scale, min_scale
    alpha = torch.clamp(0.5 * (opacity_alpha + scale_weight), 0.0, 1.0)
    dot_product = torch.sum(min_axis * dir_pp_normalized, axis=1)  # Shape: (N,)
    flip_mask = (dot_product < 0)
    adjusted_normal = torch.where(flip_mask.unsqueeze(1), -min_axis, min_axis)

    normal_bias = pc.get_normal

    normal = alpha * adjusted_normal +  (1.0 - alpha) * normal_bias
    normal = normal / normal.norm(dim=1, keepdim=True)

    material = torch.stack([pc.get_roughness, pc.get_metallic, pc.get_ao], dim = 1).squeeze()
    albedo = pc.get_albedo

    meta = torch.stack([scale_weight, torch.zeros_like(scale_weight), opacity], dim = 1).squeeze()


    out_extras = {}
    
    inputs = torch.cat([torch.zeros_like(albedo), albedo, normal * 0.5 + 0.5,  material], dim=-1)

    outs, radii = rasterizer1(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = inputs,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    out_extras["direct"] = outs[0:3, :, :]
    out_extras["albedo"] = outs[3:6, :, :]
    out_extras["normal"] = outs[6:9, :, :]
    out_extras["material"] = outs[9:12, :, :]


    one_alpha = torch.ones_like(opacity)
    out_extras["view"] = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = dir_pp_normalized * 0.5 + 0.5, 
        opacities = one_alpha,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
        )[0]
    
    out_extras["meta"] = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = meta,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
        )[0]
    


    out_extras["alpha_scale"] = out_extras["meta"][0:1, ...]
    mask_color = out_extras["meta"][1:2, ...]
    
    out_extras["mask"] = (mask_color < 1.0 - mask_eps)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    result = {
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
    result.update(out_extras)
    return result

