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
from scene import Scene
import os, time
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, pbr_shading
import torchvision
import numpy as np
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    output_types = ["normal"]

    # Always create GT directory
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(gts_path, exist_ok=True)

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)

    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")
    makedirs(normal_path, exist_ok=True)

    roughness_path = os.path.join(model_path, name, "ours_{}".format(iteration), "roughness")
    makedirs(roughness_path, exist_ok=True)

    metallic_path = os.path.join(model_path, name, "ours_{}".format(iteration), "metallic")
    makedirs(metallic_path, exist_ok=True)

    ao_path = os.path.join(model_path, name, "ours_{}".format(iteration), "ao")
    makedirs(ao_path, exist_ok=True)


    albedo_path = os.path.join(model_path, name, "ours_{}".format(iteration), "albedo")
    makedirs(albedo_path, exist_ok=True)

    diffuse_path = os.path.join(model_path, name, "ours_{}".format(iteration),"diffuse") 
    makedirs(diffuse_path, exist_ok=True)
    specular_path = os.path.join(model_path, name, "ours_{}".format(iteration),"specular") 
    makedirs(specular_path, exist_ok=True)
    prefilter_path = os.path.join(model_path, name, "ours_{}".format(iteration),"prefilter")
    makedirs(prefilter_path, exist_ok=True)
    
    brdf_path= os.path.join(model_path, name, "ours_{}".format(iteration),"brdf") 
    makedirs(brdf_path, exist_ok=True)
    
    implicit_path = os.path.join(model_path, name, "ours_{}".format(iteration),"implicit_specular") 
    makedirs(implicit_path, exist_ok=True)
 
    explicit_path = os.path.join(model_path, name, "ours_{}".format(iteration),"explicit_specular") 
    makedirs(explicit_path, exist_ok=True)



    mask_path = os.path.join(model_path, name, "ours_{}".format(iteration),"mask") 
    makedirs(mask_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        g_buffer = render(view, gaussians, pipeline, background, mode = "render")#, mask_eps = 0.5)
        masking = view.gt_alpha_mask is not None
        result = pbr_shading(g_buffer, gaussians.env,gaussians.nerual_specular, debug = True, masking = masking)
        render_res = result["render"]
        diffuse = result["diffuse"]
        specular = result["specular"]
        explicit_spec = result["explicit_specular"]
        implicit_spec = result["implicit_specular"]

        prefilter = result["prefilter"]
        brdf = result["brdf"]

        gt = view.original_image[0:3, :, :]
        normal = g_buffer["normal"]
        material = g_buffer["material"]
        roughness = material[0].unsqueeze(0).repeat_interleave(3, dim=0)
        metallic = material[1].unsqueeze(0).repeat_interleave(3, dim=0)
        ao= material[2].unsqueeze(0).repeat_interleave(3, dim=0)

        mask = g_buffer["mask"].float()
        albedo = g_buffer["albedo"]
        
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(render_res, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(normal, os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(roughness, os.path.join(roughness_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(metallic, os.path.join(metallic_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(mask, os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(ao, os.path.join(ao_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(albedo, os.path.join(albedo_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(diffuse, os.path.join(diffuse_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(specular, os.path.join(specular_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(explicit_spec, os.path.join(explicit_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(implicit_spec, os.path.join(implicit_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(prefilter, os.path.join(prefilter_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(brdf, os.path.join(brdf_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color= [1,1,1] if dataset.white_background else [0, 0, 0]

        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
