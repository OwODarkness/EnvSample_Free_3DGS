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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils.graphics_utils import rgb_to_gray

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def gaussian_blur(x, kernel_size=11, sigma=4.0):

    C, H, W = x.shape
    device = x.device

    # 1D Gaussian
    coords = torch.arange(kernel_size, device=device).float() - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    # 2D kernel
    kernel_2d = g[:, None] * g[None, :]
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1,1,k,k)

    # IMPORTANT: match image channels, NOT Gaussians
    kernel_2d = kernel_2d.repeat(C, 1, 1, 1)  # (C,1,k,k)

    padding = kernel_size // 2

    return F.conv2d(x, kernel_2d, padding=padding, groups=C)

def low_freq_loss(I_gt, D, S_low, kernel_size=11, sigma=3.0):

    residual = I_gt - D - S_low
    blurred = gaussian_blur(residual, kernel_size, sigma)

    loss = torch.mean(torch.abs(blurred))
    return loss

def entropy_loss(scale, eps=1e-6):
    m = scale.clamp(eps, 1.0-eps)
    loss = -(m * torch.log(m) + (1.0 - m) * torch.log(1.0 - m))
    return loss.mean()

def total_variation_loss(image):

    diff_h = torch.abs(image[ :, 1:, :] - image[ :, :-1, :])  # height differences
    diff_w = torch.abs(image[ :, :, 1:] - image[ :, :, :-1])  # width differences

    return torch.mean(diff_h) + torch.mean(diff_w)

def channel_grad_loss(image):
    loss = 0
    for c1, c2 in [(0, 1), (1, 2), (0, 2)]:
        grad_c1 = image[c1, 1:, :] - image[c1, :-1, :]
        grad_c2 = image[c2, 1:, :] - image[c2, :-1, :]
        loss += torch.mean(torch.abs(grad_c1 - grad_c2))
    return loss


def metal_roughness_constraint(metallic,  gt):
    gray_gt = rgb_to_gray(gt)

    very_rough_metal = torch.mean(
        (metallic > 0.8).float() * (gray_gt < 0.2).float()
    ) * 0.8

    very_glossy_non_metal = torch.mean(
        (metallic < 0.2).float() * (gray_gt > 0.8).float()
    ) * 0.2

    return very_rough_metal + very_glossy_non_metal


def correlation_aware_loss(metallic, roughness):

    wrong_correlation = torch.mean(metallic * roughness)  # Currently high
    right_correlation = torch.mean(metallic * (1 - roughness))  # Should allow this

    correlation_loss = torch.relu(wrong_correlation - right_correlation)
    return correlation_loss


def prefilter_loss(prefilter_map, roughness_map):
    """
    prefilter_map: [3, H, W]
    roughness_map: [1, H, W]
    """
    grad_x = torch.abs(prefilter_map[:, :, 1:] - prefilter_map[:, :, :-1])  # [3, H, W-1]
    grad_y = torch.abs(prefilter_map[:, 1:, :] - prefilter_map[:, :-1, :])  # [3, H-1, W]
    
    grad_x_mean = grad_x.mean(dim=0)  # [H, W-1]
    grad_y_mean = grad_y.mean(dim=0)  # [H-1, W]
    
    roughness = roughness_map.squeeze()  # [H, W]
    
    roughness_for_grad_x = roughness[:, :-1]  # [H, W-1]
    
    roughness_for_grad_y = roughness[:-1, :]  # [H-1, W]
    
    loss_x = (grad_x_mean * roughness_for_grad_x).mean()
    loss_y = (grad_y_mean * roughness_for_grad_y).mean()
    
    total_loss = (loss_x + loss_y) / 2
    
    return total_loss
