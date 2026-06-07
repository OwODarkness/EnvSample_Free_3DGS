# 面向无环境光采样的神经高斯泼溅物理渲染方法

**面向无环境光采样的神经高斯物理泼溅方法**官方实现

## Overview

**面向无环境光采样的神经高斯物理泼溅方法**提出了一种基于 **3D Gaussian Splatting（3DGS）** 的新型渲染框架，旨在提高3DGS对于镜面反射的表达，能够高效地完成场景重建，并实现具有真实感的新视角渲染。

------

## Method

 近年来，三维高斯泼溅开始通过物理反射模型增强镜面反射成分，以提升新视角合成质量。然而，这类方法通常需要在训练过程中对可微环境光进行采样和积分，使得渲染质量与计算效率难以兼顾。

为解决此类问题，提出一种面向无环境光采样的神经高斯泼溅物理渲染方法，通过轻量化双网络结构对低频镜面反射进行隐式建模，并结合球谐函数与Cook–Torrance模型补充高频信息，在无需环境光采样积分的条件下，解决同阶球谐函数难以同时刻画低频与高频信号所导致的伪影问题，实现不同频段反射成分的统一建模和高效渲染。另外，为克服对三维高斯图元进行表面法线估计的挑战，提出一种基于特征值引导的法线收缩策略，引导可靠表面法线重建。

![](docs/figs/pipeline.png)

------

## 安装

### 软硬件环境要求

- Ubuntu 22.04
- CUDA 11.8
- NVIDIA RTX 4080 / RTX 4090
- Python 3.10

### 克隆仓库

```bash
git clone https://github.com/OwODarkness/EnvSample_Free_3DGS.git
cd EnvSample_Free_3DGS
```

### 创建环境

```bash
conda env create -f environment.yml
conda activate envsample_free_3dgs
```

### 

------

## 数据集

我们的实验主要在 [Glossy Synthetic](https://liuyuan-pal.github.io/NeRO/), [Shiny Blender Real](https://storage.googleapis.com/gresearch/refraw360/ref_real.zip), [Glossy Real](https://liuyuan-pal.github.io/NeRO/)数据集上进行

## 训练



```bash
python train.py \
    -s dataset/luyu_blender \
    -m output/luyu_blender \
    --eval \
    -w
```

### 参数

| Argument    | Description  |
| ----------- | ------------ |
| `-s`        | 数据集路径   |
| `-m`        | 输出目录     |
| `--eval`    | 启用评估     |
| `-w`        | 白色背景     |
| --roughness | 粗糙度初始值 |
| --metallic  | 金属度初始值 |

> 对于 **Glossy Synthetic** 数据集，请使用 `--roughness 0.3 --metallic 0.7`，因为该数据集中的场景主要包含高反射率和金属材质表面。
>
> 对于 **Shiny Blender Real** 和 **Glossy Real** 数据集，请使用 `--roughness 0.7 --metallic 0.3`，因为这些数据集通常具有更粗糙的表面特性以及较弱的金属反射属性。

## 渲染

```bash
python render.py -m output/luyu_blender
```

------

## 结果

Glossy Synthetic

### 定性评估

![](docs/figs/compare.png)

### 定量评估

| Method  | PSNR ↑    | SSIM ↑    | LPIPS ↓   | Train Time | FPS  |
| ------- | --------- | --------- | --------- | ---------- | ---- |
| 3DGS    | 26.17     | 0.915     | 0.087     | 00:06:15   | 131  |
| GShader | 27.07     | 0.923     | 0.083     | 01:04:00   | 39   |
| Ours    | **27.75** | **0.929** | **0.075** | 00:20:21   | 57   |




