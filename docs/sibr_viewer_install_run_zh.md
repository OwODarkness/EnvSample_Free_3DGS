# SIBR Viewer 安装与运行指南（Windows）

本文介绍如何编译仓库中的 `SIBR_viewers/` 交互式查看器，并加载 EnvSample-Free-3DGS 训练得到的场景。步骤对应当前 Windows 版本代码。

## 环境要求

- 64 位 Windows 10 或 Windows 11。
- Visual Studio 2022，并安装 **使用 C++ 的桌面开发**。
- CMake 3.22 或更高版本。
- NVIDIA CUDA Toolkit 11.8 和较新的 NVIDIA 驱动。本查看器是在 CUDA 11.8 下编译的。
- 支持 CUDA 的 NVIDIA GPU。

查看器直接读取训练好的模型，运行时不需要 Python Conda 环境。训练模型或使用 Python 脚本渲染时才需要配置 Conda 环境。

## 配置与编译

先在 PowerShell 中切换到仓库根目录，再运行以下命令。构建目录位于仓库旁边，不会把 CMake 生成文件写入源码目录。

```powershell
$repo = (Resolve-Path .).Path
$viewer = Join-Path $repo 'SIBR_viewers'
$build = Join-Path (Split-Path $repo -Parent) 'EnvSample_Free_3DGS_SIBR_build'

cmake -S $viewer -B $build `
  -G 'Visual Studio 17 2022' -A x64 `
  -DCMAKE_CUDA_ARCHITECTURES=89 `
  -DBUILD_IBR_OURS=ON `
  -DBUILD_IBR_GAUSSIANVIEWER=ON

cmake --build $build --config Release --target SIBR_ours_app --parallel
```

`CMAKE_CUDA_ARCHITECTURES=89` 对应 Ada 架构显卡，例如 RTX 4070。如果使用其他 GPU，请按其 CUDA Compute Capability 修改；例如，Ampere 显卡通常使用 `86`。

SIBR 会将可运行程序和运行时 DLL 输出到：

```text
SIBR_viewers\install\bin\SIBR_ours_app.exe
```

CMake 生成的构建文件位于 `$build`。

## 加载训练场景

模型目录通常应包含以下 PLY 文件：

```text
<模型目录>\point_cloud\iteration_<编号>\point_cloud.ply
```

该 PLY 必须由本仓库的 PBR 训练流程生成，其中需要包含材质、法线、环境光和神经网络数据。普通 3DGS PLY 不包含这些信息，不能直接用于此查看器。

请从 `install\bin` 目录启动程序，使 Windows 能找到运行时 DLL：

```powershell
$repo = (Resolve-Path .).Path
$viewerBin = Join-Path $repo 'SIBR_viewers\install\bin'
Set-Location $viewerBin

& .\SIBR_ours_app.exe --model-path '<trained-scene-directory>' --vsync 0
```

省略 `--iteration` 时，程序会选择编号最大的 `iteration_*` 目录。也可以指定迭代编号：

```powershell
& .\SIBR_ours_app.exe --model-path '<trained-scene-directory>' --iteration 30000 --vsync 0
```

如果场景数据集元数据与模型不在同一目录，可用 `--path` 指定数据集目录：

```powershell
& .\SIBR_ours_app.exe --model-path '<trained-scene-directory>' --path '<dataset-directory>' --vsync 0
```

`--vsync 0` 会关闭垂直同步，适合测量未封顶帧率；省略该参数则使用默认垂直同步设置。`--device 0` 选择 CUDA 设备 0，也是默认值；需要时可更换设备编号。

## 相机控制

- 按 **O** 开始或停止围绕模型的自动环绕视角。
- 按 **Y** 在标准 FPS 控制和 Trackball 导航之间切换。
- 按 **Esc** 关闭查看器。

