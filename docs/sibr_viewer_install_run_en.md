# SIBR Viewer: Build and Run (Windows)

This guide builds the interactive SIBR viewer included in `SIBR_viewers/` and runs it with a trained EnvSample-Free-3DGS scene. It describes the Windows setup used for this checkout.

## Requirements

- Windows 10 or 11, 64-bit.
- Visual Studio 2022 with **Desktop development with C++** installed.
- CMake 3.22 or newer.
- NVIDIA CUDA Toolkit 11.8 and a current NVIDIA driver. CUDA 11.8 is the version used to build this viewer.
- An NVIDIA CUDA-capable GPU.

The viewer uses the trained model directly. The Python Conda environment is not needed to run it. A Conda environment is only needed if you also plan to train or render with the Python scripts.

## Configure and build

Open PowerShell at the repository root before running these commands. The build directory is a sibling of the repository, so generated CMake files do not go inside the source checkout.

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

`CMAKE_CUDA_ARCHITECTURES=89` targets Ada GPUs such as the RTX 4070. Change it to your GPU's CUDA compute capability if needed. For example, Ampere GPUs commonly use `86`.

SIBR writes the runnable executable and its runtime DLLs to:

```text
SIBR_viewers\install\bin\SIBR_ours_app.exe
```

The CMake-generated build files remain in `$build`.

## Run a trained scene

The model directory must contain the trained PLY, normally at:

```text
<model-directory>\point_cloud\iteration_<number>\point_cloud.ply
```

The PLY must be produced by this repository's PBR training pipeline. A standard 3DGS PLY without the material, normal, environment, and neural-network data is not sufficient.

Run the viewer from its `install\bin` directory so Windows can find the runtime DLLs:

```powershell
$repo = (Resolve-Path .).Path
$viewerBin = Join-Path $repo 'SIBR_viewers\install\bin'
Set-Location $viewerBin

& .\SIBR_ours_app.exe --model-path '<trained-scene-directory>' --vsync 0
```

When `--iteration` is omitted, the viewer selects the highest-numbered `iteration_*` folder. To choose one explicitly:

```powershell
& .\SIBR_ours_app.exe --model-path '<trained-scene-directory>' --iteration 30000 --vsync 0
```

If the scene dataset metadata is stored separately from the model directory, pass it with `--path`:

```powershell
& .\SIBR_ours_app.exe --model-path '<trained-scene-directory>' --path '<dataset-directory>' --vsync 0
```

`--vsync 0` disables vertical synchronization for uncapped frame-rate measurement. Omit it to use the default V-sync setting. `--device 0` selects CUDA device 0 and is the default; use another index when needed.

## Camera controls

- Press **O** to toggle automatic orbit around the model. Press **O** again to stop.
- Press **Y** to switch between the standard FPS controls and trackball navigation.
- Press **Esc** to close the viewer.

