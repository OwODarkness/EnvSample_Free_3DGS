project(sibr_ours LANGUAGES CXX)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Reuse the existing SIBR CUDA rasterizer target from gaussianviewer.
if(NOT TARGET sibr_gaussian)
	message(FATAL_ERROR "sibr_ours requires the gaussianviewer project to be enabled")
endif()

add_subdirectory(renderer)
add_subdirectory(apps)
