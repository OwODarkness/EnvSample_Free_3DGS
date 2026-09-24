#pragma once

#include <core/system/Config.hpp>
#include <core/system/CommandLineArgs.hpp>

#if defined(SIBR_OS_WINDOWS) && !defined(SIBR_STATIC_DEFINE)
#  ifdef SIBR_EXP_OURS_EXPORTS
#    define SIBR_EXP_OURS_EXPORT __declspec(dllexport)
#  else
#    define SIBR_EXP_OURS_EXPORT __declspec(dllimport)
#  endif
#else
#  define SIBR_EXP_OURS_EXPORT
#endif

namespace sibr {

struct OursAppArgs : virtual BasicIBRAppArgs {
	RequiredArg<std::string> modelPath = { "model-path", "Model directory" };
	RequiredArg<std::string> modelPathShort = { "m", "Model directory" };
	Arg<std::string> iteration = { "iteration", "", "Iteration to load from model; empty selects the latest" };
	RequiredArg<std::string> pathShort = { "s", "Path to the dataset root" };
	Arg<int> device = { "device", 0, "CUDA device index" };
	Arg<bool> loadImages = { "load_images", "Whether or not to load images for scene overview." };
	Arg<bool> noInterop = { "no_interop", "Don't use CUDA/OpenGL interop." };
	Arg<int> shDegree = { "sh_degree", 3, "Spherical-harmonics degree stored in the model" };
	Arg<bool> whiteBackground = { "white_background", "Use a white rasterization background" };
};

} // namespace sibr
