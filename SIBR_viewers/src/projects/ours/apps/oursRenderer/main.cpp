#include "projects/ours/renderer/Config.hpp"
#include "projects/ours/renderer/OursView.hpp"

#include <core/graphics/Window.hpp>
#include <core/view/MultiViewManager.hpp>
#include <core/view/SceneDebugView.hpp>

#include <boost/filesystem.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <regex>
#include <string>

namespace fs = boost::filesystem;
using namespace sibr;

static std::string findLatestIteration(const std::string& root) {
	int best = -1;
	std::string result;
	const fs::path path(root);
	if (!fs::exists(path) || !fs::is_directory(path)) {
		return result;
	}
	const std::regex pattern(R"(^iteration_(\d+)$)");
	for (const auto& entry : fs::directory_iterator(path)) {
		if (!fs::is_directory(entry)) {
			continue;
		}
		std::smatch match;
		const std::string name = entry.path().filename().string();
		if (std::regex_match(name, match, pattern)) {
			const int value = std::stoi(match[1].str());
			if (value > best) {
				best = value;
				result = name;
			}
		}
	}
	return result;
}

static std::string resolvePly(const std::string& model, const std::string& iteration) {
	fs::path point_cloud = fs::path(model) / "point_cloud";
	std::string selected = iteration;
	if (selected.empty() || selected == "-1") {
		selected = findLatestIteration(point_cloud.string());
	}
	if (selected.empty()) {
		throw std::runtime_error("No iteration_* directory found under " + point_cloud.string());
	}
	return (point_cloud / selected / "point_cloud.ply").string();
}

int main(int argc, char** argv) {
	CommandLineArgs::parseMainArgs(argc, argv);
	OursAppArgs args;
	args.displayHelpIfRequired();
	if (!args.modelPath.isInit() && args.modelPathShort.isInit()) {
		args.modelPath = args.modelPathShort.get();
	}
	if (!args.dataset_path.isInit() && args.pathShort.isInit()) {
		args.dataset_path = args.pathShort.get();
	}
	if (!args.modelPath.isInit()) {
		SIBR_ERR << "Missing --model-path (or -m).";
		return EXIT_FAILURE;
	}

	if (!args.dataset_path.isInit()) {
		args.dataset_path = args.modelPath.get();
	}

	uint render_w = args.rendering_size.get()[0] > 0 ? static_cast<uint>(args.rendering_size.get()[0]) : static_cast<uint>(args.win_width.get());
	uint render_h = args.rendering_size.get()[1] > 0 ? static_cast<uint>(args.rendering_size.get()[1]) : static_cast<uint>(args.win_height.get());
	BasicIBRScene::SceneOptions options;
	options.renderTargets = args.loadImages;
	options.mesh = false;
	options.images = args.loadImages;
	options.cameras = true;
	options.texture = false;
	BasicIBRScene::Ptr scene(new BasicIBRScene(args, options));
	if (args.rendering_size.get()[0] <= 0 || args.rendering_size.get()[1] <= 0) {
		const auto& cameras = scene->cameras()->inputCameras();
		if (!cameras.empty()) {
			render_w = cameras[0]->w();
			render_h = cameras[0]->h();
		}
	}

	const std::string ply = resolvePly(args.modelPath.get(), args.iteration.get());
	Window window("SIBR ours renderer", Vector2i(50, 50), args,
		getResourcesDirectory() + "/ours/sibr_ours_app.ini");
	const Vector2u resolution(render_w, render_h);
	OursView::Ptr view(new OursView(scene, render_w, render_h, ply.c_str(),
		args.shDegree, args.whiteBackground, !args.noInterop, args.device));

	InteractiveCameraHandler::Ptr camera(new InteractiveCameraHandler());
	camera->setup(scene->cameras()->inputCameras(),
		Viewport(0, 0, static_cast<float>(render_w), static_cast<float>(render_h)), nullptr);
	MultiViewManager manager(window, false);
	manager.addIBRSubView("OURS", view, resolution,
		ImGuiWindowFlags_ResizeFromAnySide | ImGuiWindowFlags_NoBringToFrontOnFocus);
	manager.addCameraForView("OURS", camera);
	bool auto_orbit = false;
	float orbit_angle = 0.0f;
	const Vector3f orbit_center = view->modelCenter();
	const Vector3f orbit_up = camera->getCamera().up().normalized();
	const Vector3f initial_offset = camera->getCamera().position() - orbit_center;
	const float orbit_height = initial_offset.dot(orbit_up);
	const Vector3f initial_radial = initial_offset - orbit_height * orbit_up;
	const Vector3f orbit_radial = initial_radial.norm() > 1e-4f ? initial_radial.normalized() : Vector3f::UnitX();
	const float orbit_radius = std::max(initial_radial.norm(), 1.8f * view->modelRadius());
	auto previous_frame = std::chrono::steady_clock::now();
	SIBR_LOG << "Press O to toggle automatic camera orbit." << std::endl;

	while (window.isOpened()) {
		Input::poll();
		window.makeContextCurrent();
		if (Input::global().key().isPressed(Key::Escape)) {
			window.close();
		}
		if (Input::global().key().isReleased(Key::O)) {
			auto_orbit = !auto_orbit;
			SIBR_LOG << "Automatic camera orbit " << (auto_orbit ? "enabled" : "disabled") << std::endl;
		}
		manager.onUpdate(Input::global());
		const auto now = std::chrono::steady_clock::now();
		const float delta_seconds = std::chrono::duration<float>(now - previous_frame).count();
		previous_frame = now;
		if (auto_orbit) {
			orbit_angle += delta_seconds * 0.25f;
			InputCamera orbit_camera = camera->getCamera();
			const Vector3f radial = std::cos(orbit_angle) * orbit_radial +
				std::sin(orbit_angle) * orbit_up.cross(orbit_radial);
			const Vector3f eye = orbit_center + orbit_height * orbit_up + orbit_radius * radial;
			orbit_camera.setLookAt(eye, orbit_center, orbit_up);
			camera->fromCamera(orbit_camera, false, false);
		}
		manager.onRender(window);
		window.swapBuffer();
	}
	return EXIT_SUCCESS;
}

