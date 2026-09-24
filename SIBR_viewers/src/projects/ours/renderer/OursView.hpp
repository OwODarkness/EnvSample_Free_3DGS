#pragma once

#include "Config.hpp"

#include <core/scene/BasicIBRScene.hpp>
#include <core/view/ViewBase.hpp>
#include <core/graphics/Texture.hpp>
#include <core/renderer/CopyRenderer.hpp>

#include <cuda_runtime.h>
#include <array>
#include <cstdint>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sibr {

/**
 * Loads the paper renderer's extended Gaussian PLY and applies its learned
 * environment and neural PBR shading in a CUDA post-process.
 */
class SIBR_EXP_OURS_EXPORT OursView final : public ViewBase {
	SIBR_CLASS_PTR(OursView);

public:
	OursView(const BasicIBRScene::Ptr& scene,
		uint render_w,
		uint render_h,
		const char* ply_file,
		int sh_degree,
		bool white_background,
		bool use_interop,
		int device);
	~OursView() override;

	void onRenderIBR(IRenderTarget& dst, const Camera& eye) override;
	void onUpdate(Input& input) override;
	void onGUI() override;
	const Vector3f& modelCenter() const { return _model_center; }
	float modelRadius() const { return _model_radius; }

private:
	class CudaBytes;

	void loadModel(const char* ply_file, int sh_degree);
	void allocateDeviceData(bool white_background);
	void releaseDeviceData();

	BasicIBRScene::Ptr _scene;
	int _device = 0;
	int _sh_degree = 3;
	int _count = 0;
	uint _render_w = 0;
	uint _render_h = 0;
	bool _use_interop = false;
	bool _white_background = false;
	Vector3f _model_center = Vector3f::Zero();
	float _model_radius = 1.0f;

	std::vector<float> _positions;
	std::vector<float> _rotations;
	std::vector<float> _scales;
	std::vector<float> _opacities;
	std::vector<float> _shs;
	std::vector<float> _albedo;
	std::vector<float> _normal_encoded;
	std::vector<float> _normal_axis;
	std::vector<float> _normal_bias;
	std::vector<float> _normal_blend;
	std::vector<float> _material;
	std::vector<float> _view_colors;
	std::vector<float> _mask_colors;
	std::vector<float> _filter0_weight, _filter0_bias, _filter1_weight, _filter1_bias, _filter2_weight, _filter2_bias;
	std::vector<float> _brdf0_weight, _brdf0_bias, _brdf1_weight, _brdf1_bias, _brdf2_weight, _brdf2_bias;
	std::vector<float> _environment_sh;

	float* _pos_cuda = nullptr;
	float* _rot_cuda = nullptr;
	float* _scale_cuda = nullptr;
	float* _opacity_cuda = nullptr;
	float* _shs_cuda = nullptr;
	float* _view_cuda = nullptr;
	float* _proj_cuda = nullptr;
	float* _cam_pos_cuda = nullptr;
	float* _background_cuda = nullptr;
	float* _image_cuda = nullptr;
	float* _raster_color_cuda = nullptr;
	float* _albedo_cuda = nullptr;
	float* _normal_cuda = nullptr;
	float* _material_cuda = nullptr;
	float* _mask_cuda = nullptr;
	float* _normal_axis_cuda = nullptr;
	float* _normal_bias_cuda = nullptr;
	float* _normal_blend_cuda = nullptr;
	float* _pbr_attributes_cuda = nullptr;
	float* _view_colors_cuda = nullptr;
	float* _white_background_cuda = nullptr;
	float* _unit_opacity_cuda = nullptr;
	float* _multi_background_cuda = nullptr;
	float* _pbr_images_cuda = nullptr;
	float* _environment_cuda = nullptr;
	std::array<float*, 12> _network_cuda{};
	float* _filter1_weight_transposed_cuda = nullptr;
	float* _brdf1_weight_transposed_cuda = nullptr;
	int* _rect_cuda = nullptr;

	std::unique_ptr<CudaBytes> _geometry;
	std::unique_ptr<CudaBytes> _binning;
	std::unique_ptr<CudaBytes> _image;
	std::function<char*(size_t)> _geometry_buffer;
	std::function<char*(size_t)> _binning_buffer;
	std::function<char*(size_t)> _image_buffer;

	std::vector<float> _image_host;
	std::shared_ptr<Texture2DRGB32F> _display_texture;
	std::unique_ptr<CopyRenderer> _copy_renderer;
	std::string _model_file;
	std::string _timing_log_path;
	bool _timing_enabled = false;
	std::array<cudaEvent_t, 8> _timing_events{};
	std::array<double, 5> _timing_accum_ms{};
	uint64_t _timing_frame_count = 0;
};

} // namespace sibr
