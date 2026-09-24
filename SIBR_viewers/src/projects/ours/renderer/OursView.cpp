#include "OursView.hpp"
#include "OursPbr.cuh"

#include <core/graphics/GUI.hpp>

#include <rasterizer.h>
#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <chrono>

namespace {

float sigmoid(float x) {
	return 1.0f / (1.0f + std::exp(-x));
}

struct PlyElementData {
	std::size_t count = 0;
	std::size_t stride = 0;
	std::unordered_map<std::string, std::size_t> fields;
	std::vector<float> values;
};

struct PlyData {
	std::unordered_map<std::string, PlyElementData> elements;
};

PlyData readBinaryPly(const char* filename, int sh_degree) {
	std::ifstream input(filename, std::ios::binary);
	if (!input) throw std::runtime_error(std::string("Unable to open PLY: ") + filename);

	struct HeaderElement { std::string name; std::size_t count; std::vector<std::string> fields; };
	std::vector<HeaderElement> header;
	std::string line;
	bool binary_little_endian = false;
	while (std::getline(input, line)) {
		std::istringstream tokens(line);
		std::string first; tokens >> first;
		if (first == "format") {
			std::string format; tokens >> format;
			binary_little_endian = format == "binary_little_endian";
		} else if (first == "element") {
			HeaderElement element; tokens >> element.name >> element.count;
			header.push_back(std::move(element));
		} else if (first == "property" && !header.empty()) {
			std::string type, name; tokens >> type >> name;
			if (type == "list" || (type != "float" && type != "float32"))
				throw std::runtime_error("Expected scalar float PLY properties");
			header.back().fields.push_back(name);
		} else if (first == "end_header") break;
	}
	if (!binary_little_endian) throw std::runtime_error("Expected a binary little-endian PLY");

	PlyData result;
	for (const HeaderElement& item : header) {
		PlyElementData element;
		element.count = item.count;
		element.stride = item.fields.size();
		for (std::size_t i = 0; i < item.fields.size(); ++i) element.fields.emplace(item.fields[i], i);
		if (element.stride && item.count > std::numeric_limits<std::size_t>::max() / element.stride)
			throw std::runtime_error("PLY element size overflows addressable memory");
		element.values.resize(item.count * element.stride);
		input.read(reinterpret_cast<char*>(element.values.data()),
			static_cast<std::streamsize>(element.values.size() * sizeof(float)));
		if (!input) throw std::runtime_error("PLY element payload is truncated: " + item.name);
		result.elements.emplace(item.name, std::move(element));
	}
	const std::size_t sh_count = static_cast<std::size_t>(sh_degree + 1) * (sh_degree + 1);
	const auto& vertex = result.elements.at("vertex");
	for (const char* name : {"x", "y", "z", "nx", "ny", "nz", "ax", "ay", "az", "opacity", "roughness", "metallic", "ao"})
		if (vertex.fields.find(name) == vertex.fields.end()) throw std::runtime_error(std::string("Missing PLY vertex property: ") + name);
	for (int i = 0; i < 3; ++i) {
		if (vertex.fields.find("scale_" + std::to_string(i)) == vertex.fields.end() ||
			vertex.fields.find("f_dc_" + std::to_string(i)) == vertex.fields.end())
			throw std::runtime_error("Missing scale or SH-DC PLY properties");
	}
	for (int i = 0; i < 4; ++i) if (vertex.fields.find("rot_" + std::to_string(i)) == vertex.fields.end()) throw std::runtime_error("Missing PLY rotation property");
	for (std::size_t i = 0; i < 3 * (sh_count - 1); ++i)
		if (vertex.fields.find("f_rest_" + std::to_string(i)) == vertex.fields.end()) throw std::runtime_error("Missing PLY SH-rest property");
	for (const char* name : {"environment", "filter_net", "brdf_net"})
		if (result.elements.find(name) == result.elements.end()) throw std::runtime_error(std::string("Missing PLY element: ") + name);
	return result;
}

const PlyElementData& elementData(const PlyData& ply, const std::string& name) {
	return ply.elements.at(name);
}
std::size_t fieldIndex(const PlyElementData& element, const std::string& name) { return element.fields.at(name); }
float fieldValue(const PlyElementData& element, const std::string& name) {
	if (element.count == 0) throw std::runtime_error("Empty PLY element");
	return element.values.at(fieldIndex(element, name));
}
void readLayer(const PlyElementData& element, const std::string& prefix, int weightLayer, int biasLayer,
	int outCount, int inCount, std::vector<float>& weights, std::vector<float>& biases) {
	weights.resize(static_cast<std::size_t>(outCount) * inCount);
	biases.resize(outCount);
	const std::string weightBase = prefix + "_layer" + std::to_string(weightLayer) + "_weight_";
	const std::string biasBase = prefix + "_layer" + std::to_string(biasLayer) + "_bias_";
	for (int o = 0; o < outCount; ++o) {
		biases[o] = fieldValue(element, biasBase + std::to_string(o));
		for (int i = 0; i < inCount; ++i)
			weights[static_cast<std::size_t>(o) * inCount + i] = fieldValue(element,
				weightBase + std::to_string(o) + "_" + std::to_string(i));
	}
}
template <typename T>
void copyToDevice(T*& dst, const std::vector<T>& src) {
	if (src.empty()) {
		return;
	}
	cudaMalloc(reinterpret_cast<void**>(&dst), src.size() * sizeof(T));
	cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);
}

} // namespace

namespace sibr {

class OursView::CudaBytes {
public:
	CudaBytes() = default;
	CudaBytes(const CudaBytes&) = delete;
	CudaBytes& operator=(const CudaBytes&) = delete;
	~CudaBytes() {
		if (_ptr != nullptr) {
			cudaFree(_ptr);
		}
	}

	char* resize(std::size_t bytes) {
		if (bytes > _size) {
			if (_ptr != nullptr) {
				cudaFree(_ptr);
			}
			cudaMalloc(&_ptr, bytes * 2);
			_size = bytes * 2;
		}
		return reinterpret_cast<char*>(_ptr);
	}

private:
	void* _ptr = nullptr;
	std::size_t _size = 0;
};

OursView::OursView(const BasicIBRScene::Ptr& scene,
	uint render_w,
	uint render_h,
	const char* ply_file,
	int sh_degree,
	bool white_background,
	bool use_interop,
	int device)
	: ViewBase(render_w, render_h),
	_scene(scene),
	_device(device),
	_sh_degree(std::max(0, std::min(sh_degree, 3))),
	_render_w(render_w),
	_render_h(render_h),
	_use_interop(use_interop),
	_white_background(white_background),
	_model_file(ply_file != nullptr ? ply_file : "") {
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	if (device_count <= 0 || _device < 0 || _device >= device_count) {
		throw std::runtime_error("No valid CUDA device is available for SIBR_ours");
	}
	cudaSetDevice(_device);
	if (const char* timing_path = std::getenv("SIBR_OURS_TIMING_LOG")) {
		_timing_enabled = true;
		_timing_log_path = timing_path;
		for (cudaEvent_t& event : _timing_events) cudaEventCreate(&event);
	}
	loadModel(ply_file, _sh_degree);
	allocateDeviceData(white_background);

	_image_host.resize(static_cast<std::size_t>(_render_w) * _render_h * 3);
	ImageRGB32F initial(_render_w, _render_h);
	_display_texture.reset(new Texture2DRGB32F(initial));
	_copy_renderer.reset(new CopyRenderer());
	_copy_renderer->flip() = true;
}

OursView::~OursView() {
	releaseDeviceData();
}

void OursView::loadModel(const char* ply_file, int sh_degree) {
	const PlyData ply = readBinaryPly(ply_file, sh_degree);
	const PlyElementData& vertex = elementData(ply, "vertex");
	const PlyElementData& environment = elementData(ply, "environment");
	const std::size_t stride = vertex.stride;
	_count = static_cast<int>(vertex.count);
	const std::size_t sh_count = static_cast<std::size_t>(sh_degree + 1) * (sh_degree + 1);
	const std::size_t x_index = fieldIndex(vertex, "x"), y_index = fieldIndex(vertex, "y"), z_index = fieldIndex(vertex, "z");
	const std::size_t opacity_index = fieldIndex(vertex, "opacity");
	std::size_t scale_index[3], rotation_index[4], dc_index[3];
	std::vector<std::size_t> rest_index(3 * (sh_count - 1));
	for (int i = 0; i < 3; ++i) {
		scale_index[i] = fieldIndex(vertex, "scale_" + std::to_string(i));
		dc_index[i] = fieldIndex(vertex, "f_dc_" + std::to_string(i));
	}
	for (int i = 0; i < 4; ++i) rotation_index[i] = fieldIndex(vertex, "rot_" + std::to_string(i));
	for (std::size_t i = 0; i < 3 * (sh_count - 1); ++i) rest_index[i] = fieldIndex(vertex, "f_rest_" + std::to_string(i));

	_positions.resize(vertex.count * 3);
	Vector3f bounds_min = Vector3f::Constant(std::numeric_limits<float>::max());
	Vector3f bounds_max = Vector3f::Constant(std::numeric_limits<float>::lowest());
	_rotations.resize(vertex.count * 4);
	_scales.resize(vertex.count * 3);
	_opacities.resize(vertex.count);
	_shs.assign(vertex.count * sh_count * 3, 0.0f);
	_albedo.resize(vertex.count * 3);
	_normal_encoded.resize(vertex.count * 3);
	_normal_axis.resize(vertex.count * 3);
	_normal_bias.resize(vertex.count * 3);
	_normal_blend.resize(vertex.count);
	_material.resize(vertex.count * 3);
	float opacity_mean = 0.0f;
	for (std::size_t i = 0; i < vertex.count; ++i) opacity_mean += sigmoid(vertex.values[i * stride + opacity_index]);
	opacity_mean /= static_cast<float>(vertex.count);
	float opacity_variance = 0.0f;
	for (std::size_t i = 0; i < vertex.count; ++i) {
		const float delta = sigmoid(vertex.values[i * stride + opacity_index]) - opacity_mean;
		opacity_variance += delta * delta;
	}
	if (vertex.count > 1) opacity_variance /= static_cast<float>(vertex.count - 1);
	const float opacity_bound = opacity_mean + 3.0f * opacity_variance;
	for (std::size_t i = 0; i < vertex.count; ++i) {
		const float* src = vertex.values.data() + i * stride;
		_positions[i * 3 + 0] = src[x_index];
		_positions[i * 3 + 1] = src[y_index];
		_positions[i * 3 + 2] = src[z_index];
		for (int axis = 0; axis < 3; ++axis) {
			bounds_min[axis] = std::min(bounds_min[axis], _positions[i * 3 + axis]);
			bounds_max[axis] = std::max(bounds_max[axis], _positions[i * 3 + axis]);
		}
		for (int c = 0; c < 3; ++c) {
			_scales[i * 3 + c] = std::exp(src[scale_index[c]]);
			_shs[i * sh_count * 3 + c] = src[dc_index[c]];
			_albedo[i * 3 + c] = sigmoid(src[fieldIndex(vertex, std::string("a") + "xyz"[c])]);
			_material[i * 3 + c] = sigmoid(src[fieldIndex(vertex, c == 0 ? "roughness" : (c == 1 ? "metallic" : "ao"))]);
		}
		float qnorm = 0.0f;
		for (int c = 0; c < 4; ++c) { _rotations[i * 4 + c] = src[rotation_index[c]]; qnorm += src[rotation_index[c]] * src[rotation_index[c]]; }
		qnorm = std::sqrt(std::max(qnorm, 1e-12f));
		for (int c = 0; c < 4; ++c) _rotations[i * 4 + c] /= qnorm;
		_opacities[i] = sigmoid(src[opacity_index]);
		for (std::size_t sh = 1; sh < sh_count; ++sh)
			for (std::size_t channel = 0; channel < 3; ++channel)
				_shs[i * sh_count * 3 + sh * 3 + channel] = src[rest_index[channel * (sh_count - 1) + sh - 1]];

		const float sx = _scales[i * 3], sy = _scales[i * 3 + 1], sz = _scales[i * 3 + 2];
		const int min_axis = sx <= sy && sx <= sz ? 0 : (sy <= sz ? 1 : 2);
		const float q0=_rotations[i*4], q1=_rotations[i*4+1], q2=_rotations[i*4+2], q3=_rotations[i*4+3];
		const float r[3][3] = {
			{1-2*(q2*q2+q3*q3), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)},
			{2*(q1*q2+q0*q3), 1-2*(q1*q1+q3*q3), 2*(q2*q3-q0*q1)},
			{2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1*q1+q2*q2)}
		};
		float axis[3] = { r[0][min_axis], r[1][min_axis], r[2][min_axis] };
		const float normal[3] = { src[fieldIndex(vertex, "nx")], src[fieldIndex(vertex, "ny")], src[fieldIndex(vertex, "nz")] };
		const float max_scale = std::max(sx, std::max(sy, sz));
		const float min_scale = std::min(sx, std::min(sy, sz));
		const float scale_weight = 1.0f - min_scale / (max_scale + 1e-6f);
		const float opacity_alpha = std::exp(-std::abs(opacity_bound - _opacities[i]));
		_normal_blend[i] = std::max(0.0f, std::min(1.0f, 0.5f * (opacity_alpha + scale_weight)));
		for (int c = 0; c < 3; ++c) { _normal_axis[i*3+c] = axis[c]; _normal_bias[i*3+c] = normal[c]; }
	}

	_model_center = 0.5f * (bounds_min + bounds_max);
	_model_radius = std::max(0.01f, 0.5f * (bounds_max - bounds_min).norm());
	_environment_sh.resize(27);
	for (int channel = 0; channel < 3; ++channel) {
		_environment_sh[channel * 9] = fieldValue(environment, "f_dc_" + std::to_string(channel));
		for (int sh = 1; sh < 9; ++sh)
			_environment_sh[channel * 9 + sh] = fieldValue(environment, "f_rest_" + std::to_string((sh - 1) * 3 + channel));
	}
	const PlyElementData& filter = elementData(ply, "filter_net");
	const PlyElementData& brdf = elementData(ply, "brdf_net");
	readLayer(filter, "filter", 0, 1, 128, 7, _filter0_weight, _filter0_bias);
	readLayer(filter, "filter", 2, 3, 128, 128, _filter1_weight, _filter1_bias);
	readLayer(filter, "filter", 4, 5, 3, 128, _filter2_weight, _filter2_bias);
	readLayer(brdf, "brdf", 0, 1, 32, 2, _brdf0_weight, _brdf0_bias);
	readLayer(brdf, "brdf", 2, 3, 32, 32, _brdf1_weight, _brdf1_bias);
	readLayer(brdf, "brdf", 4, 5, 2, 32, _brdf2_weight, _brdf2_bias);
	_view_colors.resize(vertex.count * 3);
	_mask_colors.assign(vertex.count * 3, 0.0f);
	SIBR_LOG << "SIBR_ours loaded " << _count << " PBR splats from " << _model_file << std::endl;
}void OursView::allocateDeviceData(bool white_background) {
	copyToDevice(_pos_cuda, _positions);
	copyToDevice(_rot_cuda, _rotations);
	copyToDevice(_scale_cuda, _scales);
	copyToDevice(_opacity_cuda, _opacities);
	copyToDevice(_shs_cuda, _shs);
	copyToDevice(_environment_cuda, _environment_sh);
	std::vector<float>* network_data[12] = {
		&_filter0_weight, &_filter0_bias, &_filter1_weight, &_filter1_bias, &_filter2_weight, &_filter2_bias,
		&_brdf0_weight, &_brdf0_bias, &_brdf1_weight, &_brdf1_bias, &_brdf2_weight, &_brdf2_bias
	};
	for (std::size_t i = 0; i < _network_cuda.size(); ++i) copyToDevice(_network_cuda[i], *network_data[i]);
	std::vector<float> filter1_transposed(_filter1_weight.size());
	for (int o = 0; o < 128; ++o)
		for (int i = 0; i < 128; ++i)
			filter1_transposed[static_cast<std::size_t>(i) * 128 + o] = _filter1_weight[static_cast<std::size_t>(o) * 128 + i];
	copyToDevice(_filter1_weight_transposed_cuda, filter1_transposed);
	std::vector<float> brdf1_transposed(_brdf1_weight.size());
	for (int o = 0; o < 32; ++o)
		for (int i = 0; i < 32; ++i)
			brdf1_transposed[static_cast<std::size_t>(i) * 32 + o] = _brdf1_weight[static_cast<std::size_t>(o) * 32 + i];
	copyToDevice(_brdf1_weight_transposed_cuda, brdf1_transposed);
	cudaMalloc(reinterpret_cast<void**>(&_view_cuda), sizeof(Matrix4f));
	cudaMalloc(reinterpret_cast<void**>(&_proj_cuda), sizeof(Matrix4f));
	cudaMalloc(reinterpret_cast<void**>(&_cam_pos_cuda), 3 * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&_background_cuda), 3 * sizeof(float));
	const std::size_t pixel_count = static_cast<std::size_t>(_render_w) * _render_h;
	cudaMalloc(reinterpret_cast<void**>(&_image_cuda), 3 * pixel_count * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&_raster_color_cuda), 3 * static_cast<std::size_t>(_count) * sizeof(float));
	copyToDevice(_albedo_cuda, _albedo);
	copyToDevice(_normal_cuda, _normal_encoded);
	copyToDevice(_normal_axis_cuda, _normal_axis);
	copyToDevice(_normal_bias_cuda, _normal_bias);
	copyToDevice(_normal_blend_cuda, _normal_blend);
	copyToDevice(_material_cuda, _material);
	copyToDevice(_mask_cuda, _mask_colors);
	cudaMalloc(reinterpret_cast<void**>(&_white_background_cuda), 3 * sizeof(float));
	const float white[3] = { 1.0f, 1.0f, 1.0f };
	cudaMemcpy(_white_background_cuda, white, sizeof(white), cudaMemcpyHostToDevice);
	cudaMalloc(reinterpret_cast<void**>(&_unit_opacity_cuda), static_cast<std::size_t>(_count) * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&_pbr_attributes_cuda), 18 * static_cast<std::size_t>(_count) * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&_view_colors_cuda), 3 * static_cast<std::size_t>(_count) * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&_multi_background_cuda), 18 * sizeof(float));
	cudaMalloc(reinterpret_cast<void**>(&_pbr_images_cuda), 18 * pixel_count * sizeof(float));
	const std::vector<float> unit_opacity(static_cast<std::size_t>(_count), 1.0f);
	cudaMemcpy(_unit_opacity_cuda, unit_opacity.data(), unit_opacity.size() * sizeof(float), cudaMemcpyHostToDevice);
	const float background[3] = { white_background ? 1.0f : 0.0f, white_background ? 1.0f : 0.0f, white_background ? 1.0f : 0.0f };
	cudaMemcpy(_background_cuda, background, sizeof(background), cudaMemcpyHostToDevice);
	float multi_background[18];
	for (int c = 0; c < 15; ++c) multi_background[c] = background[c % 3];
	for (int c = 15; c < 18; ++c) multi_background[c] = 1.0f;
	cudaMemcpy(_multi_background_cuda, multi_background, sizeof(multi_background), cudaMemcpyHostToDevice);
	_geometry.reset(new CudaBytes());
	_binning.reset(new CudaBytes());
	_image.reset(new CudaBytes());
	_geometry_buffer = [this](std::size_t n) { return _geometry->resize(n); };
	_binning_buffer = [this](std::size_t n) { return _binning->resize(n); };
	_image_buffer = [this](std::size_t n) { return _image->resize(n); };
}
void OursView::releaseDeviceData() {
	for (cudaEvent_t& event : _timing_events) {
		if (event != nullptr) { cudaEventDestroy(event); event = nullptr; }
	}
	const auto free_device = [](float*& ptr) { if (ptr != nullptr) { cudaFree(ptr); ptr = nullptr; } };
	free_device(_pos_cuda); free_device(_rot_cuda); free_device(_scale_cuda); free_device(_opacity_cuda); free_device(_shs_cuda);
	free_device(_view_cuda); free_device(_proj_cuda); free_device(_cam_pos_cuda); free_device(_background_cuda); free_device(_image_cuda);
	free_device(_raster_color_cuda); free_device(_albedo_cuda); free_device(_normal_cuda); free_device(_material_cuda);
	free_device(_mask_cuda); free_device(_normal_axis_cuda); free_device(_normal_bias_cuda); free_device(_normal_blend_cuda);
	free_device(_pbr_attributes_cuda); free_device(_view_colors_cuda); free_device(_white_background_cuda);
	free_device(_unit_opacity_cuda); free_device(_multi_background_cuda); free_device(_pbr_images_cuda); free_device(_environment_cuda);
	free_device(_filter1_weight_transposed_cuda); free_device(_brdf1_weight_transposed_cuda);
	for (float*& ptr : _network_cuda) free_device(ptr);
	if (_rect_cuda != nullptr) { cudaFree(_rect_cuda); _rect_cuda = nullptr; }
}
void OursView::onRenderIBR(IRenderTarget& dst, const Camera& eye) {
	const auto frame_start = std::chrono::steady_clock::now();
	Matrix4f view = eye.view();
	Matrix4f projection = eye.viewproj();
	view.row(1) *= -1.0f;
	view.row(2) *= -1.0f;
	projection.row(1) *= -1.0f;
	const float tan_fovy = std::tan(eye.fovy() * 0.5f);
	const float tan_fovx = tan_fovy * eye.aspect();
	cudaMemcpy(_view_cuda, view.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice);
	cudaMemcpy(_proj_cuda, projection.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice);
	cudaMemcpy(_cam_pos_cuda, eye.position().data(), 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (_timing_enabled) cudaEventRecord(_timing_events[0]);
	launchBuildPbrAttributes(_count, _sh_degree, _pos_cuda, _shs_cuda, _cam_pos_cuda,
		_albedo_cuda, _normal_axis_cuda, _normal_bias_cuda, _normal_blend_cuda, _material_cuda,
		_pbr_attributes_cuda, _view_colors_cuda);
	if (_timing_enabled) cudaEventRecord(_timing_events[1]);
	auto raster = [&](const float* shs, const float* colors, const float* opacity, const float* background, float* output, int channels) {
		CudaRasterizer::Rasterizer::forward(
			_geometry_buffer, _binning_buffer, _image_buffer,
			_count, _sh_degree, 16, background, _render_w, _render_h,
			_pos_cuda, shs, colors, opacity, _scale_cuda, 1.0f,
			_rot_cuda, nullptr, _view_cuda, _proj_cuda, _cam_pos_cuda,
			tan_fovx, tan_fovy, false, output, false, nullptr, _rect_cuda, nullptr, nullptr, channels);
	};
	const std::size_t pixels = static_cast<std::size_t>(_render_w) * _render_h;
	const std::size_t image_bytes = 3 * pixels * sizeof(float);
	if (_timing_enabled) cudaEventRecord(_timing_events[2]);
	raster(nullptr, _pbr_attributes_cuda, _opacity_cuda, _multi_background_cuda, _pbr_images_cuda, 18);
	if (_timing_enabled) cudaEventRecord(_timing_events[3]);
	if (_timing_enabled) cudaEventRecord(_timing_events[4]);
	raster(nullptr, _view_colors_cuda, _unit_opacity_cuda, _background_cuda, _image_cuda, 3);
	cudaMemcpy(_pbr_images_cuda + 12 * pixels, _image_cuda, image_bytes, cudaMemcpyDeviceToDevice);
	if (_timing_enabled) cudaEventRecord(_timing_events[5]);

	const float* base = _pbr_images_cuda;
	if (_timing_enabled) cudaEventRecord(_timing_events[6]);
	launchOursPbr(static_cast<int>(_render_w), static_cast<int>(_render_h),
		base, base + 3*pixels, base + 6*pixels, base + 9*pixels, base + 12*pixels, base + 15*pixels,
		_environment_cuda,
		_network_cuda[0], _network_cuda[1], _network_cuda[2], _filter1_weight_transposed_cuda, _network_cuda[3], _network_cuda[4], _network_cuda[5],
		_network_cuda[6], _network_cuda[7], _network_cuda[8], _brdf1_weight_transposed_cuda, _network_cuda[9], _network_cuda[10], _network_cuda[11],
		_image_cuda);
	if (_timing_enabled) cudaEventRecord(_timing_events[7]);
	cudaMemcpy(_image_host.data(), _image_cuda, image_bytes, cudaMemcpyDeviceToHost);
	ImageRGB32F image(_render_w, _render_h);
	for (uint y = 0; y < _render_h; ++y) {
		for (uint x = 0; x < _render_w; ++x) {
			const std::size_t pixel = static_cast<std::size_t>(y) * _render_w + x;
			image(x, y)[0] = _image_host[pixel];
			image(x, y)[1] = _image_host[pixels + pixel];
			image(x, y)[2] = _image_host[2 * pixels + pixel];
		}
	}
	_display_texture->update(image);
	_copy_renderer->process(_display_texture->handle(), dst);
	if (_timing_enabled) {
		float stage_ms[4]{};
		for (int stage = 0; stage < 4; ++stage)
			cudaEventElapsedTime(&stage_ms[stage], _timing_events[stage*2], _timing_events[stage*2+1]);
		const double frame_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frame_start).count();
		_timing_accum_ms[0] += frame_ms;
		for (int stage = 0; stage < 4; ++stage) _timing_accum_ms[stage+1] += stage_ms[stage];
		if (++_timing_frame_count == 60) {
			std::ifstream existing_file(_timing_log_path, std::ios::ate);
			const bool needs_header = !existing_file || existing_file.tellg() == 0;
			std::ofstream timing_file(_timing_log_path, std::ios::app);
			if (needs_header)
				timing_file << "frames,frame_ms,attributes_gpu_ms,multichannel_raster_gpu_ms,view_raster_gpu_ms,pbr_gpu_ms\n";
			timing_file << _timing_frame_count;
			for (double value : _timing_accum_ms) timing_file << ',' << value / _timing_frame_count;
			timing_file << '\n';
			_timing_accum_ms.fill(0.0);
			_timing_frame_count = 0;
		}
	}
}
void OursView::onUpdate(Input&) {}

void OursView::onGUI() {
	if (ImGui::Begin("OURS renderer")) {
		ImGui::Text("Loaded model: %s", _model_file.c_str());
		ImGui::Text("Splats: %d", _count);
		ImGui::Text("Backend: SIBR CUDA Gaussian rasterization");
		ImGui::Text("Appearance: learned neural PBR + environment SH.");
	}
	ImGui::End();
}

} // namespace sibr


