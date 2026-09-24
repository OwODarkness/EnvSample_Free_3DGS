#pragma once

#include <cuda_runtime.h>

void launchOursPbr(
    int width, int height,
    const float* direct,
    const float* albedo,
    const float* normalEncoded,
    const float* material,
    const float* view,
    const float* maskTransmittance,
    const float* environmentSh,
    const float* filter0Weight, const float* filter0Bias,
    const float* filter1Weight, const float* filter1WeightTransposed, const float* filter1Bias,
    const float* filter2Weight, const float* filter2Bias,
    const float* brdf0Weight, const float* brdf0Bias,
    const float* brdf1Weight, const float* brdf1WeightTransposed, const float* brdf1Bias,
    const float* brdf2Weight, const float* brdf2Bias,
    float* output,
    cudaStream_t stream = 0);

void launchBuildPbrAttributes(
    int count, int shDegree,
    const float* positions, const float* shs, const float* cameraPosition,
    const float* albedo, const float* normalAxis, const float* normalBias,
    const float* normalBlend, const float* material,
    float* attributes18, float* viewColors,
    cudaStream_t stream = 0);
