#include "OursPbr.cuh"

#include <cmath>

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kC0 = 0.28209479177387814f;
constexpr float kC1 = 0.4886025119029199f;


__device__ float clamp01(float x) { return fminf(1.0f, fmaxf(0.0f, x)); }
__device__ float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
__device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ float dot3(const float* a, const float* b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
__device__ void normalize3(float* a) {
    const float n = sqrtf(dot3(a, a));
    if (n > 1e-8f) { a[0] /= n; a[1] /= n; a[2] /= n; }
}
__device__ void evalEnvSh(const float* sh, const float* n, float* out) {
    const float x=n[0], y=n[1], z=n[2], xx=x*x, yy=y*y, zz=z*z, xy=x*y, yz=y*z, xz=x*z;
    for (int c=0; c<3; ++c) {
        const float* f=sh+c*9;
        out[c]=kC0*f[0] - kC1*y*f[1] + kC1*z*f[2] - kC1*x*f[3]
            + 1.0925484305920792f*xy*f[4] - 1.0925484305920792f*yz*f[5] + 0.31539156525252005f*(2.0f*zz-xx-yy)*f[6] - 1.0925484305920792f*xz*f[7] + 0.5462742152960396f*(xx-yy)*f[8];
    }
}
__device__ void dense(const float* input, int inCount, int outCount, const float* weight, const float* bias, float* output, bool relu) {
    for (int o=0; o<outCount; ++o) {
        float v=bias[o];
        for (int i=0; i<inCount; ++i) v += weight[o*inCount+i]*input[i];
        output[o]=relu ? fmaxf(v, 0.0f) : v;
    }
}

__device__ float evalGaussianSh(const float* sh, int degree, const float* dir) {
    const float x = dir[0], y = dir[1], z = dir[2];
    const float xx = x*x, yy = y*y, zz = z*z, xy = x*y, yz = y*z, xz = x*z;
    float value = kC0 * sh[0];
    if (degree > 0) {
        value += -kC1*y*sh[1] + kC1*z*sh[2] - kC1*x*sh[3];
    }
    if (degree > 1) {
        value += 1.0925484305920792f*xy*sh[4] - 1.0925484305920792f*yz*sh[5]
            + 0.31539156525252005f*(2.0f*zz-xx-yy)*sh[6] - 1.0925484305920792f*xz*sh[7]
            + 0.5462742152960396f*(xx-yy)*sh[8];
    }
    if (degree > 2) {
        value += -0.5900435899266435f*y*(3.0f*xx-yy)*sh[9] + 2.890611442640554f*xy*z*sh[10]
            - 0.4570457994644658f*y*(4.0f*zz-xx-yy)*sh[11]
            + 0.3731763325901154f*z*(2.0f*zz-3.0f*xx-3.0f*yy)*sh[12]
            - 0.4570457994644658f*x*(4.0f*zz-xx-yy)*sh[13]
            + 1.445305721320277f*z*(xx-yy)*sh[14]
            - 0.5900435899266435f*x*(xx-3.0f*yy)*sh[15];
    }
    return fmaxf(value + 0.5f, 0.0f);
}

__global__ void buildPbrAttributesKernel(
    int count, int degree, const float* positions, const float* shs, const float* camera,
    const float* albedo, const float* normalAxis, const float* normalBias,
    const float* normalBlend, const float* material, float* attributes, float* viewColors) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float toCamera[3] = { camera[0]-positions[3*i], camera[1]-positions[3*i+1], camera[2]-positions[3*i+2] };
    normalize3(toCamera);
    const int coeffCount = (degree + 1) * (degree + 1);
    const float* pointSh = shs + static_cast<std::size_t>(i) * coeffCount * 3;
    for (int c = 0; c < 3; ++c) {
        float coefficients[16];
        for (int k = 0; k < coeffCount; ++k) coefficients[k] = pointSh[3*k+c];
        const float dir[3] = {-toCamera[0], -toCamera[1], -toCamera[2]};
        attributes[i*18+c] = evalGaussianSh(coefficients, degree, dir);
        attributes[i*18+3+c] = albedo[3*i+c];
        attributes[i*18+9+c] = material[3*i+c];
        viewColors[3*i+c] = toCamera[c];
        attributes[i*18+12+c] = toCamera[c];
        attributes[i*18+15+c] = 0.0f;
    }

    float normal[3];
    const float facing = toCamera[0]*normalAxis[3*i] + toCamera[1]*normalAxis[3*i+1] + toCamera[2]*normalAxis[3*i+2];
    const float axisSign = facing < 0.0f ? -1.0f : 1.0f;
    for (int c = 0; c < 3; ++c)
        normal[c] = normalBlend[i] * normalAxis[3*i+c] * axisSign + (1.0f-normalBlend[i]) * normalBias[3*i+c];
    normalize3(normal);
    for (int c = 0; c < 3; ++c) attributes[i*18+6+c] = 0.5f * (normal[c]+1.0f);
}
__global__ void pbrKernel(
    int width, int height, const float* direct, const float* albedo, const float* normalEncoded,
    const float* material, const float* view, const float* maskT, const float* env,
    const float* fw0, const float* fb0, const float* fw1, const float* fb1, const float* fw2, const float* fb2,
    const float* bw0, const float* bb0, const float* bw1, const float* bb1, const float* bw2, const float* bb2,
    float* output) {
    const int p=blockIdx.x*blockDim.x+threadIdx.x, n=width*height;
    if (p>=n) return;
    if (maskT[p] >= 0.995f) { output[p]=output[n+p]=output[2*n+p]=1.0f; return; }

    float N[3], V[3], R[3], irradiance[3];
    for (int c=0;c<3;++c) {
        N[c]=2.0f*normalEncoded[c*n+p]-1.0f;
        V[c]=view[c*n+p];
    }
    normalize3(N); normalize3(V);
    const float rough=fmaxf(0.0f, material[p]);
    const float metal=material[n+p], ao=material[2*n+p];
    for (int c=0;c<3;++c) R[c]=2.0f*dot3(N,V)*N[c]-V[c];
    normalize3(R);
    evalEnvSh(env,N,irradiance);

    float F0[3], Fview[3], kd[3], diffuse[3];
    const float ndv=clamp01(dot3(N,V));
    for (int c=0;c<3;++c) {
        const float alb=albedo[c*n+p];
        F0[c]=0.04f*(1.0f-metal)+alb*metal;
        Fview[c]=F0[c]+(1.0f-F0[c])*powf(1.0f-ndv,5.0f);
        kd[c]=(1.0f-Fview[c])*(1.0f-metal);
        diffuse[c]=kd[c]*ao*(irradiance[c]*alb+direct[c*n+p]);
    }

    float filterIn[7]={R[0],R[1],R[2],R[0],R[1],R[2],rough};
    float h0[128], h1[128], pre[3];
    dense(filterIn,7,128,fw0,fb0,h0,true);
    dense(h0,128,128,fw1,fb1,h1,true);
    dense(h1,128,3,fw2,fb2,pre,false);
    for (int c=0;c<3;++c) pre[c]=softplus(pre[c]);

    float brdfIn[2]={ndv,rough}, b0[32], b1[32], bout[2];
    dense(brdfIn,2,32,bw0,bb0,b0,true);
    dense(b0,32,32,bw1,bb1,b1,true);
    dense(b1,32,2,bw2,bb2,bout,false);
    const float scale=sigmoid(bout[0]), bias=sigmoid(bout[1]);

    const float LdotN=ndv;
    float H[3]; for (int c=0;c<3;++c) H[c]=V[c]+R[c]; normalize3(H);
    const float ndh=clamp01(dot3(N,H)), ndl=clamp01(LdotN);
    const float a=rough*rough, a2=a*a;
    const float denomD=kPi*powf(ndh*ndh*(a2-1.0f)+1.0f,2.0f);
    const float ndf=a2/fmaxf(denomD,1e-8f);
    const float k=(rough+1.0f)*(rough+1.0f)/8.0f;
    const float gV=ndv/(ndv*(1.0f-k)+k+1e-8f), gL=ndl/(ndl*(1.0f-k)+k+1e-8f);
    const float geo=gV*gL, denom=4.0f*ndv*ndl+1e-6f;
    float intensity=0.299f*direct[p]+0.587f*direct[n+p]+0.114f*direct[2*n+p];
    for (int c=0;c<3;++c) {
        const float brdf=Fview[c]*scale+bias;
        const float fd=F0[c]+(1.0f-F0[c])*powf(1.0f-ndh,5.0f);
        const float directSpec=(ndf*geo*fd)/denom;
        const float spec=(pre[c]*brdf + (directSpec*intensity*ndl+direct[c*n+p])/(2.0f*kPi));
        output[c*n+p]=diffuse[c]+spec;
    }
}

__global__ void pbrKernelTiled(
    int width, int height, const float* direct, const float* albedo, const float* normalEncoded,
    const float* material, const float* view, const float* maskT, const float* env,
    const float* fw0, const float* fb0, const float* fw1T, const float* fb1, const float* fw2, const float* fb2,
    const float* bw0, const float* bb0, const float* bw1T, const float* bb1, const float* bw2, const float* bb2,
    float* output) {
    constexpr int pixelsPerBlock = 16;
    const int pixelBase = blockIdx.x * pixelsPerBlock;
    const int tid = threadIdx.x;
    const int pixelCount = width * height;
    __shared__ float filterInput[pixelsPerBlock][7];
    __shared__ float brdfInput[pixelsPerBlock][2];
    __shared__ float filterHidden0[pixelsPerBlock][128];
    __shared__ float filterHidden1[pixelsPerBlock][128];
    __shared__ float filterOutput[pixelsPerBlock][3];
    __shared__ float brdfHidden0[pixelsPerBlock][32];
    __shared__ float brdfHidden1[pixelsPerBlock][32];
    __shared__ float brdfOutput[pixelsPerBlock][2];
    __shared__ int active[pixelsPerBlock];

    if (tid < pixelsPerBlock) {
        const int p = pixelBase + tid;
        active[tid] = p < pixelCount && maskT[p] < 0.995f;
        if (p < pixelCount && !active[tid]) output[p] = output[pixelCount+p] = output[2*pixelCount+p] = 1.0f;
        if (active[tid]) {
            float N[3], V[3], R[3];
            for (int c = 0; c < 3; ++c) {
                N[c] = 2.0f*normalEncoded[c*pixelCount+p]-1.0f;
                V[c] = view[c*pixelCount+p];
            }
            normalize3(N); normalize3(V);
            const float rough = fmaxf(0.0f, material[p]);
            const float ndv = clamp01(dot3(N,V));
            for (int c = 0; c < 3; ++c) R[c] = 2.0f*dot3(N,V)*N[c]-V[c];
            normalize3(R);
            for (int c = 0; c < 3; ++c) filterInput[tid][c] = R[c];
            for (int c = 0; c < 3; ++c) filterInput[tid][3+c] = R[c];
            filterInput[tid][6] = rough;
            brdfInput[tid][0] = ndv;
            brdfInput[tid][1] = rough;
        }
    }
    __syncthreads();
    const int activeCount = __syncthreads_count(tid < pixelsPerBlock && active[tid]);
    if (activeCount == 0) return;

    if (tid < 128) {
        float sums[pixelsPerBlock] = {};
        for (int i = 0; i < 7; ++i) {
            const float w = fw0[tid*7+i];
            for (int p = 0; p < pixelsPerBlock; ++p)
                if (active[p]) sums[p] += w*filterInput[p][i];
        }
        for (int p = 0; p < pixelsPerBlock; ++p)
            if (active[p]) filterHidden0[p][tid] = fmaxf(sums[p]+fb0[tid],0.0f);
    }
    __syncthreads();

    if (tid < 128) {
        float sums[pixelsPerBlock] = {};
        for (int i = 0; i < 128; ++i) {
            const float w = fw1T[i*128+tid];
            for (int p = 0; p < pixelsPerBlock; ++p)
                if (active[p]) sums[p] += w*filterHidden0[p][i];
        }
        for (int p = 0; p < pixelsPerBlock; ++p)
            if (active[p]) filterHidden1[p][tid] = fmaxf(sums[p]+fb1[tid],0.0f);
    }
    __syncthreads();

    if (tid < pixelsPerBlock*3) {
        const int p = tid/3, channel = tid%3;
        if (active[p]) {
            float value = fb2[channel];
            for (int i = 0; i < 128; ++i) value += fw2[channel*128+i]*filterHidden1[p][i];
            filterOutput[p][channel] = softplus(value);
        }
    }
    __syncthreads();

    if (tid < 32) {
        float sums[pixelsPerBlock] = {};
        for (int i = 0; i < 2; ++i) {
            const float w = bw0[tid*2+i];
            for (int p = 0; p < pixelsPerBlock; ++p)
                if (active[p]) sums[p] += w*brdfInput[p][i];
        }
        for (int p = 0; p < pixelsPerBlock; ++p)
            if (active[p]) brdfHidden0[p][tid] = fmaxf(sums[p]+bb0[tid],0.0f);
    }
    __syncthreads();

    if (tid < 32) {
        float sums[pixelsPerBlock] = {};
        for (int i = 0; i < 32; ++i) {
            const float w = bw1T[i*32+tid];
            for (int p = 0; p < pixelsPerBlock; ++p)
                if (active[p]) sums[p] += w*brdfHidden0[p][i];
        }
        for (int p = 0; p < pixelsPerBlock; ++p)
            if (active[p]) brdfHidden1[p][tid] = fmaxf(sums[p]+bb1[tid],0.0f);
    }
    __syncthreads();

    if (tid < pixelsPerBlock*2) {
        const int p = tid/2, channel = tid%2;
        if (active[p]) {
            float value = bb2[channel];
            for (int i = 0; i < 32; ++i) value += bw2[channel*32+i]*brdfHidden1[p][i];
            brdfOutput[p][channel] = sigmoid(value);
        }
    }
    __syncthreads();

    if (tid < pixelsPerBlock && active[tid]) {
        const int p = pixelBase+tid;
        float N[3], V[3], R[3], irradiance[3];
        for (int c = 0; c < 3; ++c) {
            N[c] = 2.0f*normalEncoded[c*pixelCount+p]-1.0f;
            V[c] = view[c*pixelCount+p];
        }
        normalize3(N); normalize3(V);
        const float rough = fmaxf(0.0f,material[p]);
        const float metal = material[pixelCount+p], ao = material[2*pixelCount+p];
        const float ndv = clamp01(dot3(N,V));
        for (int c = 0; c < 3; ++c) R[c] = 2.0f*dot3(N,V)*N[c]-V[c];
        normalize3(R); evalEnvSh(env,N,irradiance);

        float F0[3], Fview[3], kd[3], diffuse[3];
        for (int c = 0; c < 3; ++c) {
            const float alb = albedo[c*pixelCount+p];
            F0[c] = 0.04f*(1.0f-metal)+alb*metal;
            Fview[c] = F0[c]+(1.0f-F0[c])*powf(1.0f-ndv,5.0f);
            kd[c] = (1.0f-Fview[c])*(1.0f-metal);
            diffuse[c] = kd[c]*ao*(irradiance[c]*alb+direct[c*pixelCount+p]);
        }
        float H[3]; for (int c = 0; c < 3; ++c) H[c]=V[c]+R[c]; normalize3(H);
        const float ndh=clamp01(dot3(N,H)), ndl=ndv;
        const float a=rough*rough, a2=a*a;
        const float denomD=kPi*powf(ndh*ndh*(a2-1.0f)+1.0f,2.0f);
        const float ndf=a2/fmaxf(denomD,1e-8f);
        const float k=(rough+1.0f)*(rough+1.0f)/8.0f;
        const float gV=ndv/(ndv*(1.0f-k)+k+1e-8f), gL=ndl/(ndl*(1.0f-k)+k+1e-8f);
        const float geo=gV*gL, denom=4.0f*ndv*ndl+1e-6f;
        const float intensity=0.299f*direct[p]+0.587f*direct[pixelCount+p]+0.114f*direct[2*pixelCount+p];
        const float scale=brdfOutput[tid][0], bias=brdfOutput[tid][1];
        for (int c = 0; c < 3; ++c) {
            const float brdf=Fview[c]*scale+bias;
            const float fd=F0[c]+(1.0f-F0[c])*powf(1.0f-ndh,5.0f);
            const float directSpec=(ndf*geo*fd)/denom;
            const float spec=(filterOutput[tid][c]*brdf+(directSpec*intensity*ndl+direct[c*pixelCount+p])/(2.0f*kPi));
            output[c*pixelCount+p]=diffuse[c]+spec;
        }
    }
}
}

void launchOursPbr(
    int width, int height, const float* direct, const float* albedo, const float* normalEncoded,
    const float* material, const float* view, const float* maskT, const float* env,
    const float* fw0, const float* fb0, const float* fw1, const float* fw1t, const float* fb1, const float* fw2, const float* fb2,
    const float* bw0, const float* bb0, const float* bw1, const float* bw1t, const float* bb1, const float* bw2, const float* bb2,
    float* output, cudaStream_t stream) {
    const int count=width*height;
    (void)fw1; (void)bw1;
    pbrKernelTiled<<<(count+15)/16,128,0,stream>>>(width,height,direct,albedo,normalEncoded,material,view,maskT,env,
        fw0,fb0,fw1t,fb1,fw2,fb2,bw0,bb0,bw1t,bb1,bw2,bb2,output);
}

void launchBuildPbrAttributes(
    int count, int shDegree, const float* positions, const float* shs, const float* cameraPosition,
    const float* albedo, const float* normalAxis, const float* normalBias,
    const float* normalBlend, const float* material, float* attributes18, float* viewColors,
    cudaStream_t stream) {
    buildPbrAttributesKernel<<<(count+255)/256,256,0,stream>>>(count, shDegree, positions, shs,
        cameraPosition, albedo, normalAxis, normalBias, normalBlend, material, attributes18, viewColors);
}
