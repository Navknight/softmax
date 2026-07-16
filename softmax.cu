#include <cstdio>
#include <cmath>
#include <cuda.h>

 #define NEG_BIG -3.402823466e38f

__device__ float dev_max(float a, float b) {
    return a > b ? a : b;
}

__device__ float warpReduceSum(float v) {
    for(int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ float warpReduceMax(float v) {
    for(int offset = 16; offset > 0; offset /= 2) {
        v = dev_max(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ float blockReduceSum(float v) {
    __shared__ float s[32];
    v = warpReduceSum(v);
    if(threadIdx.x % 32 == 0) s[threadIdx.x/32] = v;
    __syncthreads();
    int nwarps = blockDim.x / 32;
    v = (threadIdx.x < nwarps) ? s[threadIdx.x] : 0.0f;
    if(threadIdx.x / 32 == 0) v = warpReduceSum(v);
    return v;
}

__device__ float blockReduceMax(float v) {
    __shared__ float s[32];
    v = warpReduceMax(v);
    if(threadIdx.x % 32 == 0) s[threadIdx.x/32] = v;
    __syncthreads();
    int nwarps = blockDim.x / 32;
    v = (threadIdx.x < nwarps) ? s[threadIdx.x] : NEG_BIG;
    if(threadIdx.x / 32 == 0) v = warpReduceMax(v);
    return v;
}

__global__ void k_max_partial (float *a, int n, float *result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float acc = NEG_BIG;

    for(int i = tid; i < n; i += gridDim.x * blockDim.x) {
        acc = dev_max(acc, a[i]);
    }

    acc = blockReduceMax(acc);
    if(threadIdx.x == 0) result[blockIdx.x] = acc;
}

__global__ void k_max_final(float *partial, int n, float *result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float acc = NEG_BIG;

    for(int i = tid; i < n; i += gridDim.x * blockDim.x) {
        acc = dev_max(acc, partial[i]);
    }

    acc = blockReduceMax(acc);
    if(threadIdx.x == 0) result[blockIdx.x] = acc;
}

__global__ void k_sum_partial(float *a, int n, float *d_max, float *result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float m = *d_max;
    float acc = 0.0f;

    for(int i = tid; i < n; i += gridDim.x * blockDim.x) {
        acc += expf(a[i] - m);
    }

    acc = blockReduceSum(acc);
    if(threadIdx.x == 0) result[blockIdx.x] = acc;
}

__global__ void k_sum_final(float *partial, int n, float *result) {
    float acc = 0.0f;

    for(int i = threadIdx.x; i < n; i += blockDim.x) {
        acc += partial[i];
    }

    acc = blockReduceSum(acc);
    if(threadIdx.x == 0) result[0] = acc;
}

__global__ void k_norm(float *a, float *y, int n, float *d_max, float *d_sum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float m = *d_max;
    float inv = 1.0f / *d_sum;

    for(int i = tid; i < n; i += gridDim.x * blockDim.x) {
        y[i] = expf(a[i] - m) * inv;
    }
}

int main() {
    int n = 1000000;
    float *h_x = (float *)malloc(n * sizeof(float));
    float *h_y = (float *)malloc(n * sizeof(float));
    for(int i = 0; i < n; i++)
        h_x[i] = sinf((float)i) * 10.0f;

    float *d_x, *d_y, *d_partial, *d_max, *d_sum;
    cudaMalloc((void **)&d_x, n * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));
    cudaMalloc((void **)&d_partial, 40 * sizeof(float));
    cudaMalloc((void **)&d_max, sizeof(float));
    cudaMalloc((void **)&d_sum, sizeof(float));
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

    k_max_partial<<<40, 256>>>(d_x, n, d_partial);
    k_max_final  <<<1, 256>>>(d_partial, 40, d_max);
    k_sum_partial<<<40, 256>>>(d_x, n, d_max, d_partial);
    k_sum_final  <<<1, 256>>>(d_partial, 40, d_sum);
    k_norm       <<<40, 256>>>(d_x, d_y, n, d_max, d_sum);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) printf("launch error: %s\n", cudaGetErrorString(err));

    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU reference in double
    double m = -1e300;
    for(int i = 0; i < n; i++) if(h_x[i] > m) m = h_x[i];
    double d = 0.0;
    for(int i = 0; i < n; i++) d += exp((double)h_x[i] - m);

    double maxdiff = 0.0, ysum = 0.0;
    for(int i = 0; i < n; i++) {
        double ref = exp((double)h_x[i] - m) / d;
        double diff = fabs((double)h_y[i] - ref);
        if(diff > maxdiff) maxdiff = diff;
        ysum += h_y[i];
    }

    printf("max diff vs CPU: %.3e\n", maxdiff);
    printf("sum(y) = %.6f (want 1.0)\n", ysum);
    if(maxdiff < 1e-6 && fabs(ysum - 1.0) < 1e-4) printf("PASS\n");
    else printf("FAIL\n");

    free(h_x); free(h_y);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_partial);
    cudaFree(d_max); cudaFree(d_sum);
    return 0;
}
