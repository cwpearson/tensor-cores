#include "benchmark_gpu_float_rrr.hpp"

#include "benchmark_cpu_float_rrr.hpp"
#include "numeric.hpp"
#include "time.hpp"
#include "cuda_runtime.hpp"

#include <curand.h>
#include <mma.h>

#include <iostream>

/* a*b=c
   a[m x p]
   b[p x n]
   c[m x n]

   all row-major
*/
__global__ void mm(float *_c, const float *_a, const float *_b, const int m, const int n, const int p)
{
#define a(_i, _j) _a[(_i)*p + (_j)]
#define b(_i, _j) _b[(_i)*n + (_j)]
#define c(_i, _j) _c[(_i)*n + (_j)]

    for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < m; i += blockDim.y * gridDim.y)
    {
        for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += blockDim.x * gridDim.x)
        {
            float acc = 0;
            for (int k = 0; k < p; ++k)
            {
                acc += a(i, k) * b(k, j);
            }
            c(i, j) = acc;
        }
    }

#undef a
#undef b
#undef c
}

GPUFloatRRR::GPUFloatRRR()
{
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
    CUDA_RUNTIME(cudaEventCreate(&start_));
    CUDA_RUNTIME(cudaEventCreate(&stop_));
}

GPUFloatRRR::~GPUFloatRRR()
{
    CUDA_RUNTIME(cudaStreamDestroy(stream_));
    CUDA_RUNTIME(cudaEventDestroy(start_));
    CUDA_RUNTIME(cudaEventDestroy(stop_));
}

bool GPUFloatRRR::check()
{
    bool success = true;

    // compute expected
    std::vector<float> _ce(m_ * n_);
    CPURRR::mm(_ce.data(), a_, b_, m_, n_, k_);

#define ca(i, j) (c_[(i)*n_ + (j)])
#define ce(i, j) (_ce[(i)*n_ + (j)])

    for (int i = 0; i < m_; ++i)
    {
        for (int j = 0; j < n_; ++j)
        {
            if (!almost_equal(ca(i, j), ce(i, j), 1e-2))
            {
                std::cerr << "ERR at " << i << " " << j << " "
                          << "ce=" << ce(i, j) << " ca=" << ca(i, j) << std::endl;
                success = false;
            }
        }
    }

#undef ca
#undef ce

    // send ca back to GPU
    CUDA_RUNTIME(cudaMemPrefetchAsync(c_, sizeof(*c_) * m_ * n_, 0, 0));
    CUDA_RUNTIME(cudaDeviceSynchronize());
    return success;
}

void GPUFloatRRR::initialize(const Spec &spec)
{
    m_ = spec.m;
    n_ = spec.n;
    k_ = spec.k;

    // generate random numbers on CPU
    CUDA_RUNTIME(cudaMallocManaged(&a_, sizeof(*a_) * m_ * k_));
    CUDA_RUNTIME(cudaMallocManaged(&b_, sizeof(*b_) * k_ * n_));
    fill(a_, m_ * k_);
    fill(b_, k_ * n_);

    // send to GPU
    CUDA_RUNTIME(cudaMemPrefetchAsync(a_, sizeof(*a_) * m_ * k_, 0, 0));
    CUDA_RUNTIME(cudaMemPrefetchAsync(b_, sizeof(*b_) * k_ * n_, 0, 0));
    CUDA_RUNTIME(cudaDeviceSynchronize());

    // GPU output
    CUDA_RUNTIME(cudaMallocManaged(&c_, sizeof(*c_) * m_ * n_));
}

void GPUFloatRRR::finalize()
{
    CUDA_RUNTIME(cudaFree(a_));
    CUDA_RUNTIME(cudaFree(b_));
    CUDA_RUNTIME(cudaFree(c_));
}

double GPUFloatRRR::sample()
{
            dim3 bd(32, 8, 1);
            dim3 gd((m_ + bd.y - 1) / bd.y, (n_ + bd.x - 1) / bd.x, 1);

                cudaEventRecord(start_, stream_);
                mm<<<gd, bd, 0, stream_>>>(c_, a_, b_, m_, n_, k_);
                CUDA_RUNTIME(cudaEventRecord(stop_, stream_));
                CUDA_RUNTIME(cudaGetLastError());
                CUDA_RUNTIME(cudaEventSynchronize(stop_));
                float millis;
                CUDA_RUNTIME(cudaEventElapsedTime(&millis, start_, stop_));
                return millis / 1e3;
}
