#include "benchmark_rrc_cublas_gemmex.hpp"

#include "benchmark_cpu_float_rrr.hpp"
#include "numeric.hpp"
#include "time.hpp"
#include "cuda_runtime.hpp"

#include <iostream>



static __global__ void float_to_half(half *h, const float *f, size_t n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        h[i] = f[i];
    }
}

RRC_cuBLAS_GemmEx::RRC_cuBLAS_GemmEx()
{

    CUBLAS(cublasCreate(&handle_));
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
    CUDA_RUNTIME(cudaEventCreate(&start_));
    CUDA_RUNTIME(cudaEventCreate(&stop_));
    CUBLAS(cublasSetStream(handle_, stream_));
}

RRC_cuBLAS_GemmEx::~RRC_cuBLAS_GemmEx()
{
    CUDA_RUNTIME(cudaStreamDestroy(stream_));
    CUDA_RUNTIME(cudaEventDestroy(start_));
    CUDA_RUNTIME(cudaEventDestroy(stop_));
    CUBLAS(cublasDestroy(handle_));
}

bool RRC_cuBLAS_GemmEx::check()
{
    bool success = true;

    // compute expected
    std::vector<float> _ce(m_ * n_);
    CPURRR::mm(_ce.data(), a32_, b32_, m_, n_, k_);

    // result produced in column-major storage
#define ca(i, j) (ca_[(j)*m_ + (i)])
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
    CUDA_RUNTIME(cudaMemPrefetchAsync(ca_, sizeof(*ca_) *m_ *n_, 0, 0));
    CUDA_RUNTIME(cudaDeviceSynchronize());
    return success;
}

void RRC_cuBLAS_GemmEx::initialize(const Spec &spec)
{
    // pad out to next multiple of 16 in each dimension
    m_ = (spec.m + 15) / 16 * 16;
    n_ = (spec.n + 15) / 16 * 16;
    k_ = (spec.k + 15) / 16 * 16;

    // generate random numbers on CPU
    CUDA_RUNTIME(cudaMallocManaged(&a32_, sizeof(*a32_) * m_ * k_));
    CUDA_RUNTIME(cudaMallocManaged(&b32_, sizeof(*b32_) * k_ * n_));
    fill(a32_, m_ * k_);
    fill(b32_, k_ * n_);

    // convert to half-precision GPU inputs
    CUDA_RUNTIME(cudaMalloc(&a_, sizeof(*a_) * m_ * k_));
    CUDA_RUNTIME(cudaMalloc(&b_, sizeof(*b_) * k_ * n_));
    float_to_half<<<256,256>>>(a_, a32_, m_ * k_);
    float_to_half<<<256,256>>>(b_, b32_, k_ * n_);
    CUDA_RUNTIME(cudaDeviceSynchronize());

    // GPU output
    CUDA_RUNTIME(cudaMallocManaged(&ca_, sizeof(*ca_) * m_ * n_));

    // Use tensor cores
   CUBLAS(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));

}

void RRC_cuBLAS_GemmEx::finalize()
{
    CUDA_RUNTIME(cudaFree(a_));
    CUDA_RUNTIME(cudaFree(b_));
    CUDA_RUNTIME(cudaFree(ca_));
    CUDA_RUNTIME(cudaFree(a32_));
    CUDA_RUNTIME(cudaFree(b32_));
}

double RRC_cuBLAS_GemmEx::sample()
{

    float alpha = 1.0f;
   float beta = 0.0f;

    cudaEventRecord(start_, stream_);
    // by default, is col*col=col, so change to row*row=col
    CUBLAS(cublasGemmEx(handle_, CUBLAS_OP_T, CUBLAS_OP_T, 
                m_, n_, k_, 
                &alpha,
                a_, CUDA_R_16F, k_,
                b_, CUDA_R_16F, n_,
                &beta, 
                ca_, CUDA_R_32F, m_,
                CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));

    CUDA_RUNTIME(cudaEventRecord(stop_, stream_));
    CUDA_RUNTIME(cudaGetLastError());
    CUDA_RUNTIME(cudaEventSynchronize(stop_));
    float millis;
    CUDA_RUNTIME(cudaEventElapsedTime(&millis, start_, stop_));
    return millis/1e3;
}
