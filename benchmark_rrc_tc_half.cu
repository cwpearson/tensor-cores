#include "benchmark_rrc_tc_half.hpp"

#include "benchmark_cpu_float_rrr.hpp"
#include "numeric.hpp"
#include "time.hpp"
#include "cuda_runtime.hpp"

#include <curand.h>
#include <mma.h>

#include <iostream>

constexpr int WMMA_TILE_M = 16;
constexpr int WMMA_TILE_N = 16;
constexpr int WMMA_TILE_K = 16;

static __global__ void mm(float * __restrict__ _c, const half * __restrict__ _a, const half * __restrict__ _b, const int M, const int N, const int K)
{
#define a(_i, _j) (_a[(_i)*K + (_j)])
#define b(_i, _j) (_b[(_i)*N + (_j)])
#define c(_i, _j) (_c[(_j)*M + (_i)]) // column major

    using nvcuda::wmma::accumulator;
    using nvcuda::wmma::fill_fragment;
    using nvcuda::wmma::fragment;
    using nvcuda::wmma::load_matrix_sync;
    using nvcuda::wmma::matrix_a;
    using nvcuda::wmma::matrix_b;
    using nvcuda::wmma::mem_col_major; // a layout_t
    using nvcuda::wmma::mma_sync;
    using nvcuda::wmma::row_major; // a type
    using nvcuda::wmma::store_matrix_sync;

    typedef fragment<matrix_a, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, half, row_major> FragA;
    typedef fragment<matrix_b, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, half, row_major> FragB;
    typedef fragment<accumulator, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, float> FragC;

    FragA fa;
    FragB fb;
    FragC fc;

    fill_fragment(fc, 0.0f);

    // TODO -- loop over matrices with warps
    const int wx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int wy = blockIdx.y * blockDim.y + threadIdx.y;

    for (int t = 0; t < K; t += WMMA_TILE_K)
    {
        // row and col of beginning of tile
        const int aRow = wy * WMMA_TILE_M;
        const int aCol = t;
        const int bRow = t;
        const int bCol = wx * WMMA_TILE_N;

        if (aRow < M && aCol < K && bRow < K && bCol < N)
        {

            // cast to half for now
            load_matrix_sync(fa, &a(aRow, aCol), unsigned(K));
            load_matrix_sync(fb, &b(bRow, bCol), unsigned(N));
            mma_sync(fc, fa, fb, fc);
        }
    }

    const int cRow = wy * WMMA_TILE_M;
    const int cCol = wx * WMMA_TILE_N;

    if (cRow < M && cCol < N)
    {
        store_matrix_sync(&c(cRow, cCol), fc, M, mem_col_major);
    }

#undef a
#undef b
#undef c
}

static __global__ void float_to_half(half *h, const float *f, size_t n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        h[i] = f[i];
    }
}

RRC_TC_half::RRC_TC_half()
{
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
    CUDA_RUNTIME(cudaEventCreate(&start_));
    CUDA_RUNTIME(cudaEventCreate(&stop_));
}

RRC_TC_half::~RRC_TC_half()
{
    CUDA_RUNTIME(cudaStreamDestroy(stream_));
    CUDA_RUNTIME(cudaEventDestroy(start_));
    CUDA_RUNTIME(cudaEventDestroy(stop_));
}

bool RRC_TC_half::check()
{
    bool success = true;

    // compute expected
    std::vector<float> _ce(m_ * n_);
    CPURRR::mm(_ce.data(), a32_, b32_, m_, n_, k_);

#define ca(i, j) (ca_[(j)*m_ + (i)]) // actual is column-major
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

void RRC_TC_half::initialize(const Spec &spec)
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

}

void RRC_TC_half::finalize()
{
    CUDA_RUNTIME(cudaFree(a_));
    CUDA_RUNTIME(cudaFree(b_));
    CUDA_RUNTIME(cudaFree(ca_));
    CUDA_RUNTIME(cudaFree(a32_));
    CUDA_RUNTIME(cudaFree(b32_));
}

double RRC_TC_half::sample()
{
    // 1 warp in x, 8 warps in y
    constexpr dim3 bd(32, 8, 1);
    const dim3 gd((n_ + WMMA_TILE_N - 1) / WMMA_TILE_N, (m_ + WMMA_TILE_N * bd.y - 1) / (WMMA_TILE_N * bd.y), 1);

    cudaEventRecord(start_, stream_);
    mm<<<gd, bd, 0, stream_>>>(ca_, a_, b_, m_, n_, k_);
    CUDA_RUNTIME(cudaEventRecord(stop_, stream_));
    CUDA_RUNTIME(cudaGetLastError());
    CUDA_RUNTIME(cudaEventSynchronize(stop_));
    float millis;
    CUDA_RUNTIME(cudaEventElapsedTime(&millis, start_, stop_));
    return millis/1e3;
}
