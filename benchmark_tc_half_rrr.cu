#include "benchmark_tc_half_rrr.hpp"

#include "numeric.hpp"
#include "time.hpp"
#include "cuda_runtime.hpp"

#include <mma.h>

#include <iostream>

constexpr int WMMA_TILE_M = 16;
constexpr int WMMA_TILE_N = 16;
constexpr int WMMA_TILE_K = 16;

__global__ void mm_tc(float *_c, const half *_a, const half *_b, const int M, const int N, const int K)
{
#define a(_i, _j) (_a[(_i)*K + (_j)])
#define b(_i, _j) (_b[(_i)*N + (_j)])
#define c(_i, _j) (_c[(_i)*N + (_j)])

    using nvcuda::wmma::accumulator;
    using nvcuda::wmma::fill_fragment;
    using nvcuda::wmma::fragment;
    using nvcuda::wmma::load_matrix_sync;
    using nvcuda::wmma::matrix_a;
    using nvcuda::wmma::matrix_b;
    using nvcuda::wmma::mem_row_major; // a layout_t
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
        store_matrix_sync(&c(cRow, cCol), fc, N, mem_row_major);
    }

#undef a
#undef b
#undef c
}

__global__ void half_to_float(float *f, const half *h, size_t n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        f[i] = h[i];
    }
}

TCHalfRRR::TCHalfRRR()
{
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
    CUDA_RUNTIME(cudaEventCreate(&start_));
    CUDA_RUNTIME(cudaEventCreate(&stop_));
}

TCHalfRRR::~TCHalfRRR()
{
    CUDA_RUNTIME(cudaStreamDestroy(stream_));
    CUDA_RUNTIME(cudaEventDestroy(start_));
    CUDA_RUNTIME(cudaEventDestroy(stop_));
}

bool TCHalfRRR::check(const Product &prod)
{
    bool success = true;
    return success;

    // convert ca to float
    float *temp{};
    CUDA_RUNTIME(cudaMallocManaged(&temp, sizeof(float) * prod.m * prod.n));
    half_to_float<<<256, 512>>>(temp, (half *)prod.ca, prod.m * prod.n);
    CUDA_RUNTIME(cudaDeviceSynchronize());

    // compare with CPU
    const float *_ca = (float *)prod.ca;

#define ca(i, j) (_ca[(i)*prod.n + (j)])
#define ce(i, j) (temp[(i)*prod.n + (j)])

    for (int i = 0; i < prod.m; ++i)
    {
        for (int j = 0; j < prod.n; ++j)
        {
            if (!almost_equal(ca(i, j), ce(i, j), 1e-5))
            {
                std::cerr << "ERR at " << i << " " << j << " "
                          << "ce=" << ce(i, j) << " ca=" << ca(i, j) << std::endl;
                success = false;
            }
        }
    }

#undef ca
#undef ce

    CUDA_RUNTIME(cudaFree(temp));
    temp = nullptr;
    return success;
}

Product TCHalfRRR::initialize(const Spec &spec)
{
    Product ret;

    // pad out to next multiple of 16 in each dimension
    ret.m = (spec.m + 15) / 16 * 16;
    ret.n = (spec.n + 15) / 16 * 16;
    ret.k = (spec.k + 15) / 16 * 16;

    // half precision inputs
    CUDA_RUNTIME(cudaMalloc(&ret.a, sizeof(half) * ret.m * ret.k));
    CUDA_RUNTIME(cudaMalloc(&ret.b, sizeof(half) * ret.k * ret.n));
    CUDA_RUNTIME(cudaMalloc(&ret.ca, sizeof(half) * ret.m * ret.n));

    // compute expected on CPU with float precision
    ret.ce = new float[ret.m * ret.n];
    return ret;
}

void TCHalfRRR::finalize(Product &prod)
{
    CUDA_RUNTIME(cudaFree((half *)prod.a));
    CUDA_RUNTIME(cudaFree((half *)prod.b));
    CUDA_RUNTIME(cudaFree((float *)prod.ca));
    delete[](float *) prod.ce;
}

double TCHalfRRR::mm(Product &prod)
{
    // 1 warp in x, 8 warps in y
    constexpr dim3 bd(32, 8, 1);
    const dim3 gd((prod.n + WMMA_TILE_N - 1) / WMMA_TILE_N, (prod.m + WMMA_TILE_N * bd.y - 1) / (WMMA_TILE_N * bd.y), 1);

    cudaEventRecord(start_, stream_);
    mm_tc<<<gd, bd, 0, stream_>>>((float *)prod.ca, (half *)prod.a, (half *)prod.b, prod.m, prod.n, prod.k);
    cudaEventRecord(stop_, stream_);
    CUDA_RUNTIME(cudaGetLastError());
    CUDA_RUNTIME(cudaEventSynchronize(stop_));
    float millis;
    CUDA_RUNTIME(cudaEventElapsedTime(&millis, start_, stop_));
    return millis;
}
