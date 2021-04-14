#include "benchmark_tc_half_rrr_shmem.hpp"

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

constexpr int TB_TILE_M = 32;
constexpr int TB_TILE_N = 64;
constexpr int TB_TILE_K = 32;

/* blockDim.y = TB_TILE_M
   blockDim.x = TB_TILE_N
*/
static __global__ void mm(float *_c, const half *_a, const half *_b, const int M, const int N, const int K)
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

    __shared__ half sa[TB_TILE_M][TB_TILE_K];
    __shared__ half sb[TB_TILE_K][TB_TILE_N];

    // warp index in the global space
    const int wx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int wy = blockIdx.y * blockDim.y + threadIdx.y;

    // start of the block in C that the thread block will create
    const int tbCol = blockIdx.x * TB_TILE_N;
    const int tbRow = blockIdx.y * TB_TILE_M;

    // start of the block in C that the warp will create
    // TODO - i and j?
    const int cCol = wx * WMMA_TILE_N;
    const int cRow = wy * WMMA_TILE_M;

    // index of warp within thread block
    const int wbx = threadIdx.x / 32;
    const int wby = threadIdx.y;

    // if (wbx == 0 && wby == 0)
    // {
    //     printf("cRow=%d cCol=%d\n", cRow, cCol);
    // }

    // block tile
    for (int bt = 0; bt < K; bt += TB_TILE_K) {

// thread block collab load of A. Thread block needs to cover whole shared memory area
#if 1
        for (int s = threadIdx.y * blockDim.x + threadIdx.x; s < TB_TILE_M * TB_TILE_K; s += blockDim.x * blockDim.y) {
            int sy = s / TB_TILE_K;
            int sx = s % TB_TILE_K;
            if (tbRow + sy <  M && bt + sx < K) {
            sa[sy][sx] = a(tbRow + sy, bt+sx) ;
            }
        }
#else
        
        for (int sy = threadIdx.y; sy < TB_TILE_M && tbRow + sy < M; sy += blockDim.y) {
            for (int sx = threadIdx.x; sx < TB_TILE_K && bt + sx < K; sx += blockDim.x)
            {
                // printf("sa[%d][%d] = a(%d, %d)\n", sy, sx, tbRow + sy, tbCol + sx);
                sa[sy][sx] = a(tbRow + sy, bt + sx);
            }
        }
#endif

        // collab load tile of B
#if 1
        for (int s = threadIdx.y * blockDim.x + threadIdx.x; s < TB_TILE_K * TB_TILE_N; s += blockDim.x * blockDim.y) {
            int sy = s / TB_TILE_N;
            int sx = s % TB_TILE_N;
            if (bt + sy < K && tbCol + sx < N) {
            sb[sy][sx] = b(bt + sy, tbCol + sx);
            }
        }
#else
        for (int sy = threadIdx.y; sy < TB_TILE_K && bt + sy < K; sy += blockDim.y) {
            for (int sx = threadIdx.x; sx < TB_TILE_N && tbCol + sx < N; sx += blockDim.x)
            {
                sb[sy][sx] = b(bt + sy, tbCol + sx);
            }
        }
#endif
        __syncthreads();

        // loop over wmma tiles in the threab block tile
        for (int wt = 0; wt < TB_TILE_K; wt += WMMA_TILE_K) {

            // row and column within the shared memory tile
            const int saRow = wby * WMMA_TILE_M;
            const int saCol = wt;
            const int sbRow = wt;
            const int sbCol = wbx * WMMA_TILE_N;

            // position in the A and B matrix that corresponds to the shared memory load
            const int aRow = tbRow + saRow;
            const int aCol = bt + saCol;
            const int bRow = bt + sbRow;
            const int bCol = tbCol + sbCol;

            // if (blockIdx.y == 1)
            //     printf("wt=%d c(%d, %d) += sa[%d][%d] ~ a(%d, %d) * sb[%d][%d] ~ b(%d, %d)\n", wt, cRow, cCol, saRow, saCol, aRow, aCol, sbRow, sbCol, bRow, bCol);

            if (aRow < M && aCol < K && bRow < K && bCol < N)
            {

            // if (blockIdx.y == 1)
            //     printf("wt=%d c(%d, %d) += sa[%d][%d] ~ a(%d, %d) * sb[%d][%d] ~ b(%d, %d)\n", wt, cRow, cCol, saRow, saCol, aRow, aCol, sbRow, sbCol, bRow, bCol);


                load_matrix_sync(fa, &sa[saRow][saCol], TB_TILE_K);
                load_matrix_sync(fb, &sb[sbRow][sbCol], TB_TILE_N);
                mma_sync(fc, fa, fb, fc);
            }
        }
        __syncthreads();
    }

    if (cRow < M && cCol < N)
    {
        store_matrix_sync(&c(cRow, cCol), fc, N, mem_row_major);
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

TCHalfRRRShmem::TCHalfRRRShmem()
{
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
    CUDA_RUNTIME(cudaEventCreate(&start_));
    CUDA_RUNTIME(cudaEventCreate(&stop_));
}

TCHalfRRRShmem::~TCHalfRRRShmem()
{
    CUDA_RUNTIME(cudaStreamDestroy(stream_));
    CUDA_RUNTIME(cudaEventDestroy(start_));
    CUDA_RUNTIME(cudaEventDestroy(stop_));
}

bool TCHalfRRRShmem::check()
{
    bool success = true;

    // compute expected
    std::vector<float> _ce(m_ * n_);
    CPURRR::mm(_ce.data(), a32_, b32_, m_, n_, k_);

#define ca(i, j) (ca_[(i)*n_ + (j)])
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

void TCHalfRRRShmem::initialize(const Spec &spec)
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

void TCHalfRRRShmem::finalize()
{
    CUDA_RUNTIME(cudaFree(a_));
    CUDA_RUNTIME(cudaFree(b_));
    CUDA_RUNTIME(cudaFree(ca_));
    CUDA_RUNTIME(cudaFree(a32_));
    CUDA_RUNTIME(cudaFree(b32_));
}

double TCHalfRRRShmem::sample()
{
    // extend x dimension by warp
    constexpr dim3 bd(32 * TB_TILE_N / WMMA_TILE_N, TB_TILE_M / WMMA_TILE_M, 1);
    const dim3 gd((n_ + TB_TILE_N - 1) / TB_TILE_N, (m_ + TB_TILE_M - 1) / TB_TILE_M, 1);

    // printf("bd=%d,%d,%d", bd.x, bd.y, bd.z);
    // printf("gd=%d,%d,%d", gd.x, gd.y, gd.z);

    cudaEventRecord(start_, stream_);
    mm<<<gd, bd, 0, stream_>>>(ca_, a_, b_, m_, n_, k_);
    CUDA_RUNTIME(cudaEventRecord(stop_, stream_));
    CUDA_RUNTIME(cudaGetLastError());
    CUDA_RUNTIME(cudaEventSynchronize(stop_));
    float millis;
    CUDA_RUNTIME(cudaEventElapsedTime(&millis, start_, stop_));
    return millis/1e3;
}
