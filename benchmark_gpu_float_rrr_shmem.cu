#include "benchmark_gpu_float_rrr_shmem.hpp"

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

   each thread-block creates a TILE_Y by TILE_X block of the product matrix
   The tile loaded from A is BDY x TILE_P and from B is BDX x TILE_P

   The x and y block dim should match BDX and BDY respectively
*/
template <int TILE_X, int TILE_Y, int TILE_P>
__global__ void mm_s(float *_c, const float *_a, const float *_b, const int m, const int n, const int p)
{
#define a(_i, _j) _a[(_i)*p + (_j)]
#define b(_i, _j) _b[(_i)*n + (_j)]
#define c(_i, _j) _c[(_i)*n + (_j)]

    __shared__ float sA[TILE_Y][TILE_P];
    __shared__ float sB[TILE_P][TILE_X];

    // TODO - loop, but all threads should be active as long as at least one is contributing a
    const int i = blockIdx.y * TILE_Y + threadIdx.y;
    const int j = blockIdx.x * TILE_X + threadIdx.x;

    // t-th tile in A and B
    float acc = 0;
    for (int t = 0; t < (p + TILE_P - 1) / TILE_P; ++t)
    {

        // collab load tile of A. X dim needs to cover TILE_P
        if (i < m)
        {
            for (int x = threadIdx.x; x < TILE_P && t * TILE_P + x < p; x += TILE_X)
            {
                sA[threadIdx.y][x] = a(i, t * TILE_P + x);

                // if (i == 0)
                // {
                //     printf("t=%d sA(%d,%d) = a(%d,%d)\n", t, threadIdx.y, x, i, t * TILE_P + x);
                // }
            }
        }
        // collab load tile of B. Y dim needs to cover TILE_P
        if (j < n)
        {
            for (int y = threadIdx.y; y < TILE_P && t * TILE_P + y < p; y += TILE_Y)
            {
                sB[y][threadIdx.x] = b(t * TILE_P + y, j);

                // if (j == 0)
                // {
                //     printf("t=%d ty=%d TILE_Y=%d, sB(%d,%d) = b(%d,%d)\n", t, threadIdx.y, TILE_Y, y, threadIdx.x, t * TILE_P + y, j);
                // }
            }
        }
        __syncthreads();

        // partial product from this tile
        for (int k = 0; k < TILE_P && t * TILE_P + k < p; ++k)
        {

            // if (i == 0 && j == 0)
            // {
            //     printf("t=%d sa(%d,%d) * sb(%d,%d)\n", t, threadIdx.y, k, k, threadIdx.x);
            // }

            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (i < m && j < n)
    {
        c(i, j) = acc;
    }

#undef a
#undef b
#undef c
}

GPUFloatRRRShmem::GPUFloatRRRShmem()
{
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
    CUDA_RUNTIME(cudaEventCreate(&start_));
    CUDA_RUNTIME(cudaEventCreate(&stop_));
}

GPUFloatRRRShmem::~GPUFloatRRRShmem()
{
    CUDA_RUNTIME(cudaStreamDestroy(stream_));
    CUDA_RUNTIME(cudaEventDestroy(start_));
    CUDA_RUNTIME(cudaEventDestroy(stop_));
}

bool GPUFloatRRRShmem::check()
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

void GPUFloatRRRShmem::initialize(const Spec &spec)
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

void GPUFloatRRRShmem::finalize()
{
    CUDA_RUNTIME(cudaFree(a_));
    CUDA_RUNTIME(cudaFree(b_));
    CUDA_RUNTIME(cudaFree(c_));
}

double GPUFloatRRRShmem::sample()
{
        
            constexpr int SH_PER_BLOCK = 4 * 1024; // target shared memory useage per block
            constexpr int TILE_X = 32;
            constexpr int TILE_Y = 8;
            constexpr int TILE_P = SH_PER_BLOCK / (TILE_X + TILE_Y) / sizeof(float);
            constexpr dim3 bd(TILE_X, TILE_Y, 1);
            const dim3 gd((n_ + bd.x - 1) / bd.x, (m_ + bd.y - 1) / bd.y, 1);


                cudaEventRecord(start_, stream_);
                mm_s<TILE_X, TILE_Y, TILE_P><<<gd, bd, 0, stream_>>>(c_, a_, b_, m_, n_, k_);
                CUDA_RUNTIME(cudaEventRecord(stop_, stream_));
                CUDA_RUNTIME(cudaGetLastError());
                CUDA_RUNTIME(cudaEventSynchronize(stop_));
                float millis;
                CUDA_RUNTIME(cudaEventElapsedTime(&millis, start_, stop_));
                return millis / 1e3;



}
