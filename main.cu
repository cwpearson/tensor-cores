#include <iostream>
#include <cstdlib>
#include <cmath>

#include "cuda_runtime.hpp"
#include "benchmark.hpp"
#include "benchmark_cpu_float_rcr.hpp"
#include "benchmark_tc_half_rrr.hpp"
#include "numeric.hpp"
#include "time.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::flush;

/* a*b=c
   a[m x p]
   b[p x n]
   c[m x n]
*/
Product create(const Spec &s)
{
    Product ret;
    ret.m = s.m;
    ret.n = s.n;
    ret.k = s.k;
    cudaMallocManaged(&ret.a, sizeof(float) * s.m * s.k);
    fill((float*)ret.a, s.m, s.k);
    cudaMallocManaged(&ret.b, sizeof(float) * s.k * s.n);
    fill((float*)ret.b, s.k, s.n);
    cudaMallocManaged(&ret.ce, sizeof(float) * s.m * s.n);
    cudaMallocManaged(&ret.ca, sizeof(float) * s.m * s.n);
    return ret;
}

void destroy(Product &prod)
{
    CUDA_RUNTIME(cudaFree(prod.a));
    CUDA_RUNTIME(cudaFree(prod.b));
    CUDA_RUNTIME(cudaFree(prod.ce));
    CUDA_RUNTIME(cudaFree(prod.ca));
    prod.m = 0;
    prod.n = 0;
    prod.k = 0;
}

/* a*b=c
   a[m x p]
   b[p x n]
   c[m x n]

   all row-major
*/
void cpu(float *_c, const float *_a, const float *_b, const int m, const int n, const int p)
{
#define a(_i, _j) _a[(_i)*p + (_j)]
#define b(_i, _j) _b[(_i)*n + (_j)]
#define c(_i, _j) _c[(_i)*n + (_j)]

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
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

void check(const Product &product)
{

    float *_ca = (float*)product.ca;
    float *_ce = (float*)product.ca;

#define ca(i, j) (_ca[(i) * product.n + (j)])
#define ce(i, j) (_ce[(i) * product.n + (j)])

    for (int i = 0; i < product.m; ++i)
    {
        for (int j = 0; j < product.n; ++j)
        {
            if (!almost_equal(ca(i, j), ce(i, j), 1e-5))
            {
                cerr << "ERR at " << i << " " << j << " "
                     << "ce=" << ce(i, j) << " ca=" << ca(i, j) << endl;
            }
        }
    }

#undef ca
#undef ce
}

int main(void)
{

    cudaStream_t stream;
    CUDA_RUNTIME(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t eStart, eStop;
    CUDA_RUNTIME(cudaEventCreate(&eStart));
    CUDA_RUNTIME(cudaEventCreate(&eStop));

    srand(100);

    for (int ti = 0; ti < 1000; ++ti)
    {

        Spec spec = Spec::create_log_uniform(2, 10);

        cout << spec.m << "," << spec.n << "," << spec.k << "," << spec.flop() << flush;

#if 1

        {
            CPURCR bench;
            Result res = bench.run(spec);
            cout << "," << res.med() << flush;
        }
#endif
        cout << endl;
        continue;

        Product product = create(spec);

        {
            auto start = Clock::now();
            for (int i = 0; i < 10; ++i)
            {
                cpu((float*)product.ce, (float*)product.a, (float*)product.b, product.m, product.n, product.k);
            }
            Duration elapsed = Clock::now() - start;
            cout << "," << elapsed.count() / 10 << flush;
        }

#if 1
        {
            double elapsed = 0;
            dim3 bd(32, 8, 1);
            dim3 gd((product.m + bd.y - 1) / bd.y, (product.n + bd.x - 1) / bd.x, 1);
            for (int i = 0; i < 10; ++i)
            {
                cudaEventRecord(eStart, stream);
                mm<<<gd, bd, 0, stream>>>((float*)product.ca, (float*)product.a, (float*)product.b, product.m, product.n, product.k);
                cudaEventRecord(eStop, stream);
                CUDA_RUNTIME(cudaEventSynchronize(eStop));
                float millis;
                CUDA_RUNTIME(cudaEventElapsedTime(&millis, eStart, eStop));
                elapsed += millis / 1e3;
            }
            cout << "," << elapsed / 10 << flush;

            check(product);
        }
#endif

#if 1
        {
            constexpr int SH_PER_BLOCK = 2 * 1024; // target shared memory useage per block
            constexpr int TILE_X = 32;
            constexpr int TILE_Y = 8;
            constexpr int TILE_P = SH_PER_BLOCK / (TILE_X + TILE_Y) / sizeof(float);
            constexpr dim3 bd(TILE_X, TILE_Y, 1);
            const dim3 gd((product.n + bd.x - 1) / bd.x, (product.m + bd.y - 1) / bd.y, 1);

            double elapsed = 0;
            for (int i = 0; i < 10; ++i)
            {
                cudaEventRecord(eStart, stream);
                mm_s<TILE_X, TILE_Y, TILE_P><<<gd, bd, 0, stream>>>((float*)product.ca, (float*)product.a, (float*)product.b, product.m, product.n, product.k);
                cudaEventRecord(eStop, stream);
                CUDA_RUNTIME(cudaEventSynchronize(eStop));
                float millis;
                CUDA_RUNTIME(cudaEventElapsedTime(&millis, eStart, eStop));
                elapsed += millis / 1e3;
            }
            cout << "," << elapsed / 10 << flush;

            check(product);
        }
#endif

        destroy(product);
        cout << endl;
    }

    CUDA_RUNTIME(cudaStreamDestroy(stream));
    CUDA_RUNTIME(cudaEventDestroy(eStart));
    CUDA_RUNTIME(cudaEventDestroy(eStop));
}