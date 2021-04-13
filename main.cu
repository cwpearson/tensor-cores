#include <iostream>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <numeric>

#include <cuda_runtime.h>
#include <mma.h>

inline void checkCuda(cudaError_t result, const char *file, const int line)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "%s:%d: CUDA Runtime Error %d: %s\n", file, line, int(result), cudaGetErrorString(result));
        exit(-1);
    }
}
#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);

using std::cerr;
using std::cout;
using std::endl;
using std::flush;

void fill(float *a, const int m, const int n)
{
    for (int i = 0; i < m * n; ++i)
    {
        a[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

struct Product
{
    float *a;
    float *b;
    float *ce; // expected
    float *ca; // actual
    int m;
    int n;
    int p;

    size_t flop() const
    {
        return size_t(m) * size_t(n) * size_t(p) * 2;
    }
};

/* a*b=c
   a[m x p]
   b[p x n]
   c[m x n]
*/
Product create(const int m, const int n, const int p)
{
    Product ret;
    ret.m = m;
    ret.n = n;
    ret.p = p;
    cudaMallocManaged(&ret.a, sizeof(float) * m * p);
    fill(ret.a, m, p);
    cudaMallocManaged(&ret.b, sizeof(float) * p * n);
    fill(ret.b, p, n);
    cudaMallocManaged(&ret.ce, sizeof(float) * m * n);
    cudaMallocManaged(&ret.ca, sizeof(float) * m * n);
    return ret;
}

Product create(const int lo, const int hi)
{

    float logm = (float(rand()) / RAND_MAX) * (hi-lo) + lo;
    float logn = (float(rand()) / RAND_MAX) * (hi-lo) + lo;
    float logp = (float(rand()) / RAND_MAX) * (hi-lo) + lo;

    int m = std::pow(2, logm);
    int n = std::pow(2, logn);
    int p = std::pow(2, logp);

    // m = 35;
    // n = 12;
    // p = 734;
    return create(m, n, p);
}

void destroy(Product &prod)
{
    CUDA_RUNTIME(cudaFree(prod.a));
    CUDA_RUNTIME(cudaFree(prod.b));
    CUDA_RUNTIME(cudaFree(prod.ce));
    CUDA_RUNTIME(cudaFree(prod.ca));
    prod.m = 0;
    prod.n = 0;
    prod.p = 0;
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

   one tile per warp
*/
__global__ void mm_tc(half *_c, const half *_a, const half *_b, const int m, const int n, const int p)
{
#define a(_i, _j) (_a[(_i)*p + (_j)])
#define b(_i, _j) (_b[(_i)*n + (_j)])
#define c(_i, _j) (_c[(_i)*n + (_j)])

    constexpr int WMMA_TILE_M = 16;
    constexpr int WMMA_TILE_N = 16;
    constexpr int WMMA_TILE_P = 16;

    using nvcuda::wmma::matrix_a;
    using nvcuda::wmma::matrix_b;
    using nvcuda::wmma::accumulator;
    using nvcuda::wmma::fragment;
    using nvcuda::wmma::col_major; // a type
    using nvcuda::wmma::row_major; // a type
    using nvcuda::wmma::mem_row_major; // a layout_t
    using nvcuda::wmma::fill_fragment;
    using nvcuda::wmma::load_matrix_sync;
    using nvcuda::wmma::store_matrix_sync;
    using nvcuda::wmma::mma_sync;

    typedef fragment<matrix_a, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_P, half, row_major> FragA;
    typedef fragment<matrix_b, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_P, half, row_major> FragB;
    typedef fragment<accumulator, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_P, float> FragC;

    FragA fa;
    FragB fb;
    FragC fc;

    fill_fragment(fc, 0.0f);

    const int wx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int wy = blockIdx.y * blockDim.y + threadIdx.y;

    for (int t = 0; t < p; t += WMMA_TILE_P)
    {
        // row and col of beginning of tile
        int aRow = wy * WMMA_TILE_M;
        int aCol = t;
        const int bRow = t;
        const int bCol = wx * WMMA_TILE_N;


        // TODO -- tile may go outside of the matrix
        if (aRow < m && aCol < p && bRow < p && bCol < n) {

            // cast to half for now
            load_matrix_sync(fa, (half*)&a(aRow, aCol), unsigned(p));
            load_matrix_sync(fb, (half*)&b(bRow, bCol), unsigned(n));
            mma_sync(fc, fa, fb, fc);
        }

    }

    int cRow = wy * WMMA_TILE_M;
    int cCol = wx * WMMA_TILE_N;

    if (cRow < m && cCol < n) {
        store_matrix_sync(&c(cRow, cCol), fc, n, mem_row_major);
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

bool almost_equal(float a, float b,
                  float maxRelativeDiff = std::numeric_limits<float>::epsilon())
{
    if (std::isnan(a) || std::isnan(b))
    {
        return false;
    }

    const float difference = fabs(a - b);

    // Scale to the largest value.
    a = fabs(a);
    b = fabs(b);
    const float scaledEpsilon =
        maxRelativeDiff * max(a, b);

    return difference <= scaledEpsilon;
}

void check(const Product &product)
{
#define ca(i, j) product.ca[i * product.n + j]
#define ce(i, j) product.ce[i * product.n + j]

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
    typedef std::chrono::system_clock Clock;
    typedef std::chrono::duration<double> Duration;
    typedef std::chrono::time_point<Clock, Duration> Time;

    cudaStream_t stream;
    CUDA_RUNTIME(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t eStart, eStop;
    CUDA_RUNTIME(cudaEventCreate(&eStart));
    CUDA_RUNTIME(cudaEventCreate(&eStop));

    srand(100);

    for (int ti = 0; ti < 1000; ++ti)
    {
        // while(true) {

        Product product = create(2, 10);

        cout << product.m << "," << product.n << "," << product.p << "," << product.flop() << flush;

        {
            auto start = Clock::now();
            for (int i = 0; i < 10; ++i)
            {
                cpu(product.ce, product.a, product.b, product.m, product.n, product.p);
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
                mm<<<gd, bd, 0, stream>>>(product.ca, product.a, product.b, product.m, product.n, product.p);
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
                mm_s<TILE_X, TILE_Y, TILE_P><<<gd, bd, 0, stream>>>(product.ca, product.a, product.b, product.m, product.n, product.p);
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