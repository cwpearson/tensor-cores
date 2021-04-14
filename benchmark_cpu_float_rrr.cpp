#include "benchmark_cpu_float_rrr.hpp"

#include "numeric.hpp"
#include "time.hpp"

Product CPURRR::initialize(const Spec &spec)
{
    Product ret;

    ret.m = spec.m;
    ret.n = spec.n;
    ret.k = spec.k;
    ret.a = new float[ret.m * ret.k];
    fill((float *)ret.a, ret.m * ret.k);
    ret.b = new float[ret.k * ret.n];
    fill((float *)ret.b, ret.k * ret.n);
    ret.ce = new float[ret.m * ret.n];
    ret.ca = new float[ret.m * ret.n];
    return ret;
}

void CPURRR::finalize(Product &prod)
{
    delete[](float *) prod.a;
    delete[](float *) prod.b;
    delete[](float *) prod.ca;
    delete[](float *) prod.ce;
}

void CPURRR::mm(float *_c, const float *_a, const float *_b, const int M, const int N, const int K) {
#define a(_i, _j) (_a[(_i)*K + (_j)]) // row-major
#define b(_i, _j) (_b[(_i)*N + (_j)]) // row-major
#define c(_i, _j) (_c[(_i)*N + (_j)]) // row-major

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float acc = 0;
            for (int k = 0; k < K; ++k)
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

double CPURRR::sample(Product &prod)
{

    float *_c = (float *)prod.ce;
    float *_a = (float *)prod.a;
    float *_b = (float *)prod.b;
    const int M = prod.m;
    const int N = prod.n;
    const int K = prod.k;

    auto start = Clock::now();
    mm(_c, _a, _b, M, N, K);
    Duration elapsed = Clock::now() - start;
    return elapsed.count();

}
