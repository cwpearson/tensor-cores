#include "benchmark_cpu_float_rcr.hpp"

#include "numeric.hpp"
#include "time.hpp"

Product CPURCR::initialize(const Spec &spec)
{
    Product ret;

    ret.m = spec.m;
    ret.n = spec.n;
    ret.k = spec.k;
    ret.a = new float[ret.m * ret.k];
    fill((float *)ret.a, ret.m, ret.k);
    ret.b = new float[ret.k * ret.n];
    fill((float *)ret.b, ret.k, ret.n);
    ret.ce = new float[ret.m * ret.n];
    ret.ca = new float[ret.m * ret.n];
    return ret;
}

void CPURCR::finalize(Product &prod)
{
    delete[](float *) prod.a;
    delete[](float *) prod.b;
    delete[](float *) prod.ca;
    delete[](float *) prod.ce;
}

double CPURCR::mm(Product &prod)
{

    float *_c = (float *)prod.ce;
    float *_a = (float *)prod.a;
    float *_b = (float *)prod.b;
    const int M = prod.m;
    const int N = prod.n;
    const int K = prod.k;

#define a(_i, _j) (_a[(_i)*K + (_j)]) // row major
#define b(_i, _j) (_b[(_j)*K + (_i)]) // colum major
#define c(_i, _j) (_c[(_i)*N + (_j)]) // row-major

    auto start = Clock::now();
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
    Duration elapsed = Clock::now() - start;
    return elapsed.count();

#undef a
#undef b
#undef c
}
