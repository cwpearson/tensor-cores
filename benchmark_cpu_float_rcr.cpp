#include "benchmark_cpu_float_rcr.hpp"

#include "numeric.hpp"
#include "time.hpp"

Result CPURCR::run(const Spec &spec)
{
    Result ret;
    ret.status = Result::Status::error;
    Product prod = initialize(spec);

    for (int i = 0; i < 10; ++i)
    {
        auto start = Clock::now();
        mm((float*)prod.ce, (float*)prod.a, (float*)prod.b, prod.m, prod.n, prod.k);
        Duration elapsed = Clock::now() - start;
        ret.add_sample(elapsed.count());
    }

    finalize(prod);
    ret.status = Result::Status::success;
    return ret;
}

Product CPURCR::initialize(const Spec &spec)
{
    Product ret;

    ret.m = spec.m;
    ret.n = spec.n;
    ret.k = spec.k;
    ret.a = new float[ret.m * ret.k];
    fill((float*)ret.a, ret.m, ret.k);
    ret.b = new float[ret.k * ret.n];
    fill((float*)ret.b, ret.k, ret.n);
    ret.ce = new float[ret.m * ret.n];
    ret.ca = new float[ret.m * ret.n];
    return ret;
}

void CPURCR::finalize(Product &prod)
{
    delete[] (float*)prod.a;
    delete[] (float*)prod.b;
    delete[] (float*)prod.ca;
    delete[] (float*)prod.ce;
}

void CPURCR::mm(float *_c, const float *_a, const float *_b, const int M, const int N, const int K)
{
#define a(_i, _j) (_a[(_i)*K + (_j)]) // row major
#define b(_i, _j) (_b[(_j)*K + (_i)]) // colum major
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
