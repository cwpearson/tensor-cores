#include "benchmark_cpu_float_rcr.hpp"

#include "numeric.hpp"
#include "time.hpp"

void CPURCR::initialize(const Spec &spec)
{
    m_ = spec.m;
    n_ = spec.n;
    k_ = spec.k;

    a_ = new float[m_ * k_];
    fill(a_, m_ * k_);

    b_ = new float[k_ * n_];
    fill(b_, k_ * n_);

    c_ = new float[m_ * n_];
}

void CPURCR::finalize()
{
    delete[] a_;
    delete[] b_;
    delete[] c_;
}

void CPURCR::mm(float * __restrict__ _c, const float * __restrict__ _a, const float * __restrict__ _b, const int M, const int N, const int K) {
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


double CPURCR::sample()
{
    auto start = Clock::now();
    mm(c_, a_, b_, m_, n_, k_);
    Duration elapsed = Clock::now() - start;
    return elapsed.count();
}
