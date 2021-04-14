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

double CPURCR::sample()
{

#define a(_i, _j) (a_[(_i)*k_ + (_j)]) // row major
#define b(_i, _j) (b_[(_j)*k_ + (_i)]) // colum major
#define c(_i, _j) (c_[(_i)*n_ + (_j)]) // row-major

    auto start = Clock::now();
    for (int i = 0; i < m_; ++i)
    {
        for (int j = 0; j < n_; ++j)
        {
            float acc = 0;
            for (int k = 0; k < k_; ++k)
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
