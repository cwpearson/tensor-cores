#include "benchmark_rcr_cpu_tiled_float.hpp"

#include "benchmark_cpu_float_rcr.hpp"
#include "numeric.hpp"
#include "time.hpp"

#include <iostream>

void RCR_CPU_tiled_float::initialize(const Spec &spec)
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

void RCR_CPU_tiled_float::finalize()
{
    delete[] a_;
    delete[] b_;
    delete[] c_;
}

bool RCR_CPU_tiled_float::check() {

    bool success = true;

    std::vector<float> _e(m_ *n_);
    CPURCR::mm(_e.data(), a_, b_, m_, n_, k_);

#define ce(_i, _j) (_e[(_i)*n_ + (_j)]) // row-major
#define ca(_i, _j) (c_[(_i)*n_ + (_j)]) // row-major

    for (int i = 0; i < m_; ++i)
    {
        for (int j = 0; j < n_; ++j)
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

    return success;

}

void RCR_CPU_tiled_float::mm(float * __restrict__ c, const float * __restrict__ a, const float * __restrict__ b, const int M, const int N, const int K) {
#define a(_i, _j) (a[(_i)*K + (_j)]) // row major
#define b(_i, _j) (b[(_j)*K + (_i)]) // colum major
#define c(_i, _j) (c[(_i)*N + (_j)]) // row-major

    constexpr int TILE_M = 32;
    constexpr int TILE_N = 32;
    constexpr int TILE_K = 32;

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            c(i,j) = 0;
        }
    }


    for (int ti = 0; ti < M; ti += TILE_M) {
        for (int tj = 0; tj < N; tj += TILE_N) {
            for (int t = 0; t < K; t += TILE_K) {
                for (int i = ti; i < M && i < ti+TILE_M; ++i)
                {
                    for (int j = tj; j < N && j < tj + TILE_N; ++j)
                    {
                        float acc = c(i,j);
                        for (int k = t; k < K && k < t + TILE_K; ++k)
                        {
                            acc += a(i, k) * b(k, j);
                        }
                        c(i, j) = acc;
                    }
                }
            }
            
        }
    }
#undef a
#undef b
#undef c
}

double RCR_CPU_tiled_float::sample()
{
    auto start = Clock::now();
    mm(c_,a_,b_,m_,n_,k_);
    Duration elapsed = Clock::now() - start;
    return elapsed.count();
}
