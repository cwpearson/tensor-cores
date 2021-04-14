#pragma once

#include "benchmark.hpp"

class CPURRR : public Benchmark
{

public:
    // public so other benchmarks can use it to check correctness
    static void mm(float *c, const float *a, const float *b, const int M, const int N, const int K);

protected:
    bool check(const Product &product) override
    {
        (void)product;
        return true;
    }
    Product initialize(const Spec &spec) override;
    void finalize(Product &product) override;
    double sample(Product &prod) override;
};