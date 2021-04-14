#pragma once

#include "benchmark.hpp"

class CPURCR : public Benchmark
{

public:
    virtual Result run(const Spec &spec) override;

protected:
    bool check(const Product &product) override
    {
        (void)product;
        return true;
    }
    Product initialize(const Spec &spec) override;
    void finalize(Product &product) override;

private:
    void mm(float *_c, const float *_a, const float *_b, const int M, const int N, const int K);
};