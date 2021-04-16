#pragma once

#include "benchmark.hpp"

class RCR_CPU_tiled_float : public Benchmark
{
public:
    // public so other benchmarks can use it to check correctness
    static void mm(float *c, const float *a, const float *b, const int M, const int N, const int K);
    
protected:
    bool check() override;
    void initialize(const Spec &spec) override;
    void finalize() override;
    double sample() override;

    int m_, n_, k_;
    float *a_, *b_, *c_;

};