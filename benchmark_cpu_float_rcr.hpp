#pragma once

#include "benchmark.hpp"

class CPURCR : public Benchmark
{

public:
    // public so other benchmarks can use it to check correctness
    static void mm(float *c, const float *a, const float *b, const int M, const int N, const int K);

protected:
    bool check() override // no check
    {
        return true;
    }
    void initialize(const Spec &spec) override;
    void finalize() override;
    double sample() override;
    std::string name() override {return "RCR CPU";}

private:
    int m_, n_, k_;
    float *a_, *b_, *c_;
};