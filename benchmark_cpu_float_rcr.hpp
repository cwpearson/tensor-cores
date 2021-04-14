#pragma once

#include "benchmark.hpp"

class CPURCR : public Benchmark
{

protected:
    bool check() override // no check
    {
        return true;
    }
    void initialize(const Spec &spec) override;
    void finalize() override;
    double sample() override;

    int m_, n_, k_;
    float *a_, *b_, *c_;

};