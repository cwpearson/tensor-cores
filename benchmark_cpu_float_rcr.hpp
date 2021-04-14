#pragma once

#include "benchmark.hpp"

class CPURCR : public Benchmark
{

protected:
    bool check(const Product &product) override
    {
        (void)product;
        return true;
    }
    Product initialize(const Spec &spec) override;
    void finalize(Product &product) override;
    double mm(Product &prod) override;
};