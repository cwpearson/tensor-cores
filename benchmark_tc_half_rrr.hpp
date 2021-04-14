#pragma once

#include "benchmark.hpp"

class TCHalfRRR : public Benchmark
{
public:
    TCHalfRRR();
    ~TCHalfRRR() override;

protected:
    cudaStream_t stream_;
    cudaEvent_t start_;
    cudaEvent_t stop_;

    bool check(const Product &product) override;
    Product initialize(const Spec &spec) override;
    void finalize(Product &product) override;
    double mm(Product &product) override;
};