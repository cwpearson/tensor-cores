#pragma once

#include "benchmark.hpp"

class TCHalfRRR : public Benchmark
{
public:
    TCHalfRRR();
    ~TCHalfRRR() override;
    Result run(const Spec &spec) override;
    bool check(const Product &product) override;

protected:
    cudaStream_t stream_;
    cudaEvent_t start_;
    cudaEvent_t stop_;

    Product initialize(const Spec &spec) override;
    void finalize(Product &product) override;

private:
    double mm(const Product &product);
};