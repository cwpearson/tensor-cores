#pragma once

#include "benchmark.hpp"

#include <cuda_fp16.h>

class GPUFloatRRRShmem : public Benchmark
{
public:
    GPUFloatRRRShmem();
    ~GPUFloatRRRShmem() override;

protected:
    cudaStream_t stream_;
    cudaEvent_t start_;
    cudaEvent_t stop_;

    int m_, n_, k_;

    float *a_, *b_; // GPU
    float *c_; // UM

    bool check() override;
    void initialize(const Spec &spec) override;
    void finalize() override;
    double sample() override;
    std::string name() override {return "RRR GPU float shmem";}
};