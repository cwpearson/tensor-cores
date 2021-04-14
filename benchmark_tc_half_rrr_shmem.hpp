#pragma once

#include "benchmark.hpp"

#include <cuda_fp16.h>

class TCHalfRRRShmem : public Benchmark
{
public:
    TCHalfRRRShmem();
    ~TCHalfRRRShmem() override;

protected:
    cudaStream_t stream_;
    cudaEvent_t start_;
    cudaEvent_t stop_;

    int m_, n_, k_;

    __half *a_; // GPU
    __half *b_; // GPU
    float *ca_; // UM

    // 32-bit for checking correctness
    float *a32_; // UM
    float *b32_; // UM

    bool check() override;
    void initialize(const Spec &spec) override;
    void finalize() override;
    double sample() override;
};