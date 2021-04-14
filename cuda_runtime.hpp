#pragma once

#include <cstdio>

#include <cuda_runtime.h>
#include <curand.h>

inline void checkCuda(cudaError_t result, const char *file, const int line)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "%s:%d: CUDA Runtime Error %d: %s\n", file, line, int(result), cudaGetErrorString(result));
        exit(-1);
    }
}
#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);


inline void checkCurand(curandStatus_t result, const char *file, const int line)
{
    if (result != CURAND_STATUS_SUCCESS)
    {
        fprintf(stderr, "%s:%d: cuRand Error %d\n", file, line, int(result));
        exit(-1);
    }
}
#define CURAND(stmt) checkCurand(stmt, __FILE__, __LINE__);