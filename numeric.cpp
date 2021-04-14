#include "numeric.hpp"

#include <cstdlib>

void fill(float *a, const size_t n)
{
    for (int i = 0; i < n; ++i)
    {
        a[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}