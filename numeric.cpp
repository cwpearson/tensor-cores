#include "numeric.hpp"

#include <cstdlib>

void fill(float *a, const int m, const int n)
{
    for (int i = 0; i < m * n; ++i)
    {
        a[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}