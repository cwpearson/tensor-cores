#pragma once

#include <limits>
#include <cmath>

void fill(float *a, const int m, const int n);



inline bool almost_equal(float a, float b,
                  float maxRelativeDiff = std::numeric_limits<float>::epsilon())
{
    if (std::isnan(a) || std::isnan(b))
    {
        return false;
    }

    const float difference = fabs(a - b);

    // Scale to the largest value.
    a = fabs(a);
    b = fabs(b);
    const float scaledEpsilon =
        maxRelativeDiff * std::fmax(a, b);

    return difference <= scaledEpsilon;
}