#pragma once

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>

/* c(m,n) = a(m,k) * b(k,n)

   definition of a matrix multiplication to measure
*/
struct Spec
{
    int m;
    int n;
    int k;

    Spec(int _m, int _n, int _k) : m(_m), n(_n), k(_k) {}

    // actual flops required by the problem
    size_t flop() const
    {
        return size_t(m) * size_t(n) * size_t(k) * 2;
    }

    static Spec create_log_uniform(int lo, int hi)
    {
        float logm = (float(rand()) / RAND_MAX) * (hi - lo) + lo;
        float logn = (float(rand()) / RAND_MAX) * (hi - lo) + lo;
        float logk = (float(rand()) / RAND_MAX) * (hi - lo) + lo;
        int m = std::pow(2, logm);
        int n = std::pow(2, logn);
        int k = std::pow(2, logk);
        // return Spec(33,16,16);
        return Spec(m, n, k);
    }
};

struct Result
{
    enum class Status
    {
        skipped,
        success,
        error
    } status;
    std::vector<double> samples;

    void add_sample(double d)
    {
        samples.push_back(d);
    }

    double med()
    {
        if (samples.empty())
        {
            throw std::logic_error("no samples");
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    }
};

class Benchmark
{
protected:
    // check the result for correctness (called on first sample)
    virtual bool check() = 0;

    // create needed allocations
    virtual void initialize(const Spec &spec) = 0;

    // release allocations
    virtual void finalize() = 0;

    // return the time taken by a single matmul
    virtual double sample() = 0;

public:
    virtual ~Benchmark() {}

    // run a benchmark
    virtual Result run(const Spec &spec)
    {
        Result ret;
        ret.status = Result::Status::success;
        initialize(spec);

        for (int i = 0; i < 20; ++i)
        {
            double secs = sample();

            if (0 == i)
            {
                if (!check())
                {
                    ret.status = Result::Status::error;
                    throw std::logic_error("correctness failed");
                }
            }

            ret.add_sample(secs);
        }

        finalize();
        return ret;
    }
};