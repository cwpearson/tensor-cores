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
        return Spec(m, n, k);
    }
};

/* these are void since internally, the benchmark may use types not known to the C++ compiler (like half)
*/
struct Product
{
    void *a;
    void *b;
    void *ce; // expected
    void *ca; // actual
    int m;
    int n;
    int k;

    Product() : a(nullptr), b(nullptr), ce(nullptr), ca(nullptr), m(0), n(0), k(0)
    {
    }

    // real flops in the multiplication
    size_t flop() const
    {
        return size_t(m) * size_t(n) * size_t(k) * 2;
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
    // check a product for correctness
    virtual bool check(const Product &product) = 0;

    // create needed allocations
    virtual Product initialize(const Spec &spec) = 0;
    // release allocations
    virtual void finalize(Product &product) = 0;

public:
    virtual ~Benchmark() {}

    // run a benchmark
    virtual Result run(const Spec &s) = 0;
};