#include <iostream>
#include <cstdlib>

#include "cuda_runtime.hpp"
#include "benchmark.hpp"
#include "benchmark_cpu_float_rcr.hpp"
#include "benchmark_cpu_float_rrr.hpp"
#include "benchmark_cublas.hpp"
#include "benchmark_gpu_float_rrr_shmem.hpp"
#include "benchmark_gpu_float_rrr.hpp"
#include "benchmark_tc_half_rrr_shmem.hpp"
#include "benchmark_tc_half_rrr.hpp"
#include "numeric.hpp"
#include "time.hpp"



using std::cerr;
using std::cout;
using std::endl;
using std::flush;

int main(void)
{
    srand(100);

    cout << "M,N,K,FLOP,RRR CPU,RCR CPU, RRR GPU Global, RRR GPU Shmem, RRR GPU TC, RRR GPU TC/Shmem, RRC Cublas\n";

    for (int ti = 0; ti < 1000; ++ti)
    {

        Spec spec = Spec::create_log_uniform(3, 11);

        cout << spec.m << "," << spec.n << "," << spec.k << "," << spec.flop() << flush;

#if 1

        {
            CPURRR bench;
            Result res = bench.run(spec);
            cout << "," << res.med() << flush;
        }
#endif

#if 1

        {
            CPURCR bench;
            Result res = bench.run(spec);
            cout << "," << res.med() << flush;
        }
#endif

#if 1
        {
            GPUFloatRRR bench;
            Result res = bench.run(spec);
            cout << "," << res.med() << flush;
        }
#endif

#if 1
        {
            GPUFloatRRRShmem bench;
            Result res = bench.run(spec);
            cout << "," << res.med() << flush;
        }
#endif

#if 1
        {
            TCHalfRRR bench;
            Result res = bench.run(spec);
            cout << "," << res.med() << flush;
        }
#endif

#if 1
        {
            TCHalfRRRShmem bench;
            Result res = bench.run(spec);
            cout << "," << res.med() << flush;
        }
#endif

#if 1
        {
            Cublas bench;
            Result res = bench.run(spec);
            cout << "," << res.med() << flush;
        }
#endif

        cout << endl;

    }


}