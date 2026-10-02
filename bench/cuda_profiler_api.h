// Compatibility shim for the CUDA profiler runtime entry points.
//
// CUDA 13 removed <cuda_profiler_api.h> while libcudart keeps exporting cudaProfilerStart/Stop, so
// bench targets that include the header directly fail to compile on CUDA 13. bench/ is on the
// include path of every bench target, so this file answers that include: it forwards to the real
// toolkit header when one is still reachable, and otherwise declares the two exported entry points.
// Bench-only; no production target sees this directory on its include path.
#ifndef NINFER_BENCH_CUDA_PROFILER_API_H_
#define NINFER_BENCH_CUDA_PROFILER_API_H_

#if defined(__has_include_next)
#if __has_include_next(<cuda_profiler_api.h>)
#include_next <cuda_profiler_api.h>
#define NINFER_BENCH_CUDA_PROFILER_API_FORWARDED 1
#endif
#endif

#ifndef NINFER_BENCH_CUDA_PROFILER_API_FORWARDED
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t CUDARTAPI cudaProfilerStart(void);
cudaError_t CUDARTAPI cudaProfilerStop(void);

#ifdef __cplusplus
}
#endif
#endif

#endif
