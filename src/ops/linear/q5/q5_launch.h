#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q5Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q5_gemv_r16_s2_x(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_simt_split2_exact(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_q5_simt_split4_exact(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_q5_mma_r64_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_ksplit_n5120_k6144_c4(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream);
void launch_q5_ksplit_n5120_k6144_c8(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream);
void launch_q5_ksplit_n5120_k6144_c16(const Tensor& x, const Weight& w, Tensor& out,
                                     cudaStream_t stream);
void launch_q5_ksplit_n5120_k17408_c4(const Tensor& x, const Weight& w, Tensor& out,
                                     cudaStream_t stream);
void launch_q5_ksplit_n5120_k17408_c8(const Tensor& x, const Weight& w, Tensor& out,
                                     cudaStream_t stream);
void launch_q5_ksplit_n5120_k17408_c16(const Tensor& x, const Weight& w, Tensor& out,
                                      cudaStream_t stream);
// Q5 GDN value/z projection: split the 12288 logical rows into two 6144-row outputs.
void launch_q5_ksplit_gdn_value_z_t8(const Tensor& x, const Weight& w, Tensor& value, Tensor& z,
                                     cudaStream_t stream);
// Q5 attention gate/value projection: 7168 rows = gate(6144) + value(1024), K=5120, T=8.
void launch_q5_ksplit_attn_gate_value_t8(const Tensor& x, const Weight& w, Tensor& gate,
                                         Tensor& value, cudaStream_t stream);

} // namespace ninfer::ops::detail
