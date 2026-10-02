#include "ops/linear/q5/q5_launch.h"

#include "ops/linear/q5/q5_ksplit_mma.cuh"

namespace ninfer::ops::detail {

void launch_q5_ksplit_n5120_k6144_c4(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream) {
    launch_q5_ksplit_mma<5120, 6144, 4>(x, w, out, stream);
}

void launch_q5_ksplit_n5120_k6144_c8(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream) {
    launch_q5_ksplit_mma<5120, 6144, 8>(x, w, out, stream);
}

void launch_q5_ksplit_n5120_k6144_c16(const Tensor& x, const Weight& w, Tensor& out,
                                     cudaStream_t stream) {
    launch_q5_ksplit_mma<5120, 6144, 16>(x, w, out, stream);
}

void launch_q5_ksplit_n5120_k17408_c4(const Tensor& x, const Weight& w, Tensor& out,
                                     cudaStream_t stream) {
    launch_q5_ksplit_mma<5120, 17408, 4>(x, w, out, stream);
}

void launch_q5_ksplit_n5120_k17408_c8(const Tensor& x, const Weight& w, Tensor& out,
                                     cudaStream_t stream) {
    launch_q5_ksplit_mma<5120, 17408, 8>(x, w, out, stream);
}

void launch_q5_ksplit_n5120_k17408_c16(const Tensor& x, const Weight& w, Tensor& out,
                                     cudaStream_t stream) {
    launch_q5_ksplit_mma<5120, 17408, 16>(x, w, out, stream);
}

void launch_q5_ksplit_gdn_value_z_t8(const Tensor& x, const Weight& w, Tensor& value, Tensor& z,
                                     cudaStream_t stream) {
    launch_q5_ksplit_mma_split<12288, 6144, 5120, 8>(x, w, value, z, stream);
}

void launch_q5_ksplit_attn_gate_value_t8(const Tensor& x, const Weight& w, Tensor& gate,
                                         Tensor& value, cudaStream_t stream) {
    launch_q5_ksplit_mma_split<7168, 6144, 5120, 8>(x, w, gate, value, stream);
}

} // namespace ninfer::ops::detail
