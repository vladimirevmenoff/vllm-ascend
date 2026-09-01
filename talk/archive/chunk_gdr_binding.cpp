#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <cmath>

#include "aclnn_torch_adapter/op_api_common.h"

namespace chunk_test {

static aclTensor* make_acl_tensor_nd(const at::Tensor& t) {
    if (!t.defined()) return nullptr;
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (!aclCreateTensor) return nullptr;
    auto dtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(t.scalar_type())];
    auto itemsize = t.itemsize();
    int64_t storage_dim = t.storage().nbytes() / itemsize;
    void* data_ptr = const_cast<void*>(t.storage().data());
    auto acl_t = aclCreateTensor(
        t.sizes().data(), t.sizes().size(), dtype,
        t.strides().data(), t.storage_offset(),
        ACL_FORMAT_ND,
        &storage_dim, 1,
        data_ptr);
    return acl_t;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_gated_delta_rule_fwd_h(
    const at::Tensor & k,
    const at::Tensor & w,
    const at::Tensor & u,
    const c10::optional<at::Tensor> & g,
    const c10::optional<at::Tensor> & gk,
    const c10::optional<at::Tensor> & initial_state,
    c10::optional<bool> output_final_state,
    c10::optional<int64_t> chunk_size,
    c10::optional<bool> save_new_value,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices,
    c10::optional<bool> use_exp2,
    c10::optional<bool> transpose_state_layout)
{
    bool output_final_state_ = false;
    const at::Tensor &initial_state_ = c10::value_or_else(initial_state, [] { return at::Tensor(); });
    int64_t chunk_size_ = chunk_size.has_value() ? chunk_size.value() : 64;
    const at::Tensor &g_ = c10::value_or_else(g, [] { return at::Tensor(); });
    const at::Tensor &gk_ = c10::value_or_else(gk, [] { return at::Tensor(); });

    auto k_sizes = k.sizes();
    auto u_sizes = u.sizes();
    int K = k_sizes[3];
    int B = k_sizes[0];
    int T = k_sizes[2];
    int HV = u_sizes[1];
    int V = u_sizes[3];
    int NT = (T + chunk_size_ - 1) / chunk_size_;

    at::Tensor h_out = at::empty({B, HV, NT, K, V}, k.options());
    h_out.fill_(42.0);
    at::Tensor v_new_out = at::empty(u.sizes(), u.options());
    v_new_out.fill_(42.0);
    at::Tensor final_state_out = at::empty({1}, k.options());

    bool save_new_value_ = save_new_value.value_or(true);
    fprintf(stderr, "DBG: B=%d T=%d K=%d V=%d HV=%d NT=%d ofs=%d csz=%ld\n", B, T, K, V, HV, NT, (int)output_final_state_, chunk_size_);
    fflush(stderr);

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    aclrtSynchronizeStream(acl_stream);
    fprintf(stderr, "DBG: stream ok\n"); fflush(stderr);

    // Create aclTensors
    aclTensor* acl_k = make_acl_tensor_nd(k);
    fprintf(stderr, "DBG: acl_k=%p\n", acl_k); fflush(stderr);
    aclTensor* acl_w = make_acl_tensor_nd(w);
    aclTensor* acl_u = make_acl_tensor_nd(u);
    aclTensor* acl_g = make_acl_tensor_nd(g_);
    aclTensor* acl_h = make_acl_tensor_nd(h_out);
    aclTensor* acl_vn = make_acl_tensor_nd(v_new_out);
    aclTensor* acl_fs = make_acl_tensor_nd(final_state_out);
    fprintf(stderr, "DBG: all tensors created\n"); fflush(stderr);

    static const auto getWsSize = GetOpApiFuncAddr("aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize");
    static const auto execFn = GetOpApiFuncAddr("aclnnChunkGatedDeltaRuleFwdH");
    fprintf(stderr, "DBG: getWsSize=%p execFn=%p\n", getWsSize, execFn); fflush(stderr);

    typedef int (*GetWsSizeFn)(
        const aclTensor*, const aclTensor*, const aclTensor*,
        const aclTensor*, const aclTensor*, const aclTensor*,
        bool, int64_t, bool,
        const aclIntArray*, const aclIntArray*,
        bool, bool,
        const aclTensor*, const aclTensor*, const aclTensor*,
        uint64_t*, aclOpExecutor**);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    fprintf(stderr, "DBG: calling GetWS...\n"); fflush(stderr);

    int ws_ret = reinterpret_cast<GetWsSizeFn>(getWsSize)(
        acl_k, acl_w, acl_u,
        acl_g, nullptr, nullptr,
        output_final_state_, chunk_size_, save_new_value_,
        nullptr, nullptr,
        false, false,
        acl_h, acl_vn, acl_fs,
        &ws_size, &executor);

    fprintf(stderr, "GetWS ret=%d ws_size=%lu\n", ws_ret, ws_size);

    // Allocate workspace as a REAL tensor so we can inspect it later
    at::Tensor ws_tensor;
    void* ws_addr = nullptr;
    if (ws_size > 0) {
        ws_tensor = at::empty({(long)ws_size},
            at::TensorOptions(torch_npu::utils::get_npu_device_type()).dtype(at::kByte));
        ws_addr = const_cast<void*>(ws_tensor.storage().data());
    }

    // Execute
    typedef int (*ExecFn)(void*, uint64_t, aclOpExecutor*, aclrtStream);
    int exec_ret = reinterpret_cast<ExecFn>(execFn)(ws_addr, ws_size, executor, acl_stream);
    fprintf(stderr, "Exec ret=%d\n", exec_ret);

    aclrtSynchronizeStream(acl_stream);

    // Check h_out
    {
        auto h_cpu = h_out.cpu().to(at::kFloat);
        float* hd = h_cpu.data_ptr<float>();
        int total = h_cpu.numel();
        int non42 = 0;
        for (int i = 0; i < total; i++) {
            if (std::abs(hd[i] - 42.0f) > 0.01f) non42++;
        }
        fprintf(stderr, "h_out: total=%d non42=%d first8=[%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f]\n",
                total, non42, hd[0], hd[1], hd[2], hd[3], hd[4], hd[5], hd[6], hd[7]);
    }

    // Check workspace for non-zero content
    if (ws_size > 0) {
        auto ws_cpu = ws_tensor.cpu();
        uint8_t* wd = ws_cpu.data_ptr<uint8_t>();
        int nonzero = 0;
        for (uint64_t i = 0; i < std::min(ws_size, (uint64_t)1000000); i++) {
            if (wd[i] != 0) nonzero++;
        }
        // Check first 64 bytes as float16
        fprintf(stderr, "workspace: size=%lu nonzero_in_first_1M=%d\n", ws_size, nonzero);
        float* wf = reinterpret_cast<float*>(wd);
        fprintf(stderr, "ws as float32[0:8] = [%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f]\n",
                wf[0], wf[1], wf[2], wf[3], wf[4], wf[5], wf[6], wf[7]);
    }

    return std::make_tuple(h_out, v_new_out, at::Tensor());
}

TORCH_LIBRARY(_C_chunk_test, m) {
    m.def("chunk_gated_delta_rule_fwd_h(Tensor k, Tensor w, Tensor u, "
          "Tensor? g=None, Tensor? gk=None, Tensor? initial_state=None, "
          "bool? output_final_state=False, int? chunk_size=64, "
          "bool? save_new_value=False, int[]? cu_seqlens=None, "
          "int[]? chunk_indices=None, bool? use_exp2=False, "
          "bool? transpose_state_layout=False) -> (Tensor, Tensor, Tensor)");
    m.impl("chunk_gated_delta_rule_fwd_h", &chunk_gated_delta_rule_fwd_h);
}

} // namespace chunk_test
