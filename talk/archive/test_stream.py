"""Test: verify stream is valid and try raw aclnn call with different stream"""
import torch, torch_npu, ctypes

torch_npu.npu.set_device(0)

# Get torch_npu stream
stream = torch_npu.npu.current_stream()
print(f"torch_npu stream: {stream}")
print(f"npu_stream ptr: {hex(stream.npu_stream)}")

# Create a fresh CANN stream
acl = ctypes.CDLL("libascendcl.so")

acl.aclrtCreateStream.restype = ctypes.c_int
raw_stream = ctypes.c_void_p(0)
ret = acl.aclrtCreateStream(ctypes.byref(raw_stream))
print(f"aclrtCreateStream ret={ret} stream={hex(raw_stream.value)}")

# Simple test with InplaceZero on raw stream
x = torch.full((4,), 99.0, dtype=torch.float16, device="npu:0")
torch.npu.synchronize()

nnop = ctypes.CDLL("libnnopbase.so")
create_fn = nnop.aclCreateTensor
create_fn.restype = ctypes.c_void_p

shape = (ctypes.c_int64 * 1)(4)
strides = (ctypes.c_int64 * 1)(1)
storage_shape = (ctypes.c_int64 * 1)(4)

acl_x = create_fn(shape, 1, 1, strides, 0, 2, storage_shape, 1, ctypes.c_void_p(x.data_ptr()))

lib = ctypes.CDLL("libopapi.so")
ws_fn = lib.aclnnInplaceZeroGetWorkspaceSize
ws_fn.restype = ctypes.c_int
ws_size = ctypes.c_uint64(0)
executor = ctypes.c_void_p(0)
ret = ws_fn(ctypes.c_void_p(acl_x), ctypes.byref(ws_size), ctypes.byref(executor))

# Execute on RAW stream
exec_fn = lib.aclnnInplaceZero
exec_fn.restype = ctypes.c_int
ret2 = exec_fn(None, ctypes.c_uint64(0), executor, raw_stream)
print(f"InplaceZero on raw stream ret={ret2}")
acl.aclrtSynchronizeStream(raw_stream)
print(f"After InplaceZero on raw stream: {x.cpu().tolist()}")

# Now test chunk kernel with raw stream
cust = ctypes.CDLL("libcust_opapi.so")
ws_fn2 = cust.aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize

k = (torch.randn(1,1,64,128, dtype=torch.float16)*0.1).npu()
w = (-torch.rand(1,1,64,128, dtype=torch.float16)*0.5).npu()
u = (torch.randn(1,1,64,128, dtype=torch.float16)*0.1).npu()
g = (-torch.rand(1,1,64)*0.1).float().npu()
h = torch.full((1,1,1,128,128), 42.0, dtype=torch.float16, device="npu:0")
vn = torch.full((1,1,64,128), 42.0, dtype=torch.float16, device="npu:0")
fs = torch.empty((1,), dtype=torch.float16, device="npu:0")
torch.npu.synchronize()

def make_acl(t):
    if t is None: return None
    s = (ctypes.c_int64 * t.dim())(*t.shape)
    st = (ctypes.c_int64 * t.dim())(*t.stride())
    ss = (ctypes.c_int64 * 1)(t.storage().nbytes() // t.element_size())
    dtype_map = {torch.float16: 1, torch.float32: 0}
    return create_fn(s, t.dim(), dtype_map[t.dtype], st, 0, 2, ss, 1, ctypes.c_void_p(t.data_ptr()))

ws_fn2.restype = ctypes.c_int
ws_size2 = ctypes.c_uint64(0)
executor2 = ctypes.c_void_p(0)

ret = ws_fn2(
    ctypes.c_void_p(make_acl(k)),
    ctypes.c_void_p(make_acl(w)),
    ctypes.c_void_p(make_acl(u)),
    ctypes.c_void_p(make_acl(g)),
    None, None,  # gk, initial_state
    ctypes.c_bool(False),  # output_final_state
    ctypes.c_int64(64),  # chunk_size
    ctypes.c_bool(True),  # save_new_value
    None, None,  # cu_seqlens, chunk_indices
    ctypes.c_bool(False),  # use_exp2
    ctypes.c_bool(False),  # transpose_state_layout
    ctypes.c_void_p(make_acl(h)),
    ctypes.c_void_p(make_acl(vn)),
    ctypes.c_void_p(make_acl(fs)),
    ctypes.byref(ws_size2),
    ctypes.byref(executor2))
print(f"chunk GetWS ret={ret} ws_size={ws_size2.value} executor={hex(executor2.value) if executor2.value else 'NULL'}")

if ws_size2.value > 0:
    ws_t = torch.empty(ws_size2.value, dtype=torch.uint8, device="npu:0")
    ws_ptr = ctypes.c_void_p(ws_t.data_ptr())
else:
    ws_ptr = None

exec_fn2 = cust.aclnnChunkGatedDeltaRuleFwdH
exec_fn2.restype = ctypes.c_int
ret2 = exec_fn2(ws_ptr, ws_size2, executor2, raw_stream)
print(f"chunk exec on RAW stream ret={ret2}")

acl.aclrtSynchronizeStream(raw_stream)
h_cpu = h.cpu().float()
print(f"h[0,0,0,0,:4] = {h_cpu[0,0,0,0,:4].tolist()}")
print(f"h still 42? {h_cpu[0,0,0,0,0].item() == 42.0}")
