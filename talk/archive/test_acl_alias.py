"""Test if aclCreateTensor aliases torch_npu tensor storage"""
import torch, torch_npu, ctypes

torch_npu.npu.set_device(0)

x = torch.full((4,), 99.0, dtype=torch.float16, device="npu:0")
torch.npu.synchronize()
print("before:", x.cpu().tolist())

lib = ctypes.CDLL("libopapi.so")
nnop = ctypes.CDLL("libnnopbase.so")

create_fn = nnop.aclCreateTensor
create_fn.restype = ctypes.c_void_p

shape = (ctypes.c_int64 * 1)(4)
strides = (ctypes.c_int64 * 1)(1)
storage_shape = (ctypes.c_int64 * 1)(4)
data_ptr = ctypes.c_void_p(x.data_ptr())
print(f"x.data_ptr = {hex(x.data_ptr())}")

acl_x = create_fn(shape, 1, 1, strides, 0, 2, storage_shape, 1, data_ptr)
print(f"acl_x ptr: {hex(acl_x) if acl_x else 'NULL'}")

ws_fn = lib.aclnnInplaceZeroGetWorkspaceSize
ws_fn.restype = ctypes.c_int
ws_size = ctypes.c_uint64(0)
executor = ctypes.c_void_p(0)
ret = ws_fn(ctypes.c_void_p(acl_x), ctypes.byref(ws_size), ctypes.byref(executor))
print(f"ws ret={ret} ws_size={ws_size.value}")

stream = torch_npu.npu.current_stream().npu_stream
exec_fn = lib.aclnnInplaceZero
exec_fn.restype = ctypes.c_int
ret2 = exec_fn(None, ctypes.c_uint64(0), executor, ctypes.c_void_p(stream))
print(f"exec ret={ret2}")

acl_lib = ctypes.CDLL("libascendcl.so")
acl_lib.aclrtSynchronizeStream(ctypes.c_void_p(stream))

print("after InplaceZero:", x.cpu().tolist())
