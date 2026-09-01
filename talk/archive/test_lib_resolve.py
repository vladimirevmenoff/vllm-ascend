"""Check which library aclnn functions resolve from"""
import torch, torch_npu, ctypes, ctypes.util

torch_npu.npu.set_device(0)

# Check dladdr for the aclnn functions
cust = ctypes.CDLL("libcust_opapi.so")
opapi = ctypes.CDLL("libopapi.so")

# Get function addresses
ws_fn_cust = ctypes.cast(cust.aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize, ctypes.c_void_p).value
ex_fn_cust = ctypes.cast(cust.aclnnChunkGatedDeltaRuleFwdH, ctypes.c_void_p).value

print(f"cust lib: ws_fn={hex(ws_fn_cust)} exec_fn={hex(ex_fn_cust)}")

# Try to find these in opapi too
try:
    ws_fn_opapi = ctypes.cast(opapi.aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize, ctypes.c_void_p).value
    print(f"opapi lib: ws_fn={hex(ws_fn_opapi)} (CONFLICT!)")
except:
    print("opapi lib: not found (good)")

# Use dladdr to find which .so the function is in
libc = ctypes.CDLL("libc.so.6")

class Dl_info(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]

dladdr = libc.dladdr
dladdr.restype = ctypes.c_int
dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(Dl_info)]

info = Dl_info()
ret = dladdr(ctypes.c_void_p(ws_fn_cust), ctypes.byref(info))
if ret:
    print(f"GetWorkspaceSize resolves to: {info.dli_fname.decode()}")
    print(f"  symbol: {info.dli_sname.decode() if info.dli_sname else 'N/A'}")
    print(f"  addr: {hex(info.dli_saddr)}")

ret2 = dladdr(ctypes.c_void_p(ex_fn_cust), ctypes.byref(info))
if ret2:
    print(f"Execute resolves to: {info.dli_fname.decode()}")
    print(f"  symbol: {info.dli_sname.decode() if info.dli_sname else 'N/A'}")

# Also check CommonOpExecutorRun
try:
    common_run = ctypes.cast(opapi.CommonOpExecutorRun, ctypes.c_void_p).value
    print(f"\nCommonOpExecutorRun in opapi: {hex(common_run)}")
except:
    print("\nCommonOpExecutorRun not in opapi")

# Check what GetOpApiFuncAddr would resolve to
# This is what the EXEC_NPU_CMD macro uses
libdl = ctypes.CDLL("libdl.so.2")
dlsym = libdl.dlsym
dlsym.restype = ctypes.c_void_p
dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

# Try RTLD_DEFAULT
RTLD_DEFAULT = ctypes.c_void_p(0)
resolved = dlsym(RTLD_DEFAULT, b"aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize")
print(f"\nRTLD_DEFAULT resolves GetWorkspaceSize to: {hex(resolved) if resolved else 'NULL'}")
if resolved:
    ret3 = dladdr(ctypes.c_void_p(resolved), ctypes.byref(info))
    if ret3:
        print(f"  from: {info.dli_fname.decode()}")
