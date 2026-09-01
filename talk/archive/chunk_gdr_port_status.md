# chunk_gated_delta_rule_fwd_h — 310P port status

## Fixed — BUILD NOW SUCCEEDS
1. def.cpp: added `ascend310p` config, fp16-only dtype combos
2. compat_310p.h: dummy `bfloat16_t` struct so catlass headers compile
3. compat_310p.h: `#define PIPE_FIX PIPE_MTE3` — 310P has no fixpipe unit
4. compat_310p.h: `#define LoadDataWithSparse LoadDataWithSparseCal` — API rename on 310P
5. block_epilogue_gdn_fwdh_update.hpp: `AscendC::ToFloat` → `(float)` cast (bf16 dead code path)
6. chunk_gated_delta_rule_fwd_h.cpp: `#include "compat_310p.h"` — CMake `-include` doesn't reach ccec/opc
7. CMakeLists.txt: `-include compat_310p.h` (host-side only, kernel uses direct include)

## Key discovery: -include flag doesn't reach kernel compiler
`add_ops_compile_options` in CMakeLists.txt only passes flags to the host-side compiler (g++).
The kernel is compiled by `opc` which invokes `ccec` with its own flags.
Fix: include compat_310p.h directly from the kernel source file.

## Key discovery: __CCE_AICORE__ == 310 is NOT 310P
- `__CCE_AICORE__ == 310` → Ascend950 (A5 chip, arch35)
- 310P falls through to `#else` → AtlasA2 path (CATLASS_ARCH=2201)
- AtlasA2 arch struct (UB=192K, L0C=128K) is close enough for 310P3

## Build command
```
cd csrc && rm -rf build && bash build.sh --pkg --soc=ascend310p --ops=chunk_gated_delta_rule_fwd_h
```

## Install
```
cd build && ./cann-ops-transformer-custom_linux-aarch64.run --quiet --install-for-all
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
```

## Next: precision test
- Need to test with torch_npu calling aclnnChunkGatedDeltaRuleFwdH
- Compare NPU output vs fp32 reference (PyTorch CPU)
- Focus on per-chunk state writeback (line 159 of block_epilogue_gdn_fwdh_update.hpp — fp32→fp16 cast)
- Also need chunk_fwd_o op for the full pipeline
