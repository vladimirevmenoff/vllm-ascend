# Hand-rolled m200 matmul recipe (from CANN 9.1 matmul lib study)

Key findings (full recipe in agent report):
- m200 lib path: UB->L1 (MTE3, NZ) -> load3dv2 L1->L0A/L0B (MTE1) -> Mmad (M) -> DataCopy CO1->UB on PIPE_V (BLOCK_MODE_MATRIX) -> per-row DataCopy NZ->ND in UB (V).
- L1 layout both A+B: NZ, offset(row,colFrac)=colFrac*rows*16+row*16. Producible w/ per-16-col DataCopy {blockCount=64, blockLen=1, srcStride=cols/16-1}.
- Lib dispatch on m200 = load3dv2 (loadInstr variants), NOT load2d; but plain LoadData2d equally valid + simpler: A {rep=K1,srcStride=M1,ifTranspose=false} x M1 calls; B {rep=N1,srcStride=K1,ifTranspose=true} x K1 calls.
- FMatrix reg shared A/B on m200 -> re-Set between loads if using load3dv2.
- Mmad: m/k/n, cmatrixInitVal=true, cmatrixSource=false, unitFlag=0. PipeBarrier<PIPE_M> only if (m/16)(n/16)<10 (not for 64x64).
- CO1->UB is PIPE_V (proven by V_MTE3 deque in lib). fp32 GetBlockCount()=16.
- Missing from snapshot: copy_tile_to_cube (exact UB->L1 lib call), feature_trait, scheduler event wiring, LoadData3DParamsV2Pro defaults.
- 310P ini: L1 1MB, L0A/B 64KB, L0C 256KB, UB 256KB.
