/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_MICRO_MM_H
#define CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_MICRO_MM_H

#include "kernel_operator.h"

namespace ChunkGatedDeltaRuleComputeWy {

using namespace AscendC;

// Hand-rolled m200 cube path for the two fixed shapes this kernel needs:
// C[64,n] fp32 = A[64,64] half x B[64,n] half, n in {64,128}, operands ND in
// UB. Bypasses the matmul library, whose per-call overhead (~1.8us measured)
// dwarfs the ~65ns of arithmetic at these sizes.
class WyMicroMm {
 public:
  __aicore__ inline void Init(TPipe *pipe) {
    pipe->InitBuffer(l1ABuf_, 64 * 64 * sizeof(half));
    pipe->InitBuffer(l1BBuf_, 64 * 128 * sizeof(half));
    pipe->InitBuffer(l0ABuf_, 64 * 64 * sizeof(half));
    pipe->InitBuffer(l0BBuf_, 64 * 128 * sizeof(half));
    pipe->InitBuffer(coBuf_, 64 * 128 * sizeof(float));
  }

  // aUb: [64,64] half, row stride 64. bUb: [64,n] half, row stride n (both
  // contiguous ND). cUb: [64,n] fp32, row stride ldc. cNz: fp32 scratch >= 64*n.
  __aicore__ inline void Mm(LocalTensor<float> cUb, LocalTensor<half> aUb, LocalTensor<half> bUb,
                            LocalTensor<float> cNz, uint32_t n) {
    Mm(cUb, aUb, bUb, cNz, n, n);
  }

  __aicore__ inline void Mm(LocalTensor<float> cUb, LocalTensor<half> aUb, LocalTensor<half> bUb,
                            LocalTensor<float> cNz, uint32_t n, uint32_t ldc) {
    LocalTensor<half> l1A = l1ABuf_.Get<half>();
    LocalTensor<half> l1B = l1BBuf_.Get<half>();
    LocalTensor<half> l0A = l0ABuf_.Get<half>();
    LocalTensor<half> l0B = l0BBuf_.Get<half>();
    LocalTensor<float> co = coBuf_.Get<float>();
    const uint16_t n1 = static_cast<uint16_t>(n / 16);
    // A/B were produced on V; the L1 staging below reads them. UB->L1 nd2nz:
    // one DataCopy per 16-column fractal, all 64 rows land contiguously.
    Evt<HardEvent::V_MTE3>();
    for (uint32_t j = 0; j < 4; ++j) {
      DataCopy(l1A[j * 64 * 16], aUb[j * 16], {64, 1, 3, 0});
    }
    for (uint32_t j = 0; j < n1; ++j) {
      DataCopy(l1B[j * 64 * 16], bUb[j * 16], {64, 1, static_cast<uint16_t>(n1 - 1), 0});
    }
    Evt<HardEvent::MTE3_MTE1>();
    // L1(NZ) -> L0A (Zz): each m-fractal-row gathers its 4 k-fractals.
    LoadData2dParams pa(0, 4, 4, 0, 0, false, 0);
    for (uint32_t i = 0; i < 4; ++i) {
      LoadData(l0A[i * 4 * 256], l1A[i * 256], pa);
    }
    // L1(NZ) -> L0B (Zn): each k-fractal gathers its n1 n-fractals; the
    // transpose flag is how load2d expresses non-transposed B (see
    // load_to_l0b_load2d.h in the toolkit).
    LoadData2dParams pb(0, static_cast<uint8_t>(n1), 4, 0, 0, true, 0);
    for (uint32_t i = 0; i < 4; ++i) {
      LoadData(l0B[i * n1 * 256], l1B[i * 256], pb);
    }
    Evt<HardEvent::MTE1_M>();
    MmadParams mp(64, static_cast<uint16_t>(n), 64, /*unitFlag=*/0, /*cmatrixSource=*/false,
                  /*cmatrixInitVal=*/true);
    Mmad(co, l0A, l0B, mp);
    Evt<HardEvent::M_V>();
    // CO1 (fp32 NZ) -> UB NZ via matrix block mode, then NZ->ND with one Muls
    // per n-fractal column (16 fp32 lanes x 64 rows each).
    DataCopyEnhancedParams enh;
    enh.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(cNz, co, {n1, 4, 0, 0}, enh);
    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < n1; ++j) {
      Muls(cUb[j * 16], cNz[j * 64 * 16], 1.0f, static_cast<uint64_t>(16), 64,
           {1, 1, static_cast<uint8_t>(ldc * sizeof(float) / 32), 2});
    }
    PipeBarrier<PIPE_V>();
    Evt<HardEvent::V_M>();
  }

 private:
  template <HardEvent E>
  __aicore__ inline void Evt() {
    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
    SetFlag<E>(evt);
    WaitFlag<E>(evt);
  }

  TBuf<TPosition::A1> l1ABuf_;
  TBuf<TPosition::B1> l1BBuf_;
  TBuf<TPosition::A2> l0ABuf_;
  TBuf<TPosition::B2> l0BBuf_;
  TBuf<TPosition::CO1> coBuf_;
};

}  // namespace ChunkGatedDeltaRuleComputeWy

#endif  // CHUNK_GATED_DELTA_RULE_COMPUTE_WY_ARCH20_MICRO_MM_H
