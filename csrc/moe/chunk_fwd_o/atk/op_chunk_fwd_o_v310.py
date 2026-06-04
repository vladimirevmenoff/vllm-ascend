"""ATK custom generator for ChunkFwdO (310P).

Cross-input constraints YAML can't express:
  - B=1; T = NT*chunk_size, T >= chunk_size
  - shared Hv across q/k/v/h/g
  - shared Kdim between q/k/h
  - shared Vdim between v/h
  - h shape exactly [B, Hv, NT, Kdim, Vdim]
  - scale = 1/sqrt(Kdim)
"""

from __future__ import annotations

import math
import random

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig

CHUNK_SIZE = 64
HV_CHOICES = [4, 8]
K_CHOICES = [128, 192]
V_CHOICES = [128]


def _pick_T(rng: random.Random, lo: int, hi: int) -> int:
    n_lo = max(1, lo // CHUNK_SIZE)
    n_hi = max(n_lo, hi // CHUNK_SIZE)
    return rng.randint(n_lo, n_hi) * CHUNK_SIZE


@GENERATOR_REGISTRY.register("chunk_fwd_o_v310")
class ChunkFwdOGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        rng = random.Random(getattr(case_config, "seed", 0) or 0)

        bucket = getattr(case_config, "size_range", (64, 2048))
        T = _pick_T(rng, bucket[0], bucket[1])
        NT = T // CHUNK_SIZE
        B = 1
        Hv = rng.choice(HV_CHOICES)
        K = rng.choice(K_CHOICES)
        V = rng.choice(V_CHOICES)

        shape_map = {
            "q": [B, Hv, T, K],
            "k": [B, Hv, T, K],
            "v": [B, Hv, T, V],
            "h": [B, Hv, NT, K, V],
            "g": [B, Hv, T],
        }
        for inp in case_config.inputs:
            if inp.name in shape_map:
                inp.shape = shape_map[inp.name]

        for attr in case_config.attrs:
            if attr.name == "chunk_size":
                attr.value = CHUNK_SIZE
            elif attr.name == "scale":
                attr.value = 1.0 / math.sqrt(K)

        return case_config
