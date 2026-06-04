"""ATK custom generator for ChunkGatedDeltaRuleFwdH (310P).

Enforces cross-input constraints the YAML cannot express:
  - B = 1
  - same T across k/w/u/g, T multiple of chunk_size, T >= chunk_size
  - same Hv across k/w/u/g/initial_state
  - same Kdim across k/w/initial_state
  - same Vdim across u/initial_state
  - g in [-0.1, 0]
"""

from __future__ import annotations

import random

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig

CHUNK_SIZE = 64

# Qwen3.5-4B GDN head/dim. Add more here if kernel supports them.
HV_CHOICES = [4, 8]
K_CHOICES = [128, 192]
V_CHOICES = [128]


def _pick_T(rng: random.Random, lo: int, hi: int) -> int:
    n_chunks_lo = max(1, lo // CHUNK_SIZE)
    n_chunks_hi = max(n_chunks_lo, hi // CHUNK_SIZE)
    return rng.randint(n_chunks_lo, n_chunks_hi) * CHUNK_SIZE


@GENERATOR_REGISTRY.register("cgdr_fwd_h_v310")
class CgdrFwdHGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        rng = random.Random(getattr(case_config, "seed", 0) or 0)

        size_bucket = getattr(case_config, "size_range", (64, 2048))
        T = _pick_T(rng, size_bucket[0], size_bucket[1])
        B = 1
        Hv = rng.choice(HV_CHOICES)
        K = rng.choice(K_CHOICES)
        V = rng.choice(V_CHOICES)

        shape_map = {
            "k": [B, Hv, T, K],
            "w": [B, Hv, T, K],
            "u": [B, Hv, T, V],
            "g": [B, Hv, T],
            "initial_state": [B, Hv, K, V],
        }

        for inp in case_config.inputs:
            if inp.name in shape_map:
                inp.shape = shape_map[inp.name]

        for attr in case_config.attrs:
            if attr.name == "chunk_size":
                attr.value = CHUNK_SIZE
            elif attr.name == "output_final_state":
                attr.value = rng.choice([True, False])

        return case_config
