# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The prefix store's keys (`ttnn/prefix_store.py`): no ttnn, no torch, so that vLLM's scheduler
(tools/glm53_vllm_scheduler.py) and the adapter hash a prompt the same way the store's entries were keyed."""

from __future__ import annotations

import hashlib
import os
from array import array

GRID = 8192  # snapshot / resume positions: multiples of the chunked prefill's grid (tools/glm53_vllm.py CHUNK_GRID)
AUTO = "auto"


def entries_setting(default: str = "0") -> int | str:
    """GLM53_PREFIX_ENTRIES: "auto" (the store sized from the free DRAM at start-up, `prefix_store.auto_entries`), or a
    number of entries (0: no store); unset or empty: `default` (the vLLM serving's "auto", the generator's "0")."""
    raw = os.environ.get("GLM53_PREFIX_ENTRIES", "").strip().lower() or default
    if raw == AUTO:
        return AUTO
    try:
        n = int(raw)
    except ValueError as err:
        raise ValueError(f"GLM53_PREFIX_ENTRIES={raw!r}: 'auto' or a number of entries (0: off)") from err
    if n < 0:
        raise ValueError(f"GLM53_PREFIX_ENTRIES={n}: 'auto' or a number of entries (0: off)")
    return n


def prefix_keys(tokens, salt: str = "", grid: int = GRID) -> list[str]:
    """Chained hashes of a prompt's whole blocks of `grid` tokens (ids as int32): keys[j] stands for its first (j + 1)
    grid tokens (each key covers every block before it, as vLLM's block hashes)."""
    ids = [int(t) for t in tokens]
    out, prev = [], hashlib.sha256(salt.encode()).digest()
    for j in range(len(ids) // grid):
        h = hashlib.sha256(prev)
        h.update(array("i", ids[j * grid : (j + 1) * grid]).tobytes())
        prev = h.digest()
        out.append(prev.hex())
    return out
