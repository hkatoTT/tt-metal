# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tensor layout of CFG batching in the Wan2.2 transformer.

With CFG batching (`WanTransformer3DModel.inner_step_cfg`), the cond and the uncond pass of classifier-free guidance
run as one forward. Both passes share the latent and the timestep and differ only in the text, so the per-device rows
of the two passes are stacked into one tensor, the cond rows first: [1, 1, 2n, D]. Row-wise ops (norms, projections,
FFN) then run once on the 2n rows.

Head-split tensors keep that order within each head ([1, H, 2n, E]). Where each head has to attend within its own
pass, attention folds the passes into the heads ([1, 2H, n, E], head 2h + b is pass b of head h).
"""

from __future__ import annotations

import ttnn


def join_passes(cond: ttnn.Tensor, uncond: ttnn.Tensor) -> ttnn.Tensor:
    """Stack the rows of the two passes: [A, B, n, F] each -> [A, B, 2n, F]."""
    return ttnn.concat([cond, uncond], dim=2)


def stack_passes(x: ttnn.Tensor) -> ttnn.Tensor:
    """The same rows for both passes: [A, B, n, F] -> [A, B, 2n, F]."""
    return join_passes(x, x)


def split_passes(x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """The cond and the uncond rows: [A, B, 2n, F] -> [A, B, n, F] each."""
    n = x.shape[2] // 2
    return tuple(ttnn.slice(x, [0, 0, start, 0], [x.shape[0], x.shape[1], start + n, x.shape[3]]) for start in (0, n))


def passes_as_batch(x: ttnn.Tensor) -> ttnn.Tensor:
    """[1, 1, 2n, F] -> [1, 2, n, F] (a view), the batched layout the fused distributed RMSNorm takes, so that RoPE
    applies to each pass."""
    return ttnn.reshape(x, (1, 2, x.shape[2] // 2, x.shape[3]))


def fold_heads(x: ttnn.Tensor) -> ttnn.Tensor:
    """[1, H, 2n, E] -> [1, 2H, n, E] (a view): head 2h + b becomes pass b of head h."""
    return ttnn.reshape(x, (1, 2 * x.shape[1], x.shape[2] // 2, x.shape[3]))


def unfold_heads(x: ttnn.Tensor) -> ttnn.Tensor:
    """[1, 2H, n, E] -> [1, H, 2n, E] (a view), the inverse of `fold_heads`."""
    return ttnn.reshape(x, (1, x.shape[1] // 2, 2 * x.shape[2], x.shape[3]))
