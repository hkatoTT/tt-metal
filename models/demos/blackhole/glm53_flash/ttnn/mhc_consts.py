"""The constant tiles of `ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn` (torch only, no device)."""

from __future__ import annotations

import torch

W = 32  # one tile: the packed per-token 4 x 4 matrix (16 entries) and the 24 projection outputs fit in it


def split_sinkhorn_consts(n: int, scale: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """The constant tiles of `mhc_split_sinkhorn`, [8, 32, 32] fp32, in the op's order:

    0-2: selection matrices that pick the pre / post / comb outputs out of the projection (left-aligned) and multiply
    them by their scale; 3-5: the matching biases, repeated down every token row; 6, 7: the row-sum and column-sum
    broadcast matrices of the matmul-form Sinkhorn (entry (i, j) of a token's 4 x 4 matrix sits at column 4 i + j;
    the pad columns carry an identity block so no divisor is zero).
    """
    scale, base = scale.float(), base.float()
    sel = torch.zeros(3, W, W)
    for p in range(n):
        sel[0, p, p] = scale[0]
        sel[1, n + p, p] = scale[1]
    for p in range(n * n):
        sel[2, 2 * n + p, p] = scale[2]
    bias = torch.zeros(3, W, W)
    bias[0, :, :n] = base[:n]
    bias[1, :, :n] = base[n : 2 * n]
    bias[2, :, : n * n] = base[2 * n :]
    row_sum = torch.zeros(W, W)
    col_sum = torch.zeros(W, W)
    for p in range(n * n):
        for q in range(n * n):
            if q // n == p // n:
                row_sum[q, p] = 1.0
            if q % n == p % n:
                col_sum[q, p] = 1.0
    for p in range(n * n, W):
        row_sum[p, p] = col_sum[p, p] = 1.0
    return torch.cat([sel, bias, row_sum[None], col_sum[None]])


def emulate_split_sinkhorn(mixes: torch.Tensor, consts: torch.Tensor, n: int, iters: int, eps: float):
    """The op's arithmetic in torch on [T, 32] fp32 mixes, as its constants define it (to check the constants).

    The op adds a bias tile to every 32-token tile; every row of a bias tile is the same, so row 0 stands for all.
    """
    sel, bias, row_sum, col_sum = consts[0:3], consts[3:6], consts[6], consts[7]
    pre = torch.sigmoid(mixes @ sel[0] + bias[0][0]) + eps
    post = 2 * torch.sigmoid(mixes @ sel[1] + bias[1][0])
    e = torch.exp(mixes @ sel[2] + bias[2][0])
    comb = e / (e @ row_sum) + eps
    comb = comb / (comb @ col_sum + eps)
    for _ in range(iters - 1):
        comb = comb / (comb @ row_sum + eps)
        comb = comb / (comb @ col_sum + eps)
    return pre[:, :n], post[:, :n], comb[:, : n * n]


def stream_mix_consts(n: int = 4, mix: int = 24) -> torch.Tensor:
    """The constant tiles of `mhc_stream_mix`, [13, 32, 32] fp32: 0-2 the own-block masks of the projection's column
    tiles (row (t, i) keeps columns mix i .. mix i + mix - 1), 3-5 the block sums (projection column g -> mix column
    g % mix), 6 the identity, 7 all ones, 8 the pre repeat (pre column i -> every column c with c % n == i), 9-12 the
    token sums of row tiles 0-3 (token t <- its n rows, 32 / n tokens per row tile)."""
    r = torch.arange(W)[:, None]
    c = torch.arange(W)[None, :]
    t = torch.zeros(13, W, W)
    for k in range(3):
        g = W * k + c
        t[k] = ((g // mix == r % n) & (g < n * mix)).float()
        g = W * k + r
        t[3 + k] = ((g % mix == c) & (g < n * mix) & (c < mix)).float()
    t[6] = torch.eye(W)
    t[7] = 1.0
    t[8] = ((r < n) & (c % n == r)).float()
    for rt in range(4):
        t[9 + rt] = (rt * (W // n) + c // n == r).float()
    return t
