"""Temperature / top-k / top-p sampling on the device after the LM head: the pick by tt-metal's
common sampler's core op (`ttnn.sampling`, models/common/sampling), the candidates gathered for our head's layout.

The common `TTSampling` takes logits split over one mesh axis; `TtLmHead.greedy_mesh` splits the vocabulary over all
32 chips (chip k = 4 r + c holds the ids k w .. k w + w - 1, every mesh row's tokens).  Here each chip takes the top 32
of its w logits (`ttnn.topk`), the (value, id) pairs are gathered over the row's chips and then the mesh rows (32 x 32
= 1,024 candidates a token, chip k's at columns 32 k ..), the chip offsets added to the ids, and `ttnn.sampling` picks
one a token (softmax with the temperature, top-k <= 32, top-p, a draw seeded per token by `ttnn.manual_seed`), as the
common sampler does with its devices' candidates.  The distribution is the vocabulary's cut to its top 32 (the op's
top-k limit; the common sampler's max_top_k).  At most 32 tokens a `ttnn.sampling` call (one core a token): more are
taken 32 at a time.

`SamplingInputs` holds the per-token parameters as device tensors at fixed addresses (written before each step, an
`InputPack`'s fields): k (uint32), p and 1/temperature (bf16), and the seeds (uint32).  A greedy token takes k = 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.demos.blackhole.glm53_flash.ttnn.common import (
    COLS,
    HIFI4_FP32,
    ROW_AXIS,
    ROWS,
    TP_AXIS,
    mapper,
    put,
    write_rows,
)

TOP = 32  # candidates a chip, and the op's top-k limit
CALL = 32  # tokens a ttnn.sampling call (one core a token)


@dataclass
class SamplingInputs:
    """A step's sampling parameters, one a token of every mesh row ([n] each, the same on every chip)."""

    k: object  # uint32 row-major
    p: object  # bf16 row-major
    inv_temp: object  # bf16 row-major: 1 / temperature
    seeds: object  # uint32 row-major
    user_ids: object  # uint32 row-major [CALL]: 0 .. 31, the cores of a call (constant)

    @classmethod
    def alloc(cls, mesh, n: int) -> "SamplingInputs":
        rm = dict(dims=(None, None), layout=ttnn.ROW_MAJOR_LAYOUT)
        return cls(
            k=put(mesh, torch.ones(n, dtype=torch.int32), dtype=ttnn.uint32, **rm),
            p=put(mesh, torch.ones(n), dtype=ttnn.bfloat16, **rm),
            inv_temp=put(mesh, torch.ones(n), dtype=ttnn.bfloat16, **rm),
            seeds=put(mesh, torch.arange(n, dtype=torch.int32), dtype=ttnn.uint32, **rm),
            user_ids=put(mesh, torch.arange(CALL, dtype=torch.int32), dtype=ttnn.uint32, **rm),
        )

    def fields(self) -> list:
        """The tensors written each step (for `InputPack`)."""
        return [self.k, self.p, self.inv_temp, self.seeds]

    @staticmethod
    def host_values(k, p, temperature, seeds) -> dict:
        """Per-token parameters (sequences of n) -> each field's host values: greedy (temperature 0 or k 1) as k 1;
        k 0 (no top-k) as the op's 32."""
        ks, ps, invs = [], [], []
        for kk, pp, t in zip(k, p, temperature):
            greedy = t <= 1e-5 or kk == 1
            ks.append(1 if greedy else (TOP if kk <= 0 else min(int(kk), TOP)))
            ps.append(1.0 if greedy else float(pp))
            invs.append(1.0 if greedy else 1.0 / float(t))
        return {
            "k": torch.tensor(ks, dtype=torch.int32),
            "p": torch.tensor(ps),
            "inv_temp": torch.tensor(invs),
            "seeds": torch.tensor([int(s) & 0x7FFFFFFF for s in seeds], dtype=torch.int32),
        }


class CandidateSampler:
    """The device constants (made once, before any capture): each candidate column's chip offset."""

    def __init__(self, mesh, width: int, valid: list[int]) -> None:
        self.mesh, self.width = mesh, width
        chips = ROWS * COLS
        self._offsets_row = torch.arange(chips, dtype=torch.int32).repeat_interleave(TOP) * width  # [1,024]
        self.offsets = {}  # tokens n -> [1, 1, n, 1,024] int32 (`prepare`)
        # each chip's padded columns (past the vocabulary) as -inf before its top 32
        ids = torch.arange(chips * width).view(ROWS, 1, 1, COLS * width)
        vocab = sum(valid)
        self.pad = put(mesh, torch.where(ids < vocab, 0.0, -1e30), dims=(0, 3), dtype=ttnn.bfloat16)

    def prepare(self, n: int) -> None:
        """The chip offsets for n tokens a step, made before any capture."""
        if n not in self.offsets:
            t = self._offsets_row.view(1, 1, 1, -1).expand(1, 1, n, -1).contiguous()
            self.offsets[n] = put(self.mesh, t, dims=(None, None), dtype=ttnn.int32)

    def candidates(self, part, tiled: bool = False):
        """Each chip's logits [1, 1, n, w] (fp32 or bf16) -> every token's 1,024 candidates: values [1, 1, n, 1,024]
        bf16 tiled and global ids [1, 1, n, 1,024] uint32 row-major (`tiled`: fp32 tiles), the same on every chip."""
        x = part if part.dtype == ttnn.bfloat16 else ttnn.typecast(part, ttnn.bfloat16)
        x = ttnn.add(x, self.pad)
        values, idx = ttnn.topk(x, k=TOP, dim=-1)
        ttnn.deallocate(x)
        gather = lambda t: ttnn.all_gather(  # noqa: E731  (the row's chips, then the mesh rows: chip 4 r + c at 32 (4 r + c))
            ttnn.all_gather(t, dim=3, cluster_axis=TP_AXIS), dim=3, cluster_axis=ROW_AXIS
        )
        values_all, idx_all = gather(values), gather(idx)
        ttnn.deallocate(values)
        ttnn.deallocate(idx)
        local = ttnn.typecast(idx_all, ttnn.int32)
        ttnn.deallocate(idx_all)
        self.prepare(local.shape[-2])
        if tiled:  # (the global ids as fp32 tiles: exact below 2^24, for `StepSampling`'s row pick)
            ids = ttnn.add(local, self.offsets[local.shape[-2]])
            ttnn.deallocate(local)
            ids_f = ttnn.typecast(ids, ttnn.float32)
            ttnn.deallocate(ids)
            return values_all, ids_f
        ids = ttnn.add(local, self.offsets[local.shape[-2]], dtype=ttnn.uint32)
        ttnn.deallocate(local)
        ids_rm = ttnn.untilize(ids)
        ttnn.deallocate(ids)
        return values_all, ids_rm

    def sample(self, part, inputs: SamplingInputs):
        """Each chip's logits [1, 1, n, w] -> the sampled ids [1, 1, 1, n] uint32 row-major, the same on every chip."""
        values, ids = self.candidates(part)
        n = values.shape[-2]
        outs = []
        for s in range(0, n, CALL):
            e = min(n, s + CALL)
            v = values if (s, e) == (0, n) else ttnn.slice(values, [0, 0, s, 0], [1, 1, e, values.shape[-1]])
            i = ids if (s, e) == (0, n) else ttnn.slice(ids, [0, 0, s, 0], [1, 1, e, ids.shape[-1]])
            par = lambda t: t if (s, e) == (0, n) else ttnn.slice(t, [s], [e])  # noqa: E731
            ttnn.manual_seed(seeds=par(inputs.seeds), user_ids=ttnn.slice(inputs.user_ids, [0], [e - s]))
            outs.append(ttnn.sampling(v, i, k=par(inputs.k), p=par(inputs.p), temp=par(inputs.inv_temp)))
        ttnn.deallocate(values)
        ttnn.deallocate(ids)
        return outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=-1)


def reference_distribution(logits: torch.Tensor, k: int, p: float, temperature: float) -> torch.Tensor:
    """The distribution the device draws from (CPU): the top 32 logits (of the whole vocabulary), softmax with the
    temperature, the top k, then the smallest top-p set (`ttnn.sampling`'s order), renormalized -> [V] probabilities."""
    v, i = torch.topk(logits.float(), TOP)
    if temperature <= 1e-5 or k == 1:
        out = torch.zeros_like(logits, dtype=torch.float32)
        out[i[0]] = 1.0
        return out
    probs = torch.softmax(v / temperature, -1)
    kk = TOP if k <= 0 else min(k, TOP)
    probs[kk:] = 0
    probs = probs / probs.sum()
    cum = torch.cumsum(probs, -1)
    keep = (cum - probs) < p  # (the tokens before the cumulative sum passes p, and the one that passes it)
    probs = torch.where(keep, probs, torch.zeros_like(probs))
    probs = probs / probs.sum()
    out = torch.zeros_like(logits, dtype=torch.float32)
    out[i] = probs
    return out


class StepSampling:
    """DFlash2's verify step with sampling: every position of every user draws a token from the
    target's distribution (`CandidateSampler`'s candidates, `ttnn.sampling`), and the step's acceptance stays as it is
    (a draft is taken while it equals the token drawn one position before it).  The drafts are the drafter's top picks
    (one token, probability 1), so a draft equals the draw with the target's probability of it, and the draw where it
    does not is one from the target's distribution without the draft: speculative sampling's acceptance and its
    residual draw, the output distributed as the target's own draws.  Greedy users (k 1) take the greedy head's ids
    (`greedy` 1), as the greedy step.

    Each mesh row draws its own b = (K + 1) U tokens (virtual user j U + u): the candidates of every row's tokens are
    gathered on every chip (the vocabulary is split over all 32), the row's b taken by a one-hot matmul (`pick`, exact),
    one `ttnn.sampling` call per 32 tokens, and the ids gathered over the rows for the acceptance.  The parameters and
    a base seed per token, a row's on its chips, written by the host when the requests change (`host_rows`); the seeds
    made on the device each step (`seeds`: base + position, so each position its own draw, also when the host does not
    write the step's inputs, the run-ahead)."""

    def __init__(self, mesh, head, b: int) -> None:
        self.mesh, self.b = mesh, b
        n_all = ROWS * b
        self.n_all = n_all
        rm = dict(dims=(0, None), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.inputs = SamplingInputs(
            k=put(mesh, torch.ones(ROWS * b, dtype=torch.int32), dtype=ttnn.uint32, **rm),  # [b] a row
            p=put(mesh, torch.ones(ROWS * b), dtype=ttnn.bfloat16, **rm),
            inv_temp=put(mesh, torch.ones(ROWS * b), dtype=ttnn.bfloat16, **rm),
            seeds=None,
            user_ids=put(mesh, torch.arange(CALL, dtype=torch.int32), dims=(None, None), dtype=ttnn.uint32,
                         layout=ttnn.ROW_MAJOR_LAYOUT),
        )  # fmt: skip
        self.base = put(mesh, torch.zeros(ROWS, 1, 1, b, dtype=torch.int32), dtype=ttnn.int32, **rm)
        self.greedy = put(mesh, torch.ones(1, 1, n_all, 1), dims=(None, None), dtype=ttnn.float32)
        self.drawn = put(mesh, torch.zeros(1, 1, n_all, 1), dims=(None, None), dtype=ttnn.float32)  # 1 - greedy
        pick = torch.zeros(ROWS, 1, b, n_all)  # row r's: its b tokens of the 8 b
        for r in range(ROWS):
            pick[r, 0, :, r * b : (r + 1) * b] = torch.eye(b)
        self.pick = put(mesh, pick, dims=(0, None), dtype=ttnn.bfloat16)  # [1, 1, b, 8 b] a row
        self.pick_f32 = put(mesh, pick, dims=(0, None), dtype=ttnn.float32)
        self.sampler = CandidateSampler(mesh, head.mesh_ids_width, head.mesh_valid)
        self.sampler.prepare(n_all)

    def seeds(self, ctx_pos):
        """The step's positions ([1, b] int32 row-major per chip, this row's: DFlash2Step.ctx_pos) -> this row's seeds
        [b] uint32 row-major."""
        pos = ttnn.to_layout(ttnn.reshape(ctx_pos, [1, 1, 1, self.b]), ttnn.TILE_LAYOUT)
        base = ttnn.to_layout(self.base, ttnn.TILE_LAYOUT)
        s = ttnn.add(pos, base)
        for t in (pos, base):
            ttnn.deallocate(t)
        out = ttnn.reshape(ttnn.to_layout(ttnn.typecast(s, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [self.b])
        ttnn.deallocate(s)
        return out

    def host_rows(self, k, p, temperature, seed, drafts: int) -> dict:
        """Per user [ROWS, U] -> each row's token values (position j, user u at j U + u; row r's at r b): k / p /
        1/temperature [ROWS b], the base seeds [ROWS, 1, 1, b]; and the greedy flags of every row's tokens
        [1, 1, 8 b, 1]."""
        rows, u = k.shape
        rep = lambda t: t[:, None, :].expand(rows, drafts + 1, u).reshape(-1)  # noqa: E731
        vals = SamplingInputs.host_values(
            rep(k).tolist(), rep(p).tolist(), rep(temperature).tolist(), rep(seed).tolist()
        )
        out = {name: vals[name] for name in ("k", "p", "inv_temp")}  # [ROWS b], row r's b at r b
        out["base"] = vals["seeds"].view(rows, 1, 1, -1)
        out["greedy"] = (vals["k"] == 1).float().view(1, 1, -1, 1)
        return out

    def write(self, values: dict) -> None:
        """`host_rows`' values -> the device tensors (between steps, outside any trace)."""
        for name, t, dtype in (("k", self.inputs.k, ttnn.uint32), ("base", self.base, ttnn.int32)):
            write_rows(self.mesh, values[name], t, dtype)
        rows = mapper(self.mesh, (0, None))  # (p and 1/temperature as floats: write_rows takes integers)
        for name, t in (("p", self.inputs.p), ("inv_temp", self.inputs.inv_temp)):
            host = ttnn.from_torch(
                values[name].float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rows
            )
            ttnn.copy_host_to_device_tensor(host, t)
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        for t, v in ((self.greedy, values["greedy"]), (self.drawn, 1.0 - values["greedy"])):
            host = ttnn.from_torch(v, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
            ttnn.copy_host_to_device_tensor(host, t)

    def _pick_ids(self, ids):
        """Every row's candidate ids [1, 1, 8 b, 1,024] fp32 -> this row's [1, 1, b, 1,024] fp32, exact.  A matmul's
        inputs are tf32 in the FPU (a 10-bit mantissa: the ids, up to 154,880, would round), so the ids go as
        id // 1,024 and id % 1,024 (each below 2,048: exact), picked apart and joined by elementwise fp32 ops (exact).
        """
        hi = ttnn.floor(ttnn.multiply(ids, 1.0 / 1024))
        lo = ttnn.subtract(ids, ttnn.multiply(hi, 1024.0))
        mm = dict(dtype=ttnn.float32, compute_kernel_config=HIFI4_FP32)
        own_hi = ttnn.matmul(self.pick_f32, hi, **mm)
        own_lo = ttnn.matmul(self.pick_f32, lo, **mm)
        for t in (hi, lo):
            ttnn.deallocate(t)
        scaled = ttnn.multiply(own_hi, 1024.0)
        out = ttnn.add(scaled, own_lo)
        for t in (own_hi, own_lo, scaled):
            ttnn.deallocate(t)
        return out

    def draw(self, part, greedy_ids, seeds):
        """Each chip's logits [1, 1, 8 b, w] and the greedy head's ids [1, 1, 8 b, 1] fp32 -> the step's ids
        [1, 1, 8 b, 1] fp32 (greedy rows the greedy ids, the others a draw), the same on every chip."""
        values, ids = self.sampler.candidates(part, tiled=True)  # [1, 1, 8 b, 1,024] bf16, fp32
        own_v = ttnn.matmul(self.pick, values)  # [1, 1, b, 1,024] (one-hot: exact)
        own_i = self._pick_ids(ids)
        for t in (values, ids):
            ttnn.deallocate(t)
        ids_rm = ttnn.untilize(ttnn.typecast(own_i, ttnn.uint32))
        ttnn.deallocate(own_i)
        b, outs = self.b, []
        for s0 in range(0, b, CALL):
            e = min(b, s0 + CALL)
            whole = (s0, e) == (0, b)
            v = own_v if whole else ttnn.slice(own_v, [0, 0, s0, 0], [1, 1, e, own_v.shape[-1]])
            i = ids_rm if whole else ttnn.slice(ids_rm, [0, 0, s0, 0], [1, 1, e, ids_rm.shape[-1]])
            par = lambda t: t if whole else ttnn.slice(t, [s0], [e])  # noqa: E731
            ttnn.manual_seed(seeds=par(seeds), user_ids=ttnn.slice(self.inputs.user_ids, [0], [e - s0]))
            outs.append(ttnn.sampling(v, i, k=par(self.inputs.k), p=par(self.inputs.p), temp=par(self.inputs.inv_temp)))
        for t in (own_v, ids_rm):
            ttnn.deallocate(t)
        mine = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=-1)  # [1, 1, 1, b] uint32 row-major
        every = ttnn.all_gather(mine, dim=3, cluster_axis=ROW_AXIS)  # [1, 1, 1, 8 b]
        ttnn.deallocate(mine)
        col = ttnn.reshape(ttnn.to_layout(every, ttnn.TILE_LAYOUT), [1, 1, self.n_all, 1])
        ttnn.deallocate(every)
        f = ttnn.typecast(col, ttnn.float32)
        ttnn.deallocate(col)
        # greedy g + (1 - g) draw, g 0 / 1 (exact in fp32: the ids are below 2^24)
        keep = ttnn.multiply(greedy_ids, self.greedy)
        take = ttnn.multiply(f, self.drawn)
        ttnn.deallocate(f)
        out = ttnn.add(keep, take)
        for t in (keep, take):
            ttnn.deallocate(t)
        return out
