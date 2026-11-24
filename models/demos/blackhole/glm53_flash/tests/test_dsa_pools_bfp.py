# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The DSA indexer's pools' keys stored in BFP8_b / BFP4_b (dsa.POOLS_DTYPE): the three ops that touch the pools
cache, each against the bf16 form and torch's rounding (tools/kv_round.bfp_rna: each face row of 16 values on its
largest exponent, to nearest, ties away from zero), on the 8x4 mesh (replicated inputs, chips 0 and 31 read):

  bfp_pack (prefill's writes, kernels/dsa/bfp_pack.cpp): bf16 tiles -> BFP tiles equal to bfp_rna of the input, value
      for value (inputs: normal, exponents spread within each group, exact ties of the BFP steps, all-zero groups,
      negatives; the keys' shapes of a chunk of 1,024 and of a sequence-parallel wave of 8 x 1,024 tokens); BFP8 also
      against the packer's form (dsa_indexer.bfp_cast: its BFP8 rounds so too).
  index_update (decode's writes, the split form's owner rounding the kept keys' rows into its BFP pages): step after
      step on the same random states as the bf16 form, the BFP pools equal to bfp_rna of the bf16 form's pools (every
      row: the written ones rounded, the others as they were), the recent pages bit for bit; T segments a user 1 / 4 /
      8, users 1 / 4 / 16 a row, the verify and random position patterns of test_dsa_indexer_ops.py.
  pool_scores (the reads: the unpacker takes CB_K in the cache's format): a BFP cache against a bf16 cache of the same
      (rounded) values, bit for bit: decode one query a user (1 / 16 / 64 users), decode's groups (4 users x 8 tokens
      over 32K pools), and a prefill chunk's shared form; and the times.

Prints PBFP lines and a PBFP RESULT; fails on a difference.
    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_dsa_pools_bfp.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.tests._trace import traced as trace_capture  # noqa: E402
from models.demos.blackhole.glm53_flash.tools.kv_round import bfp_rna  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, put  # noqa: E402

HEADS, DIM, KPOOL = 32, 128, 4
REP = dict(dims=(None, None))
FORMS = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


def chips(t, which=(0, 31)) -> list[torch.Tensor]:
    dev = ttnn.get_device_tensors(t)
    return [ttnn.to_torch(dev[i]).float() for i in which]


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal values (+0.0 / -0.0 alike; -inf alike)."""
    return a.shape == b.shape and bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())


def bits16(a: torch.Tensor) -> torch.Tensor:
    return a.to(torch.bfloat16).view(torch.int16)


def timed(mesh, fn, reps: int = 10) -> float:
    ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    with trace_capture(mesh) as trace:
        outs = [fn() for _ in range(reps)]
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    sec = (time.perf_counter() - t0) / reps
    ttnn.release_trace(mesh, trace)
    for o in outs:
        ttnn.deallocate(o)
    return sec


def _inputs(g, rows: int) -> dict:
    shape = (1, 1, rows, DIM)
    normal = torch.randn(shape, generator=g)
    spread = torch.randn(shape, generator=g) * torch.pow(2.0, torch.randint(-14, 6, shape, generator=g).float())
    # exact ties of the BFP8 / BFP4 steps: a group's largest value 1.9375 2^e, the others halfway between two steps
    ties = torch.zeros(shape)
    flat = ties.view(-1, 16)
    for i in range(flat.shape[0]):
        e = float(i % 9 - 4)
        flat[i, 0] = 1.9375 * 2.0**e
        for j in range(1, 16):
            k = (i + j) % 7
            step8, step4 = 2.0 ** (e - 6), 2.0 ** (e - 2)
            v = (k + 0.5) * (step8 if j % 2 else step4)
            flat[i, j] = v if (i + j) % 3 else -v
    zeros = torch.randn(shape, generator=g)
    zeros.view(-1, 16)[::3] = 0.0  # every third group all zero
    return {"normal": normal, "spread": spread, "ties": ties, "zeros": zeros}


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_pools_bfp(mesh_device):
    from models.demos.blackhole.glm53_flash.ttnn.dsa import PAGE
    from models.demos.blackhole.glm53_flash.ttnn.dsa_indexer import (
        PARAMS,
        bfp_cast,
        bfp_pack,
        index_update,
        pool_scores,
    )

    g = torch.Generator().manual_seed(12)
    bad = []

    # -- bfp_pack ------------------------------------------------------------------------------------------------
    for rows in (256, 2048):  # a chunk of 1,024 tokens' pools, a wave of 8 x 1,024 (sequence parallel)
        for name, x in _inputs(g, rows).items():
            xb = x.to(torch.bfloat16)
            d_x = put(mesh_device, xb, **REP)
            for fmt, dt in FORMS.items():
                want = bfp_rna(xb.float(), fmt)
                out = bfp_pack(mesh_device, d_x, dt)
                got = chips(out)
                ok = all(same(c, want) for c in got)
                share = float((got[0] == want).float().mean())
                line = f"PBFP pack {fmt} {name} [{rows}, 128]: = bfp_rna {ok} ({share:.6f} of the values)"
                if fmt == "bfp8":
                    cast = bfp_cast(mesh_device, d_x, dt)
                    line += f", = bfp_cast (the packer's) {same(chips(cast)[0], got[0])}"
                    ttnn.deallocate(cast)
                if not ok:
                    d = (got[0] - want).abs()
                    i = int(d.argmax())
                    line += f"; max |d| {float(d.max()):.3g} at {i}: x {float(xb.view(-1)[i]):.6g} got "
                    line += f"{float(got[0].view(-1)[i]):.6g} want {float(want.view(-1)[i]):.6g}"
                    bad.append(f"pack {fmt} {name} {rows}")
                print(line, flush=True)
                ttnn.deallocate(out)
            if rows == 2048 and name == "normal":
                for fmt, dt in FORMS.items():
                    sec = timed(mesh_device, lambda dt=dt: bfp_pack(mesh_device, d_x, dt))
                    print(f"PBFP pack {fmt} [{rows}, 128] ({rows * DIM // 1024} tiles): {sec * 1e6:.1f} us", flush=True)
            ttnn.deallocate(d_x)

    # -- index_update --------------------------------------------------------------------------------------------
    from models.demos.blackhole.glm53_flash.tests.test_dsa_indexer_ops import _split_positions

    pp, max_seq = 8, 8 * 32 * KPOOL  # pool pages a user (1,024 positions)
    steps = int(os.environ.get("GLM53_PBFP_STEPS", "6"))
    bias = torch.full((1, 1, PAGE, DIM), float("-inf"))
    bias[0, 0, :KPOOL] = torch.randn(KPOOL, DIM, generator=g) * 0.5
    d_bias = put(mesh_device, bias, dtype=ttnn.float32, **REP)
    for segs in (1, 4, 8):
        for users in (1, 4, 16):
            for pattern in ("verify", "random"):
                shapes = ([users + 1, 1, PAGE, DIM], [users + 1, 1, PAGE, DIM], [users * pp + 1, 1, PAGE, DIM])
                init = [torch.randn(s, generator=g).to(torch.bfloat16) for s in shapes]
                ref = [put(mesh_device, x, **REP) for x in init]
                states = {
                    fmt: [put(mesh_device, init[0], **REP), put(mesh_device, init[1], **REP)]
                    + [put(mesh_device, bfp_rna(init[2].float(), fmt), dtype=dt, **REP)]
                    for fmt, dt in FORMS.items()
                }
                starts = [(125, 1, 2, 3)[u % 4] + 8 * (u // 4) for u in range(users)]
                pos = list(starts)
                differ = {fmt: [] for fmt in FORMS}
                for step in range(steps):
                    b = users * segs
                    if pattern == "verify":
                        positions = [pos[v % users] + v // users for v in range(b)]
                    else:
                        per_user = []
                        for u in range(users):
                            mode = int(torch.randint(0, 5, (1,), generator=g))
                            base = int(torch.randint(0, max_seq - 16 * segs, (1,), generator=g))
                            per_user.append(_split_positions(g, mode, base, segs, max_seq))
                        positions = [per_user[v % users][v // users] for v in range(b)]
                    owner = [v % users for v in range(b)]
                    p = torch.zeros(1, 1, b, PARAMS, dtype=torch.int64)
                    p[0, 0, :, 0] = torch.tensor(positions)
                    p[0, 0, :, 1] = torch.tensor(owner) * max_seq
                    p[0, 0, :, 2] = torch.tensor(owner) * pp
                    d_p = put(mesh_device, p.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, **REP)
                    ik = put(mesh_device, torch.randn(1, 1, b, DIM, generator=g).to(torch.bfloat16), **REP)
                    ig = put(mesh_device, torch.randn(1, 1, b, DIM, generator=g).to(torch.bfloat16), **REP)
                    index_update(mesh_device, ik, ig, d_p, *ref, d_bias, users, segs)
                    want = [chips(x) for x in ref]
                    for fmt in FORMS:
                        index_update(mesh_device, ik, ig, d_p, *states[fmt], d_bias, users, segs)
                        got = [chips(x) for x in states[fmt]]
                        rows = [
                            sum(int((bits16(w) != bits16(h)).any(-1).sum()) for w, h in zip(want[i], got[i]))
                            for i in (0, 1)
                        ]
                        rounded = [bfp_rna(w, fmt) for w in want[2]]
                        rows.append(sum(int((r != h).any(-1).sum()) for r, h in zip(rounded, got[2])))
                        if any(rows):
                            differ[fmt].append((step, rows))
                    for t in (ik, ig, d_p):
                        ttnn.deallocate(t)
                    if pattern == "verify":
                        acc = [int(x) for x in torch.randint(0, segs, (users,), generator=g)]
                        if step % 3 == 2:
                            acc = [segs - 1] * users
                        pos = [pos[u] + acc[u] + 1 for u in range(users)]
                        if max(pos) + segs >= max_seq:
                            pos = list(starts)
                for fmt in FORMS:
                    if differ[fmt]:
                        bad.append(f"index_update {fmt} T={segs} users={users} {pattern}")
                    print(
                        f"PBFP index_update {fmt} T={segs} users={users} {pattern}: {steps} steps, rows differing "
                        f"(rkey, rgate, pools against bfp_rna of the bf16 form's) "
                        f"{differ[fmt][:3] if differ[fmt] else 'none'}",
                        flush=True,
                    )
                for t in ref + [x for fmt in FORMS for x in states[fmt]]:
                    ttnn.deallocate(t)
    ttnn.deallocate(d_bias)

    # -- pool_scores ---------------------------------------------------------------------------------------------
    def scores_case(label, keys_pages, owners, positions, n_pools, pp_user, shared, group=None, computed=None):
        n = owners.numel()
        q = torch.randn(n, 1, HEADS, DIM, generator=g).to(torch.bfloat16)
        hw = (torch.randn(1, 1, n, HEADS, generator=g) / 64).to(torch.bfloat16)
        params = torch.zeros(1, 1, n, PARAMS, dtype=torch.int64)
        params[0, 0, :, 0] = positions
        params[0, 0, :, 2] = owners * pp_user
        d_hw = put(mesh_device, hw, **REP)
        d_p = put(mesh_device, params.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, **REP)
        if shared:
            d_q = put(mesh_device, q, **REP)
        else:  # (decode: row-major queries [1, 1, N, 32 x 128])
            d_q = put(mesh_device, q.reshape(1, 1, n, HEADS * DIM), layout=ttnn.ROW_MAJOR_LAYOUT, **REP)
        line = f"PBFP scores {label}:"
        for fmt, dt in FORMS.items():
            rounded = bfp_rna(keys_pages.float(), fmt)
            d_ref = put(mesh_device, rounded.to(torch.bfloat16), **REP)
            d_low = put(mesh_device, rounded, dtype=dt, **REP)
            if not same(chips(d_low)[0], rounded):
                bad.append(f"scores {label} {fmt}: the uploaded cache")
            run = lambda c: pool_scores(  # noqa: E731
                mesh_device, d_q, c, d_hw, d_p, n_pools, shared, computed, group=group
            )
            a, b = run(d_ref), run(d_low)
            ok = all(torch.equal(bits16(x), bits16(y)) for x, y in zip(chips(a), chips(b)))
            if not ok:
                bad.append(f"scores {label} {fmt}")
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            t_ref = timed(mesh_device, lambda: run(d_ref))
            t_low = timed(mesh_device, lambda: run(d_low))
            line += f" {fmt} bit for bit {ok} ({t_ref * 1e6:.1f} -> {t_low * 1e6:.1f} us);"
            ttnn.deallocate(d_ref)
            ttnn.deallocate(d_low)
        print(line, flush=True)
        for t in (d_q, d_hw, d_p):
            ttnn.deallocate(t)

    pp = 64  # pages a user: 2,048 pools (a cache of 8,192 tokens)
    keys = (torch.randn(64 * pp + 1, 1, 32, DIM, generator=g) * 0.7).to(torch.bfloat16)
    for users in (1, 16, 64):
        owners = torch.arange(users)
        positions = torch.randint(0, pp * 32 * KPOOL, (users,), generator=g)
        positions[0] = pp * 32 * KPOOL - 1
        scores_case(f"decode {users} users x 2,048 pools", keys, owners, positions, pp * 32, pp, False)
    start, t = 4096, 512
    scores_case(
        "prefill 512 queries at 4,096 (shared)",
        keys,
        torch.full((t,), 5),
        start + torch.arange(t),
        pp * 32,
        pp,
        True,
        computed=(start + t) // KPOOL,
    )
    pp_long, users, tokens = 256, 4, 8  # 8,192 pools a user (a 32K cache), the verify step's groups
    keys_long = (torch.randn(users * pp_long + 1, 1, 32, DIM, generator=g) * 0.7).to(torch.bfloat16)
    owners = torch.arange(users * tokens) % users
    positions = (pp_long * 32 * KPOOL - tokens + torch.arange(users * tokens) // users).clamp(max=pp_long * 128 - 1)
    scores_case(
        f"decode groups {users} users x {tokens} tokens x 8,192 pools",
        keys_long,
        owners,
        positions,
        pp_long * 32,
        pp_long,
        False,
        group=(users, tokens),
    )
    print(f"PBFP RESULT {'all as expected' if not bad else 'DIFFER: ' + '; '.join(bad)}", flush=True)
    assert not bad
