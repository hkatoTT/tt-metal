# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.from_torch's block-float packing in one pass (tt-metal's bfp_pack_fast.hpp, TT_BFP_PACK_FAST default 1) against
the general form (TT_BFP_PACK_FAST=0, read at each call): the same host tensor, spec and bytes, for the shapes and
values the weight load converts and for the ones that take the general way (padded shapes, other tiles), then the time
of a MoE layer's 96 shards packed by 16 threads as `common.upload_shards`.  Host only, but ttnn's conversion opens the
device context: run it as a device test.
Run: GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_bfp_pack_fast.py -s
"""

from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")


def set_fast(on: bool) -> None:
    os.environ["TT_BFP_PACK_FAST"] = "1" if on else "0"


def described(t) -> tuple:
    """The host tensor's spec and its bytes' digest."""
    shard = t.host_buffer().get_shard(ttnn.MeshCoordinate(0, 0))
    raw = torch.from_dlpack(shard).contiguous()
    digest = hashlib.blake2b(raw.view(torch.uint8).numpy(), digest_size=16).hexdigest()
    return (tuple(t.shape), tuple(t.padded_shape), str(t.dtype), str(t.layout), str(t.tile), raw.numel(), digest)


def cases():
    g = torch.Generator().manual_seed(5)

    def normal(*shape, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=g) * 0.02).to(dtype)

    def bits16(*shape):
        return torch.randint(-32768, 32768, shape, generator=g, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)

    def bits32(*shape):
        return (
            torch.randint(-(2**31), 2**31, shape, generator=g, dtype=torch.int64)
            .to(torch.int32)
            .view(torch.float32)
        )

    zeros_rows = normal(256, 512)
    zeros_rows[::7] = 0
    zeros_rows[3::7] = -0.0
    yield "moe-gate-shard", normal(1, 1, 4096, 18432)
    yield "moe-down-shard", normal(1, 1, 18432, 4096)
    yield "bf16-random-bits", bits16(256, 1024)
    yield "bf16-zero-rows", zeros_rows
    yield "bf16-batched-3d", normal(3, 64, 96)
    yield "bf16-4d", normal(2, 3, 32, 64)
    yield "f32-normal", normal(512, 2048, dtype=torch.float32)
    yield "f32-random-bits", bits32(128, 512)
    yield "bf16-padded (general way, tilized fast pack)", normal(1, 1, 100, 200)
    yield "bf16-padded-3d (general way, tilized fast pack)", normal(5, 33, 70)
    yield "f32-padded (general way, tilized fast pack)", bits32(1, 40, 72)


@pytest.mark.timeout(600)
def test_bfp_pack_fast_bytes():
    differ = 0
    n = 0
    for name, x in cases():
        for dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
            for tile in (None, ttnn.Tile([16, 32])) if "padded" not in name and x.dim() == 2 else (None,):
                kw = {} if tile is None else {"tile": tile}
                got = []
                secs = []
                for fast in (False, True):
                    set_fast(fast)
                    t0 = time.perf_counter()
                    t = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, **kw)
                    secs.append(time.perf_counter() - t0)
                    got.append(described(t))
                n += 1
                same = got[0] == got[1]
                differ += not same
                print(f"BFP {name} {list(x.shape)} {x.dtype} -> {dtype}{'' if tile is None else ' tile 16x32'}: "
                      f"{'same' if same else 'DIFFERENT'} (general {secs[0]:.3f} s, fast {secs[1]:.3f} s)"
                      f"{'' if same else f' {got[0]} / {got[1]}'}", flush=True)  # fmt: skip
    set_fast(True)
    print(f"BFP RESULT: {n - differ} of {n} conversions the same spec and bytes", flush=True)
    assert differ == 0


@pytest.mark.timeout(600)
def test_bfp_pack_fast_layer_time():
    """A MoE layer's 96 [4096, 18432] bf16 shards to BFP8 by 16 threads (8 distinct inputs), both forms."""
    g = torch.Generator().manual_seed(6)
    inputs = [(torch.randn(1, 1, 4096, 18432, generator=g) * 0.02).to(torch.bfloat16) for _ in range(8)]

    def one(i):
        return ttnn.from_torch(inputs[i % 8], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    for fast in (False, True, False, True):
        set_fast(fast)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(16) as pool:
            out = list(pool.map(one, range(96)))
        dt = time.perf_counter() - t0
        del out
        print(f"BFP layer time ({'fast' if fast else 'general'}): 96 shards in {dt:.2f} s", flush=True)
    set_fast(True)


@pytest.mark.timeout(600)
def test_pack_transposed_slices():
    """moe.PACK_SLICES (GLM53_MOE_PACK_SLICES: the MoE shards packed straight from the experts' slices): ttnn
    bfp_utils.pack_transposed_slices against from_torch(torch.cat([s.T ..], dim)) for the MoE shards' two forms (gate /
    up: 72 [256, 4096] row blocks, dim 1; down: 72 [4096, 256] column blocks of [4096, 2048] experts, dim 0) and small
    ones, BFP8 and BFP4: the same spec and bytes; and a layer's 96 shards' time."""
    g = torch.Generator().manual_seed(9)

    def experts(n, rows, cols):
        return [(torch.randn(rows, cols, generator=g) * 0.02).to(torch.bfloat16) for _ in range(n)]

    gate = experts(72, 2048, 4096)  # an expert's gate [intermediate, hidden]
    down = experts(72, 4096, 2048)  # an expert's down [hidden, intermediate]
    small = experts(5, 96, 160)
    forms = [
        ("gate shard", [w[256:512] for w in gate], 1),
        ("down shard", [w[:, 512:768] for w in down], 0),
        ("small dim 1", [w[32:96, 32:160] for w in small], 1),
        ("small dim 0", [w[:, 64:128] for w in small], 0),
    ]
    differ = 0
    n = 0
    for name, slices, dim in forms:
        for dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
            t0 = time.perf_counter()
            ref = ttnn.from_torch(
                torch.cat([s.T for s in slices], dim=dim)[None, None], dtype=dtype, layout=ttnn.TILE_LAYOUT
            )
            t1 = time.perf_counter()
            got = ttnn._ttnn.bfp_utils.pack_transposed_slices(slices, dim, dtype)
            t2 = time.perf_counter()
            a, b = described(ref), described(got)
            same = a == b
            differ += not same
            n += 1
            print(f"BFP2 {name} {list(slices[0].shape)} x {len(slices)} dim {dim} -> {dtype}: "
                  f"{'same' if same else 'DIFFERENT'} (cat + from_torch {t1 - t0:.3f} s, packed {t2 - t1:.3f} s)"
                  f"{'' if same else f' {a} / {b}'}", flush=True)  # fmt: skip
    print(f"BFP2 RESULT: {n - differ} of {n} the same spec and bytes", flush=True)
    assert differ == 0
    # a layer's 96 shards (gate, up, down x 32 chips) by 16 threads, both ways
    jobs = [([w[256:512] for w in gate], 1)] * 64 + [([w[:, 512:768] for w in down], 0)] * 32
    for way in ("cat + from_torch", "packed"):

        def one(job):
            slices, dim = job
            if way == "packed":
                return ttnn._ttnn.bfp_utils.pack_transposed_slices(slices, dim, ttnn.bfloat8_b)
            return ttnn.from_torch(
                torch.cat([s.T for s in slices], dim=dim)[None, None], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
            )

        t0 = time.perf_counter()
        with ThreadPoolExecutor(16) as pool:
            out = list(pool.map(one, jobs))
        print(f"BFP2 layer time ({way}): 96 shards in {time.perf_counter() - t0:.2f} s", flush=True)
        del out
