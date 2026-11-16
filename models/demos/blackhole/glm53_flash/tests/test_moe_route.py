"""`ttnn.experimental.deepseek_prefill.moe_decode_route` against torch on random logits: each chip's top-8 choices and
normalized weights for its 9 experts; and its packed reading against the tile-row one.  No checkpoint.

Run (a device test):
    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_moe_route.py -s
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.ttnn.common import COLS, DEVICE_PARAMS, MESH_SHAPE, ROWS, put  # noqa: E402

LOCAL = 9  # whole experts a chip (the op's whole-expert placement: chip (r, c) holds experts 72 c + 9 r + j)


def expert_id(row: int, col: int, j: int) -> int:
    return 72 * col + LOCAL * row + j


E, K, SCALE = 288, 8, 2.5


def route_reference(logits, bias, base, tokens):
    """[N, E] fp32 -> [N, LOCAL]: the chip's columns of the normalized top-k weights, rounded to bf16."""
    scores = torch.sigmoid(logits)
    ids = torch.topk(scores + bias, K, dim=-1).indices.sort(dim=-1).values
    total = torch.zeros(logits.shape[0])
    for i in range(K):  # in expert order, one fp32 add at a time
        total = total + scores.gather(1, ids[:, i : i + 1])[:, 0]
    inv = 1.0 / (total + 1e-20)
    w = torch.zeros(logits.shape[0], LOCAL)
    for i in range(K):
        e = ids[:, i]
        mine = (e >= base) & (e < base + LOCAL)
        val = ((scores.gather(1, e[:, None])[:, 0] * inv) * SCALE).to(torch.bfloat16).float()
        rows = torch.nonzero(mine)[:, 0]
        w[rows, e[rows] - base] = val[rows]
    w[tokens:] = 0
    return w


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("n, tokens", [(32, 8), (32, 32), (128, 128), (256, 256)])
def test_moe_route_values(mesh_device, n, tokens):
    g = torch.Generator().manual_seed(n + tokens)
    logits = torch.randn(n, E, generator=g) * 2.0
    bias = torch.randn(E, generator=g) * 0.1
    base = torch.zeros(ROWS, COLS, 1, 8, dtype=torch.int32)
    for r in range(ROWS):
        for c in range(COLS):
            base[r, c, 0, 0] = expert_id(r, c, 0)
    rep = dict(dims=(None, None), dtype=ttnn.float32)
    d_logits = put(mesh_device, logits[None, None], **rep)
    d_bias = put(mesh_device, bias.view(1, 1, 1, -1), **rep)
    d_base = put(mesh_device, base, dims=(0, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    w = ttnn.experimental.deepseek_prefill.moe_decode_route(d_logits, d_bias, d_base, tokens, K, LOCAL, SCALE)

    chips = lambda t: ttnn.get_device_tensors(t)  # noqa: E731
    for i in (0, 5, 17, 31):
        chip_base = int(ttnn.to_torch(chips(d_base)[i]).flatten()[0])
        want = route_reference(logits, bias, chip_base, tokens)
        got = ttnn.to_torch(chips(w)[i]).float().reshape(n, 32)
        sets = ((got[:, :LOCAL] > 0) != (want > 0)).sum().item()
        diff = (got[:, :LOCAL] - want).abs()
        ulps = (diff > 0).sum().item()
        print(
            f"n {n} tokens {tokens} chip {i} base {chip_base}: chosen {int((want > 0).sum())}, choice mismatches "
            f"{sets}, weights differing {ulps}, max |d| {diff.max().item():.3e}, rest zero "
            f"{bool((got[:, LOCAL:] == 0).all())}",
            flush=True,
        )
        assert sets == 0
        assert diff.max().item() <= 2**-8 * want.abs().max().item()
        assert (got[:, LOCAL:] == 0).all()


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_moe_route_pack(mesh_device):
    """The packed reading (the default up to 8 rows a core: each core's rows only, three a tile) against the tile-row
    one (MOE_ROUTE_PACK=0), bit for bit on every chip, with the column-sliced ranges (local 72, the shared expert on
    column 1), every chosen id (local 0) and the chip's 9; and both forms' times (trace).  The program cache is cleared
    between the forms (the switch is read when the program is made)."""
    import time

    def traced(fn, reps=10):
        ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh_device)
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        outs = [fn() for _ in range(reps)]
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        seconds = (time.perf_counter() - t0) / reps
        ttnn.release_trace(mesh_device, trace)
        for o in outs:
            ttnn.deallocate(o)
        return seconds

    rep = dict(dims=(None, None), dtype=ttnn.float32)
    layouts = {
        "column": (72, True, lambda r, c: (72 * c, int(c == 1))),
        "every id": (0, False, lambda r, c: (0, 0)),
        "chip's 9": (LOCAL, False, lambda r, c: (expert_id(r, c, 0), 0)),
    }
    for name, (local, shared, first) in layouts.items():
        base = torch.zeros(ROWS, COLS, 1, 8, dtype=torch.int32)
        for r in range(ROWS):
            for c in range(COLS):
                base[r, c, 0, 0], base[r, c, 0, 1] = first(r, c)
        d_base = put(mesh_device, base, dims=(0, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        for n, tokens in [(32, 1), (32, 8), (32, 32), (128, 128), (512, 512), (1024, 1024)]:
            g = torch.Generator().manual_seed(7 * n + tokens)
            logits = torch.randn(n, E, generator=g) * 2.0
            bias = torch.randn(E, generator=g) * 0.1
            d_logits = put(mesh_device, logits[None, None], **rep)
            d_bias = put(mesh_device, bias.view(1, 1, 1, -1), **rep)
            run = lambda: ttnn.experimental.deepseek_prefill.moe_decode_route(  # noqa: E731
                d_logits, d_bias, d_base, tokens, K, local, SCALE, shared=shared
            )
            got, times = {}, {}
            for form in ("0", "1"):
                os.environ["MOE_ROUTE_PACK"] = form
                mesh_device.clear_program_cache()
                w = run()
                got[form] = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(w)]
                ttnn.deallocate(w)
                times[form] = traced(run) * 1e6
            os.environ.pop("MOE_ROUTE_PACK")
            mesh_device.clear_program_cache()
            same = all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(got["0"], got["1"]))
            print(
                f"{name}, n {n} tokens {tokens}: bit-identical {same}; tile rows {times['0']:.1f} us, packed "
                f"{times['1']:.1f} us",
                flush=True,
            )
            assert same
            ttnn.deallocate(d_logits)
            ttnn.deallocate(d_bias)
        ttnn.deallocate(d_base)
