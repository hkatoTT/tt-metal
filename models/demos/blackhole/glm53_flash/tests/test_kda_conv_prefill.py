"""The KDA prefill's convolution (`qkv_causal_conv1d_silu`, as `TtKda.prefill` calls it: proj [1, T, 6432] bf16, the
history [1, 3, 6144], four taps, q / k / v of 2,048 channels a chip, channel blocks of 768) on random inputs, and the
padded tokens' mask (`real`) on k, v and the gate.

    GLM53_CONV_T          the chunk's rows (default 512,128)
    GLM53_CONV_VALID      the real tokens of each form, comma-separated (default "-37": t - 37; a negative value counts
                          from t; 0: a row without a request, every token masked; t: none masked), each a pass of
                          the bits and times below (the masked rows' writes)
    GLM53_CONV_BLOCKS     channel_chunk_size values (default 768: the model's), each a form: the same values whatever
                          the blocks (a channel's taps and rows only), so each is compared bit for bit with the first
    GLM53_KDA_ZONES=1     two eager calls a chunk length for the kernels' zones (the profiler on), nothing else
                          (GLM53_KDA_ZONES_REAL=1: with the mask folded in; GLM53_KDA_ZONES_BLOCK: the
                          channel_chunk_size, default 768; the model: kda._conv_block, 64 from 1,024 tokens, else 32)
    otherwise: each form's time from a trace (the median of 3 replays of 20 calls) and, for the forms of the same
    values, whether q / k / v are the same bits as the first form's.  KCONV lines, no assert.

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_kda_conv_prefill.py -s
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

from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, put  # noqa: E402

WIDTH, PROJ = 2048, 6432
REPS = 20


def traced(mesh, fn) -> float:
    for t in fn():
        ttnn.deallocate(t)
    ttnn.synchronize_device(mesh)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        for _ in range(REPS):
            for t in fn():
                ttnn.deallocate(t)
    finally:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        ts.append((time.perf_counter() - t0) / REPS)
    ttnn.release_trace(mesh, trace)
    return sorted(ts)[1]


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_kda_conv_prefill(mesh_device):
    """Runs `_conv` (the module's docstring), an exception printed and the test ending normally."""
    try:
        _conv(mesh_device)
    except Exception as err:  # noqa: BLE001
        print(f"KDA_CONV_PREFILL FAILED: {err!r}"[:3000], flush=True)


def _conv(mesh):
    chans = 3 * WIDTH
    zones = os.environ.get("GLM53_KDA_ZONES") == "1"
    for t in [int(x) for x in os.environ.get("GLM53_CONV_T", "512,128").split(",")]:
        g = torch.Generator().manual_seed(700 + t)
        proj = put(mesh, torch.randn(1, t, PROJ, generator=g), dims=(None, None), dtype=ttnn.bfloat16)
        hist = put(mesh, torch.randn(1, 3, chans, generator=g), dims=(None, None), dtype=ttnn.bfloat16,
                   layout=ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
        taps = [put(mesh, torch.randn(1, 1, chans, generator=g) * 0.5, dims=(None, None), dtype=ttnn.bfloat16)
                for _ in range(4)]  # fmt: skip
        zero = ttnn.from_torch(torch.zeros(1, dtype=torch.int64), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
        valids = [int(x) for x in os.environ.get("GLM53_CONV_VALID", "-37").split(",")]
        for valid in [v + t if v < 0 else v for v in valids]:
            _conv_valid(mesh, t, g, proj, hist, taps, zero, valid, zones)
        for x in (proj, hist, *taps, zero):
            ttnn.deallocate(x)


def _conv_valid(mesh, t, g, proj, hist, taps, zero, valid, zones):
    """One mask (the first `valid` of the t tokens real): the bits of the mask in the conv and the gate against the
    multiplies, and the times."""
    print(f"KCONV T={t}: {valid} real tokens", flush=True)
    real = put(mesh, (torch.arange(t) < valid).float().view(1, t, 1), dims=(None, None), dtype=ttnn.bfloat16)

    def conv(block=768, **kw):
        return ttnn.experimental.kda.qkv_causal_conv1d_silu(
            proj,
            hist,
            *taps,
            WIDTH,
            WIDTH,
            WIDTH,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=block),
            actual_start=zero,
            predecessor_carry=hist,
            **kw,
        )

    def masked():  # (the earlier form: the conv, then k and v times the mask)
        q, k, v = conv()
        out = [q, ttnn.multiply(k, real), ttnn.multiply(v, real)]
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        return out

    if zones:
        for _ in range(2):
            # (GLM53_KDA_ZONES_REAL=1: the mask folded in, the model's form with padded tokens; else without, the
            # form a wave without padding takes)
            zb = int(os.environ.get("GLM53_KDA_ZONES_BLOCK", "768"))  # (the model: kda._conv_block)
            for x in conv(zb, real=real) if os.environ.get("GLM53_KDA_ZONES_REAL") == "1" else conv(zb):
                ttnn.deallocate(x)
            ttnn.synchronize_device(mesh)
        print(f"KCONV T={t}: zone runs done", flush=True)
    else:
        blocks = [int(x) for x in os.environ.get("GLM53_CONV_BLOCKS", "768").split(",")]
        forms = {"conv+mask": masked}
        forms.update({f"conv {b}": (lambda b=b: conv(block=b)) for b in blocks})
        # the mask folded into the conv's writer (the model's), against the conv and the two multiplies (the
        # earlier form)
        want = [ttnn.to_torch(ttnn.get_device_tensors(x)[0]) for x in masked()]
        for b in blocks:
            outs = conv(block=b, real=real)
            got = [ttnn.to_torch(ttnn.get_device_tensors(x)[0]) for x in outs]
            for x in outs:
                ttnn.deallocate(x)
            same = [bool(torch.equal(a, c)) for a, c in zip(want, got)]
            print(f"KCONV T={t} block {b}: the mask in the conv against the conv and multiplies, q / k / v the same "
                  f"bits {same}", flush=True)  # fmt: skip
        forms.update({f"conv {b} real": (lambda b=b: conv(block=b, real=real)) for b in blocks})
        # the gate op with the mask, against the gate op and a multiply
        from models.demos.blackhole.glm53_flash.ttnn.kda_prefill_ops import gate, gate_constant

        fg = put(mesh, torch.randn(1, t, 2 * WIDTH, generator=g) * 2, dims=(None, None), dtype=ttnn.bfloat16)
        bias = gate_constant(mesh, torch.randn(4 * WIDTH, generator=g), WIDTH)
        rate = gate_constant(mesh, torch.rand(4 * WIDTH, generator=g) + 0.5, WIDTH)
        g0 = gate(mesh, fg, bias, rate, WIDTH)
        g_want = ttnn.multiply(g0, real)
        g_got = gate(mesh, fg, bias, rate, WIDTH, real=real)
        same = torch.equal(
            ttnn.to_torch(ttnn.get_device_tensors(g_want)[0]), ttnn.to_torch(ttnn.get_device_tensors(g_got)[0])
        )
        print(f"KCONV T={t}: the gate with the mask against the gate and a multiply, the same bits {same}", flush=True)
        for x in (g0, g_want, g_got):
            ttnn.deallocate(x)

        def gate_mul():
            a = gate(mesh, fg, bias, rate, WIDTH)
            out = ttnn.multiply(a, real)
            ttnn.deallocate(a)
            return [out]

        forms["gate+mul"] = gate_mul
        forms["gate real"] = lambda: [gate(mesh, fg, bias, rate, WIDTH, real=real)]
        times, first = {}, None
        for name, fn in forms.items():
            try:
                if name.startswith("conv ") and not name.endswith("real"):
                    outs = fn()
                    got = [ttnn.to_torch(ttnn.get_device_tensors(x)[0]) for x in outs]
                    for x in outs:
                        ttnn.deallocate(x)
                    if first is None:
                        first = got
                    else:
                        same = all(torch.equal(a, b) for a, b in zip(first, got))
                        print(f"KCONV T={t} {name}: q / k / v the same bits as the first block size's: {same}",
                              flush=True)  # fmt: skip
                times[name] = traced(mesh, fn)
            except Exception as err:  # noqa: BLE001
                print(f"KCONV T={t} {name}: {type(err).__name__}: {str(err).splitlines()[0][:150]}", flush=True)
        print(f"KCONV T={t}: " + ", ".join(f"{k} {v * 1e6:.1f} us" for k, v in times.items()), flush=True)
        for x in (fg, bias, rate):
            ttnn.deallocate(x)
    ttnn.deallocate(real)
