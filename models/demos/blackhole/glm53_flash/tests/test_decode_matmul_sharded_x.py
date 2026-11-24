"""decode_matmul with its x width-sharded in L1 (GLM53_DFLASH2_NORM_SHARD: the drafter's sublayer reads its normed
input from the rms_norm's sharded output, no S2I) against the same x DRAM interleaved (the earlier form), bit for bit
on all 32 chips: the drafter's kernel projection with side 0's convolution inside (`conv_side` .. `conv_out`, the
model's form), delta and the conv's output, at M = 8 / 32 / 64 rows (8 / 32 / 128 users); x sharded as TtRmsNorm's
sharded form (16 cores, width-sharded, `sharded_configs`).  Then GLM53_DMSX_REPEAT more sharded calls each against
the first, and both forms traced and timed.

    GLM53_DMSX_M       the row counts (default 8,32,64)
    GLM53_DMSX_TIME=0  no timing (e.g. under the watcher)
    GLM53_DMSX_REPEAT  the sharded calls compared with the first (default 20)

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_decode_matmul_sharded_x.py -s

Prints "DMSX M ..: delta the same / conv the same .." lines and "DMSX ALL SAME" when every comparison held.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.tests.test_decode_matmul_conv import BLOCK, K, N, same_bits, timed  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import dflash2_ops as ops  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, mapper  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.head import TtRmsNorm  # noqa: E402


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_decode_matmul_sharded_x(mesh_device):
    mesh = mesh_device
    op = ttnn.experimental.deepseek_prefill.decode_matmul
    rep = mapper(mesh, (None, None))
    timing = os.environ.get("GLM53_DMSX_TIME", "1") != "0"
    repeat = int(os.environ.get("GLM53_DMSX_REPEAT", "20"))
    g = torch.Generator().manual_seed(1829)
    kp = ttnn.from_torch(
        torch.randn(1, 1, K, N, generator=g) * (1.0 / K**0.5),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=rep,
    )
    base = ops.conv_base(mesh, torch.randn(2, K, generator=g) * 0.5)
    norm = TtRmsNorm(mesh, torch.ones(K), 1e-6)  # (its sharded form's configs only)
    ok = True
    for m in [int(v) for v in os.environ.get("GLM53_DMSX_M", "8,32,64").split(",")]:
        e, s = ops.conv_constants(mesh, BLOCK[m])
        memory, _ = norm.sharded_configs(m)
        h = ttnn.from_torch(
            torch.randn(1, 1, m, K, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=rep,
        )  # fmt: skip
        hs = ttnn.to_memory_config(h, memory)
        spec = ttnn.TensorSpec(
            ttnn.Shape([1, 1, m, K]), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM
        )

        def run(x):
            hc = ttnn.allocate_tensor_on_device(spec, mesh)
            delta = op(x, kp, conv_side=0, conv_base=base, conv_e=e, conv_s=s, conv_out=hc)
            return delta, hc

        d0, c0 = run(h)
        d1, c1 = run(hs)
        d_same, c_same = same_bits(d0, d1), same_bits(c0, c1)
        print(
            f"DMSX M {m}: x sharded {hs.memory_config().memory_layout}, delta the {'same' if d_same else 'DIFFER'}, "
            f"conv the {'same' if c_same else 'DIFFER'} on {len(ttnn.get_device_tensors(c1))} chips",
            flush=True,
        )
        ok = ok and d_same and c_same
        again = 0
        for _ in range(repeat):
            d2, c2 = run(hs)
            again += int(same_bits(d2, d1) and same_bits(c2, c1))
            ttnn.deallocate(d2)
            ttnn.deallocate(c2)
        if repeat:
            print(f"DMSX M {m}: {again} of {repeat} more sharded calls the same", flush=True)
        ok = ok and again == repeat
        for t in (d0, c0, d1, c1):
            ttnn.deallocate(t)
        if timing:
            t_dram, t_l1 = timed(mesh, lambda: run(h)), timed(mesh, lambda: run(hs))
            t_dram2, t_l12 = timed(mesh, lambda: run(h)), timed(mesh, lambda: run(hs))
            print(
                f"DMSX TIME M {m}: x in DRAM {t_dram * 1e6:.1f} / {t_dram2 * 1e6:.1f} us, x sharded in L1 "
                f"{t_l1 * 1e6:.1f} / {t_l12 * 1e6:.1f} us",
                flush=True,
            )
        for t in (h, hs, e, s):
            ttnn.deallocate(t)
    # (printed, not asserted: a probe's result is its printed lines)
    print("DMSX ALL SAME" if ok else "DMSX NOT ALL SAME", flush=True)
