"""DSA's prefill head moves (GLM53_DSA_PREFILL_MOVES: whole-tile copies) against the ttnn ops they replace,
bit for bit, and their times (traced): `dsa_latent.q_heads` (q [1, 1, T, 16 x 256] -> [1, 16, T, 256], whole or a row
range) against nlp_create_q_heads_split + concat (and a slice of its heads), `dsa_latent.join_heads` (one or two
[1, 16, T_i, 256] -> [1, 1, sum T_i, 4,096]) against concat + nlp_concat_heads.  Random bf16.  Print only (KDMV lines).

    GLM53_DMV_T     tokens (default "128,512,1024")
    GLM53_DMV_REPS  calls a trace (default 20)

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_dsa_moves.py -s
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

from models.demos.blackhole.glm53_flash.tests._trace import is_device_timeout, traced  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.dsa_latent import join_heads, q_heads  # noqa: E402

TS = [int(x) for x in os.environ.get("GLM53_DMV_T", "128,512,1024").split(",")]
REPS = int(os.environ.get("GLM53_DMV_REPS", "20"))
H, D, WAVE = 16, 256, 512


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_moves(mesh_device):
    try:
        _probe(mesh_device)
    except Exception as err:  # noqa: BLE001
        if is_device_timeout(err):
            raise
        print(f"KDMV FAILED: {err!r}"[:3000], flush=True)


def _timed(mesh, fn) -> float:
    outs = fn()
    ttnn.synchronize_device(mesh)
    for o in outs:
        ttnn.deallocate(o)
    with traced(mesh) as trace:
        kept = [fn() for _ in range(REPS)]
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    s = (time.perf_counter() - t0) / REPS
    ttnn.release_trace(mesh, trace)
    for outs in kept:
        for o in outs:
            ttnn.deallocate(o)
    return s


def _bits(a, b) -> int:
    """Differing elements over every chip (-1: the shapes differ)."""
    n = 0
    for x, y in zip(ttnn.get_device_tensors(a), ttnn.get_device_tensors(b)):
        tx, ty = ttnn.to_torch(x), ttnn.to_torch(y)
        if tx.shape != ty.shape:
            return -1
        n += int((tx.view(torch.int16) != ty.view(torch.int16)).sum())
    return n


def _probe(mesh):
    g = torch.Generator().manual_seed(17)
    rep = dict(
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)
    )
    chips = mesh.get_num_devices()
    for t in TS:
        q = ttnn.from_torch(torch.randn(chips, 1, t, H * D, generator=g), **rep)  # (a different q on each chip)

        def old_heads(q=q):
            lo, hi = ttnn.experimental.nlp_create_q_heads_split(q, num_heads=H, split_head_dim=D // 2)
            heads = ttnn.concat([lo, hi], dim=-1)
            ttnn.deallocate(lo)
            ttnn.deallocate(hi)
            return [heads]

        def new_heads(q=q):
            return [q_heads(mesh, q, H, cores=120)]

        a, b = old_heads()[0], new_heads()[0]
        diff = _bits(a, b)
        print(f"KDMV T {t} q heads: differing {diff}; old {_timed(mesh, old_heads) * 1e6:.1f} us, new "
              f"{_timed(mesh, new_heads) * 1e6:.1f} us", flush=True)  # fmt: skip
        if t > WAVE:
            for r0, r1 in ((0, WAVE), (WAVE, t)):
                want = ttnn.slice(a, [0, 0, r0, 0], [1, H, r1, D])
                got = q_heads(mesh, q, H, rows=(r0, r1), cores=120)
                print(f"KDMV T {t} q heads rows {r0}..{r1} against a slice of the heads: differing {_bits(want, got)}",
                      flush=True)  # fmt: skip
                ttnn.deallocate(want)
                ttnn.deallocate(got)

            def old_halves(q=q):
                heads = old_heads(q)[0]
                qa = ttnn.slice(heads, [0, 0, 0, 0], [1, H, WAVE, D])
                qb = ttnn.slice(heads, [0, 0, WAVE, 0], [1, H, t, D])
                ttnn.deallocate(heads)
                return [qa, qb]

            def new_halves(q=q):
                return [q_heads(mesh, q, H, rows=(0, WAVE), cores=120), q_heads(mesh, q, H, rows=(WAVE, t), cores=120)]

            print(f"KDMV T {t} the first wave's q halves: old {_timed(mesh, old_halves) * 1e6:.1f} us, new "
                  f"{_timed(mesh, new_halves) * 1e6:.1f} us", flush=True)  # fmt: skip
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        # the join: one part (T rows), and (past WAVE) two parts (WAVE and T - WAVE rows)
        forms = [[t]] + ([[WAVE, t - WAVE]] if t > WAVE else [])
        for sizes in forms:
            parts = [ttnn.from_torch(torch.randn(chips, H, n, D, generator=g), **rep) for n in sizes]

            def old_join(parts=parts):
                cat = ttnn.concat(parts, dim=2) if len(parts) > 1 else parts[0]
                out = ttnn.experimental.nlp_concat_heads(cat)
                if len(parts) > 1:
                    ttnn.deallocate(cat)
                return [out]

            def new_join(parts=parts):
                return [join_heads(mesh, parts, cores=120)]

            a, b = old_join()[0], new_join()[0]
            print(f"KDMV T {t} join {sizes}: differing {_bits(a, b)}; old {_timed(mesh, old_join) * 1e6:.1f} us, new "
                  f"{_timed(mesh, new_join) * 1e6:.1f} us", flush=True)  # fmt: skip
            for x in [a, b] + parts:
                ttnn.deallocate(x)
        ttnn.deallocate(q)
    print("KDMV done", flush=True)
