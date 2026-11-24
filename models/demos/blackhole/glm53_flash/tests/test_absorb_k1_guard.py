"""The DSA per-head absorb matmuls on the batched reuse config (GLM53_DSA_ABSORB_K1: in0_block_w 1) with its guard
(`dsa.absorb_k1_config` taking the tensor and the output's memory config: the K1 config only where ttnn's default adds
the K tiles one by one onto the L1 sum too): each call as `TtDsa._absorb` (q [1, 16, T, 256] @ W_kb [1, 16, 256, 512],
optionally to L1) and `TtDsa._values` (out [1, 16, T, 512], DRAM or L1, @ W_vb^T [1, 16, 512, 256]) make it, bit for
bit against ttnn's default (the call without ABSORB_K1), and the unguarded K1 config (K1 for any whole number of tile
rows) against the default too, to show where the guard is needed.  Times (traced) of the default and the guarded
call.  Random bf16 inputs, BFP8 weights as the model's.  Print only (KAKG lines).

    GLM53_AKG_REPS  calls a trace (default 20)

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_absorb_k1_guard.py -s
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
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, WEIGHT_MM  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.dsa import ABSORB_K1, absorb_k1_config, absorb_k1_rows  # noqa: E402

REPS = int(os.environ.get("GLM53_AKG_REPS", "20"))
H = 16
# (matmul, tokens, input in L1, output in L1): the serving prefill's chunks (128 / 512 / 1,024, also the SP waves' and
# the long cache's), the decode past 64 tokens a mesh row (96 / 128: the transposed path, DRAM; 96 heads-first: q~ to
# L1, the SDPA output read from L1), and the narrow shapes (values from 2,304 rows, absorb from 4,608)
CASES = [
    ("absorb", 96, False, True),
    ("values", 96, True, False),
    ("absorb", 96, False, False),
    ("values", 96, False, False),
    ("absorb", 128, False, False),
    ("values", 128, False, False),
    ("absorb", 512, False, False),
    ("values", 512, False, False),
    ("absorb", 1024, False, False),
    ("values", 1024, False, False),
    ("values", 2048, False, False),
    ("values", 2304, False, False),
    ("values", 4096, False, False),
    ("absorb", 2048, False, False),
    ("absorb", 4096, False, False),
    ("absorb", 4608, False, False),
    ("absorb", 8192, False, False),
]


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_absorb_k1_guard(mesh_device):
    old = os.environ.get(ABSORB_K1)
    os.environ[ABSORB_K1] = "1"
    try:
        _probe(mesh_device)
    except Exception as err:  # noqa: BLE001
        if is_device_timeout(err):
            raise
        print(f"KAKG FAILED: {err!r}"[:3000], flush=True)
    finally:
        if old is None:
            os.environ.pop(ABSORB_K1, None)
        else:
            os.environ[ABSORB_K1] = old


def _timed(mesh, fn) -> float:
    out = fn()
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(out)
    with traced(mesh) as trace:
        outs = [fn() for _ in range(REPS)]
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    s = (time.perf_counter() - t0) / REPS
    ttnn.release_trace(mesh, trace)
    for o in outs:
        ttnn.deallocate(o)
    return s


def _host(t) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def _unguarded(grid, rows: int, n_tiles: int):
    """The K1 config without the guard (any whole number of tile rows)."""
    m = absorb_k1_rows(rows // 32)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=max(w for w in (4, 2, 1) if n_tiles % w == 0),
        per_core_M=m,
        per_core_N=n_tiles,
    )


def _probe(mesh):
    grid = mesh.compute_with_storage_grid_size()
    g = torch.Generator().manual_seed(29)
    rep = dict(layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    w = {
        "absorb": ttnn.from_torch(torch.randn(1, H, 256, 512, generator=g) * 0.05, dtype=ttnn.bfloat8_b, **rep),
        "values": ttnn.from_torch(torch.randn(1, H, 512, 256, generator=g) * 0.05, dtype=ttnn.bfloat8_b, **rep),
    }
    print(f"KAKG grid {grid.x} x {grid.y}, {len(CASES)} calls", flush=True)
    for name, t, in_l1, out_l1 in CASES:
        k = 256 if name == "absorb" else 512
        n_tiles = w[name].shape[-1] // 32
        mem_in = ttnn.L1_MEMORY_CONFIG if in_l1 else ttnn.DRAM_MEMORY_CONFIG
        mc = ttnn.L1_MEMORY_CONFIG if out_l1 else None
        a = ttnn.from_torch(torch.randn(1, H, t, k, generator=g), dtype=ttnn.bfloat16, memory_config=mem_in, **rep)
        # as TtDsa._absorb / _values call it (the values' call never passes a memory config)
        k1 = absorb_k1_config(mesh, a, n_tiles, mc) if name == "absorb" else absorb_k1_config(mesh, a, n_tiles)
        base = {} if mc is None else {"memory_config": mc}
        arms = {"default": dict(base), "guarded": dict(base, **({} if k1 is None else {"program_config": k1}))}
        arms["unguarded"] = dict(base, program_config=_unguarded(grid, t, n_tiles))
        tag = f"KAKG {name} T {t:5d} in {'L1' if in_l1 else 'DRAM'} out {'L1' if out_l1 else 'DRAM'}"
        got, us, err = {}, {}, {}
        for arm, kw in arms.items():

            def fn(a=a, wt=w[name], kw=kw):
                return ttnn.matmul(a, wt, compute_kernel_config=WEIGHT_MM, **kw)

            try:
                out = fn()
                got[arm] = _host(out)
                ttnn.deallocate(out)
                if arm != "unguarded" or k1 is None:
                    us[arm] = _timed(mesh, fn) * 1e6
            except Exception as e:  # noqa: BLE001
                if is_device_timeout(e):
                    raise
                err[arm] = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"

        def diff(x):
            if x not in got or "default" not in got:
                return f"n/a ({err.get(x, err.get('default'))})"
            return int((got[x].view(torch.int32) != got["default"].view(torch.int32)).sum())

        def time_of(x):
            return f"{us[x]:.1f} us" if x in us else "n/a"

        numel = got["default"].numel() if "default" in got else 0
        print(
            f"{tag}: guard {'K1 m ' + str(k1.per_core_M) if k1 is not None else 'ttnn default'}; guarded "
            f"differing {diff('guarded')}, unguarded K1 differing {diff('unguarded')} of {numel}; default "
            f"{time_of('default')}, guarded {time_of('guarded')}, unguarded {time_of('unguarded')}",
            flush=True,
        )
        ttnn.deallocate(a)
    print("KAKG done", flush=True)
