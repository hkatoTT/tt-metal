"""Device time of the decode step's weight-bound matmuls ([32, K] @ [K, N], one op per captured trace, replayed).

Per shape: ttnn's default config on interleaved DRAM weights (what the model runs now), and the DRAM-sharded matmul
(weights split by column over the DRAM banks, the activation width-sharded in L1, one reader per bank region), in
bf16 and with BFP8 weights.  Every chip of the mesh runs the same op; the weights are random.

    GLM53_MM_BENCH_SHAPES = K:N,... (default: MoE gate|up and down, KDA's x projection, o_proj, the LM head quarter)
Run (a device test): GLM53_DEVICE_TEST=1 pytest ... -s
"""

from __future__ import annotations

import math
import os
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    DEVICE_PARAMS_TRACE,
    HIFI4_FP32,
    MESH_SHAPE,
    mapper,
)

REPS = 10
SHAPES = "4096:36864,18432:4096,4096:6432,2048:4096,4096:38720"


def timed(mesh, fn) -> float:
    fn()
    ttnn.synchronize_device(mesh)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = [fn() for _ in range(REPS)]
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    seconds = (time.perf_counter() - t0) / REPS
    ttnn.release_trace(mesh, trace)
    for o in outs:
        ttnn.deallocate(o)
    return seconds


def width_cores(k_tiles: int, grid) -> int:
    """The most cores (up to 64, within the grid) that split K's tiles evenly: the activation's width shards."""
    for n in range(min(64, grid.x * grid.y), 0, -1):
        if k_tiles % n == 0:
            return n
    return 1


def core_grid(n: int, grid) -> ttnn.CoreGrid:
    for y in range(min(grid.y, n), 0, -1):
        if n % y == 0 and n // y <= grid.x:
            return ttnn.CoreGrid(y=y, x=n // y)
    raise ValueError(n)


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_matmul_bench(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    banks = mesh_device.dram_grid_size().x
    print(f"compute grid {grid.x}x{grid.y}, DRAM banks {banks}", flush=True)
    rep = mapper(mesh_device, (None, None))
    for item in os.environ.get("GLM53_MM_BENCH_SHAPES", SHAPES).split(","):
        k, n = (int(v) for v in item.split(":"))
        x_host = torch.randn(1, 1, 32, k)
        w_host = torch.randn(1, 1, k, n) * 0.02
        x = ttnn.from_torch(x_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        res = {}
        for dtype, label in ((ttnn.bfloat16, "bf16"), (ttnn.bfloat8_b, "bfp8")):
            gb = k * n * (2 if dtype == ttnn.bfloat16 else 1.0625) / 1e9
            w = ttnn.from_torch(w_host, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
            sec = timed(mesh_device, lambda: ttnn.matmul(x, w, compute_kernel_config=HIFI4_FP32))
            res[f"{label} interleaved"] = (sec, gb)
            ttnn.deallocate(w)

            # DRAM-sharded: N padded to a multiple of 32 x lcm(banks, worker cores)
            cores = width_cores(k // 32, grid)
            unit = 32 * math.lcm(banks, cores)
            n_pad = math.ceil(n / unit) * unit
            per_core_n = n_pad // (32 * cores)
            gb_pad = gb * n_pad / n
            w_pad = torch.nn.functional.pad(w_host, (0, n_pad - n))
            banks_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
            w_mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(banks_grid, (k, n_pad // banks), ttnn.ShardOrientation.ROW_MAJOR),
            )
            try:
                wd = ttnn.from_torch(
                    w_pad,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    mesh_mapper=rep,
                    memory_config=w_mem,
                )
                in_mem = ttnn.create_sharded_memory_config(
                    (32, k // cores),
                    core_grid=core_grid(cores, grid),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                xs = ttnn.to_memory_config(x, in_mem)
                in0_block_w = max(d for d in range(1, 9) if (k // 32 // cores) % d == 0)
                cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=in0_block_w, per_core_M=1, per_core_N=per_core_n, fused_activation=None
                )

                def sharded():
                    return ttnn.matmul(
                        xs,
                        wd,
                        program_config=cfg,
                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                        compute_kernel_config=HIFI4_FP32,
                    )

                got = ttnn.to_torch(ttnn.get_device_tensors(sharded())[0])[..., :n].float()
                want = (x_host.bfloat16().float() @ w_host.float())[0, 0]
                p = torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1].item()
                res[f"{label} DRAM-sharded ({cores} cores, N {n_pad})"] = (timed(mesh_device, sharded), gb_pad)
                print(f"K {k} N {n} {label}: DRAM-sharded pcc {p:.5f}", flush=True)
                ttnn.deallocate(wd)
                ttnn.deallocate(xs)
            except Exception as e:  # noqa: BLE001  (a config the op rejects is a result too)
                print(f"K {k} N {n} {label}: DRAM-sharded failed: {str(e).splitlines()[0][:240]}", flush=True)
        print(f"[32, {k}] @ [{k}, {n}]:", flush=True)
        for name, (sec, gb) in res.items():
            print(f"  {name:48s} {sec * 1e6:8.1f} us  {gb / sec:7.1f} GB/s of weights read", flush=True)
        ttnn.deallocate(x)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_decode_linear_shapes(mesh_device):
    """The model's decode matmul (`common.decode_linear`, BFP8 weights) on [M, K] @ [K, N], trace-timed: the KDA
    projections as today (a row's t tokens, a quarter of the heads) against one head pair a chip for the mesh's 8 t
    tokens (heads over the 32 chips).

    GLM53_DL_SHAPES = M:K:N,... (default: x_proj 4096 x 6432 vs 4096 x 1088, o_proj 2048 vs 256 x 4096, at 8 / 32 /
    128 users)"""
    from models.demos.blackhole.glm53_flash.ttnn.common import decode_linear

    default = ",".join(f"{t}:4096:6432,{8 * t}:4096:1088,{t}:2048:4096,{8 * t}:256:4096" for t in (1, 4, 16))
    rep = mapper(mesh_device, (None, None))
    for item in os.environ.get("GLM53_DL_SHAPES", default).split(","):
        m, k, n = (int(v) for v in item.split(":"))
        x_host = torch.randn(1, 1, m, k)
        w_host = torch.randn(1, 1, k, n) * 0.02
        x = ttnn.from_torch(x_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        w = ttnn.from_torch(w_host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()  # noqa: E731
        y = decode_linear(mesh_device, x, w, compute_kernel_config=HIFI4_FP32)
        want = x_host[0, 0].bfloat16().float() @ first(w)[0, 0]
        got = first(y)[0, 0]
        ttnn.deallocate(y)
        p = torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1].item()
        sec = timed(mesh_device, lambda: decode_linear(mesh_device, x, w, compute_kernel_config=HIFI4_FP32))
        mb = k * n * 1.0625 / 1e6
        print(
            f"M {m:4d} K {k:5d} N {n:5d}: {sec * 1e6:7.1f} us ({mb:5.1f} MB, {mb / 1e3 / sec:6.1f} GB/s), pcc {p:.6f}",
            flush=True,
        )
        assert p > 0.9999
        ttnn.deallocate(x)
        ttnn.deallocate(w)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_batched_head_configs(mesh_device):
    """DSA's per-head absorption matmuls ([1, 16, M, K] @ [1, 16, K, N], BFP8 weights, one tile row of tokens), trace-
    timed: the model's config (one head a core, `batched_matmul_config`) and ttnn's default; values against torch on the
    dequantized weights.  (ttnn's plain MatmulMultiCoreProgramConfig was slower: 39.9 / 42.9 us against
    24.1 / 24.2.)"""
    from models.demos.blackhole.glm53_flash.ttnn.common import batched_matmul_config

    rep = mapper(mesh_device, (None, None))
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()  # noqa: E731
    for k, n in ((256, 512), (512, 256)):
        x_host = torch.randn(1, 16, 32, k)
        w_host = torch.randn(1, 16, k, n) * 0.05
        x = ttnn.from_torch(x_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        w = ttnn.from_torch(w_host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        want = x_host.bfloat16().float() @ first(w)
        configs = {
            "one head a core (model)": batched_matmul_config(mesh_device, k_tiles=k // 32, n_tiles=n // 32),
            "ttnn default": None,
        }
        mb = 16 * k * n * 1.0625 / 1e6
        for name, cfg in configs.items():
            kw = {"program_config": cfg} if cfg is not None else {}

            def run(kw=kw):
                return ttnn.matmul(x, w, compute_kernel_config=HIFI4_FP32, **kw)

            got = first(run())
            p = torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1].item()
            sec = timed(mesh_device, run)
            print(
                f"[16, 32, {k}] @ [16, {k}, {n}] {name:26s}: {sec * 1e6:7.1f} us ({mb / 1e3 / sec:6.1f} GB/s), "
                f"pcc {p:.6f}",
                flush=True,
            )
        ttnn.deallocate(x)
        ttnn.deallocate(w)
