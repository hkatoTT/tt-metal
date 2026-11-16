# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The MoE router's matmul as a generic_op (x [1, 1, n, 4096] bf16 @ w [1, 1, 4096, 288] bf16 -> [1, 1, n, 288] fp32),
the arithmetic of ttnn.matmul with moe.py's `_router_config` bit for bit (per core one output tile, K in blocks of 16
tiles, HiFi4, fp32 Dest, packer L1 accumulation: kernels/router/rm_compute.cpp), as the first part of the router and
routing in one program (`router_route`).  Cores: column c of the
output on grid column c, tile row r of x on the grid row ROW0 + r (from the grid's last rows); each row's column-0 core
reads its tile row of x from DRAM and multicasts it to the row (kernels/router/rm_x.cpp, RISCV_1); every core reads its
column of w (rm_w.cpp, RISCV_0) and writes its output tile.  The NoCs as ttnn's 1D matmul: x on NoC 1, w on NoC 0
(15.9 / 15.6 / 21.0 us against ttnn's 17.6 / 18.3 / 19.3 at 8 / 32 / 64 rows; the other way round, 32.1 / 31.9 /
38.0: the weight stream on NoC 1 ran at ~10 GB/s a core; that form, GLM53_ROUTER_MM_X_NOC1=0, was removed)."""

from __future__ import annotations

import ttnn

from .common import COLS, ROWS

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/router/"
CB_X, CB_W, CB_OUT, CB_PART, CB_SCR, CB_FLAG = 0, 1, 16, 24, 25, 26
SEM_X = 0
SEM_GO = 2  # (router_route: the collector's go to the other router cores, rm_w.cpp)
KB = 16  # in0_block_w (moe.py `_router_config`)
AHEAD = 3  # weight blocks in flight
X_CHUNK = 16  # x's tiles a multicast
# (the NoCs as ttnn's 1D matmul: in0, x, RISCV_1 on NoC 1, its reads and multicast; in1, w, RISCV_0 on NoC 0,
# preferred_noc_for_dram_read)


def _alloc(mesh, shape, dtype):
    spec = ttnn.TensorSpec(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM)
    return ttnn.allocate_tensor_on_device(spec, mesh)


def _cb(index, fmt, page, pages, cores):
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
    )


def _dm(processor, noc):
    cfg = ttnn.DataMovementConfigDescriptor()
    cfg.processor, cfg.noc = processor, noc
    return cfg


def router_parts(mesh, x, w, out, notify=(), sem_logits=0):
    """The router matmul's kernels, CBs and semaphores (x [1, 1, n, K] bf16, w [1, 1, K, N] bf16 -> out [1, 1, n, N]
    fp32, all tiled in DRAM) on the grid's last rows, its cores and their count; `notify` (NoC (x, y) of other cores of
    the same program): their semaphore `sem_logits` set to 1 once every output tile is written (by writes through router
    core 0, rm_w.cpp)."""
    n, k = int(x.shape[-2]), int(x.shape[-1])
    nt, kt, mt = int(w.shape[-1]) // 32, k // 32, -(-n // 32)
    assert kt % KB == 0, kt
    grid = mesh.compute_with_storage_grid_size()
    assert nt <= grid.x and mt <= grid.y, (nt, mt, grid)
    row0 = grid.y - mt
    allc = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, row0), ttnn.CoreCoord(nt - 1, row0 + mt - 1))])
    xr, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    count = nt * mt
    routers = [mesh.worker_core_from_logical_core(ttnn.CoreCoord(c, row0 + r)) for r in range(mt) for c in range(nt)]
    tail_all = [v for xy in routers for v in (xy.x, xy.y)] + [v for xy in notify for v in xy]
    for r in range(mt):
        y = row0 + r
        # the x kernel's multicast in its NoC's corner order: NoC 1's, the far corner first, as ttnn's matmul swaps
        # them for NOC_1.  (NoC 1's order passed to a NoC 0 multicast gives no acks: the sender waited at its write
        # barrier and the receivers at their semaphores, a hang)
        near = mesh.worker_core_from_logical_core(ttnn.CoreCoord(nt - 1, y))
        far = mesh.worker_core_from_logical_core(ttnn.CoreCoord(1, y))
        for c in range(nt):
            xr[c][y] = [x.buffer_address(), r, int(c == 0), nt, near.x, near.y, far.x, far.y]
            wr[c][y] = [w.buffer_address(), out.buffer_address(), c, r, len(notify)] + (
                [r * nt + c, count] + tail_all if notify else []
            )
    acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731
    nb = kt // KB
    chunks = -(-kt // X_CHUNK)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[CB_PART] = ttnn.UnpackToDestMode.UnpackToDestFp32  # (the reload of the fp32 partials: no TF32 rounding)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    cfg.unpack_to_dest_mode = modes
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "rm_x.cpp",
            core_ranges=allc,
            compile_time_args=[CB_X, CB_SCR, SEM_X, kt, X_CHUNK] + acc(x),
            runtime_args=xr,
            config=_dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "rm_w.cpp",
            core_ranges=allc,
            compile_time_args=[CB_W, CB_OUT, KB, nb, nt, sem_logits, CB_FLAG, SEM_GO] + acc(w) + acc(out),
            defines=[("AHEAD", str(AHEAD))],
            runtime_args=wr,
            config=_dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "rm_compute.cpp",
            core_ranges=allc,
            compile_time_args=[KB, nb],
            runtime_args=ttnn.RuntimeArgs(),
            config=cfg,
        ),
    ]
    cbs = [
        _cb(CB_X, ttnn.bfloat16, 2048, kt, allc),
        _cb(CB_W, ttnn.bfloat16, 2048, (AHEAD + 1) * KB, allc),
        _cb(CB_OUT, ttnn.float32, 4096, 1, allc),
        _cb(CB_PART, ttnn.float32, 4096, 1, allc),
        _cb(CB_SCR, ttnn.uint32, 16 * chunks, 1, allc),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=SEM_X, core_ranges=allc, initial_value=0)]
    if notify:
        cbs.append(_cb(CB_FLAG, ttnn.uint32, 16 * count, 1, allc))
        sems.append(ttnn.SemaphoreDescriptor(id=SEM_GO, core_ranges=allc, initial_value=0))
    cores = [ttnn.CoreCoord(c, row0 + r) for r in range(mt) for c in range(nt)]
    return kernels, cbs, sems, cores, nt * mt


def router_matmul(mesh, x, w):
    """x [1, 1, n, K] bf16 tiled (DRAM), w [1, 1, K, N] bf16 tiled (DRAM) -> [1, 1, n, N] fp32 tiled (DRAM)."""
    out = _alloc(mesh, [1, 1, int(x.shape[-2]), int(w.shape[-1])], ttnn.float32)
    kernels, cbs, sems, _, _ = router_parts(mesh, x, w, out)
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    mesh_program = ttnn.MeshProgramDescriptor()
    whole = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(ROWS - 1, COLS - 1))
    mesh_program[whole] = program
    ttnn.generic_op([x, w, out], mesh_program)
    return out


def router_route(mesh, moe, x, tokens: int, h1=None):
    """moe.router_logits(x) and moe_route.route_only(mesh, moe, logits, tokens, h1) in one program (the same values
    bit for bit: the router's arithmetic as ttnn.matmul's, the routing's kernels as route_only's, its readers waiting
    for the router's cores): -> route_only's w (or h1's w_all)."""
    from . import moe_route

    logits = _alloc(mesh, [1, 1, int(x.shape[-2]), int(moe.router_wt.shape[-1])], ttnn.float32)
    out = moe_route.route_only(mesh, moe, logits, tokens, h1=h1, router=(x, moe.router_wt))
    ttnn.deallocate(logits)
    return out


__all__ = ["router_matmul", "router_parts", "router_route"]
