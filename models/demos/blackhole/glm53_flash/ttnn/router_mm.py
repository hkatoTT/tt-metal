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

import os

import ttnn

from .common import COLS, ROWS

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/router/"
CB_X, CB_W, CB_OUT, CB_PART, CB_SCR, CB_FLAG, CB_WS = 0, 1, 16, 24, 25, 26, 27
CB_X8 = 28  # (GLM53_MOE_X_BFP8: the BFP8_b copy of x, rm_compute.cpp / rm_w.cpp RM_X8)
SEM_X = 0
SEM_GO = 2  # (router_route: the collector's go to the other router cores, rm_w.cpp)
SEM_RDY, SEM_ARR = 3, 4  # (WSHARE: the column's two cores' counts, rm_w.cpp)


def wshare() -> bool:
    """GLM53_ROUTER_WSHARE (read at each call; on by default): at 2 tile rows of x (the decode's 64 tokens a mesh row,
    128 users), the column's two cores read each weight block from DRAM once between them and write it on to the other
    (rm_w.cpp SHARE), in place of each reading the column's whole 2.36 / 9 MB: the call's DRAM reads 4.7 -> 2.36 MB, the
    arithmetic and the values the same (the op 31.3 -> 28.3 us, the verify step at 128 users -0.11 ms; long stresses
    clean).  0: each core reads all, the earlier form."""
    return os.environ.get("GLM53_ROUTER_WSHARE", "1") == "1"


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


def router_parts(mesh, x, w, out, notify=(), sem_logits=0, x8=None):
    """The router matmul's kernels, CBs and semaphores (x [1, 1, n, K] bf16, w [1, 1, K, N] bf16 -> out [1, 1, n, N]
    fp32, all tiled in DRAM) on the grid's last rows, its cores and their count; `notify` (NoC (x, y) of other cores of
    the same program): their semaphore `sem_logits` set to 1 once every output tile is written (by writes through router
    core 0, rm_w.cpp).  `x8` ([1, 1, n, K] bfloat8_b tiled in DRAM; GLM53_MOE_X_BFP8=1): each core packs its
    share of its tile row of x (K tiles c, c + N tiles, ..) into it after its output tile (RM_X8)."""
    n, k = int(x.shape[-2]), int(x.shape[-1])
    nt, kt, mt = int(w.shape[-1]) // 32, k // 32, -(-n // 32)
    assert kt % KB == 0, kt
    grid = mesh.compute_with_storage_grid_size()
    assert nt <= grid.x and mt <= grid.y, (nt, mt, grid)
    row0 = grid.y - mt
    allc = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, row0), ttnn.CoreCoord(nt - 1, row0 + mt - 1))])
    xr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    x8_arg = 0
    count = nt * mt
    share = wshare() and mt == 2
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
            peer = []
            if share:  # (the column's other core, rm_w.cpp SHARE)
                xy = mesh.worker_core_from_logical_core(ttnn.CoreCoord(c, row0 + 1 - r))
                peer = [xy.x, xy.y]
            wr[c][y] = (
                [w.buffer_address(), out.buffer_address(), c, r, len(notify)]
                + ([r * nt + c, count] + tail_all if notify else [])
                + peer
            )
            x8_arg = len(wr[c][y])  # (the same for every core: x8's address after the others)
            if x8 is not None:
                wr[c][y] = list(wr[c][y]) + [x8.buffer_address()]
                cr[c][y] = [c]
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
            compile_time_args=[CB_W, CB_OUT, KB, nb, nt, sem_logits, CB_FLAG, SEM_GO]
            + acc(w)
            + acc(out)
            + (acc(x8) if x8 is not None else []),
            defines=[("AHEAD", str(AHEAD))]
            + (
                [("SHARE", "1"), ("SHARE_CB", str(CB_WS)), ("SHARE_RDY", str(SEM_RDY)), ("SHARE_ARR", str(SEM_ARR))]
                if share
                else []
            )
            + (
                [("RM_X8", "1"), ("RM_X8_CB", str(CB_X8)), ("RM_X8_KT", str(kt)), ("RM_X8_ARG", str(x8_arg))]
                if x8 is not None
                else []
            ),
            runtime_args=wr,
            config=_dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "rm_compute.cpp",
            core_ranges=allc,
            compile_time_args=[KB, nb],
            defines=[("RM_X8", "1"), ("RM_X8_CB", str(CB_X8)), ("RM_X8_NT", str(nt))] if x8 is not None else [],
            runtime_args=cr,
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
    if x8 is not None:  # (a core's share: K tiles c, c + nt, .., at most ceil(kt / nt))
        assert tuple(x8.shape) == tuple(x.shape) and x8.dtype == ttnn.bfloat8_b, (x8.shape, x8.dtype)
        cbs.append(_cb(CB_X8, ttnn.bfloat8_b, 1088, -(-kt // nt), allc))
    sems = [ttnn.SemaphoreDescriptor(id=SEM_X, core_ranges=allc, initial_value=0)]
    if share:
        cbs.append(_cb(CB_WS, ttnn.uint32, 16 * 2 * nb, 1, allc))
        sems += [ttnn.SemaphoreDescriptor(id=i, core_ranges=allc, initial_value=0) for i in (SEM_RDY, SEM_ARR)]
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


def router_route(mesh, moe, x, tokens: int, h1=None, x8=None):
    """moe.router_logits(x) and moe_route.route_only(mesh, moe, logits, tokens, h1) in one program (the same values
    bit for bit: the router's arithmetic as ttnn.matmul's, the routing's kernels as route_only's, its readers waiting
    for the router's cores): -> route_only's w (or h1's w_all).  `x8` (GLM53_MOE_X_BFP8): x's BFP8_b copy written by
    the router's cores too (router_parts)."""
    from . import moe_route

    logits = _alloc(mesh, [1, 1, int(x.shape[-2]), int(moe.router_wt.shape[-1])], ttnn.float32)
    out = moe_route.route_only(mesh, moe, logits, tokens, h1=h1, router=(x, moe.router_wt, x8))
    ttnn.deallocate(logits)
    return out


__all__ = ["router_matmul", "router_parts", "router_route"]
