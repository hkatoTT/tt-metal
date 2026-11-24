# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""A decode matmul with a long K split over 3 cores a column of the output, the values of decode_linear's 1D config bit
for bit (the drafter's context fc, x [1, 1, R, 36,864] bf16 @ w [36,864, 1,024] BFP8, R = 32 / 64 at 32 / 128 users,
ttnn's 1D matmul on 32 cores: 125 / 192 us, its x multicast from one sender and each core's whole column of w the
bounds).

Arithmetic (kernels/fc_ksplit/fk_compute.cpp): as ttnn.matmul with decode_linear's config for at most two tile rows
(per core MT x 1 output tiles, out_subblock 1 x 1, in0_block_w KB, the given compute config's fidelity, fp32 Dest,
packer L1 accumulation): P = B_0, P += B_1 .. B_{NB-2} (the packer, fp32), out = P reloaded + the last block.  Block b <
NB - 1 is summed by slice b % 3 (slice 0, the owner, also takes the last block): the helpers write their B_b (fp32) into
the owner's ring of theirs (all of them: no credits), and the owner unpacks each to Dest and packs it onto P with the
same L1 accumulation in the same order.

Cores (12 x 10 grid): the slices as column bands, slice s on x 4 s .. 4 s + 3, y 0..7 (32 cores each, column n the n-th
core row-major); each slice's x sender right above its band's far corner, on (4 s + 3, 8): it reads the slice's blocks
of x from DRAM and multicasts them on NoC 1 (kernels/fc_ksplit/fk_x.cpp, CB_X a ring of XS slots, counts by plain
writes); the weights on NoC 0 and the partials written to the owner over NoC 0 (fk_w.cpp).  NoC 1 runs toward the
smaller coordinates and a multicast starts at its rectangle's far corner, so each sender enters its own band in one
hop and the three multicast trees never cross.  The first layout (slices 0 / 1 on x 0..7, y 0..3 / 4..7, slice 2 on
x 8..11, senders on (s, 8)) had the senders' ways cross the other slices' rectangles round the torus, three linked
multicasts at once: one sender (slice 2's) stuck inside its multicast, a hang within seconds of back-to-back calls
and once in the DFlash2 step at 128 users."""

from __future__ import annotations

import os

import ttnn

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/fc_ksplit/"
CB_X, CB_W, CB_OUT, CB_PART, CB_SEND, CB_RECV0, CB_XF, CB_XW, CB_WW = 0, 1, 16, 24, 25, 26, 29, 30, 31
# (helper h's partials at the owner: CB_RECV0 + h, h = 1 .. S - 1)
SEM_X, SEM_RDY0 = 0, 1  # (the owner's counts of helper h's partials: SEM_RDY0 + h)
S = 3
XS = 3  # CB_X's slots
SS = 4  # CB_SEND's slots
AHEAD = 3  # weight blocks in flight
RECTS = [((0, 0), (3, 7)), ((4, 0), (7, 7)), ((8, 0), (11, 7))]  # each slice's rectangle (logical x0, y0), (x1, y1)
SENDERS = [(3, 8), (7, 8), (11, 8)]  # (right above each band's far corner: NoC 1's multicasts in one hop, disjoint)


def pattern() -> list:
    """GLM53_FC_KSPLIT_PATTERN (read at each call; default 0,1,2): block b < NB - 1 summed by slice PAT[b % len(PAT)]
    (0 the owner; e.g. 0,1,2,1,2 gives the owner a fifth of the blocks, its merges the rest)."""
    pat = [int(v) for v in os.environ.get("GLM53_FC_KSPLIT_PATTERN", "0,1,2").split(",") if v != ""]
    assert 1 <= len(pat) <= 14 and set(pat) == set(range(S)), pat
    return pat


def _pat_word(pat: list) -> int:
    return len(pat) | sum(v << (4 + 2 * i) for i, v in enumerate(pat))


def on() -> bool:
    """GLM53_DFLASH2_FC_KSPLIT (read at each call; default 1): the drafter's context fc through `fc_ksplit` when it
    takes it (`takes`)."""
    return os.environ.get("GLM53_DFLASH2_FC_KSPLIT", "1") == "1"


def takes(mesh, x, w) -> bool:
    """x [.., R, K] (32 < R <= 64: at one tile row ttnn's 1D matmul is as fast, 127.7 / 129.0 against 131.5 / 129.9 us
    at R 32 / 8), w [K, N] with N / 32 = 32 (one slice's cores), K / 32 a multiple of 16, on a 12 x 10
    grid.  GLM53_FC_KSPLIT_ROWS_MIN (tests; default 33) the smallest R."""
    grid = mesh.compute_with_storage_grid_size()
    n, k, r = int(w.shape[-1]) // 32, int(w.shape[-2]) // 32, int(x.shape[-2])
    r_min = int(os.environ.get("GLM53_FC_KSPLIT_ROWS_MIN", "33"))
    return grid.x >= 12 and grid.y >= 9 and n == 32 and k % 16 == 0 and k // 16 > S and r_min <= r <= 64


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


def _core(s: int, n: int) -> tuple:
    (x0, y0), (x1, _) = RECTS[s]
    w = x1 - x0 + 1
    return x0 + n % w, y0 + n // w


def fc_ksplit(mesh, x, w, fidelity=ttnn.MathFidelity.HiFi4):
    """x [1, 1, R, K] bf16 tiled (DRAM), w [1, 1, K, N] (or [K, N]) BFP8 / BFP4 / bf16 tiled (DRAM) -> [1, 1, R, N] bf16
    tiled (DRAM): ttnn.linear(x, w) with decode_linear's 1D config (in0_block_w 16, fp32 Dest, packer L1
    accumulation, `fidelity`), bit for bit."""
    r, k, n = int(x.shape[-2]), int(x.shape[-1]), int(w.shape[-1])
    pat = pattern()
    defines = [("FK_PAT", str(_pat_word(pat)) + "u")]
    mt, kt, nt = -(-r // 32), k // 32, n // 32
    kb = 16  # decode_linear: 16 K tiles a block for 32 output tiles or fewer
    nb = kt // kb
    assert nt == 32 and kt % kb == 0 and nb > S and mt <= 2, (r, k, n)
    spec = ttnn.TensorSpec(ttnn.Shape([1, 1, r, n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM)
    out = ttnn.allocate_tensor_on_device(spec, mesh)
    phys = lambda c: mesh.worker_core_from_logical_core(ttnn.CoreCoord(*c))  # noqa: E731
    comp_ranges = [ttnn.CoreRange(ttnn.CoreCoord(*a), ttnn.CoreCoord(*b)) for a, b in RECTS]
    snd_ranges = [ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in SENDERS]
    comp, allc = ttnn.CoreRangeSet(comp_ranges), ttnn.CoreRangeSet(comp_ranges + snd_ranges)
    xr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for s in range(S):
        (x0, y0), (x1, y1) = RECTS[s]
        hi, lo = phys((x1, y1)), phys((x0, y0))  # (NoC 1's corner order: the far corner first, as router_mm)
        sp = phys(SENDERS[s])
        common = [hi.x, hi.y, lo.x, lo.y, sp.x, sp.y, s, int(s == 0)]
        sx, sy = SENDERS[s]
        xr[sx][sy] = [x.buffer_address(), 0, 32] + common
        for c in range(32):
            cx, cy = _core(s, c)
            xr[cx][cy] = [x.buffer_address(), 1 + c, 32] + common
            owner = phys(_core(0, c))
            wr[cx][cy] = [w.buffer_address(), out.buffer_address(), c, s, owner.x, owner.y]
            cr[cx][cy] = [s]
    acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[CB_PART] = ttnn.UnpackToDestMode.UnpackToDestFp32  # (the partials and the pulled B_b: no TF32 rounding)
    for h in range(1, S):
        modes[CB_RECV0 + h] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True)
    cfg.unpack_to_dest_mode = modes
    wfmt = {
        ttnn.bfloat8_b: (ttnn.bfloat8_b, 1088),
        ttnn.bfloat4_b: (ttnn.bfloat4_b, 576),
        ttnn.bfloat16: (ttnn.bfloat16, 2048),
    }
    wdt, wpage = wfmt[w.dtype]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "fk_x.cpp",
            core_ranges=allc,
            compile_time_args=[CB_X, CB_XF, CB_XW, SEM_X, kt, kb, mt, nb, XS] + acc(x),
            defines=defines,
            runtime_args=xr,
            config=_dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "fk_w.cpp",
            core_ranges=comp,
            compile_time_args=[CB_W, CB_OUT, CB_SEND, CB_RECV0, CB_WW, SEM_RDY0, kb, nb, mt, S, nt] + acc(w) + acc(out),
            defines=[("AHEAD", str(AHEAD))] + defines,
            runtime_args=wr,
            config=_dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "fk_compute.cpp",
            core_ranges=comp,
            compile_time_args=[kb, nb, mt, S],
            defines=defines,
            runtime_args=cr,
            config=cfg,
        ),
    ]
    count = [sum(1 for b in range(nb - 1) if pat[b % len(pat)] == h) for h in range(S)]
    blocks = max(count[0] + 1, *count[1:])
    cbs = [
        _cb(CB_X, ttnn.bfloat16, 2048, XS * mt * kb, allc),
        _cb(CB_XF, ttnn.uint32, 16 * 33, 1, allc),
        _cb(CB_XW, ttnn.uint32, 16 * blocks, 1, allc),
        _cb(CB_W, wdt, wpage, (AHEAD + 1) * kb, comp),
        _cb(CB_OUT, ttnn.bfloat16, 2048, mt, comp),
        _cb(CB_PART, ttnn.float32, 4096, mt, comp),
        _cb(CB_SEND, ttnn.float32, 4096, SS * mt, comp),
        _cb(CB_WW, ttnn.uint32, 16 * nb, 1, comp),
    ]
    for h in range(1, S):  # (all of helper h's partials: no credits)
        cbs.append(_cb(CB_RECV0 + h, ttnn.float32, 4096, count[h] * mt, comp))
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=allc, initial_value=0) for i in range(SEM_RDY0 + S)]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    mesh_program = ttnn.MeshProgramDescriptor()
    rows, cols = tuple(mesh.shape)  # (8 x 4; 1 x 1 on a single-chip simulator)
    whole = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(rows - 1, cols - 1))
    mesh_program[whole] = program
    ttnn.generic_op([x, w, out], mesh_program)
    return out


__all__ = ["fc_ksplit", "on", "takes"]
