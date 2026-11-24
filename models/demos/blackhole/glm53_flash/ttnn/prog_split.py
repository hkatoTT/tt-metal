# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The progressive form's weight split on the device (moe.py, GLM53_MOE_PROG_F): a bf16 weight (tiled, DRAM interleaved)
-> base = BFP4_b(w) and correction = BFP4_b(w - base), each packed exactly as the host packer (round to nearest-even;
tools/bfp_study.py's bfp_round), by the RISC-Vs of every core (kernels/prog/prog_split.cpp).  ttnn.typecast's BFP4
truncates (its base 1.8x the host's error, base + correction 3x BFP8's), and two host packings would double the load's
host time."""

from __future__ import annotations

import ttnn

from .common import COLS, ROWS

KERNEL = "models/demos/blackhole/glm53_flash/ttnn/kernels/prog/prog_split.cpp"
BF16_TILE, BFP4_TILE = 2048, 576


def _alloc(mesh, shape, dtype):
    return ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM), mesh
    )


def split(mesh, t, out=None, mode=0, e_off=0, slots=0, d_tiles=0, half=0, corr_off=0):
    """[.., H, W] bf16 tiled in DRAM -> (base, correction), both bfloat4_b tiled in DRAM, the same shape; or (`out`,
    mode 1 / 2) both written into the experts op's w1 / w2 (`out`, BFP4 ND-sharded) at their places: the kernel's
    `place` (kernels/prog/prog_split.cpp), the correction `corr_off` tiles past the base.  (The plain copy of a dense
    copy's tiles to their places, `relocate` for GLM53_MOE_W1_COPY, was removed.)"""
    shape = [int(d) for d in t.shape]
    if out is None:
        base = _alloc(mesh, shape, ttnn.bfloat4_b)
        corr = _alloc(mesh, shape, ttnn.bfloat4_b)
    else:
        base = corr = out
    width = shape[-1] // 32
    tiles = 1
    for d in shape[:-2]:
        tiles *= d
    tiles *= (shape[-2] // 32) * (shape[-1] // 32)
    g = mesh.compute_with_storage_grid_size()
    coords = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]
    per, extra = divmod(tiles, len(coords))
    rt = ttnn.RuntimeArgs()
    first = 0
    used = []
    for i, c in enumerate(coords):
        count = per + (1 if i < extra else 0)
        if count == 0:
            continue
        rt[c.x][c.y] = [
            t.buffer_address(),
            base.buffer_address(),
            corr.buffer_address(),
            first,
            count,
            mode,
            width,
            e_off,
            slots,
            d_tiles,
            half,
            corr_off,
        ]
        first += count
        used.append(c)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in used])
    ct = [0]
    for x in (t, base, corr):
        ct += ttnn.TensorAccessorArgs(x).get_compile_time_args()
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KERNEL,
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
    ]
    l1 = 2 * BF16_TILE + 4 * BFP4_TILE
    cbs = [
        ttnn.CBDescriptor(
            total_size=l1,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=l1)],
        )
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    mesh_program = ttnn.MeshProgramDescriptor()
    whole = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(ROWS - 1, COLS - 1))
    mesh_program[whole] = program
    ttnn.generic_op([t, base, corr] if out is None else [t, out], mesh_program)
    return base, corr


__all__ = ["split"]
