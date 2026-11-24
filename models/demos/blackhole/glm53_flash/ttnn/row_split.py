# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Each chip of a mesh row taking its own part of a tensor the row's chips hold alike (generic_op, a program per mesh
column with its own offset): kernels/split/take_blocks.cpp.

  take(mesh, src, shape, blocks, block_pages, block_stride, offsets): out [shape] of src's pages, `blocks` blocks of
      `block_pages` pages, block b from src page offsets[c] + b block_stride on the chips of mesh column c.
"""

from __future__ import annotations

import ttnn

from .common import COLS, ROWS, core_rects

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/split/"
BATCH = 8


def take(mesh, src, shape, blocks: int, block_pages: int, block_stride: int, offsets: list[int]):
    """src (DRAM interleaved, tiled or row-major) -> a new tensor of `shape` (src's dtype and layout) whose pages are
    src's pages offsets[c] + b block_stride + p (b < blocks, p < block_pages) on mesh column c."""
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape(shape), src.dtype, src.layout, buffer_type=ttnn.BufferType.DRAM), mesh
    )
    if src.layout == ttnn.TILE_LAYOUT:
        page = 32 * 32 * src.element_size()
    else:
        page = shape[-1] * src.element_size()
    total = blocks * block_pages
    grid = mesh.compute_with_storage_grid_size()
    n = min(grid.x * grid.y, -(-total // BATCH))
    per = -(-total // n)
    parts = [(o, min(total, o + per)) for o in range(0, total, per)]
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(parts))]
    cores = core_rects(coords)
    mesh_program = ttnn.MeshProgramDescriptor()
    for col in range(COLS):
        rt = ttnn.RuntimeArgs()
        for c, (o0, o1) in zip(coords, parts):
            rt[c.x][c.y] = [src.buffer_address(), out.buffer_address(), o0, o1, offsets[col]]
        kernel = ttnn.KernelDescriptor(
            kernel_source=KDIR + "take_blocks.cpp",
            core_ranges=cores,
            # (the split and the block geometry in the compile-time args: the program cache does not see runtime values)
            compile_time_args=[page, block_pages, block_stride, BATCH, 0]
            + ttnn.TensorAccessorArgs(src).get_compile_time_args()
            + ttnn.TensorAccessorArgs(out).get_compile_time_args()
            + [total, per, offsets[col]],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        cb = ttnn.CBDescriptor(
            total_size=BATCH * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=page)],
        )
        program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])
        rng = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, col), ttnn.MeshCoordinate(ROWS - 1, col))
        mesh_program[rng] = program
    ttnn.generic_op([src, out], mesh_program)
    return out


__all__ = ["take"]
