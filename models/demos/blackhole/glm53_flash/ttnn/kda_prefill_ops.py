# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Small fused steps of the KDA prefill (`TtKda._terms`), through ttnn.generic_op, the same program on every chip:

  gate(mesh, fg, bias, rate, width): the decay gate bf16(sigmoid((fp32(f) + dt_bias) * rate)) of f = fg's first `width`
      columns, in one op with the same SFPU steps and rounding as the six ttnn ops it replaces (slice, typecast to
      fp32, add, multiply, sigmoid, typecast to bf16: kernels/kda/gate_*.cpp).
  history_commit(mesh, history, dsts, table): a request's last K conv inputs (rows of history [1, K, C] bf16) into
      row `user` of the K history tensors (dsts, [1, 1, B, C] fp32), `user` from the commit table (-1: none),
      in one op of copies (kernels/kda/history_commit.cpp), in place of each row's slice and typecast and the one-hot
      multiply and addcmul over all B users (`TtKda._commit`: about 45 us a row, 3 rows a KDA layer).
"""

from __future__ import annotations

import ttnn

from .common import COLS, ROWS, core_rects

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/kda/"
CB_F, CB_B, CB_RT, CB_O = 0, 1, 2, 3


def _run(mesh, tensors: list, program) -> None:
    mesh_program = ttnn.MeshProgramDescriptor()
    whole = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(ROWS - 1, COLS - 1))
    mesh_program[whole] = program
    ttnn.generic_op(tensors, mesh_program)


def _cb(index: int, fmt, page: int, pages: int, cores) -> ttnn.CBDescriptor:
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
    )


def gate_constant(mesh, values, width: int):
    """A per-channel fp32 vector [1, 1, 1, 4 width] (the chips' slices along the last dim, as `TtKda.dt_bias`) as the
    gate op's operand: [1, 1, 32, width] per chip, the row repeated in all 32 rows (what binary_ng's row broadcast
    reads)."""
    from .common import put

    return put(
        mesh, values.float().reshape(1, 1, 1, -1).expand(1, 1, 32, -1).contiguous(), dims=(None, 3), dtype=ttnn.float32
    )


def gate(mesh, fg, bias, rate, width: int, real=None):
    """fg [1, T, F] bf16 tiled (f in its first `width` columns), bias and rate [1, 1, 32, width] fp32 (`gate_constant`)
    -> [1, T, width] bf16 tiled.  `real` ([1, T, 1] bf16 tiled of 1 and 0, or None): the rows of a 0 come out times 0
    (the sign kept), as `ttnn.multiply(gate, real)`."""
    t, f_cols = fg.shape[-2], fg.shape[-1] // 32
    r_tiles, w_cols = t // 32, width // 32
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape([1, t, width]), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM),
        mesh,
    )
    grid = mesh.compute_with_storage_grid_size()
    total = r_tiles * w_cols
    n = min(grid.x * grid.y, total)
    per = -(-total // n)
    ranges = [(i * per, min(total, (i + 1) * per)) for i in range(n) if i * per < total]
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(ranges))]
    cores = core_rects(coords)
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (i0, i1) in zip(coords, ranges):
        rd[c.x][c.y] = [fg.buffer_address(), bias.buffer_address(), rate.buffer_address(), i0, i1]
        cp[c.x][c.y] = [i0, i1]
        wr[c.x][c.y] = [out.buffer_address(), i0, i1, real.buffer_address() if real is not None else 0]
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[CB_B] = ttnn.UnpackToDestMode.UnpackToDestFp32
    modes[CB_RT] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    cfg.unpack_to_dest_mode = modes
    # (T, the width and the per-core split in the compile-time args: the program cache does not see runtime values)
    key = [r_tiles, w_cols, per]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "gate_reader.cpp",
            core_ranges=cores,
            compile_time_args=[r_tiles, f_cols, CB_F, CB_B, CB_RT]
            + ttnn.TensorAccessorArgs(fg).get_compile_time_args()
            + ttnn.TensorAccessorArgs(bias).get_compile_time_args()
            + ttnn.TensorAccessorArgs(rate).get_compile_time_args()
            + key,
            runtime_args=rd,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "gate_compute.cpp",
            core_ranges=cores,
            compile_time_args=[r_tiles, CB_F, CB_B, CB_RT, CB_O] + key,
            runtime_args=cp,
            config=cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "gate_writer.cpp",
            core_ranges=cores,
            compile_time_args=[r_tiles, w_cols, CB_O]
            + ttnn.TensorAccessorArgs(out).get_compile_time_args()
            + key
            + [int(real is not None)]
            + ttnn.TensorAccessorArgs(real if real is not None else out).get_compile_time_args(),
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    cbs = [
        _cb(CB_F, ttnn.bfloat16, 2048, 2, cores),
        _cb(CB_B, ttnn.float32, 4096, 2, cores),
        _cb(CB_RT, ttnn.float32, 4096, 2, cores),
        _cb(CB_O, ttnn.bfloat16, 2048, 2, cores),
        _cb(CB_O + 1, ttnn.bfloat16, 1024, 1, cores),  # the writer's scratch for the mask's faces
    ]
    _run(mesh, [fg, bias, rate, out] + ([real] if real is not None else []),
         ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))  # fmt: skip
    return out


HISTORY_CORES = 32


def history_commit(mesh, history, dsts: list, table) -> None:
    """history [1, K, C] bf16 tiled (a mesh row's request's last K conv inputs, oldest first), dsts K tensors [1, 1, B, C]
    fp32 tiled (every user's history rows, oldest first, updated in place), table [1, 1] int32 row-major per chip (the
    row's user; -1: no request, nothing written) -> row k of history into row `user` of dsts[k] (bf16 -> fp32: exact).
    """
    k, c = history.shape[-2], history.shape[-1]
    assert history.dtype == ttnn.bfloat16 and history.layout == ttnn.TILE_LAYOUT and k == len(dsts) and k < 16, history
    for d in dsts:
        assert d.dtype == ttnn.float32 and d.layout == ttnn.TILE_LAYOUT and d.shape[-1] == c, d
    assert table.dtype == ttnn.int32 and table.layout == ttnn.ROW_MAJOR_LAYOUT, table
    tiles = c // 32
    grid = mesh.compute_with_storage_grid_size()
    n = min(HISTORY_CORES, tiles, grid.x * grid.y)
    per = -(-tiles // n)
    parts = [(t, min(tiles, t + per)) for t in range(0, tiles, per)]
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(parts))]
    cores = core_rects(coords)
    table_page = 4 * table.shape[-1]
    rt = ttnn.RuntimeArgs()
    for x, (t0, t1) in zip(coords, parts):
        rt[x.x][x.y] = [history.buffer_address(), table.buffer_address(), t0, t1] + [d.buffer_address() for d in dsts]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KDIR + "history_commit.cpp",
        core_ranges=cores,
        compile_time_args=[tiles, k, table_page, per, 0]
        + ttnn.TensorAccessorArgs(history).get_compile_time_args()
        + ttnn.TensorAccessorArgs(table).get_compile_time_args()
        + ttnn.TensorAccessorArgs(dsts[0]).get_compile_time_args(),
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    size = 64 + 2 * (k * per * 2) * 64  # the table, then a source and a destination slot per face row
    cb = ttnn.CBDescriptor(
        total_size=size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=size)],
    )
    _run(mesh, [history, table, *dsts], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb]))


__all__ = ["gate", "gate_constant", "history_commit"]
