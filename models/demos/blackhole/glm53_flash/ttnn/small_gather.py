# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A small row-major gather over the mesh column in one op: x [1, 1, R, W] row-major (each chip its own, R x W x its
element size at most a landing page, 2 KB) -> [1, 1, 8 R, W] on every chip, mesh row r's rows at r R: the same bytes as
ttnn.all_gather(x, dim=2, cluster_axis=ROW_AXIS), which took ~68 us for the DFlash2 step's 8 U floats a row
(`dflash2_step.gathered_results`: the ring over the column's 8 chips).

One core a chip, one-shot over the 2D fabric (kernels/dflash2/small_gather.cpp, head_pick's column stage): the payload
to the column's 7 other chips (a 4-hop and a 3-hop multicast) into their DRAM landing pages, each packet a write fused
with an increment of the receive semaphore for the sender.  The landing pages and semaphores are head_pick's "head_col"
family's (2 KB pages; made at the load, `head_pick.prealloc`): no new DRAM or L1 (the global semaphores take L1 on every
core, and ttnn's default matmul configs read the L1 left).  The family's calls alternate parities; two calls of the
same parity are apart by the model's collectives over the same columns (as head_pick's)."""

from __future__ import annotations

import ttnn

from .common import COLS, ROWS
from .row_all_reduce import oneshot_area

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/dflash2/"
PAGE = 2048  # head_pick.PAGE: the "head_col" family's landing pages
CB_SCR = 0


class SmallGather:
    def __init__(self, mesh, rows: int, row_bytes: int):
        assert tuple(mesh.shape) == (ROWS, COLS) and ROWS == 8, mesh.shape
        assert rows * row_bytes <= PAGE and row_bytes % 32 == 0, (rows, row_bytes)
        assert rows == 1 or row_bytes % 64 == 0, (rows, row_bytes)  # (the rows' reads and writes 64-byte aligned)
        self.mesh, self.rows, self.row_bytes = mesh, rows, row_bytes
        self.send_bytes = max(64, -(-rows * row_bytes // 64) * 64)
        self.core = ttnn.CoreCoord(0, 0)
        self.core_set = ttnn.CoreRangeSet([ttnn.CoreRange(self.core, self.core)])
        self.core_xy = mesh.worker_core_from_logical_core(self.core)
        self.area = oneshot_area(mesh, "head_col", ROWS, 2 * ROWS, PAGE)
        self.nodes = {
            (r, c): mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for r in range(ROWS) for c in range(COLS)
        }
        self.scr_bytes = 64 + (ROWS + 1) * PAGE  # (64: the kernel aligns its start) the payload, a landed one a row

    def _program(self, r: int, c: int, x, out, parity: int) -> ttnn.ProgramDescriptor:
        node = self.nodes[(r, c)]
        cf, cb = self.nodes[((r + 1) % ROWS, c)], self.nodes[((r - 1) % ROWS, c)]
        ca = self.area
        rt = ttnn.RuntimeArgs()
        args = [x.buffer_address(), out.buffer_address(), int(node.mesh_id), int(cf.chip_id), int(cb.chip_id)]
        args += [ca.area.buffer_address(), parity * ROWS, r, ca.sem_addr[parity][r], self.core_xy.x, self.core_xy.y]
        args += [*ca.sem_addr[parity]]
        rt[self.core.x][self.core.y] = args
        kernel = ttnn.KernelDescriptor(
            kernel_source=KDIR + "small_gather.cpp",
            core_ranges=self.core_set,
            compile_time_args=[CB_SCR, self.rows, self.row_bytes, PAGE, self.send_bytes]
            + ttnn.TensorAccessorArgs(x).get_compile_time_args()
            + ttnn.TensorAccessorArgs(out).get_compile_time_args()
            + ttnn.TensorAccessorArgs(ca.area).get_compile_time_args(),
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        cbs = [
            ttnn.CBDescriptor(
                total_size=self.scr_bytes,
                core_ranges=self.core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_SCR, data_format=ttnn.uint32, page_size=self.scr_bytes)
                ],
            )
        ]
        program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
        args = program.kernels[0].runtime_args[self.core.x][self.core.y]
        for peer in (cf, cb):
            args.extend(ttnn.setup_fabric_connection(node, peer, 0, program, self.core))
        return program

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        assert x.layout == ttnn.ROW_MAJOR_LAYOUT and tuple(x.shape)[:2] == (1, 1), x
        parity = self.area.next_parity()
        out = ttnn.allocate_tensor_on_device(
            ttnn.TensorSpec(
                ttnn.Shape([1, 1, ROWS * x.shape[-2], x.shape[-1]]),
                x.dtype,
                ttnn.ROW_MAJOR_LAYOUT,
                buffer_type=ttnn.BufferType.DRAM,
            ),
            self.mesh,
        )
        mesh_program = ttnn.MeshProgramDescriptor()
        for r in range(ROWS):
            for c in range(COLS):
                coord = ttnn.MeshCoordinate(r, c)
                mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = self._program(r, c, x, out, parity)
        ttnn.generic_op([x, out], mesh_program)
        return out


_STATE: dict = {}


def small_gather(mesh, x: ttnn.Tensor) -> ttnn.Tensor:
    """x [1, 1, R, W] row-major (R x W x its element size <= 2 KB) -> [1, 1, 8 R, W] on every chip: the column's 8
    rows' x, mesh row r's at r R (ttnn.all_gather(x, dim=2, cluster_axis=ROW_AXIS)'s bytes)."""
    rows, row_bytes = int(x.shape[-2]), int(x.shape[-1]) * x.element_size()
    key = (id(mesh), rows, row_bytes)
    if key not in _STATE or _STATE[key].mesh is not mesh:
        _STATE[key] = SmallGather(mesh, rows, row_bytes)
    return _STATE[key](x)


__all__ = ["SmallGather", "small_gather"]
