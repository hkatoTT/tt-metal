# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The greedy head's pick over the mesh in one op (`TtLmHead.greedy_mesh`, GLM53_HEAD_PICK): each chip's top logit
and its global id per row of tokens (value, id [1, 1, M, 1] fp32 TILE, M = 8 t rows) -> the ids of the mesh's top
[1, 1, M, 1] fp32 (the largest value, of equal values the smallest id, as torch's argmax), the same on every chip; in
place of all-gathering the values and the ids over the row's and the column's chips (4 all_broadcast + concat pairs)
and the max, compare and min over the 32.

One core a chip, one-shot over the 2D fabric, through ttnn.generic_op (the plumbing as in `row_all_reduce.py`):
  - NCRISC reads column 0 of the value and id tiles into the payload P (M pairs of floats) and hands it to BRISC;
  - BRISC sends P to the mesh row's 3 other chips (a 2-hop multicast one way, a unicast the other) into their DRAM
    landing area (`row_all_reduce.OneshotArea` "head_row": a page for each parity and source column), each packet a
    write fused with an increment of the op core's receive semaphore for that column; it folds the 3 others' P into R
    as they land and hands R back;
  - NCRISC sends R to the mesh column's 7 other chips (a 4-hop multicast one way and a 3-hop one the other; area
    "head_col") and folds the 7 others' R into the mesh's best, then writes its ids into column 0 of the output's
    tiles (the rest zero).
The fold is exact (compares of floats), so the result is the same whatever the order.  The landing pages and
semaphores alternate between two parities from call to call (each area its own turn); two calls in a row of the same
parity are apart by the model's collectives over the same rows and columns (attention's row all-reduce, the MoE's
column gather), so a chip never writes a page its peer is still reading.
"""

from __future__ import annotations

import ttnn

from .common import COLS, ROWS
from .row_all_reduce import oneshot_area

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/head_pick/"
PAGE = 2048  # a landing page: up to 256 rows of (value, id)
# ... and past that (DFlash2's verify step at 128 users T = 8 and 256 users T = 4: 1,024 rows of the mesh, where the
# head took 4 all_broadcast + concat pairs and the reductions), a page of up to 1,024 rows (8 KB: the fabric's largest
# packet), in landing areas of their own (head.HEAD_PICK_MAX_ROWS 1024 to take them; made at the load, `prealloc`)
BIG_PAGE = 8192
CB_SCR = 0
SEM_P, SEM_R = 0, 1
MAX_ROWS = 1024


class HeadPick:
    def __init__(self, mesh, rows: int):
        assert tuple(mesh.shape) == (ROWS, COLS) and ROWS == 8 and COLS == 4, mesh.shape
        assert 1 <= rows <= MAX_ROWS and rows % 8 == 0, rows
        self.mesh, self.rows = mesh, rows
        self.trows = -(-rows // 32)
        self.page = PAGE if 8 * rows <= PAGE else BIG_PAGE
        self.core = ttnn.CoreCoord(0, 0)
        self.core_set = ttnn.CoreRangeSet([ttnn.CoreRange(self.core, self.core)])
        self.core_xy = mesh.worker_core_from_logical_core(self.core)
        self.row_area = oneshot_area(mesh, "head_row", COLS, 2 * COLS, self.page)
        self.col_area = oneshot_area(mesh, "head_col", ROWS, 2 * ROWS, self.page)
        self.nodes = {
            (r, c): mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for r in range(ROWS) for c in range(COLS)
        }
        # the scratch: P at 0, R, one landed payload, then the input's and output's tiles
        self.off_r = self.page
        self.off_in = 2 * self.page
        self.off_tiles = 3 * self.page
        self.scr_bytes = self.off_tiles + 2 * self.trows * 4096

    def _program(self, r: int, c: int, value, ids, out, rparity: int, cparity: int) -> ttnn.ProgramDescriptor:
        node = self.nodes[(r, c)]
        rf, rb = self.nodes[(r, (c + 1) % COLS)], self.nodes[(r, (c - 1) % COLS)]
        cf, cb = self.nodes[((r + 1) % ROWS, c)], self.nodes[((r - 1) % ROWS, c)]
        ra, ca = self.row_area, self.col_area
        row_rt, col_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        args = [int(node.mesh_id), int(rf.chip_id), int(rb.chip_id), ra.area.buffer_address(), rparity * COLS, c]
        args += [ra.sem_addr[rparity][c], self.core_xy.x, self.core_xy.y, *ra.sem_addr[rparity]]
        row_rt[self.core.x][self.core.y] = args
        args = [value.buffer_address(), ids.buffer_address(), out.buffer_address(), int(node.mesh_id)]
        args += [int(cf.chip_id), int(cb.chip_id), ca.area.buffer_address(), cparity * ROWS, r]
        args += [ca.sem_addr[cparity][r], self.core_xy.x, self.core_xy.y, *ca.sem_addr[cparity]]
        col_rt[self.core.x][self.core.y] = args
        common = [CB_SCR, self.rows, self.page, SEM_P, SEM_R, self.off_r, self.off_in]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=KDIR + "head_pick_row.cpp",
                core_ranges=self.core_set,
                compile_time_args=common + ttnn.TensorAccessorArgs(ra.area).get_compile_time_args(),
                runtime_args=row_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KDIR + "head_pick_col.cpp",
                core_ranges=self.core_set,
                compile_time_args=common
                + [self.off_tiles]
                + ttnn.TensorAccessorArgs(value).get_compile_time_args()
                + ttnn.TensorAccessorArgs(ids).get_compile_time_args()
                + ttnn.TensorAccessorArgs(out).get_compile_time_args()
                + ttnn.TensorAccessorArgs(ca.area).get_compile_time_args(),
                runtime_args=col_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
        ]
        cbs = [
            ttnn.CBDescriptor(
                total_size=self.scr_bytes,
                core_ranges=self.core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_SCR, data_format=ttnn.float32, page_size=self.scr_bytes)
                ],
            )
        ]
        sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=self.core_set, initial_value=0) for s in (SEM_P, SEM_R)]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        row_args = program.kernels[0].runtime_args[self.core.x][self.core.y]
        for peer in (rf, rb):
            row_args.extend(ttnn.setup_fabric_connection(node, peer, 0, program, self.core))
        col_args = program.kernels[1].runtime_args[self.core.x][self.core.y]
        for peer in (cf, cb):
            col_args.extend(ttnn.setup_fabric_connection(node, peer, 0, program, self.core))
        return program

    def __call__(self, value: ttnn.Tensor, ids: ttnn.Tensor) -> ttnn.Tensor:
        for t in (value, ids):
            assert t.dtype == ttnn.float32 and t.layout == ttnn.TILE_LAYOUT, t
            assert tuple(t.shape)[-2:] == (self.rows, 1), (t.shape, self.rows)
        rparity, cparity = self.row_area.next_parity(), self.col_area.next_parity()
        out = ttnn.allocate_tensor_on_device(
            ttnn.TensorSpec(
                ttnn.Shape([1, 1, self.rows, 1]), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM
            ),
            self.mesh,
        )
        mesh_program = ttnn.MeshProgramDescriptor()
        for r in range(ROWS):
            for c in range(COLS):
                coord = ttnn.MeshCoordinate(r, c)
                mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = self._program(
                    r, c, value, ids, out, rparity, cparity
                )
        ttnn.generic_op([value, ids, out], mesh_program)
        return out


_STATE: dict = {}


def _state(mesh, rows: int) -> HeadPick:
    key = (id(mesh), rows)
    if key not in _STATE or _STATE[key].mesh is not mesh:
        _STATE[key] = HeadPick(mesh, rows)
    return _STATE[key]


def prealloc(mesh) -> None:
    """The landing areas and semaphores now, before any program or trace (as row_all_reduce.prealloc)."""
    oneshot_area(mesh, "head_row", COLS, 2 * COLS, PAGE)
    oneshot_area(mesh, "head_col", ROWS, 2 * ROWS, PAGE)
    from .head import HEAD_PICK_MAX_ROWS  # (read at the call)

    if HEAD_PICK_MAX_ROWS > PAGE // 8:  # (the big pages' areas too)
        oneshot_area(mesh, "head_row", COLS, 2 * COLS, BIG_PAGE)
        oneshot_area(mesh, "head_col", ROWS, 2 * ROWS, BIG_PAGE)


def head_pick(mesh, value: ttnn.Tensor, ids: ttnn.Tensor) -> ttnn.Tensor:
    """value, ids [1, 1, M, 1] fp32 TILE (each chip's top logit and its global id per row of tokens) -> the mesh's top's
    ids [1, 1, M, 1] fp32, the same on every chip (of equal values the smallest id)."""
    return _state(mesh, value.shape[-2])(value, ids)


__all__ = ["HeadPick", "head_pick", "prealloc"]
