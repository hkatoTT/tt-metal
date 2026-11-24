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

import os

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
# v2 (the default, GLM53_HEAD_PICK_V2=0 for v1; kernels head_pick2_*.cpp): the same fabric plumbing and landing pages,
# with the RISC-V work cut - 64-bit keys (one compare a payload a row), the landed payloads read into slots and folded
# in one pass a row, both RISC-Vs folding half the rows of each stage, and the input's id tiles (head_argmax's: column 0
# the ids, the rest zero) as the output's with column 0 replaced, no tile zeroed.  The same ids (the keys' maximum is
# v1's rule; on 32 chips at 64 to 1,024 rows, 200 calls in a row).  Its local semaphores: P built, the row's slots in,
# the row kernel's half of R, the column's slots in, the row kernel's half of the ids.
SEM_P2, SEM_RS2, SEM_RB2, SEM_CS2, SEM_DB2 = 0, 1, 2, 3, 4
# v3, the own mesh row's tokens only (`own`, GLM53_HEAD_PICK_OWN=1 with the DFlash2 step's fused acceptance,
# GLM53_DFLASH2_ACCEPT_FUSED=1; kernels head_pick3_*.cpp): v2's keys, pages and parities, the stages swapped (the
# column's first: P to the column, each chip folds its own mesh row's t = M / 8 rows of the 8; then the row's: those t
# rows' best to the row's 3 others, folded) - 10 t folds a chip in place of 10 M.  The own rows'
# ids are v2's (the same rule, the same keys); the output's other rows are left undefined.  Its local semaphores: the
# column's slots in, the column kernel's half of Rc, the row's slots in, the row kernel's half of the ids.
SEM_CS3, SEM_RC3, SEM_RS3, SEM_DB3 = 0, 1, 2, 3


def v2_on() -> bool:
    """GLM53_HEAD_PICK_V2 (read at each call; the default 1: v2, 0: v1)."""
    return os.environ.get("GLM53_HEAD_PICK_V2", "1") == "1"


# v3 by default at these (users a mesh row, K) of the DFlash2 step: 128 users K = 3, with the fused selector
# (dflash2.SEL_FUSED_SHAPES): the DFlash2 step on the real weights at the preset C 39.865 -> 39.685 ms
OWN_SHAPES = {(16, 3)}


def own_on(shape: tuple | None = None) -> bool:
    """GLM53_HEAD_PICK_OWN (read at each call): 1 / 0 on / off; unset, on at the DFlash2 step's shapes of OWN_SHAPES
    ((users a mesh row, K): `shape`): v3, the own mesh row's ids only, where the caller takes it (the DFlash2 step with
    its fused acceptance: `dflash2_step`)."""
    v = os.environ.get("GLM53_HEAD_PICK_OWN", "")
    return v == "1" if v else shape in OWN_SHAPES


class HeadPick:
    def __init__(self, mesh, rows: int, v2: bool = False, own: bool = False):
        assert tuple(mesh.shape) == (ROWS, COLS) and ROWS == 8 and COLS == 4, mesh.shape
        assert 1 <= rows <= MAX_ROWS and rows % 8 == 0, rows
        assert not own or rows % 64 == 0, rows  # (v3: 8 rows a mesh row at least, its slices 64-byte reads)
        self.mesh, self.rows, self.v2, self.own = mesh, rows, v2 or own, own
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
        if own:  # P (all M rows), Rc (t rows), the row's 4 slots and the column's 8 of t rows, the value and id tiles
            self.slot = rows // ROWS * 8
            self.off_r = self.page
            self.off_rs = self.off_r + self.slot
            self.off_cs = self.off_rs + COLS * self.slot
            self.off_tiles = self.off_cs + ROWS * self.slot
        elif v2:  # P, R, the row's 4 slots, the column's 8 slots (a chip's own unused), the value and id tiles
            self.off_rs = 2 * self.page
            self.off_cs = self.off_rs + COLS * self.page
            self.off_tiles = self.off_cs + ROWS * self.page
        self.scr_bytes = self.off_tiles + 2 * self.trows * 4096

    def _program(self, r: int, c: int, value, ids, out, rparity: int, cparity: int) -> ttnn.ProgramDescriptor:
        node = self.nodes[(r, c)]
        rf, rb = self.nodes[(r, (c + 1) % COLS)], self.nodes[(r, (c - 1) % COLS)]
        cf, cb = self.nodes[((r + 1) % ROWS, c)], self.nodes[((r - 1) % ROWS, c)]
        ra, ca = self.row_area, self.col_area
        row_rt, col_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        args = [int(node.mesh_id), int(rf.chip_id), int(rb.chip_id), ra.area.buffer_address(), rparity * COLS, c]
        args += [ra.sem_addr[rparity][c], self.core_xy.x, self.core_xy.y]
        args += [r] if self.v2 else []  # (v2: the mesh row, the column stage's own slot)
        args += [*ra.sem_addr[rparity]]
        row_rt[self.core.x][self.core.y] = args
        args = [value.buffer_address(), ids.buffer_address(), out.buffer_address(), int(node.mesh_id)]
        args += [int(cf.chip_id), int(cb.chip_id), ca.area.buffer_address(), cparity * ROWS, r]
        args += [ca.sem_addr[cparity][r], self.core_xy.x, self.core_xy.y]
        args += [c] if self.v2 else []  # (v2: the mesh column, the row stage's own slot)
        args += [*ca.sem_addr[cparity]]
        col_rt[self.core.x][self.core.y] = args
        if self.v2:
            return self._program_v2(node, (rf, rb), (cf, cb), value, ids, out, row_rt, col_rt)
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

    def _program_v2(self, node, row_peers, col_peers, value, ids, out, row_rt, col_rt) -> ttnn.ProgramDescriptor:
        ra, ca = self.row_area, self.col_area
        if self.own:
            sems_ids = (SEM_CS3, SEM_RC3, SEM_RS3, SEM_DB3)
            common = [CB_SCR, self.rows, self.page, *sems_ids, self.off_r, self.off_rs, self.off_cs, self.off_tiles]
            common += [self.slot]
            names = ("head_pick3_row.cpp", "head_pick3_col.cpp")
        else:
            sems_ids = (SEM_P2, SEM_RS2, SEM_RB2, SEM_CS2, SEM_DB2)
            common = [CB_SCR, self.rows, self.page, *sems_ids, self.off_r, self.off_rs, self.off_cs, self.off_tiles]
            names = ("head_pick2_row.cpp", "head_pick2_col.cpp")
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=KDIR + names[0],
                core_ranges=self.core_set,
                compile_time_args=common + ttnn.TensorAccessorArgs(ra.area).get_compile_time_args(),
                runtime_args=row_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KDIR + names[1],
                core_ranges=self.core_set,
                compile_time_args=common
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
        sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=self.core_set, initial_value=0) for s in sems_ids]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        row_args = program.kernels[0].runtime_args[self.core.x][self.core.y]
        for peer in row_peers:
            row_args.extend(ttnn.setup_fabric_connection(node, peer, 0, program, self.core))
        col_args = program.kernels[1].runtime_args[self.core.x][self.core.y]
        for peer in col_peers:
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


def _state(mesh, rows: int, v2: bool = False, own: bool = False) -> HeadPick:
    key = (id(mesh), rows, v2, own)
    if key not in _STATE or _STATE[key].mesh is not mesh:
        _STATE[key] = HeadPick(mesh, rows, v2, own)
    return _STATE[key]


def prealloc(mesh) -> None:
    """The landing areas and semaphores now, before any program or trace (as row_all_reduce.prealloc)."""
    oneshot_area(mesh, "head_row", COLS, 2 * COLS, PAGE)
    oneshot_area(mesh, "head_col", ROWS, 2 * ROWS, PAGE)
    from .head import HEAD_PICK_MAX_ROWS, head_pick_max_rows  # (read at the call)

    if max(HEAD_PICK_MAX_ROWS, head_pick_max_rows()) > PAGE // 8:  # (the big pages' areas too)
        oneshot_area(mesh, "head_row", COLS, 2 * COLS, BIG_PAGE)
        oneshot_area(mesh, "head_col", ROWS, 2 * ROWS, BIG_PAGE)


def head_pick(mesh, value: ttnn.Tensor, ids: ttnn.Tensor, v2: bool | None = None, own: bool = False) -> ttnn.Tensor:
    """value, ids [1, 1, M, 1] fp32 TILE (each chip's top logit and its global id per row of tokens; v2 takes the ids'
    tiles as head_argmax writes them: column 0, the rest zero) -> the mesh's top's ids [1, 1, M, 1] fp32, the same on
    every chip (of equal values the smallest id).  v2: None reads GLM53_HEAD_PICK_V2.  own (v3, M a multiple of 64):
    each chip's own mesh row's M / 8 rows only (rows r M / 8 .. on mesh row r; the other rows undefined)."""
    if own:
        return _state(mesh, value.shape[-2], True, True)(value, ids)
    return _state(mesh, value.shape[-2], v2_on() if v2 is None else v2)(value, ids)


__all__ = ["HeadPick", "head_pick", "own_on", "prealloc"]
