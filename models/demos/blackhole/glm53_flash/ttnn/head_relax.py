# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The looser acceptance inside the greedy head's pick over the mesh (GLM53_ACCEPT_RELAX; `TtLmHead.greedy_mesh`'s
relax with RELAX_FUSED): head_pick's one-shot op (`head_pick.HeadPick`: one core a chip, the mesh row's then the
column's fold over the 2D fabric) with a third word a row of tokens, in place of the ttnn ops of `TtLmHead._relaxed`
(the 0/1 pick of the next token's logit over the chip's whole slice of the vocabulary, two all-gathers, the maxima and
the compare: +0.1-0.6 ms a step).

  head_pick_relax(value, ids, part, tokens, delta, users) -> (relaxed ids, argmax ids), [1, 1, M, 1] fp32 each:
    value, ids  head_argmax's per-chip top and its global id (M = 8 b rows of tokens: row r's b at r b);
    part        the head's product [1, 1, M, w] fp32 (this chip's ids k w ..);
    tokens      the verify step's tokens, every mesh row's [1, 1, 8, b] uint32 row-major (virtual user j U + u: user
                u's token j): row m = r b + v's next token is row r's v + U, none for v >= b - U;
    delta       the gap scalar [1, 1, 1, 1] fp32 (`TtLmHead.relax`).
The NCRISC reads, where this chip holds a row's next token, its logit (one 64-byte block of `part`) into the payload's
third word (-1e30 elsewhere, as the ttnn form's masked max); the folds keep its maximum; at the end every chip has the
mesh's top value, its id and the next token's logit, and writes the next token as the row's id where the top's logit
less the next one's is within the gap (the same fp32 subtraction and compare as the ttnn form's, on the RISC-V's IEEE
floats), else the argmax; and the argmax ids.  Its own landing areas ("head_row3" / "head_col3", pages of 12 bytes a
row), made by `prealloc` before any capture.  Up to 256 rows (head_pick's limit; past it the ttnn form).
"""

from __future__ import annotations

import ttnn

from .common import COLS, ROWS
from .row_all_reduce import oneshot_area

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/head_pick/"
MAX_ROWS = 256
PAGE = MAX_ROWS * 12  # a landing page: (value, id, the next token's logit) a row
CB_SCR = 0
SEM_P, SEM_R = 0, 1


def _round(x: int, a: int = 64) -> int:
    return -(-x // a) * a


class HeadPickRelax:
    def __init__(self, mesh, rows: int, users: int, per_row: int, width: int, valid: list, id_width: int):
        assert tuple(mesh.shape) == (ROWS, COLS) and ROWS == 8 and COLS == 4, mesh.shape
        assert 1 <= rows <= MAX_ROWS and rows == ROWS * per_row and per_row % users == 0, (rows, per_row, users)
        self.mesh, self.rows, self.users, self.per_row = mesh, rows, users, per_row
        self.wt, self.valid, self.id_width = width // 32, valid, id_width
        self.trows = -(-rows // 32)
        self.core = ttnn.CoreCoord(0, 0)
        self.core_set = ttnn.CoreRangeSet([ttnn.CoreRange(self.core, self.core)])
        self.core_xy = mesh.worker_core_from_logical_core(self.core)
        self.row_area = oneshot_area(mesh, "head_row3", COLS, 2 * COLS, PAGE)
        self.col_area = oneshot_area(mesh, "head_col3", ROWS, 2 * ROWS, PAGE)
        self.nodes = {
            (r, c): mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for r in range(ROWS) for c in range(COLS)
        }
        # the scratch: P, R, a landed payload, the tiles (value and id, then the outputs'), the tokens, the blocks of
        # the product, the gap's 64 bytes with the next ids and the offsets
        self.off_r = PAGE
        self.off_in = 2 * PAGE
        self.off_tiles = _round(3 * PAGE)
        self.tok_stride = _round(per_row * 4)
        self.off_tok = self.off_tiles + 2 * self.trows * 4096
        self.off_part = _round(self.off_tok + ROWS * self.tok_stride)
        self.off_aux = self.off_part + rows * 64
        self.scr_bytes = self.off_aux + 64 + 2 * rows * 4

    def _program(self, r, c, value, ids, out, raw, part, tokens, delta, rparity, cparity) -> ttnn.ProgramDescriptor:
        node = self.nodes[(r, c)]
        rf, rb = self.nodes[(r, (c + 1) % COLS)], self.nodes[(r, (c - 1) % COLS)]
        cf, cb = self.nodes[((r + 1) % ROWS, c)], self.nodes[((r - 1) % ROWS, c)]
        ra, ca = self.row_area, self.col_area
        k = COLS * r + c
        row_rt, col_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        args = [int(node.mesh_id), int(rf.chip_id), int(rb.chip_id), ra.area.buffer_address(), rparity * COLS, c]
        args += [ra.sem_addr[rparity][c], self.core_xy.x, self.core_xy.y, *ra.sem_addr[rparity]]
        row_rt[self.core.x][self.core.y] = args
        args = [value.buffer_address(), ids.buffer_address(), out.buffer_address(), raw.buffer_address()]
        args += [part.buffer_address(), tokens.buffer_address(), delta.buffer_address()]
        args += [k * self.id_width, self.valid[k], int(node.mesh_id), int(cf.chip_id), int(cb.chip_id)]
        args += [ca.area.buffer_address(), cparity * ROWS, r, ca.sem_addr[cparity][r], self.core_xy.x, self.core_xy.y]
        args += [*ca.sem_addr[cparity]]
        col_rt[self.core.x][self.core.y] = args
        common = [CB_SCR, self.rows, PAGE, SEM_P, SEM_R, self.off_r, self.off_in]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=KDIR + "relax_row.cpp",
                core_ranges=self.core_set,
                compile_time_args=common + ttnn.TensorAccessorArgs(ra.area).get_compile_time_args(),
                runtime_args=row_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KDIR + "relax_col.cpp",
                core_ranges=self.core_set,
                compile_time_args=common
                + [self.off_tiles, self.off_tok, self.off_part, self.off_aux, self.users, self.per_row, self.wt]
                + [self.tok_stride]
                + ttnn.TensorAccessorArgs(value).get_compile_time_args()
                + ttnn.TensorAccessorArgs(ids).get_compile_time_args()
                + ttnn.TensorAccessorArgs(out).get_compile_time_args()
                + ttnn.TensorAccessorArgs(raw).get_compile_time_args()
                + ttnn.TensorAccessorArgs(part).get_compile_time_args()
                + ttnn.TensorAccessorArgs(tokens).get_compile_time_args()
                + ttnn.TensorAccessorArgs(delta).get_compile_time_args()
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

    def __call__(self, value, ids, part, tokens, delta):
        for t in (value, ids):
            assert t.dtype == ttnn.float32 and t.layout == ttnn.TILE_LAYOUT, t
            assert tuple(t.shape)[-2:] == (self.rows, 1), (t.shape, self.rows)
        assert part.dtype == ttnn.float32 and tuple(part.shape)[-2:] == (self.rows, self.wt * 32), part.shape
        assert tokens.dtype == ttnn.uint32 and tokens.layout == ttnn.ROW_MAJOR_LAYOUT, tokens
        assert tuple(tokens.shape)[-2:] == (ROWS, self.per_row), tokens.shape
        rparity, cparity = self.row_area.next_parity(), self.col_area.next_parity()
        spec = ttnn.TensorSpec(
            ttnn.Shape([1, 1, self.rows, 1]), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM
        )
        out = ttnn.allocate_tensor_on_device(spec, self.mesh)
        raw = ttnn.allocate_tensor_on_device(spec, self.mesh)
        mesh_program = ttnn.MeshProgramDescriptor()
        for r in range(ROWS):
            for c in range(COLS):
                coord = ttnn.MeshCoordinate(r, c)
                mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = self._program(
                    r, c, value, ids, out, raw, part, tokens, delta, rparity, cparity
                )
        ttnn.generic_op([value, ids, out, raw, part, tokens, delta], mesh_program)
        return out, raw


_STATE: dict = {}


def prealloc(mesh) -> None:
    """The landing areas and semaphores now, before any program or trace (as head_pick.prealloc)."""
    oneshot_area(mesh, "head_row3", COLS, 2 * COLS, PAGE)
    oneshot_area(mesh, "head_col3", ROWS, 2 * ROWS, PAGE)


def head_pick_relax(mesh, value, ids, part, tokens, delta, users: int, valid: list, id_width: int):
    """See the module's docstring -> (relaxed ids, argmax ids), [1, 1, M, 1] fp32 each, the same on every chip."""
    rows, per_row, width = value.shape[-2], tokens.shape[-1], part.shape[-1]
    key = (id(mesh), rows, users, per_row, width)
    if key not in _STATE or _STATE[key].mesh is not mesh:
        _STATE[key] = HeadPickRelax(mesh, rows, users, per_row, width, valid, id_width)
    return _STATE[key](value, ids, part, tokens, delta)


__all__ = ["HeadPickRelax", "head_pick_relax", "prealloc", "MAX_ROWS"]
