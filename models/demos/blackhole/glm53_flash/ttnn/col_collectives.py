# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The MoE's collectives over a mesh column (8 chips, ROW_AXIS) for a few tokens a row (t <= 4), one-shot over the 2D
fabric, through ttnn.generic_op (the plumbing as in `row_all_reduce.py`):

  col_gather(mesh, x):  x [1, 1, t, D] (a row's tokens, the same on its chips; t <= 16) -> [1, 1, N, D] (N = 8 t up to
                        whole tiles): the column's 8 rows' tokens, row r's at r t .. r t + t - 1, zero rows after (as
                        `TtMoe._gather_rows`); above 4 tokens with 32 workers (4 tiles, a 4 KB slice of 16 rows each);
  col_scatter(mesh, p, t): p [1, 1, 32, D] (each chip's partial sums for those 8 t tokens) -> [1, 1, t, D]: this row's
                        tokens summed over the column's chips (fp32 adds in the order of the chips, bf16);
  col_scatter_padded(mesh, p, t): the same from p [1, 1, 8 x 32, D] whose mesh row r's t tokens (t even, <= 16) are
                        rows 0 .. t - 1 of its tile row r (the experts op's `row_tokens` y): 32 workers, each
                        tile's face 0 and 1 rows 0 .. t - 1 read straight from DRAM (no untilize: the sums do not care
                        how the values are laid out), all staged by the reader, and put back into faces 0 and 1 of the
                        output tiles.

  col_scatter_ar(mesh, p, t): either scatter (by p's height: [1, 1, 32, D] at t <= 4, the padded [1, 1, 8 x 32, D] at
                        t = 8) and then the all-reduce over the mesh row's 4 chips, in one op (`ar`): each worker's
                        summed slice (bf16, as the scatter writes it) goes on, as `row_all_reduce` sends its slices
                        (its sender kernel on a second sender core, with connections to the row's ring neighbours,
                        and its landing area and receive semaphores), to the row's other chips, and the worker adds
                        the row's four slices as `row_all_reduce` does (FPU pairs, then an SFPU add, in the order of
                        the chips) before it writes the output: the same sums as col_scatter(_padded) and then
                        row_all_reduce, without the second op, its DRAM round trip and (t = 8) all_reduce_async's
                        reshards.

Both bf16 TILE DRAM interleaved in and out.  Per chip `workers` cores (8) take D / 32 / workers tiles (columns) each:
a worker's reader reads its tiles of one tile row and its compute untilizes them (row-major rows of its columns; the
gather at t = 1 takes its row from the tiles by 64-byte reads and local copies instead); the worker copies the rows
each chip needs to the sender core's stage (the scatter: chips 0 .. 3 by the reader on NoC 1, 4 .. 7 by the writer on
NoC 0).  The sender (BRISC on link 0, NCRISC on link 1, a connection to each ring neighbour's router, half the groups
each) sends each group of workers (as many as fit a 4 KB packet: 4 at t = 1) as soon as their slices are there, in one
packet to the column's other chips (gather: the same for all, a 4-hop and a 3-hop multicast; scatter: one for each
chip, 7 unicasts), into their DRAM landing area (`row_all_reduce.OneshotArea`: persistent DRAM, the same address on
every chip, a page for each parity, source chip and group), each packet a write fused with an increment (once the
write is acknowledged) of the group leader's receive semaphore for this chip.  The leader waits for each chip in turn
and tells its members, and each reads its slice of that page from DRAM.  The worker's compute then tilizes the
gathered rows (32 rows: the 8 chips' rows in place and zero rows after) or sums the 8 slices (scatter; the writer and
reader put the rows into staging tiles, a circular buffer they zero while the slices travel); its writer and reader
write the tiles, half each.  The landing pages and semaphores alternate between two parities from call to call (the
gather and the scatter share them, and the turn), as in the row all-reduce.  The only persistent L1 is the receive
semaphores (a word a core each).
"""

from __future__ import annotations

import math

import ttnn

from . import w1_dense
from .common import COLS, ROWS, core_rects
from .row_all_reduce import oneshot_area

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/col_ar/"
ROW_KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/row_ar/"
PAGE = 2048  # a bf16 tile
CB_STAGE, CB_IN, CB_RM, CB_BUF, CB_OUT, CB_STAGING = 0, 1, 2, 3, 4, 5
CB_RSTAGE, CB_RBUF, CB_OUT2 = 6, 7, 8  # `ar`: the row sender's stage, the row's 4 slots, the row's sums
SEM_GATHER, SEM_ZERO, SEM_READY = 0, 1, 2
SEM_RGATHER = 3  # `ar`: the row sender's stage semaphore (a bit a worker)
SLOT = 4096  # a landing page: a packet
GATHER, SCATTER, SCATTER_PADDED = 0, 1, 2


class ColCollective:
    """One of the two collectives for a mesh and width: its plumbing, and the persistent state (`OneshotArea`: DRAM,
    and a few semaphores in L1) shared by the two."""

    def __init__(
        self,
        mesh,
        d: int,
        mode: int,
        workers: int | None = None,
        links: int | None = None,
        direct: bool = True,
        split_stage: int = 4,
        ar: bool = False,
    ):
        assert tuple(mesh.shape) == (ROWS, COLS) and ROWS == 8, mesh.shape
        assert not ar or mode in (SCATTER, SCATTER_PADDED), mode
        self.mesh, self.d, self.mode, self.ar = mesh, d, mode, ar
        self.tiles = d // 32
        self.workers = workers or 8
        self.links = links or 2
        assert self.tiles % self.workers == 0, (self.tiles, self.workers)
        # gather, t = 1: the row from the tiles by 64-byte reads and local copies, not untilized (32-byte DRAM reads to
        # the row's places were wrong: a DRAM read to L1 not 64-byte aligned like its source lands wrong)
        self.direct = direct
        # scatter: the reader stages chips 0 .. split_stage - 1, the writer the rest (the padded scatter: the reader all,
        # a worker one bit of the stage's semaphore, so that 32 workers fit its word)
        self.split_stage = ROWS if mode == SCATTER_PADDED else split_stage
        self.tpw = self.tiles // self.workers
        self.row_bytes = self.tpw * 64
        grid = mesh.compute_with_storage_grid_size()
        per_row = min(self.workers, grid.x - 1)
        self.worker_cores = [ttnn.CoreCoord(i % per_row, i // per_row) for i in range(self.workers)]
        self.sender = ttnn.CoreCoord(grid.x - 1, 0)
        self.worker_set = core_rects(self.worker_cores)
        self.sender_set = core_rects([self.sender])
        self.all_set = core_rects(self.worker_cores + [self.sender])
        self.worker_xy = [mesh.worker_core_from_logical_core(c) for c in self.worker_cores]
        self.sender_xy = mesh.worker_core_from_logical_core(self.sender)
        self.nodes = {
            (r, c): mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for r in range(ROWS) for c in range(COLS)
        }
        self.max_payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
        self.area = oneshot_area(mesh, "col", ROWS, 2 * ROWS * self.workers, SLOT)
        if ar:  # the row stage: row_all_reduce's landing area and semaphores for this many workers (its turn too)
            seg_max = min(512, 4096 // (2 * self.tpw))  # (as RowAllReduce sizes its slot)
            self.rslot = math.ceil(self.tpw * 2 * seg_max / PAGE) * PAGE
            self.rarea = oneshot_area(mesh, "row", COLS, 2 * COLS * self.workers, self.rslot)
            self.rsender = ttnn.CoreCoord(grid.x - 1, 1)
            self.rsender_xy = mesh.worker_core_from_logical_core(self.rsender)
            self.rsender_set = core_rects([self.rsender])
            self.all_set = core_rects(self.worker_cores + [self.sender, self.rsender])
            self.rstage_set = core_rects(self.worker_cores + [self.rsender])
            self.rmcast = True  # (row_all_reduce's in-op multicast)

    def group(self, t: int) -> int:
        """Workers a packet: the most (dividing their number) whose slices fit a 4 KB slot."""
        slice_bytes = t * self.row_bytes
        g = self.workers
        while g > 1 and (g * slice_bytes > min(SLOT, self.max_payload) or self.workers % g):
            g //= 2
        assert g * slice_bytes <= min(SLOT, self.max_payload), (t, self.row_bytes, self.max_payload)
        return g

    def _program(
        self, r: int, c: int, x, out, t: int, parity: int, rparity: int = 0, riders=()
    ) -> ttnn.ProgramDescriptor:
        slice_bytes = t * self.row_bytes
        trows = out.shape[-2] // 32 if self.mode == GATHER else 1  # the gather's output tile rows
        scatter = self.mode != GATHER
        pages = math.ceil(slice_bytes / PAGE)
        spp = SLOT // PAGE  # the scatter's slots in cb_buf
        group = self.group(t)
        split = self.tpw // 2
        direct = int(self.mode == GATHER and self.direct and t == 1)
        out_pages = self.tpw * trows if self.mode == GATHER else pages
        buf_pages = self.tpw * trows if self.mode == GATHER else ROWS * spp
        sxy = self.sender_xy
        area = self.area.area.buffer_address()
        reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for w, core in enumerate(self.worker_cores):
            lead = (w // group) * group
            members = [] if w != lead else [m for m in range(lead + 1, lead + group)]
            args = [x.buffer_address(), w * self.tpw, self.tpw, parity, r, sxy.x, sxy.y, w, self.workers]
            args += [out.buffer_address(), split, area, *self.area.sem_addr[parity], lead, w - lead, len(members)]
            for m in members:
                args += [self.worker_xy[m].x, self.worker_xy[m].y]
            if self.ar:  # the row stage: this chip's place in the row, the row sender, the parity, area, semaphores
                args += [c, self.rsender_xy.x, self.rsender_xy.y, rparity, self.rarea.area.buffer_address()]
                args += self.rarea.sem_addr[rparity]
            reader_rt[core.x][core.y] = args
            compute_rt[core.x][core.y] = []
            args = [out.buffer_address(), w * self.tpw, self.tpw, split, r, sxy.x, sxy.y, w, self.workers]
            writer_rt[core.x][core.y] = args
        node = self.nodes[(r, c)]
        fwd = self.nodes[((r + 1) % ROWS, c)]
        bwd = self.nodes[((r - 1) % ROWS, c)]
        sender_rts = []
        for link in range(self.links):
            rt = ttnn.RuntimeArgs()
            args = [int(node.mesh_id), int(fwd.chip_id), int(bwd.chip_id), link, self.links, r, area]
            args += [(parity * ROWS + r) * self.workers, self.area.sem_addr[parity][r], ROWS // 2, ROWS - 1 - ROWS // 2]
            args += [int(self.nodes[(rr, c)].chip_id) for rr in range(ROWS)]
            for lead in range(0, self.workers, group):
                args += [self.worker_xy[lead].x, self.worker_xy[lead].y]
            rt[self.sender.x][self.sender.y] = args
            sender_rts.append(rt)
        rsender_rts = []
        if self.ar:  # row_ar_sender.cpp's arguments, as RowAllReduce gives them
            rfwd, rbwd = self.nodes[(r, (c + 1) % COLS)], self.nodes[(r, (c - 1) % COLS)]
            ropp = self.nodes[(r, (c + 2) % COLS)]
            rarea = self.rarea.area.buffer_address()
            for link in range(self.links):
                rt = ttnn.RuntimeArgs()
                args = [slice_bytes, rarea, self.rarea.sem_addr[rparity][c], int(node.mesh_id), int(rfwd.chip_id)]
                args += [int(rbwd.chip_id), int(ropp.chip_id), link, self.links, (rparity * COLS + c) * self.workers]
                for xy in self.worker_xy:
                    args += [xy.x, xy.y]
                rt[self.rsender.x][self.rsender.y] = args
                rsender_rts.append(rt)
        ar = int(self.ar)
        rnpt = self.rslot // PAGE if self.ar else 1
        rarea_t = self.rarea.area if self.ar else self.area.area

        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=scatter, dst_full_sync_en=True
        )
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=KDIR + "col_reader.cpp",
                core_ranges=self.worker_set,
                compile_time_args=[self.mode, CB_IN, CB_RM, CB_STAGE, CB_OUT, SEM_GATHER, t, CB_BUF, SEM_READY]
                + [direct, self.split_stage, out_pages, SEM_ZERO, buf_pages, CB_STAGING, trows, self.tiles]
                + [ar, CB_RBUF, CB_RSTAGE, CB_OUT2, SEM_RGATHER, rnpt]
                + ttnn.TensorAccessorArgs(x).get_compile_time_args()
                + ttnn.TensorAccessorArgs(out).get_compile_time_args()
                + ttnn.TensorAccessorArgs(self.area.area).get_compile_time_args()
                + ttnn.TensorAccessorArgs(rarea_t).get_compile_time_args(),
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KDIR + "col_compute.cpp",
                core_ranges=self.worker_set,
                compile_time_args=[self.mode, CB_IN, CB_RM, CB_BUF, CB_OUT, self.tpw, buf_pages, spp, pages, direct]
                + [trows, ar, CB_RBUF, CB_OUT2, rnpt],
                runtime_args=compute_rt,
                config=compute_cfg,
            ),
            ttnn.KernelDescriptor(
                kernel_source=KDIR + "col_writer.cpp",
                core_ranges=self.worker_set,
                compile_time_args=[self.mode, CB_RM, CB_STAGE, CB_BUF, CB_OUT, SEM_GATHER, t, self.split_stage]
                + [out_pages, SEM_ZERO, CB_STAGING, trows, self.tiles, CB_OUT2 if self.ar else CB_OUT]
                + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
        ]
        for link in range(self.links):  # link 0 on BRISC, link 1 on NCRISC
            # the bits of the stage's semaphore a worker sets: the scatter's reader and writer each one, else one
            bits = 2 if self.mode == SCATTER else 1
            ct = [self.mode, CB_STAGE, self.workers, SEM_GATHER, slice_bytes, bits, group]
            ct += ttnn.TensorAccessorArgs(self.area.area).get_compile_time_args()
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=KDIR + "col_sender.cpp",
                    core_ranges=self.sender_set,
                    compile_time_args=ct,
                    runtime_args=sender_rts[link],
                    config=ttnn.WriterConfigDescriptor() if link == 0 else ttnn.ReaderConfigDescriptor(),
                )
            )
        for link in range(self.links if self.ar else 0):  # the row sender: link 0 on BRISC, link 1 on NCRISC
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=ROW_KDIR + "row_ar_sender.cpp",
                    core_ranges=self.rsender_set,
                    compile_time_args=[CB_RSTAGE, self.workers, SEM_RGATHER, self.rslot, int(self.rmcast)]
                    + ttnn.TensorAccessorArgs(self.rarea.area).get_compile_time_args(),
                    runtime_args=rsender_rts[link],
                    config=ttnn.WriterConfigDescriptor() if link == 0 else ttnn.ReaderConfigDescriptor(),
                )
            )

        def cb(index, size, cores, page=PAGE):
            return ttnn.CBDescriptor(
                total_size=size,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=page)
                ],
            )

        # the stage first, on the sender core and the workers: the same address on all of them (the workers write to
        # the sender's)
        stage_slots = self.workers * (ROWS if scatter else 1)
        cbs = [
            cb(CB_STAGE, stage_slots * slice_bytes, self.all_set, slice_bytes),
            cb(CB_IN, self.tpw * PAGE, self.worker_set),
            # the untilized tile row, or (the padded scatter) the 8 chips' slices of face rows
            cb(CB_RM, ROWS * slice_bytes if self.mode == SCATTER_PADDED else self.tpw * PAGE, self.worker_set),
            cb(CB_BUF, buf_pages * PAGE, self.worker_set),
            cb(CB_OUT, out_pages * PAGE, self.worker_set),
        ]
        if scatter:
            cbs.append(cb(CB_STAGING, self.tpw * PAGE, self.worker_set))
        if self.ar:  # (the row stage: the same address on the workers and the row sender, as CB_STAGE)
            cbs.append(cb(CB_RSTAGE, self.workers * self.rslot, self.rstage_set, self.rslot))
            cbs.append(cb(CB_RBUF, COLS * self.rslot, self.worker_set))
            cbs.append(cb(CB_OUT2, pages * PAGE, self.worker_set))
        sems = [
            ttnn.SemaphoreDescriptor(id=sem, core_ranges=self.all_set, initial_value=0)
            for sem in (SEM_GATHER, SEM_ZERO, SEM_READY) + ((SEM_RGATHER,) if self.ar else ())
        ]
        # (GLM53_W1D's riders last: the kernels above are taken by their place below)
        program = ttnn.ProgramDescriptor(kernels=kernels + list(riders), semaphores=sems, cbs=cbs)
        for link in range(self.links):
            sender_args = program.kernels[3 + link].runtime_args[self.sender.x][self.sender.y]
            for peer in (fwd, bwd):
                sender_args.extend(ttnn.setup_fabric_connection(node, peer, link, program, self.sender))
        for link in range(self.links if self.ar else 0):
            rsender_args = program.kernels[3 + self.links + link].runtime_args[self.rsender.x][self.rsender.y]
            for peer in (self.nodes[(r, (c + 1) % COLS)], self.nodes[(r, (c - 1) % COLS)]):
                rsender_args.extend(ttnn.setup_fabric_connection(node, peer, link, program, self.rsender))
        return program

    def __call__(self, x: ttnn.Tensor, t: int) -> ttnn.Tensor:
        assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT and x.shape[-1] == self.d, x
        assert 1 <= t and t * self.row_bytes <= SLOT, (t, self.row_bytes)  # a worker's slice in one packet
        parity = self.area.next_parity()
        # the row stage: the row family's turn (as a row_all_reduce call would take it)
        rparity = self.rarea.next_parity() if self.ar else 0
        assert not self.ar or t * self.row_bytes <= min(self.rslot, self.max_payload), (t, self.rslot)
        if self.mode == SCATTER_PADDED:  # (a DRAM read of fewer than 64 bytes lands right only 64-byte aligned)
            assert t % 2 == 0 and t <= 16 and x.shape[-2] == ROWS * 32, (t, x.shape)
        rows_out = -(-ROWS * t // 32) * 32 if self.mode == GATHER else t
        shape = ttnn.Shape([1, 1, rows_out, self.d])
        out = ttnn.allocate_tensor_on_device(
            ttnn.TensorSpec(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM), self.mesh
        )
        # (GLM53_W1D, part of the next KDA layer's x_proj landed in L1: on the cores this program leaves idle)
        riders = w1_dense.take(self.all_set)
        mesh_program = ttnn.MeshProgramDescriptor()
        for r in range(ROWS):
            for c in range(COLS):
                coord = ttnn.MeshCoordinate(r, c)
                mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = self._program(
                    r, c, x, out, t, parity, rparity, riders
                )
        ttnn.generic_op([x, out], mesh_program)
        return out


_STATE: dict = {}


def _state(mesh, d: int, mode: int, workers: int | None = None, ar: bool = False) -> ColCollective:
    key = (id(mesh), d, mode, workers, ar)
    if key not in _STATE or _STATE[key].mesh is not mesh:
        _STATE[key] = ColCollective(mesh, d, mode, workers, ar=ar)
    return _STATE[key]


GATHER_WIDE = 32  # the gather's workers above 4 tokens a row (a slice of up to 16 rows of 4 tiles: one 4 KB packet)


def prealloc(mesh, d: int, ar: bool = False) -> None:
    """The two ops' persistent state (`OneshotArea`) now, before the model builds or captures any program (as
    row_all_reduce.prealloc)."""
    _state(mesh, d, GATHER)
    _state(mesh, d, GATHER, GATHER_WIDE)
    _state(mesh, d, SCATTER)
    _state(mesh, d, SCATTER_PADDED, SCATTER_WIDE)
    if ar:  # col_scatter_ar (their row stage: row_all_reduce's areas for 8 and 32 workers)
        _state(mesh, d, SCATTER, ar=True)
        _state(mesh, d, SCATTER_PADDED, SCATTER_WIDE, ar=True)


def col_gather(mesh, x: ttnn.Tensor, workers: int | None = None) -> ttnn.Tensor:
    """x [1, 1, t <= 16, D] -> [1, 1, N, D] (N = 8 t up to whole tiles): the mesh column's 8 rows' t tokens (row r's
    at r t), zero rows after.  `workers`: other than the defaults (8, or GATHER_WIDE above 4 tokens), e.g. for a
    width whose 8-worker slices do not fit a packet (`prealloc_gather` it when the model loads)."""
    t = x.shape[-2]
    return _state(mesh, x.shape[-1], GATHER, workers or (None if t <= 4 else GATHER_WIDE))(x, t)


def prealloc_gather(mesh, d: int, workers: int) -> None:
    """A gather of width d with `workers` workers: its persistent state now (as `prealloc`)."""
    _state(mesh, d, GATHER, workers)


SCATTER_WIDE = 32  # the padded scatter's workers (16 rows of 4 tiles: one 4 KB packet)


def col_scatter_padded(mesh, partial: ttnn.Tensor, t: int) -> ttnn.Tensor:
    """partial [1, 1, 8 x 32, D] (mesh row r's t tokens in rows 0 .. t - 1 of tile row r; t even, <= 16) -> [1, 1, t, D]:
    this row's tokens summed over the mesh column's 8 chips (fp32 adds in the chips' order, bf16, as `col_scatter`)."""
    return _state(mesh, partial.shape[-1], SCATTER_PADDED, SCATTER_WIDE)(partial, t)


def col_scatter(mesh, partial: ttnn.Tensor, t: int) -> ttnn.Tensor:
    """partial [1, 1, 32, D] (rows r t .. r t + t - 1: mesh row r's tokens) -> [1, 1, t, D]: this row's tokens summed
    over the mesh column's 8 chips (fp32 adds in the chips' order, bf16)."""
    return _state(mesh, partial.shape[-1], SCATTER)(partial, t)


def col_scatter_ar(mesh, partial: ttnn.Tensor, t: int) -> ttnn.Tensor:
    """col_scatter (partial [1, 1, 32, D], t <= 4) or col_scatter_padded (partial [1, 1, 8 x 32, D], t = 8), then the
    all-reduce over the mesh row's 4 chips, in one op: -> [1, 1, t, D], the sum over all 32 chips of this row's tokens
    (the column's 8 in fp32 to bf16, then the row's 4 in fp32 to bf16, as col_scatter(_padded) and row_all_reduce)."""
    if partial.shape[-2] == ROWS * 32:
        return _state(mesh, partial.shape[-1], SCATTER_PADDED, SCATTER_WIDE, ar=True)(partial, t)
    return _state(mesh, partial.shape[-1], SCATTER, ar=True)(partial, t)


__all__ = [
    "ColCollective",
    "col_gather",
    "col_scatter",
    "col_scatter_ar",
    "col_scatter_padded",
    "prealloc",
    "prealloc_gather",
]
