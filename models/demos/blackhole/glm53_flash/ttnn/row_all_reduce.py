# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The all-reduce over a mesh row's 4 chips for a few tokens (t <= 16 rows of [1, 1, t, D] bf16, TILE, DRAM
interleaved; the rows below t are zero in and out): one-shot over the 2D fabric, through ttnn.generic_op.

Per chip: `workers` cores (8) take D / 32 / workers tiles each.  A worker's reader reads its tiles' first rows (faces 0
and 1, `seg` bytes each: 64 for t <= 2, 128 / 256 / 512 for t <= 4 / 8 / 16; above 4 tokens 32 workers, so that a
slice stays one 4 KB packet) from DRAM, compacted, into this chip's slot of its receive buffer (a circular buffer), and
copies that slice to the sender core's stage.  The sender (one core: its BRISC, and with 2 links (`links`, 2) its
NCRISC too, each with a fabric connection to each ring neighbour's router on its link, half the slices each) sends
each slice, as soon as it is there, to the row's three other chips, into their DRAM landing area (`OneshotArea`:
persistent DRAM, the same address on every chip, a page for each parity, source chip and worker), each packet a write
fused with an increment (once the write is acknowledged) of that worker's receive semaphore for this chip (`mcast`,
the default: a 2-hop multicast one way and a unicast the other; off: three unicasts).  The worker reads each chip's
slice from DRAM into its slot as its semaphore says it has landed; its compute adds the four slots in the order of the
chips (the same sum on every chip of the row) into fp32 Dest (`fpu` 2, the default: FPU pairs, then an SFPU add; 0: by
the SFPU in order, slower) and packs bf16, and its writer and reader put the rows into staging tiles (a circular buffer they zero while the slices travel)
and the tiles into the output, half each.  The landing pages and semaphores alternate between two parities from call
to call: a chip writes a peer's parity-p page only after that peer has finished its previous call (which needed this
chip's data), so the page it overwrites was read two calls before.  A step must make an even number of calls (a
replayed trace keeps the alternation), as for all_reduce_async.  The only persistent L1 is the receive semaphores (a
word a core each): nothing sits under the static circular buffers of other programs.
"""

from __future__ import annotations

import math

import ttnn

from .common import COLS, ROWS, core_rects

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/row_ar/"
PAGE = 2048  # a bf16 tile
CB_STAGE, CB_RECV, CB_OUT, CB_STAGING = 0, 1, 2, 3
SEM_GATHER = 0


class OneshotArea:
    """A family of one-shot fabric ops' persistent state: the DRAM landing area (`pages` pages of `page` bytes,
    interleaved, the same address on every chip), the receive semaphores (two parities x `n` chips of the ring: global
    semaphores on every core of the grid, a word a core) and the turn (the calls made: the parity)."""

    def __init__(self, mesh, n: int, pages: int, page: int):
        grid = mesh.compute_with_storage_grid_size()
        full = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        self.mesh = mesh
        spec = ttnn.TensorSpec(
            ttnn.Shape([pages, page // 2]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, buffer_type=ttnn.BufferType.DRAM
        )
        self.area = ttnn.allocate_tensor_on_device(spec, mesh)
        self.sems = [[ttnn.create_global_semaphore(mesh, full, 0) for _ in range(n)] for _ in range(2)]
        self.sem_addr = [[ttnn.get_global_semaphore_address(s) for s in row] for row in self.sems]
        self.turn = 0

    def next_parity(self) -> int:
        parity = self.turn % 2
        self.turn += 1
        return parity


_AREAS: dict = {}


def oneshot_area(mesh, family: str, n: int, pages: int, page: int) -> OneshotArea:
    """The family's state (one a family and layout: its ops share the ring, and so the parity)."""
    key = (id(mesh), family, pages, page)
    if key not in _AREAS or _AREAS[key].mesh is not mesh:
        _AREAS[key] = OneshotArea(mesh, n, pages, page)
    return _AREAS[key]


class RowAllReduce:
    """The plumbing for a mesh and width, and its persistent state (`OneshotArea`: DRAM, and a few semaphores in
    L1), made by `prealloc` when the model loads (or on the first call)."""

    def __init__(
        self,
        mesh,
        d: int,
        workers: int | None = None,
        links: int | None = None,
        mcast: bool | None = None,
        fpu: int | None = None,
        split: int | None = None,
        dry: bool = False,
    ):
        assert tuple(mesh.shape) == (ROWS, COLS), mesh.shape
        self.mesh = mesh
        self.d = d
        self.tiles = d // 32
        self.workers = workers or 8
        assert self.tiles % self.workers == 0, (self.tiles, self.workers)
        self.links = links or 2
        self.mcast = True if mcast is None else mcast
        # the sum: 2 FPU pairs and an SFPU add, 0 SFPU adds in order (a test's)
        self.fpu = 2 if fpu is None else fpu
        assert self.fpu in (0, 2), self.fpu
        self.dry = dry  # a test switch: kernels that do nothing (the launch cost)
        self.tpw = self.tiles // self.workers
        # output tiles the writer (BRISC) puts, the reader (NCRISC) the rest
        self.split = split if split is not None else self.tpw // 2
        self.seg_max = min(512, 4096 // (2 * self.tpw))  # a face's rows a slice holds: a slice a 4 KB packet
        self.npt = math.ceil(self.tpw * 2 * self.seg_max / PAGE)  # pages a slot (the largest slice)
        self.slot = self.npt * PAGE
        grid = mesh.compute_with_storage_grid_size()
        per_row = min(self.workers, grid.x - 1)
        self.worker_cores = [ttnn.CoreCoord(i % per_row, i // per_row) for i in range(self.workers)]
        self.sender = ttnn.CoreCoord(grid.x - 1, 0)
        self.worker_set = core_rects(self.worker_cores)
        self.sender_set = core_rects([self.sender])
        self.all_set = core_rects(self.worker_cores + [self.sender])
        self.worker_xy = [mesh.worker_core_from_logical_core(c) for c in self.worker_cores]
        self.sender_xy = mesh.worker_core_from_logical_core(self.sender)
        self.area = oneshot_area(mesh, "row", COLS, 2 * COLS * self.workers, self.slot)
        self.nodes = {
            (r, c): mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for r in range(ROWS) for c in range(COLS)
        }
        self.max_payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())

    def _src(self, name: str) -> str:
        return KDIR + ("row_ar_empty.cpp" if self.dry else name)

    def _program(self, r: int, c: int, x, out, t: int, seg: int, parity: int) -> ttnn.ProgramDescriptor:
        slice_bytes = self.tpw * 2 * seg
        pages = math.ceil(slice_bytes / PAGE)
        area = self.area.area.buffer_address()
        reader_rt = ttnn.RuntimeArgs()
        compute_rt = ttnn.RuntimeArgs()
        writer_rt = ttnn.RuntimeArgs()
        for w, core in enumerate(self.worker_cores):
            args = [x.buffer_address(), w * self.tpw, self.tpw, parity, c, self.sender_xy.x, self.sender_xy.y, w]
            args += [self.workers, *self.area.sem_addr[parity], out.buffer_address(), self.split, area]
            reader_rt[core.x][core.y] = args
            compute_rt[core.x][core.y] = []
            writer_rt[core.x][core.y] = [out.buffer_address(), w * self.tpw, self.split]
        node = self.nodes[(r, c)]
        fwd = self.nodes[(r, (c + 1) % COLS)]
        bwd = self.nodes[(r, (c - 1) % COLS)]
        opp = self.nodes[(r, (c + 2) % COLS)]
        sender_rts = []
        for link in range(self.links):
            rt = ttnn.RuntimeArgs()
            args = [slice_bytes, area, self.area.sem_addr[parity][c], int(node.mesh_id), int(fwd.chip_id)]
            args += [int(bwd.chip_id), int(opp.chip_id), link, self.links, (parity * COLS + c) * self.workers]
            for xy in self.worker_xy:
                args += [xy.x, xy.y]
            rt[self.sender.x][self.sender.y] = args
            sender_rts.append(rt)

        modes = [ttnn.UnpackToDestMode.Default] * 64
        if self.fpu == 0:
            modes[CB_RECV] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        compute_cfg.unpack_to_dest_mode = modes
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=self._src("row_ar_reader.cpp"),
                core_ranges=self.worker_set,
                compile_time_args=[CB_RECV, CB_STAGE, seg, SEM_GATHER, self.npt, CB_OUT, t, pages, CB_STAGING]
                + ttnn.TensorAccessorArgs(x).get_compile_time_args()
                + ttnn.TensorAccessorArgs(out).get_compile_time_args()
                + ttnn.TensorAccessorArgs(self.area.area).get_compile_time_args(),
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=self._src("row_ar_compute.cpp"),
                core_ranges=self.worker_set,
                compile_time_args=[CB_RECV, CB_OUT, self.npt, pages, int(self.fpu)],
                runtime_args=compute_rt,
                config=compute_cfg,
            ),
            ttnn.KernelDescriptor(
                kernel_source=self._src("row_ar_writer.cpp"),
                core_ranges=self.worker_set,
                compile_time_args=[CB_OUT, seg, t, pages, CB_STAGING]
                + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
        ]
        for link in range(self.links):  # link 0 on BRISC, link 1 on NCRISC
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=self._src("row_ar_sender.cpp"),
                    core_ranges=self.sender_set,
                    compile_time_args=[CB_STAGE, self.workers, SEM_GATHER, self.slot, int(self.mcast)]
                    + ttnn.TensorAccessorArgs(self.area.area).get_compile_time_args(),
                    runtime_args=sender_rts[link],
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
        cbs = [
            cb(CB_STAGE, self.workers * self.slot, self.all_set, self.slot),
            cb(CB_RECV, 4 * self.slot, self.worker_set),
            cb(CB_OUT, pages * PAGE, self.worker_set),
            cb(CB_STAGING, self.tpw * PAGE, self.worker_set),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=SEM_GATHER, core_ranges=self.all_set, initial_value=0)]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        for link in range(self.links):
            sender_args = program.kernels[3 + link].runtime_args[self.sender.x][self.sender.y]
            for peer in (fwd, bwd):
                sender_args.extend(ttnn.setup_fabric_connection(node, peer, link, program, self.sender))
        return program

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        t = x.shape[-2]
        assert 1 <= t <= 16 and x.shape[-1] == self.d, x.shape
        seg = 64 if t <= 2 else 128 if t <= 4 else 256 if t <= 8 else 512
        assert seg <= self.seg_max, (t, self.workers)
        assert self.tpw * 2 * seg <= self.max_payload, (self.tpw, seg, self.max_payload)  # a slice a packet
        parity = 0 if self.dry else self.area.next_parity()  # (the empty kernels of `dry` do not take a turn)
        out = ttnn.allocate_tensor_on_device(x.spec, self.mesh)
        mesh_program = ttnn.MeshProgramDescriptor()
        for r in range(ROWS):
            for c in range(COLS):
                coord = ttnn.MeshCoordinate(r, c)
                mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = self._program(r, c, x, out, t, seg, parity)
        ttnn.generic_op([x, out], mesh_program)
        return out


_STATE: dict = {}


def _state(mesh, d: int, workers: int | None = None) -> RowAllReduce:
    key = (id(mesh), d, workers)
    if key not in _STATE or _STATE[key].mesh is not mesh:
        _STATE[key] = RowAllReduce(mesh, d, workers)
    return _STATE[key]


WIDE = 32  # the workers above 4 tokens a row


def prealloc(mesh, d: int) -> None:
    """The op's persistent state (`OneshotArea`: its L1 only a few semaphores) now, before the model builds or captures
    any program: a program's static CBs are checked against the L1 buffers when it is enqueued, but not when a trace
    replays it."""
    _state(mesh, d)
    _state(mesh, d, WIDE)


def row_all_reduce(mesh, x: ttnn.Tensor) -> ttnn.Tensor:
    """x [1, 1, t <= 16, D] bf16 TILE DRAM interleaved, each chip's partial -> the sum over the chips of its mesh row,
    on each of them (fp32 adds, bf16)."""
    return _state(mesh, x.shape[-1], None if x.shape[-2] <= 4 else WIDE)(x)


__all__ = ["OneshotArea", "RowAllReduce", "oneshot_area", "prealloc", "row_all_reduce"]
