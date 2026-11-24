# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The sequence-parallel KDA's state maps to the rows that read them (GLM53_KDA_MAPS_X), through ttnn.generic_op
(the plumbing as in `x2d_gather.py`), in place of ttnn.all_gather over the mesh column:

  exchange(mesh, packed, table): packed [1, H, 128, 256] bf16 tiles (each chip its segment's maps [A | B],
                       `summarize_chunk_recurrence`) -> the landing [8, H, 128, 256] (persistent DRAM, the same address
                       on every chip): slot s holds row s's maps on every chip whose request has row s, as
                       ttnn.all_gather(packed, dim=0, cluster_axis=ROW_AXIS) gives them there (a copy: the same bits);
                       the other slots are left as they were (chain_affine_transforms never reads them: it streams the
                       identity for rows outside the request, `rank_mask`).

ttnn's all-gather brings every row's 1 MB (H = 16) to every chip of the column, 7 MB in over the column's links, ~110
us a layer.  A request on two rows (the serving's 4 x 1K wave) needs one other row's.  Here each row multicasts its
maps along the column's ring over the fewest hops that reach every other row of its request, `b` one way and `a` the
other (`prefill_tables.kda_sp_maps_x_table`, from the host's groups: a request alone, or a row of none, sends nothing;
the 8 rows of one request: 4 and 3, as an all-gather's multicast), and copies its own into its slot.  The chips on the
arc outside the request take the slot too and never read it.  One sender core (maps_x_send.cpp: BRISC link 0, NCRISC
link 1, the even and odd 4 KB packets of two tiles; a connection to each ring neighbour's router) and `COPIERS` copy
cores (maps_x_copy.cpp, both RISCs) a chip.  A handshake keeps a call's packets off a chip until it is in the call (the
chain of its previous call read the landing): every chip first tells every other chip of the column, and a sender sends
once all of them have told it (all, not only its arc's: that keeps every chip within one call of every other, the two
counter sets' reach).  The counters (the ready one and the data ones, a source row each) alternate between two sets by a
parity word in persistent L1 that every chip flips at the end of a call, as x2d_gather's, so the turn holds through
traces.  Persistent L1: 19 words a core; the landing 8 H MB of DRAM a chip.
"""

from __future__ import annotations

import ttnn

from .common import COLS, ROWS

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/kda/"
TILE = 2048  # a bf16 tile
SLOT = 2 * TILE  # a packet: two tiles
CB_STAGE, CB_TABLE, CB_COPY = 0, 1, 2
SEM_GO, SEM_DONE = 0, 1
COPIERS = 8
NB = 8  # the sender's stage slots a RISC
NBC = 4  # the copiers' slots a RISC


class KdaMapsX:
    """The plumbing for a mesh and maps' shape, and the persistent state (the landing and the global semaphores)."""

    def __init__(self, mesh, heads: int, dh: int):
        assert tuple(mesh.shape) == (ROWS, COLS) and ROWS == 8, mesh.shape
        self.mesh, self.heads, self.dh = mesh, heads, dh
        self.tiles = heads * (dh // 32) * (2 * dh // 32)  # a slot's tiles: [H, dh, 2 dh]
        grid = mesh.compute_with_storage_grid_size()
        full = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        self.sender = ttnn.CoreCoord(grid.x // 2, grid.y // 2)
        self.sender_set = ttnn.CoreRangeSet([ttnn.CoreRange(self.sender, self.sender)])
        self.me = mesh.worker_core_from_logical_core(self.sender)
        copy = []
        for i in range(COPIERS):  # spread over the grid: a column apart, rows in turn
            x, y = (i * grid.x) // COPIERS, (3 * i + 1) % grid.y
            while (x, y) == (self.sender.x, self.sender.y) or (x, y) in copy:
                y = (y + 1) % grid.y
            copy.append((x, y))
        self.copy_cores = [ttnn.CoreCoord(x, y) for x, y in copy]
        self.copy_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.copy_cores])
        self.nodes = {
            (r, c): mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for r in range(ROWS) for c in range(COLS)
        }
        for (r, c), node in self.nodes.items():  # (the fabric's Y axis the mesh rows)
            assert int(node.chip_id) == r * COLS + c, ((r, c), node)
        assert int(ttnn.get_tt_fabric_max_payload_size_bytes()) >= SLOT, ttnn.get_tt_fabric_max_payload_size_bytes()
        # the parity word, the ready counters [parity], the data counters [parity][source row]
        self.keep = [ttnn.create_global_semaphore(mesh, full, 0) for _ in range(1 + 2 + 2 * ROWS)]
        addr = [ttnn.get_global_semaphore_address(g) for g in self.keep]
        self.par, self.ready, self.data = addr[0], addr[1:3], addr[3:]
        spec = ttnn.TensorSpec(
            ttnn.Shape([ROWS, heads, dh, 2 * dh]), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM
        )
        self.land = ttnn.allocate_tensor_on_device(spec, mesh)

    def busy(self) -> set:
        return {(self.sender.x, self.sender.y)} | {(c.x, c.y) for c in self.copy_cores}

    def _program(self, r: int, c: int, packed, table) -> ttnn.ProgramDescriptor:
        node = self.nodes[(r, c)]
        col_fwd, col_bwd = self.nodes[((r + 1) % ROWS, c)], self.nodes[((r - 1) % ROWS, c)]
        chips = [int(self.nodes[(q, c)].chip_id) for q in range(ROWS)]
        send_rt = [ttnn.RuntimeArgs() for _ in range(2)]
        for role in range(2):
            send_rt[role][self.sender.x][self.sender.y] = [
                packed.buffer_address(),
                self.land.buffer_address(),
                table.buffer_address(),
                r,
                int(node.mesh_id),
                int(col_fwd.chip_id),
                int(col_bwd.chip_id),
                self.par,
                *self.ready,
                *self.data,
                self.me.x,
                self.me.y,
                *chips,
            ]
        copy_rt = [ttnn.RuntimeArgs() for _ in range(2)]
        n = 2 * len(self.copy_cores)
        for j, core in enumerate(self.copy_cores):
            for role in range(2):
                k = 2 * j + role
                t0, t1 = k * self.tiles // n, (k + 1) * self.tiles // n
                copy_rt[role][core.x][core.y] = [packed.buffer_address(), self.land.buffer_address(), r, t0, t1 - t0]
        acc = ttnn.TensorAccessorArgs(packed).get_compile_time_args()
        acc += ttnn.TensorAccessorArgs(self.land).get_compile_time_args()
        kernels = []
        for role in range(2):  # role 0 BRISC (NoC 0, link 0), role 1 NCRISC (NoC 1, link 1)
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=KDIR + "maps_x_send.cpp",
                    core_ranges=self.sender_set,
                    compile_time_args=[role, NB, CB_STAGE, CB_TABLE, SEM_GO, SEM_DONE, self.tiles]
                    + acc
                    + ttnn.TensorAccessorArgs(table).get_compile_time_args(),
                    runtime_args=send_rt[role],
                    config=ttnn.WriterConfigDescriptor() if role == 0 else ttnn.ReaderConfigDescriptor(),
                )
            )
        for role in range(2):
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=KDIR + "maps_x_copy.cpp",
                    core_ranges=self.copy_set,
                    compile_time_args=[role, self.tiles, NBC, CB_COPY] + acc,
                    runtime_args=copy_rt[role],
                    config=ttnn.WriterConfigDescriptor() if role == 0 else ttnn.ReaderConfigDescriptor(),
                )
            )

        def cb(index, size, cores, page):
            return ttnn.CBDescriptor(
                total_size=size,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=page)
                ],
            )

        cbs = [
            cb(CB_STAGE, 2 * NB * SLOT, self.sender_set, SLOT),
            cb(CB_TABLE, 256, self.sender_set, 256),
            cb(CB_COPY, 2 * NBC * TILE, self.copy_set, TILE),
        ]
        sems = [
            ttnn.SemaphoreDescriptor(id=SEM_GO, core_ranges=self.sender_set, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_DONE, core_ranges=self.sender_set, initial_value=0),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        for role in range(2):
            args = program.kernels[role].runtime_args[self.sender.x][self.sender.y]
            for peer in (col_fwd, col_bwd):
                args.extend(ttnn.setup_fabric_connection(node, peer, role, program, self.sender))
        return program

    def __call__(self, packed: ttnn.Tensor, table: ttnn.Tensor) -> ttnn.Tensor:
        assert packed.dtype == ttnn.bfloat16 and packed.layout == ttnn.TILE_LAYOUT, packed
        assert tuple(packed.shape) == (1, self.heads, self.dh, 2 * self.dh), packed.shape
        mesh_program = ttnn.MeshProgramDescriptor()
        for r in range(ROWS):
            for c in range(COLS):
                coord = ttnn.MeshCoordinate(r, c)
                mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = self._program(r, c, packed, table)
        ttnn.generic_op([packed, table, self.land], mesh_program)
        return self.land


_STATE: dict = {}


def _state(mesh, heads: int, dh: int) -> KdaMapsX:
    key = (id(mesh), heads, dh)
    if key not in _STATE or _STATE[key].mesh is not mesh:
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
        assert int(view.total_bytes_allocated_per_bank) == 0, (
            "kda_maps_x: its landing asked for after a trace was captured (a replay could write over it): "
            "set GLM53_KDA_MAPS_X before the model loads (model.prealloc_sp_ops), or call kda_maps_x.prealloc first"
        )
        _STATE[key] = KdaMapsX(mesh, heads, dh)
    return _STATE[key]


def prealloc(mesh, heads: int, dh: int) -> None:
    """The persistent state now (the landing and the global semaphores), before the model builds or captures any
    program (as x2d_gather.prealloc)."""
    _state(mesh, heads, dh)


def exchange(mesh, packed: ttnn.Tensor, table: ttnn.Tensor) -> ttnn.Tensor:
    """packed [1, H, dh, 2 dh] -> the landing [8, H, dh, 2 dh] (persistent: valid until the next call), slot s row s's
    maps on the chips of its request (`table`: `prefill_tables.kda_sp_maps_x_table`, a row a mesh row)."""
    return _state(mesh, packed.shape[1], packed.shape[2])(packed, table)


__all__ = ["KdaMapsX", "exchange", "prealloc"]
