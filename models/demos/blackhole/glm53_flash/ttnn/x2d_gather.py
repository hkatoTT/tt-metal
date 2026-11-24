# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The prefill MoE's x all-gather over a mesh column (8 chips, ROW_AXIS) in two dimensions, through ttnn.generic_op
(the plumbing as in `row_all_reduce.py`):

  x2d_gather(mesh, x): x [1, 1, T, D] bf16 row-major (a mesh row's T tokens, the same on its 4 chips; D = 4096, T a
                       multiple of 16) -> [1, 1, 8 T, D] row-major DRAM: the column's 8 rows' tokens, row r's at r T, as
                       ttnn.all_gather(x, dim=2, cluster_axis=ROW_AXIS) gives them (a copy: the same bits).

ttnn's all-gather goes round the column's ring: 28 MB into each chip at T = 512 over the column's links alone (2 links
a way, ~20 GB/s each: 342 us a layer at chunk 512).  Here a chip of column parity p sends only half p of K (4 KB
of each 8 KB token) to the 28 chips outside its mesh row: down its own column (a 4-hop and a 3-hop multicast, the
counts swapping from packet to packet) with 1-hop teeth into a neighbour column at every row (the first half of the
tokens to column c - 1, the second to c + 1: those neighbours are of the other parity and take that half from it), so
the column's links and the row's carry 14 MB each (both axes at once 146 GB/s a chip, 137 landing in DRAM).
One sender core (x2d_send.cpp: BRISC link 0, NCRISC link 1, a connection to each ring neighbour's router on its link,
the even and odd token pairs) and `copiers` copy cores (x2d_copy.cpp, both RISCs: the own row's whole tokens into its
place) a chip.  The output is persistent (`landing`: DRAM, the same address on every chip, one for each T taken,
`prealloc` when the model loads): the packets land in the peers' output itself, so its address must be the same on
every chip and not move under a peer that runs ahead (a peer reaches its next call only after the layer's later
collectives, which wait for this chip's experts op, the one reader of x_all).  The receive semaphores (two, on the
sender core: 112 increments a call) alternate by a flag in persistent L1 that every chip flips at the end of a call,
so the turn holds through traces whatever their number of calls.  Persistent L1: three words a core.

The teeth halve the trunks' speed (a router with 3 exits), so the form taken is the relay form: the trunks alone
(x2d_send.cpp, in blocks of G pairs, the hop counts swapping from block to block, the last packet of a block fused with
an increment of a counter (source row, role, way) on the receivers' relay core of that role's link), and relay cores
(x2d_relay.cpp; one a link, BRISC to c + 1 and NCRISC to c - 1) that read each landed pair of half p back from x_all and
pass it on to a row neighbour (1-hop unicasts along the mesh row: the even pairs of a role to c - 1, the odd ones to
c + 1); the first waits for the 28 relay-in increments, and each sets its counters back to 0.  (T = 512 / 1,024: one
relay core 426 / 822 us, a core a link 316 / 611 us, ttnn's ring 364 / 706 us.)  Its counters are global semaphores too
(two sets of 8 rows x 2 roles x 2 ways, and two relay-in ones): 69 words of persistent L1 a core.  (The trunks with
teeth, the trunks alone without relays, one relay core for both links and the GLM53_X2D_* knobs were removed.)
"""

from __future__ import annotations

import ttnn

from .common import COLS, ROWS

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/x2d/"
HALF = 4096  # bytes of half a token's K (2048 bf16)
SLOT = 2 * HALF  # a packet: a token pair's halves
PAGE = 8192  # a token (row-major bf16, D = 4096)
CB_STAGE, CB_COPY, CB_RELAY = 0, 1, 2
SEM_DONE = 0
# the copy cores, the sender's stage slots a RISC, the copiers' slots a RISC
COPIERS = 8
NB = 8
NBC = 4
# the prefill's chunk sizes a mesh row whose landing the model takes when it loads (others go to ttnn.all_gather)
TS = (128, 512, 1024)
# the relays' stage slots a RISC, the trunk's block (pairs).  The relay cores: a core a link (BRISC the neighbour
# c + 1, NCRISC c - 1: one router a RISC; one core, a RISC feeding two routers, sent at ~11 GB/s each)
NBR = 8
G = 8


class X2dGather:
    """The plumbing for a mesh and width, and the persistent state (the landings and three global semaphores)."""

    def __init__(
        self,
        mesh,
        d: int = 4096,
        nb=None,
        nbc=None,
        copiers=None,
        sender=None,
        relay=None,
        nbr=None,
        g=None,
        dbg=0,
        sdbg=0,
        rswap=0,
        relay2=None,
        ssplit=1,
        sender2=None,
    ):
        assert tuple(mesh.shape) == (ROWS, COLS) and ROWS == 8 and COLS == 4, mesh.shape
        assert d * 2 == PAGE, d
        self.mesh, self.d = mesh, d
        # (tests: the knobs below in place of the defaults)
        self.nb, self.nbc, self.nbr, self.g = nb or NB, nbc or NBC, nbr or NBR, g or G
        # (tests, the values wrong: the relay's x2d_relay.cpp DBG bits, the sender's SDBG bits)
        self.dbg, self.sdbg = dbg, sdbg
        self.rswap = rswap  # (tests: BRISC to c - 1 and NCRISC to c + 1)
        copiers = COPIERS if copiers is None else copiers
        grid = mesh.compute_with_storage_grid_size()
        full = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        if sender:
            sx, sy = (int(v) for v in sender.split(","))
        else:
            sx, sy = grid.x // 2, grid.y // 2
        self.sender = ttnn.CoreCoord(sx, sy)
        # (tests: ssplit 2, a sender core a role, role 1's at sender2)
        self.ssplit = ssplit
        s2 = tuple(int(v) for v in sender2.split(",")) if sender2 else (sx + 1, sy)
        self.senders = [self.sender, ttnn.CoreCoord(*s2)][: self.ssplit]
        if relay:
            rx, ry = (int(v) for v in relay.split(","))
        else:  # (T = 1,024: next to the sender, (5, 5) + (4, 5), 612 us; away from it, 513-522 us)
            rx, ry = 1, grid.y - 2
        # (the second relay core: by default next to the first)
        r2 = tuple(int(v) for v in relay2.split(",")) if relay2 else (rx - 1, ry)
        self.relays = [ttnn.CoreCoord(rx, ry), ttnn.CoreCoord(*r2)]
        busy = {(c.x, c.y) for c in self.senders} | {(c.x, c.y) for c in self.relays}
        copy = []
        for i in range(copiers):  # spread over the grid: a column apart, rows in turn
            x, y = (i * grid.x) // copiers, (3 * i + 1) % grid.y
            while (x, y) in busy or (x, y) in copy:
                y = (y + 1) % grid.y
            copy.append((x, y))
        self.copy_cores = [ttnn.CoreCoord(x, y) for x, y in copy]
        self.sender_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.senders])
        self.copy_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.copy_cores]) if copy else None
        self.senders_xy = [mesh.worker_core_from_logical_core(c) for c in self.senders]
        self.relay_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.relays])
        self.relay_xy = [mesh.worker_core_from_logical_core(c) for c in self.relays]
        self.done_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(c, c) for c in self.senders] + [ttnn.CoreRange(c, c) for c in self.relays]
        )
        self.nodes = {
            (r, c): mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for r in range(ROWS) for c in range(COLS)
        }
        for (r, c), node in self.nodes.items():  # (the fabric's Y axis the mesh rows: the trunks N / S, teeth E / W)
            assert int(node.chip_id) == r * COLS + c, ((r, c), node)
        assert int(ttnn.get_tt_fabric_max_payload_size_bytes()) >= SLOT, ttnn.get_tt_fabric_max_payload_size_bytes()
        # the calls' parity flag (every sender and relay core's), then the relay core's: relay-in [set], the trunk
        # counters [set][source row][role][way]
        self.keep = [ttnn.create_global_semaphore(mesh, full, 0)]
        self.par = ttnn.get_global_semaphore_address(self.keep[0])
        self.keep_relay = [ttnn.create_global_semaphore(mesh, full, 0) for _ in range(2 + 2 * ROWS * 2 * 2)]
        addr = [ttnn.get_global_semaphore_address(g) for g in self.keep_relay]
        self.relay_in = addr[:2]
        self.cnt = [
            [[[addr[2 + ((s * ROWS + q) * 2 + k) * 2 + w] for w in range(2)] for k in range(2)] for q in range(ROWS)]
            for s in range(2)
        ]
        self.land: dict = {}

    def landing(self, t: int) -> ttnn.Tensor:
        """The persistent output for t tokens a mesh row (made on first use: `prealloc` first)."""
        if t not in self.land:
            spec = ttnn.TensorSpec(
                ttnn.Shape([1, 1, ROWS * t, self.d]),
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
                buffer_type=ttnn.BufferType.DRAM,
            )
            self.land[t] = ttnn.allocate_tensor_on_device(spec, self.mesh)
        return self.land[t]

    def busy(self) -> set:
        """The (x, y) cores its programs take (another program's kernels beside them must leave these out)."""
        return {(c.x, c.y) for c in self.senders + self.relays + self.copy_cores}

    def _program(self, r: int, c: int, x, out, t: int, extra=None) -> ttnn.ProgramDescriptor:
        node = self.nodes[(r, c)]
        col_fwd, col_bwd = self.nodes[((r + 1) % ROWS, c)], self.nodes[((r - 1) % ROWS, c)]
        row_fwd, row_bwd = self.nodes[(r, (c + 1) % COLS)], self.nodes[(r, (c - 1) % COLS)]
        send_rt = [ttnn.RuntimeArgs() for _ in range(2)]
        for role in range(2):
            # the packets increment this row's counters on the receivers' relay core of this role's link: [set 0,
            # set 1] x [way 0, way 1]
            sxy = self.relay_xy[role]
            s0, s1 = self.cnt[0][r][role][0], self.cnt[1][r][role][0]
            s0b, s1b = self.cnt[0][r][role][1], self.cnt[1][r][role][1]
            core, own = self.senders[role % self.ssplit], self.senders_xy[role % self.ssplit]
            send_rt[role][core.x][core.y] = [
                x.buffer_address(),
                out.buffer_address(),
                r,
                c % 2,
                int(node.mesh_id),
                int(col_fwd.chip_id),
                int(col_bwd.chip_id),
                s0,
                s1,
                self.par,
                sxy.x,
                sxy.y,
                s0b,
                s1b,
                own.x,
                own.y,
            ]
        # the relay RISCs: (relay core, RISC, link, side, waits for the relay-in, resets the links' counters, peer)
        b_side, n_side = (0, 1) if self.rswap else (1, 0)  # (the sides of BRISC and NCRISC)
        peer = (row_bwd, row_fwd)
        specs = [
            spec
            for k in range(2)
            for spec in (
                (k, 0, k, b_side, int(k == 0), 1 << k, peer[b_side]),
                (k, 1, k, n_side, 0, 0, peer[n_side]),
            )
        ]
        relay_rt = []
        cnt = [self.cnt[s][q][k][w] for s in range(2) for q in range(ROWS) for k in range(2) for w in range(2)]
        for core_i, _, _, side, *_ in specs:
            core = self.relays[core_i]
            rt = ttnn.RuntimeArgs()
            rt[core.x][core.y] = [
                out.buffer_address(),
                r,
                c % 2,
                int(node.mesh_id),
                int((row_fwd if side else row_bwd).chip_id),
                self.par,
                self.relay_xy[0].x,
                self.relay_xy[0].y,
                *self.relay_in,
                *cnt,
            ]
            relay_rt.append(rt)
        copy_rt = [ttnn.RuntimeArgs() for _ in range(2)]
        n = 2 * len(self.copy_cores)
        for j, core in enumerate(self.copy_cores):
            for role in range(2):
                k = 2 * j + role
                t0, t1 = k * t // n, (k + 1) * t // n
                copy_rt[role][core.x][core.y] = [x.buffer_address(), out.buffer_address(), r, t0, t1 - t0]
        acc = ttnn.TensorAccessorArgs(x).get_compile_time_args() + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        kernels = []
        for role in range(2):  # role 0 BRISC (NoC 0, link 0), role 1 NCRISC (NoC 1, link 1)
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=KDIR + "x2d_send.cpp",
                    core_ranges=ttnn.CoreRangeSet(
                        [ttnn.CoreRange(self.senders[role % self.ssplit], self.senders[role % self.ssplit])]
                    ),
                    compile_time_args=[
                        role,
                        t,
                        self.nb,
                        CB_STAGE,
                        SEM_DONE,
                        self.g,
                        self.sdbg,
                        int(self.ssplit == 2),
                    ]
                    + acc,
                    runtime_args=send_rt[role],
                    config=ttnn.WriterConfigDescriptor() if role == 0 else ttnn.ReaderConfigDescriptor(),
                )
            )
        relay_k = len(kernels)
        for (core_i, risc, link, side, wait_in, reset, _), rt in zip(specs, relay_rt):
            core = self.relays[core_i]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=KDIR + "x2d_relay.cpp",
                    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
                    compile_time_args=[risc, t, self.nbr, CB_RELAY, SEM_DONE, self.g, self.dbg, link, side, wait_in]
                    + [reset]
                    + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                    runtime_args=rt,
                    config=ttnn.WriterConfigDescriptor() if risc == 0 else ttnn.ReaderConfigDescriptor(),
                )
            )
        for role in range(2 if self.copy_cores else 0):
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=KDIR + "x2d_copy.cpp",
                    core_ranges=self.copy_set,
                    compile_time_args=[role, t, self.nbc, CB_COPY] + acc,
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

        cbs = [cb(CB_STAGE, 2 * self.nb * SLOT, self.sender_set, SLOT)]
        if self.copy_cores:
            cbs.append(cb(CB_COPY, 2 * self.nbc * PAGE, self.copy_set, PAGE))
        cbs.append(cb(CB_RELAY, 2 * self.nbr * SLOT, self.relay_set, SLOT))
        sems = [ttnn.SemaphoreDescriptor(id=SEM_DONE, core_ranges=self.done_set, initial_value=0)]
        if extra is not None:  # (another program's kernels on other cores, after this one's: its indices kept)
            kernels, sems, cbs = kernels + list(extra.kernels), sems + list(extra.semaphores), cbs + list(extra.cbs)
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        for role in range(2):
            core = self.senders[role % self.ssplit]
            args = program.kernels[role].runtime_args[core.x][core.y]
            for peer in (col_fwd, col_bwd):
                args.extend(ttnn.setup_fabric_connection(node, peer, role, program, core))
        for i, (core_i, _, link, _, _, _, to) in enumerate(specs):
            core = self.relays[core_i]
            args = program.kernels[relay_k + i].runtime_args[core.x][core.y]
            args.extend(ttnn.setup_fabric_connection(node, to, link, program, core))
        return program

    def __call__(self, x: ttnn.Tensor, extra=None) -> ttnn.Tensor:
        t = x.shape[-2]
        assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.ROW_MAJOR_LAYOUT and x.shape[-1] == self.d, x
        assert t % 16 == 0 and len(x.shape) == 4 and x.shape[0] == x.shape[1] == 1, x.shape
        out = self.landing(t)
        mesh_program = ttnn.MeshProgramDescriptor()
        for r in range(ROWS):
            for c in range(COLS):
                coord = ttnn.MeshCoordinate(r, c)
                mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = self._program(r, c, x, out, t, extra)
        ttnn.generic_op([x] + (list(extra.tensors) if extra is not None else []) + [out], mesh_program)
        return out


_STATE: dict = {}


def _state(mesh, d: int) -> X2dGather:
    key = (id(mesh), d)
    if key not in _STATE or _STATE[key].mesh is not mesh:
        _STATE[key] = X2dGather(mesh, d)
    return _STATE[key]


def prealloc(mesh, d: int, ts=None) -> None:
    """The op's persistent state now (its global semaphores and the landings for `ts`, TS by default), before the model
    builds or captures any program (as row_all_reduce.prealloc)."""
    st = _state(mesh, d)
    for t in TS if ts is None else ts:
        st.landing(t)


def fits(mesh, x: ttnn.Tensor) -> bool:
    """Whether x2d_gather takes x without a new allocation (its landing taken by `prealloc`)."""
    st = _STATE.get((id(mesh), x.shape[-1]))
    return st is not None and st.mesh is mesh and x.shape[-2] in st.land


def x2d_gather(mesh, x: ttnn.Tensor, extra=None) -> ttnn.Tensor:
    """x [1, 1, T, D] row-major bf16 -> [1, 1, 8 T, D] row-major DRAM (persistent: valid until the next call with this
    T), the column's rows' tokens, as ttnn.all_gather(x, dim=2, cluster_axis=ROW_AXIS).  `extra` (kernels, semaphores,
    cbs and tensors of a program on cores it does not take, `busy_cores`: moe_prefill.PlanPush) runs in the same
    program, beside it."""
    return _state(mesh, x.shape[-1])(x, extra)


def busy_cores(mesh, d: int) -> set:
    """The (x, y) cores x2d_gather's programs take for this width."""
    return _state(mesh, d).busy()


__all__ = ["X2dGather", "busy_cores", "fits", "prealloc", "x2d_gather"]
