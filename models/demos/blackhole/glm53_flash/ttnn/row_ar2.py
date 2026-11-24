# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The decode's row all-reduce (`common._TpArAsync`, otherwise tt-metal's all_reduce_async: each chip multicasts all
of its share to the row's three other chips, 0.75 of a chip's share on each link) in two stages, through
ttnn.generic_op: a one-shot reduce-scatter, then a multicast all-gather of the sums,
0.25 + 0.1875 of a chip's share on each link.

The layout is all_reduce_async's: the input [1, 1, t, D] bf16 width-sharded in L1 over `n` cores (64 for two tile rows,
32 for one; a shard 4 tiles, 8 KB), the persistent landing buffer of the `_TpArAsync` parity (a slot of 4 tiles for
each chip of the row, at chip c's slot c), its global semaphore, and the output sharded as the input.  Link l's sender
core serves the output cores l n / 2 .. (l + 1) n / 2 - 1; within a link, core j's sum is made by chip (j mod 4) of
the row (its owner), so that each chip owns n / 8 cores a link.
  - the reduce-scatter: on each chip, the sender's reader (RISCV_1) reads the input shards of the cores the other
    chips own, the sender's writer (RISCV_0) sends each, as a fabric write fused with an increment of the owner's
    global semaphore at that core, into slot c of the owner chip's landing buffer at that core (the neighbours one
    hop each way, the opposite chip two hops through a neighbour that passes it on, the way alternating with the
    core's group of 4 so that every link carries the same bytes); an owner core copies its own share into its slot.
  - the sum: an owner core waits for the 3 shares, then all_reduce_async's own compute kernel (reduction.cpp, the same
    compile-time and runtime arguments and compute config: HiFi4, fp32 Dest, full sync) adds the 4 slots in the same
    order into the output tile: the sums are all_reduce_async's bit for bit.
  - the all-gather: the owner core copies its sums to its sender's stage and sets its bit of the sender's semaphore;
    the writer multicasts them as they come (two hops one way and one the other, alternating) into the same core's
    output shard on the three other chips, each write fused with an increment of that core's global semaphore, which
    a non-owner core waits for (1).
The landing buffers and semaphores alternate between the `_TpArAsync` parities as all_reduce_async's.  A remote write
into this call's output (the all-gather) arrives only after this chip's own program has started (the owner's sum needed
this chip's reduce-scatter share), so no earlier program of this chip still uses that L1."""

from __future__ import annotations

import os

import ttnn

from . import w1_dense
from .common import COLS, ROWS, core_rects

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/row_ar2/"
REDUCE_SRC = "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/compute/reduction.cpp"
TILE_BYTES = 2048  # a bf16 32 x 32 tile
BFP8_TILE_BYTES = 1088  # a Bfp8_b one (GLM53_AR_BFP8: the input in BFP8_b)
CB_IN, CB_LAND, CB_OUT, CB_STAGE = 0, 1, 2, 3  # (CB_LAND / CB_OUT: all_reduce_async's reduction and output CBs)
SEM_AG = 0
LINKS = 2
RING = COLS  # the row's chips


def rx_noc0_on() -> bool:
    """GLM53_AR2_RX_NOC0 (read at each call; 0 by default): the output cores' kernel (ar2_receiver.cpp) on RISCV_0 /
    NoC 0 in place of RISCV_1 / NoC 1 (the only data movement kernel on those cores either way).  An owner core's
    write of its sums into its sender's stage then leaves on NoC 0, off the links that carry the sender's reader's
    NoC 1 responses.  The same bytes to the same places."""
    return os.environ.get("GLM53_AR2_RX_NOC0", "0") == "1"


def _dm(processor, noc):
    cfg = ttnn.DataMovementConfigDescriptor()
    cfg.processor, cfg.noc = processor, noc
    return cfg


class RowAr2:
    """The plumbing for a mesh and a shard layout (n output cores, `tiles` tiles a core)."""

    def __init__(self, mesh, n: int, tiles: int):
        assert tuple(mesh.shape) == (ROWS, COLS), mesh.shape
        assert n % (LINKS * RING) == 0, n
        self.mesh = mesh
        self.n = n
        self.tiles = tiles
        self.shard = tiles * TILE_BYTES
        self.half = n // LINKS  # output cores a link
        self.n_own = self.half // RING  # the cores a chip owns on a link
        self.n_rs = self.half - self.n_own  # the shares a link's sender sends in the reduce-scatter
        assert self.n_own <= 16, self.n_own  # (a bit of the sender's semaphore each)
        grid = mesh.compute_with_storage_grid_size()
        self.out_cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n)]  # the shards' row-major order
        rows = (n - 1) // grid.x + 1
        # the sender cores where all_reduce_async puts them: the first cores past the output cores' bounding box
        self.senders = (
            [ttnn.CoreCoord(i, rows) for i in range(LINKS)]
            if rows > 1
            else [ttnn.CoreCoord(n + i, 0) for i in range(LINKS)]
        )
        self.out_set = core_rects(self.out_cores)
        self.sender_set = core_rects(self.senders)
        self.all_set = core_rects(self.out_cores + self.senders)
        self.out_xy = [mesh.worker_core_from_logical_core(c) for c in self.out_cores]
        self.sender_xy = [mesh.worker_core_from_logical_core(c) for c in self.senders]
        self.nodes = {
            (r, c): mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for r in range(ROWS) for c in range(COLS)
        }

    def _program(self, r: int, c: int, x, land, out, sem_addr: int, riders=()) -> ttnn.ProgramDescriptor:
        node = self.nodes[(r, c)]
        fwd, bwd, opp = (self.nodes[(r, (c + d) % COLS)] for d in (1, -1, 2))
        x_addr, land_addr, out_addr = x.buffer_address(), land.buffer_address(), out.buffer_address()
        # GLM53_AR_BFP8 (`common.ar_input_dtype`): an input in BFP8_b (the matmul before wrote its shard so): the
        # reduce-scatter's shares, the landing slots and the reader's pages in BFP8 (1,088-B tiles); the sums, the
        # stage and the all-gather in bf16 as before
        x8 = x.dtype == ttnn.bfloat8_b
        shard_in = self.tiles * (BFP8_TILE_BYTES if x8 else TILE_BYTES)
        slot_off = c * shard_in
        # tests' switches, read at each call: GLM53_AR2_DRY=1 the routes only (nothing sent, no one waits: the values
        # wrong), GLM53_AR2_ROUTE_CHECK=1 the writer's route bytes compared with the fabric API's encodes (DPRINT)
        dry = os.environ.get("GLM53_AR2_DRY") == "1"
        defines = [("AR2_DRY", "1")] if dry else []
        check = os.environ.get("GLM53_AR2_ROUTE_CHECK") == "1"
        rx_noc0 = rx_noc0_on()
        sender_defines = defines + ([("AR2_ROUTE_CHECK", "1")] if check else [])
        reader_rt, writer_rt, recv_rt, comp_rt = (ttnn.RuntimeArgs() for _ in range(4))
        for link in range(LINKS):
            sx, sy = self.sender_xy[link].x, self.sender_xy[link].y
            rs, rs_src, own = [], [], []
            for jj in range(self.half):
                j = link * self.half + jj
                xy = self.out_xy[j]
                core = self.out_cores[j]
                owner, group = jj % RING, jj // RING
                if owner == c:
                    own.append(xy.x | (xy.y << 8) | ((len(own) % 2) << 16))
                    recv_rt[core.x][core.y] = [1, sem_addr, x_addr, slot_off, sx, sy, len(own) - 1]
                    comp_rt[core.x][core.y] = [0, 0, 0] if dry else [1, RING, self.tiles]
                    continue
                d = (owner - c) % RING
                kind = 0 if d == 1 else 1 if d == RING - 1 else (2 if group % 2 == 0 else 3)
                rs.append(xy.x | (xy.y << 8) | (kind << 16))
                rs_src.append(xy.x | (xy.y << 8))
                recv_rt[core.x][core.y] = [0, sem_addr]
                comp_rt[core.x][core.y] = [0, 0, 0]
            assert len(rs) == self.n_rs and len(own) == self.n_own, (len(rs), len(own))
            s = self.senders[link]
            reader_rt[s.x][s.y] = [x_addr, *rs_src]
            head = [land_addr, sem_addr, out_addr, int(node.mesh_id), int(fwd.chip_id), int(bwd.chip_id)]
            writer_rt[s.x][s.y] = head + [int(opp.chip_id), slot_off, *rs, *own]

        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        compute_cfg.math_approx_mode = False
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=KDIR + "ar2_reader.cpp",
                core_ranges=self.sender_set,
                compile_time_args=[CB_IN, shard_in, self.n_rs],
                runtime_args=reader_rt,
                config=_dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KDIR + "ar2_sender.cpp",
                core_ranges=self.sender_set,
                compile_time_args=[CB_IN, CB_STAGE, SEM_AG, self.shard, self.n_rs, self.n_own, shard_in],
                defines=sender_defines,
                runtime_args=writer_rt,
                config=_dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KDIR + "ar2_receiver.cpp",
                core_ranges=self.out_set,
                compile_time_args=[
                    CB_LAND,
                    CB_OUT,
                    CB_STAGE,
                    SEM_AG,
                    self.shard,
                    self.tiles,
                    RING,
                    int(rx_noc0),
                    shard_in,
                ],
                defines=defines,
                runtime_args=recv_rt,
                config=_dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0)
                if rx_noc0
                else _dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=REDUCE_SRC,
                core_ranges=self.out_set,
                compile_time_args=[CB_LAND, CB_OUT],
                runtime_args=comp_rt,
                config=compute_cfg,
            ),
        ]

        def cb(index, size, cores, page, fmt=ttnn.bfloat16):
            return ttnn.CBDescriptor(
                total_size=size,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        land_cb = ttnn.cb_descriptor_from_sharded_tensor(CB_LAND, land, core_ranges=self.out_set)
        if x8:  # (the same landing buffer, its first RING slots of 1,088-B tiles)
            land_cb.total_size = RING * shard_in
            land_cb.format_descriptors = [
                ttnn.CBFormatDescriptor(buffer_index=CB_LAND, data_format=ttnn.bfloat8_b, page_size=BFP8_TILE_BYTES)
            ]

        # the stage first, on the senders and the output cores: the same address on all of them (an owner core writes
        # its sums to its sender's)
        cbs = [
            cb(CB_STAGE, self.n_own * self.shard, self.all_set, self.shard),
            cb(CB_IN, self.n_rs * shard_in, self.sender_set, shard_in, ttnn.bfloat8_b if x8 else ttnn.bfloat16),
            land_cb,
            ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=SEM_AG, core_ranges=self.all_set, initial_value=0)]
        program = ttnn.ProgramDescriptor(kernels=kernels + list(riders), semaphores=sems, cbs=cbs)
        for link in range(LINKS):
            s = self.senders[link]
            args = program.kernels[1].runtime_args[s.x][s.y]
            for peer in (fwd, bwd):
                args.extend(ttnn.setup_fabric_connection(node, peer, link, program, s))
        return program

    def __call__(self, x, land, sem) -> ttnn.Tensor:
        """x: the input, sharded as the output is to be; land / sem: the `_TpArAsync` parity's landing buffer and
        global semaphore."""
        shard = x.memory_config().shard_spec
        assert x.is_sharded() and shard.grid.num_cores() == self.n, (x.memory_config(), self.n)
        assert shard.shape[0] * shard.shape[1] == self.tiles * 1024, shard.shape
        # the landing (the `_TpArAsync` parity's, bf16): RING slots of a core's input share on every output core, which
        # the senders and the owners write and the compute reads as the CB over it.  all_reduce_async validates its own
        # landing; this op takes it as it comes and checks its cores and size here (right for every call form)
        land_shard = land.memory_config().shard_spec
        share = self.tiles * (BFP8_TILE_BYTES if x.dtype == ttnn.bfloat8_b else TILE_BYTES)
        assert land_shard.grid.contains(shard.grid), (land_shard.grid, shard.grid)
        assert land_shard.shape[0] * land_shard.shape[1] // 1024 * TILE_BYTES >= RING * share, (land_shard.shape, share)
        if os.environ.get("GLM53_AR2_POISON") == "1":  # (a test's, eager only: the output all NaN before the op)
            import torch

            nan = torch.full(tuple(x.shape), float("nan"), dtype=torch.bfloat16)
            out = ttnn.from_torch(
                nan,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=x.memory_config(),
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        elif x.dtype == ttnn.bfloat16:
            out = ttnn.allocate_tensor_on_device(x.spec, self.mesh)
        else:  # (GLM53_AR_BFP8: a BFP8_b input, the sums bf16)
            mc = x.memory_config()
            spec = ttnn.TensorSpec(
                ttnn.Shape(list(x.shape)),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                mc.memory_layout,
                mc.shard_spec,
                mc.buffer_type,
            )
            out = ttnn.allocate_tensor_on_device(spec, self.mesh)
        sem_addr = ttnn.get_global_semaphore_address(sem)
        # (the L1 landing of part of the next KDA layer's x_proj, GLM53_W1D: on the cores this program leaves idle)
        riders = w1_dense.take(self.all_set)
        mesh_program = ttnn.MeshProgramDescriptor()
        for r in range(ROWS):
            for c in range(COLS):
                coord = ttnn.MeshCoordinate(r, c)
                mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = self._program(
                    r, c, x, land, out, sem_addr, riders
                )
        ttnn.generic_op([x, land, out], mesh_program)
        return out


_STATE: dict = {}


def row_ar2(mesh, x, land, sem) -> ttnn.Tensor:
    """The two-stage row all-reduce of x (sharded as `_TpArAsync.shard_for` gives) into a new tensor sharded alike."""
    shard = x.memory_config().shard_spec
    n, tiles = shard.grid.num_cores(), shard.shape[0] * shard.shape[1] // 1024
    key = (id(mesh), n, tiles)
    if key not in _STATE or _STATE[key].mesh is not mesh:
        _STATE[key] = RowAr2(mesh, n, tiles)
    return _STATE[key](x, land, sem)


__all__ = ["RowAr2", "row_ar2"]
