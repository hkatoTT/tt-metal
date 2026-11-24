# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The greedy head's per-chip top in one op (`TtLmHead.greedy_mesh`), through ttnn.generic_op:

  head_argmax(logits, valid, width) -> (value, ids): logits [1, 1, M, W] fp32 (tiled; chip k = 4 r + c holds the
      global ids k width .. k width + W - 1, the first `valid[k]` of them real) -> for each row the largest real logit
      and the smallest global id where it is hit, [1, 1, M, 1] fp32 each (tiled, column 0), as the mask add, max,
      compare, multiply, id add and min made them (the same bits: exact compares, -0.0 taken as +0.0 as the masked add
      made it; a tie keeps the smallest id).

The column tiles of each tile row are split into runs over the grid; a core's NCRISC scans the rows 0-15 of its run,
its BRISC the rows 16-31, on the RISC-V, and the run 0 core of each tile row folds the runs (kernels/head_argmax/
argmax.cpp).  No multicast; the partials go to the root by NoC writes and a semaphore.
"""

from __future__ import annotations

import os

import ttnn

from .common import COLS, ROWS, core_rects

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/head_argmax/"
SEM = 0
# runs a tile row at most (each core a run of column tiles; the root folds them): fewer, longer runs trade the scan for
# the fold
MAX_RUNS = 48


def fast_on() -> bool:
    """GLM53_HEAD_ARGMAX_FAST (read at each call; on by default, 0: the keyed scan alone): a full face of 16 values
    scanned first by its raw signed bits, the keyed scan only where the face has no positive value (the same (key,
    column): kernels/head_argmax/argmax.cpp).  Alone at 64 / 256 / 512 rows 57.3 / 99.3 / 177.6 -> 50.3 / 79.6 / 139.1
    us."""
    return os.environ.get("GLM53_HEAD_ARGMAX_FAST", "1") == "1"


HALF, SLOT, TILE = 2048, 128, 4096


class HeadArgmax:
    def __init__(self, mesh, rows: int, width: int, valid: list[int], id_width: int):
        """rows M, width W (columns, whole tiles), valid[k] the real columns of chip k = 4 r + c, id_width the ids a
        chip (chip k's first global id k id_width)."""
        assert width % 32 == 0 and len(valid) == ROWS * COLS, (width, len(valid))
        self.mesh, self.rows, self.width = mesh, rows, width
        self.valid, self.id_width = valid, id_width
        grid = mesh.compute_with_storage_grid_size()
        cores = grid.x * grid.y
        tr = -(-rows // 32)
        wt = width // 32
        runs = max(1, min(wt, cores // tr, MAX_RUNS))
        per = -(-wt // runs)
        runs = -(-wt // per)
        self.tr, self.wt, self.runs, self.per = tr, wt, runs, per
        self.coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(tr * runs)]
        self.cb_bytes = 2 * per * HALF + runs * 2 * SLOT + 2 * TILE

    @property
    def cores(self) -> ttnn.CoreRangeSet:
        """The cores as rectangles (a program goes out range by range)."""
        return core_rects(self.coords)

    def _program(self, k: int, logits, value, ids) -> ttnn.ProgramDescriptor:
        rts = [ttnn.RuntimeArgs(), ttnn.RuntimeArgs()]
        for i, c in enumerate(self.coords):
            t, run = divmod(i, self.runs)
            root = self.mesh.worker_core_from_logical_core(self.coords[t * self.runs])
            ct0, ct1 = run * self.per, min(self.wt, (run + 1) * self.per)
            args = [logits.buffer_address(), value.buffer_address(), ids.buffer_address(), t, ct0, ct1, run]
            args += [root.x, root.y, self.valid[k], k * self.id_width]
            for rt in rts:
                rt[c.x][c.y] = args
        kernels = []
        for half, config in ((0, ttnn.ReaderConfigDescriptor()), (1, ttnn.WriterConfigDescriptor())):
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=KDIR + "argmax.cpp",
                    core_ranges=self.cores,
                    compile_time_args=[half, self.wt, self.runs, self.rows, self.per, 0, SEM, int(fast_on())]
                    + ttnn.TensorAccessorArgs(logits).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(value).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(ids).get_compile_time_args(),
                    runtime_args=rts[half],
                    config=config,
                )
            )
        cb = ttnn.CBDescriptor(
            total_size=self.cb_bytes,
            core_ranges=self.cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=self.cb_bytes)
            ],
        )
        sems = [ttnn.SemaphoreDescriptor(id=SEM, core_ranges=self.cores, initial_value=0)]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=[cb])

    def __call__(self, logits: ttnn.Tensor):
        assert logits.dtype == ttnn.float32 and logits.layout == ttnn.TILE_LAYOUT, logits
        assert tuple(logits.shape)[-2:] == (self.rows, self.width), (logits.shape, self.rows, self.width)
        spec = ttnn.TensorSpec(
            ttnn.Shape([1, 1, self.rows, 1]), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM
        )
        value = ttnn.allocate_tensor_on_device(spec, self.mesh)
        ids = ttnn.allocate_tensor_on_device(spec, self.mesh)
        mesh_program = ttnn.MeshProgramDescriptor()
        for r in range(ROWS):
            for c in range(COLS):
                coord = ttnn.MeshCoordinate(r, c)
                mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = self._program(COLS * r + c, logits, value, ids)
        ttnn.generic_op([logits, value, ids], mesh_program)
        return value, ids


_STATE: dict = {}


def head_argmax(mesh, logits, valid: list[int], id_width: int):
    key = (id(mesh), tuple(logits.shape), tuple(valid), id_width)
    if key not in _STATE or _STATE[key].mesh is not mesh:
        _STATE[key] = HeadArgmax(mesh, logits.shape[-2], logits.shape[-1], valid, id_width)
    return _STATE[key](logits)
