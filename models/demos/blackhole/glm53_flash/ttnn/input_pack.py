# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""A decode step's small inputs in one host write (`TtGlm53Model.write_inputs`, `write_spec_inputs`).

The inputs written from the host before every decode or verify step (the token ids, the DSA positions' buffers, MTP's
keep flags: row-major int32 / uint32 tensors, each mesh row's values on its chips) cost a write each, ~0.12-0.25 ms of
host time per write for the mesh's 32 chips, whatever their size.  `InputPack` gathers them: a host
buffer [ROWS, W] holds every field's bytes (each page of a field in a 64-byte slot), written once to one device tensor
(`write`); `unpack` (one generic op, one core a chip, the step's first op inside its trace) copies each field's pages
from there to the field's own tensor, so that every op still reads the tensors it read before (their addresses fixed
before any capture).  The op uses no NoC multicast: one DRAM read of the packed page, then a write per field page.
"""

from __future__ import annotations

import math

import torch

import ttnn
from models.demos.blackhole.glm53_flash.ttnn.common import COLS, ROWS, put, write_rows

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/"
SLOT = 64  # bytes: a field page's slot in the packed row (DRAM-aligned, the same offset mod 64 as its destination)


class InputPack:
    """The per-step input tensors `tensors` (row-major, 4-byte elements, made by `put` with dims (0, None)) packed."""

    def __init__(self, mesh, tensors: list):
        self.mesh = mesh
        self.fields = {}  # id(tensor) -> (tensor, offset in 4-byte words, page words, pages, slot words)
        offset = 0
        for t in tensors:
            if t is None or id(t) in self.fields:
                continue
            assert t.layout == ttnn.ROW_MAJOR_LAYOUT and t.element_size() == 4, (t.layout, t.dtype)
            page = t.buffer_page_size()
            pages = math.prod(t.shape) * 4 // page
            slot = -(-page // SLOT) * SLOT
            self.fields[id(t)] = (t, offset, page // 4, pages, slot // 4)
            offset += pages * slot // 4
        self.width = max(offset, SLOT // 4)
        self.host = torch.zeros(ROWS, self.width, dtype=torch.int32)
        self.device = put(mesh, self.host, dims=(0, None), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def has(self, tensor) -> bool:
        return id(tensor) in self.fields

    def set(self, tensor, values: torch.Tensor) -> None:
        """values (the field's, dim 0 over the mesh rows, as `write_rows` takes them) into the host buffer."""
        _, offset, words, pages, slot = self.fields[id(tensor)]
        v = values.reshape(ROWS, pages, words).to(torch.int32)
        self.host[:, offset : offset + pages * slot].view(ROWS, pages, slot)[:, :, :words] = v

    def write(self) -> None:
        """The host buffer to the packed device tensor: the step's one input write."""
        write_rows(self.mesh, self.host, self.device, ttnn.uint32)

    def unpack(self) -> None:
        """Each field's pages from the packed tensor to the field's tensor (a device op: inside a captured step)."""
        packed_bytes = self.width * 4
        args = [self.device.buffer_address(), packed_bytes, len(self.fields)]
        for t, offset, words, pages, _ in self.fields.values():
            args += [t.buffer_address(), words * 4, pages, offset * 4]
        core = ttnn.CoreCoord(0, 0)
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        rt = ttnn.RuntimeArgs()
        rt[0][0] = args
        kernel = ttnn.KernelDescriptor(
            kernel_source=KDIR + "input_unpack.cpp",
            core_ranges=cores,
            compile_time_args=[0, len(self.fields)],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        size = packed_bytes + SLOT  # (the slack: the copy starts at a 64-byte aligned address)
        cb = ttnn.CBDescriptor(
            total_size=size,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=size)],
        )
        program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])
        mesh_program = ttnn.MeshProgramDescriptor()
        whole = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(ROWS - 1, COLS - 1))
        mesh_program[whole] = program
        first = next(iter(self.fields.values()))[0]
        ttnn.generic_op([self.device, first], mesh_program)

    def deallocate(self) -> None:
        ttnn.deallocate(self.device)


__all__ = ["InputPack"]
