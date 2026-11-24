# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""A traced prefill's inputs in one host write (`common.FixedInputs` with a pack).

A captured prefill reads its inputs from fixed buffers (`FixedInputs.place`), each written from the host before the
replay: ~0.75 ms of host time a buffer (from_torch and the copy to the mesh's 32 chips), 45 of them for a pair of
sequence-parallel waves, with the device idle.  Here the host lays every buffer's device pages (row-major 4-byte
rows; TILE bf16 / fp32 tiles of faces, as ttnn tilizes) into one packed row per chip, `Pack.write` writes it in one
copy (`common.write_rows`, a reused host tensor), and `Pack.unpack` (one program, the captured prefill's first op)
copies each field's pages from there to its buffer, so that every op reads the buffers it read before.  The same bytes
as `place` wrote (a buffer's page size is checked; `tests/test_prefill_pack.py` compares the buffers).  No NoC
multicast: reads of the packed row and writes of the fields' pages, the pages split between CORES cores.
"""

from __future__ import annotations

import zlib

import torch

import ttnn
from models.demos.blackhole.glm53_flash.ttnn import common
from models.demos.blackhole.glm53_flash.ttnn.common import COLS, ROWS, put, write_rows

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/"
ALIGN = 64  # bytes: a field page's slot in the packed row (DRAM-aligned: the slot and the page at offset 0 mod 64)
CORES = 16  # the unpack's cores a chip
BATCH_BYTES = 16384  # an unpack's read: consecutive slots (0: a page a read)
PAGE_COST = 256  # bytes: a page's cost in splitting the pages between the cores (its write, its share of a read)


def _tiles(x: torch.Tensor) -> torch.Tensor:
    """x [.., H, W] -> its elements in ttnn's TILE order ([.., Ht, Wt] tiles of 4 faces of 16 x 16 rows, each face
    row-major), padded with zeros, flat."""
    if x.dim() < 2:
        x = x.reshape(1, -1)
    h, w = x.shape[-2], x.shape[-1]
    hp, wp = -(-h // 32) * 32, -(-w // 32) * 32
    if (hp, wp) != (h, w):
        x = torch.nn.functional.pad(x, (0, wp - w, 0, hp - h))
    lead = tuple(x.shape[:-2])
    n = len(lead)
    y = x.reshape(*lead, hp // 32, 2, 16, wp // 32, 2, 16)
    y = y.permute(*range(n), n, n + 3, n + 1, n + 4, n + 2, n + 5)  # Ht, Wt, face row, face col, 16, 16
    return y.contiguous().reshape(-1)


def chip_words(t: torch.Tensor, dtype, layout) -> torch.Tensor:
    """One chip's tensor -> its device pages' bytes as int32 words (flat), as ttnn.from_torch lays them out."""
    return rows_words(t[None], dtype, layout)[0]


def rows_words(x: torch.Tensor, dtype, layout) -> torch.Tensor:
    """x [ROWS-like n, *one chip's shape] -> [n, words]: each chip's device pages as int32 words, all chips in one
    conversion (a chip's shape of one dim is a [1, w] tensor's, as ttnn takes it)."""
    n = x.shape[0]
    if layout == ttnn.TILE_LAYOUT:
        if x.dim() < 3:
            x = x.reshape(n, 1, -1)
        if dtype == ttnn.bfloat16:
            return _tiles(x.to(torch.bfloat16)).view(torch.int16).view(torch.int32).reshape(n, -1)
        if dtype == ttnn.float32:
            return _tiles(x.to(torch.float32)).view(torch.int32).reshape(n, -1)
    elif layout == ttnn.ROW_MAJOR_LAYOUT:
        if dtype in (ttnn.int32, ttnn.uint32):
            return x.to(torch.int32).contiguous().reshape(n, -1)
        if dtype == ttnn.float32:
            return x.to(torch.float32).contiguous().reshape(n, -1).view(torch.int32)
    raise ValueError(f"prefill_pack: {dtype} {layout} is not packed")


class PackSession:
    """One prefill's fields, in the order they were first placed (a name placed again: its last values)."""

    def __init__(self):
        self.fields: dict = {}  # name -> (buffer, words [ROWS, n] int32, page bytes, pages)

    def add(self, name: str, buf, t: torch.Tensor, dims, dtype, layout) -> None:
        if dims == (0, None):  # (each mesh row's chips: its slice of dim 0)
            words = rows_words(t.reshape(ROWS, t.shape[0] // ROWS, *t.shape[1:]), dtype, layout)
        elif dims == (None, None):
            words = chip_words(t, dtype, layout)[None].expand(ROWS, -1)
        else:
            raise ValueError(f"prefill_pack: dims {dims} are not packed")
        page = buf.buffer_page_size()
        nbytes = words.shape[1] * 4
        pages = nbytes // page if page else 0
        if not page or pages * page != nbytes:
            raise ValueError(f"prefill_pack: {name}: {nbytes} bytes a chip, its buffer's pages {page} bytes")
        self.fields[name] = (buf, words, page, pages)

    def layout(self) -> tuple[list, int]:
        """([(name, buffer, page bytes, pages, slot bytes, byte offset in the packed row)], the row's bytes)."""
        out, at = [], 0
        for name, (buf, _, page, pages) in self.fields.items():
            slot = -(-page // ALIGN) * ALIGN
            out.append((name, buf, page, pages, slot, at))
            at += pages * slot
        return out, at


class Pack:
    """A FixedInputs set's packed row (`size` bytes a chip: [ROWS, size / 4] int32 row-major, one page a chip)."""

    def __init__(self, mesh, size: int):
        self.mesh = mesh
        self.width = max(ALIGN, -(-size // ALIGN) * ALIGN) // 4
        self.host = torch.zeros(ROWS, self.width, dtype=torch.int32)
        self.writes = 0  # (the packed rows written: FIXED_SKIP_SAME skips a prefill's unchanged inputs)
        self.device = put(mesh, self.host, dims=(0, None), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.synchronize_device(mesh)  # (write_rows' first write reads the buffer back: the device idle)
        write_rows(mesh, self.host, self.device, ttnn.int32)

    def write(self, session: PackSession) -> list:
        """The session's fields into the packed row on the host and the device (one copy) -> its layout."""
        layout, size = session.layout()
        if size > self.width * 4:
            raise ValueError(f"prefill_pack: {size} bytes a chip, the pack holds {self.width * 4}")
        changed = not common.FIXED_SKIP_SAME  # (read at each call)
        for name, _, page, pages, slot, at in layout:
            words = session.fields[name][1]
            w, s = page // 4, slot // 4
            dst = self.host[:, at // 4 : at // 4 + pages * s].view(ROWS, pages, s)[:, :, :w]
            new = words.reshape(ROWS, pages, w)
            if changed or not torch.equal(dst, new):  # (the device's row holds the host's: unchanged pages kept)
                dst.copy_(new)
                changed = True
        if changed:
            write_rows(self.mesh, self.host, self.device, ttnn.int32)
        self.writes += changed
        return layout

    def unpack(self, layout: list) -> None:
        """Each field's pages from the packed row to its buffer: one program over the mesh (inside a captured prefill,
        its first op), the fields' pages split between CORES cores by their cost (bytes, and PAGE_COST a page: a write
        each, a read for as many consecutive slots as fit a BATCH_BYTES buffer)."""
        cost = lambda page, n: n * (page + PAGE_COST)  # noqa: E731
        share = -(-sum(cost(page, n) for _, _, page, n, _, _ in layout) // CORES)
        per_core: list = [[] for _ in range(CORES)]
        core, used = 0, 0
        for _, buf, page, pages, slot, at in layout:
            p0 = 0
            while p0 < pages:
                if core == CORES - 1:
                    n = pages - p0
                else:
                    n = min(pages - p0, max(1, -(-(share - used) // (page + PAGE_COST))))
                per_core[core].append((buf.buffer_address(), page, p0, n, at + p0 * slot, slot))
                p0 += n
                used += cost(page, n)
                if used >= share and core < CORES - 1:
                    core, used = core + 1, 0
        n_max = max(1, max(len(c) for c in per_core))
        biggest = max(slot for _, _, _, _, slot, _ in layout)
        buf_bytes = max(biggest, BATCH_BYTES) if BATCH_BYTES else -(-max(p[2] for p in layout) // ALIGN) * ALIGN
        tag = zlib.crc32(repr([(p[2], p[3], p[4], p[5]) for p in layout]).encode()) & 0x7FFFFFFF
        cores = [ttnn.CoreCoord(c % 8, c // 8) for c in range(CORES)]
        rt = ttnn.RuntimeArgs()
        for c, work in enumerate(per_core):
            args = [self.device.buffer_address(), self.width * 4, len(work)]
            for item in work:
                args += list(item)
            rt[cores[c].x][cores[c].y] = args
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, CORES // 8 - 1))])
        kernel = ttnn.KernelDescriptor(
            kernel_source=KDIR + "prefill_unpack.cpp",
            core_ranges=core_set,
            compile_time_args=[0, n_max, buf_bytes, tag, int(BATCH_BYTES > 0)],  # (the tag: one program a layout)
            runtime_args=rt,
            config=ttnn.WriterConfigDescriptor(),
        )
        cb = ttnn.CBDescriptor(
            total_size=2 * buf_bytes + ALIGN,  # (the slack: the buffers start 64-byte aligned)
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.int32, page_size=ALIGN)],
        )
        program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])
        mesh_program = ttnn.MeshProgramDescriptor()
        whole = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(ROWS - 1, COLS - 1))
        mesh_program[whole] = program
        ttnn.generic_op([self.device, layout[0][1]], mesh_program)

    def deallocate(self) -> None:
        ttnn.deallocate(self.device)


__all__ = ["Pack", "PackSession", "chip_words", "rows_words"]
