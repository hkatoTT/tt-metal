# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The DSA decode's latents into the paged latent cache in one op (`TtDsa._step_latent`),
through ttnn.generic_op, the same program on every chip (each chip its own row's users):

  latent_write(latent, cache, cur_pos, owners, pages): row v of latent [1, 1, B, D] bf16 (tiled) into row cur_pos[v]
      % 32 of page owner * pages + cur_pos[v] // 32 of cache [U pages + 1, 1, 32, D] bf16 (tiled), owner = v mod
      owners (MTP's verify step: virtual user v on user v mod U's pages, as `TtDsa.alloc_positions`' tables);
      cur_pos [B] int32 row-major, -1: none.

In place of the reshape of the latents to [1, B, 1, D] (each token a tile row of its own: 47 us at 32 tokens a row)
and, per group of 16 users, a slice, the move to height-sharded L1 and paged_update_cache (which reads, untilizes,
tilizes and writes back each token's whole page).  The same bytes land (copies).  Kernel: kernels/dsa/latent_write.cpp.

Each core takes a range of owners (users) and a range of the D / 32 tiles across a token: a token's row is 2 face rows
of 32 bytes a tile, moved one by one, so one core a user (8 tokens: 16 tiles x 2 x 8 = 256 reads and 256 writes) took
17.8 / 32.8 / 18.0 us a DSA layer at 8 / 32 / 128 users (T = 8 / 8 / 4).  CORES (64): the cores in all, each
owner range's tiles split over CORES // ranges of them (1: one core an owner range).
"""

from __future__ import annotations

import ttnn

from .common import COLS, ROWS, core_rects

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/dsa/"
MAX_CORES = 32  # owner ranges
CORES = 64
SLOT = 64


def latent_write(mesh, latent: ttnn.Tensor, cache: ttnn.Tensor, cur_pos: ttnn.Tensor, owners: int, pages: int) -> None:
    tokens, d = latent.shape[-2], latent.shape[-1]
    assert latent.dtype == ttnn.bfloat16 and latent.layout == ttnn.TILE_LAYOUT, latent
    assert cache.dtype == ttnn.bfloat16 and cache.layout == ttnn.TILE_LAYOUT and cache.shape[-1] == d, cache
    assert cache.shape[-2] == 32 and d % 32 == 0 and tokens % owners == 0, (cache.shape, tokens, owners)
    assert cur_pos.dtype == ttnn.int32 and cur_pos.shape[-1] == tokens, cur_pos
    tiles = d // 32
    pos_page = tokens * 4
    grid = mesh.compute_with_storage_grid_size()
    n_cores = min(owners, MAX_CORES, grid.x * grid.y)
    per = -(-owners // n_cores)
    parts = [(o, min(owners, o + per)) for o in range(0, owners, per)]
    # each owner range's tiles over `split` cores (ranges of `span` tiles; the last may be shorter)
    split = max(1, min(tiles, CORES // len(parts), grid.x * grid.y // len(parts)))
    span = -(-tiles // split)
    spans = [(t, min(tiles, t + span)) for t in range(0, tiles, span)]
    work = [(o, t) for o in parts for t in spans]
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(work))]
    cores = core_rects(coords)
    rt = ttnn.RuntimeArgs()
    for c, ((o0, o1), (t0, t1)) in zip(coords, work):
        rt[c.x][c.y] = [latent.buffer_address(), cache.buffer_address(), cur_pos.buffer_address(), o0, o1, t0, t1]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KDIR + "latent_write.cpp",
        core_ranges=cores,
        compile_time_args=[tokens, owners, pages, tiles, pos_page, 0]
        + ttnn.TensorAccessorArgs(latent).get_compile_time_args()
        + ttnn.TensorAccessorArgs(cache).get_compile_time_args()
        + ttnn.TensorAccessorArgs(cur_pos).get_compile_time_args(),
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    # the positions, then a token's 2 x span 64-byte slots for each of a core's tokens
    size = -(-pos_page // SLOT) * SLOT + per * (tokens // owners) * 2 * span * SLOT
    cb = ttnn.CBDescriptor(
        total_size=size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=size)],
    )
    program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])
    mesh_program = ttnn.MeshProgramDescriptor()
    whole = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(ROWS - 1, COLS - 1))
    mesh_program[whole] = program
    ttnn.generic_op([latent, cur_pos, cache], mesh_program)


# q_heads: the decode's q into its heads by one op, in place of ttnn.reshape + transpose (up to 16 tokens a row: 11.6 +
# 4.1 us at 8) or nlp_create_qkv_heads (on 1 core a tile row: 11.7 / 14.0 us at 32 / 64 tokens)
Q_HEADS_CORES = 64
Q_HEADS_GROUP = 4  # tiles read at once by a core


def q_heads(mesh, q: ttnn.Tensor, heads: int, copies: int = 1) -> ttnn.Tensor:
    """q [1, 1, B, H D] bf16 (tiled) -> [1, H copies, B, D] bf16 (tiled, q's memory config): head h's copy s at
    h copies + s.  Whole tiles moved (the same bytes).  Kernel: kernels/dsa/q_heads.cpp."""
    rows, width = q.shape[-2], q.shape[-1]
    assert q.dtype == ttnn.bfloat16 and q.layout == ttnn.TILE_LAYOUT and width % (heads * 32) == 0, q
    d = width // heads
    m, w = -(-rows // 32), d // 32
    spec = ttnn.TensorSpec(
        ttnn.Shape([1, heads * copies, rows, d]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
        buffer_type=q.memory_config().buffer_type,
    )  # fmt: skip
    out = ttnn.allocate_tensor_on_device(spec, mesh)
    tiles = heads * copies * m * w
    n = max(1, min(Q_HEADS_CORES, tiles))
    per = -(-tiles // n)
    parts = [(a, min(tiles, a + per)) for a in range(0, tiles, per)]
    grid = mesh.compute_with_storage_grid_size()
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(parts))]
    cores = core_rects(coords)
    rt = ttnn.RuntimeArgs()
    for c, (a, b) in zip(coords, parts):
        rt[c.x][c.y] = [q.buffer_address(), out.buffer_address(), a, b]
    group = min(Q_HEADS_GROUP, per)
    kernel = ttnn.KernelDescriptor(
        kernel_source=KDIR + "q_heads.cpp",
        core_ranges=cores,
        compile_time_args=[m, heads, w, copies, group, 0]
        + ttnn.TensorAccessorArgs(q).get_compile_time_args()
        + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    cb = ttnn.CBDescriptor(
        total_size=group * 2048,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=group * 2048)],
    )
    program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])
    mesh_program = ttnn.MeshProgramDescriptor()
    whole = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(ROWS - 1, COLS - 1))
    mesh_program[whole] = program
    ttnn.generic_op([q, out], mesh_program)
    return out
