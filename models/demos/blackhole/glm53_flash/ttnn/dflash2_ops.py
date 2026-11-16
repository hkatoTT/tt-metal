# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The DFlash2 drafter's own ops (`dflash2.TtDFlash2`), through ttnn.generic_op; data movement and RISC-V scans only
(no multicast, no fabric), the same program on every chip unless noted:

  rope_rows(idx, cos_tab, sin_tab, rows, div, mod, cur=(n, div, add)) -> (cos, sin[, cur_pos]): row o of the outputs
      [1, 1, rows, 128] fp32 (tiled) is the tables' row idx[o // div] + (o % mod if mod); cur_pos [n] int32 the
      positions idx[v // div] + add (the paged sdpa decode's).  One table of the halves (sin_tab None: the drafter's)
      or cos and sin apart.  kernels/dflash2/rope_rows.cpp.
  kv_write(src, cache, idx, users, ...): head rows of src (tiled bf16) into the paged cache [pages, 1, 32, 128]
      (kv_write.cpp: the source and destination maps there).
  top16(logits, valid, id_width) -> [1, 1, M, 32] fp32 row-major: per row the chip's 16 largest logits (descending)
      and their global ids; per chip (chip k = 4 r + c holds ids k id_width ..).  top16.cpp.
  top16_merge(gathered, bonus, users, slots) -> (sel, cand, pred): the 32 chips' lists merged for this mesh row's slot
      rows (per chip: its mesh row's users), and the codebooks' id rows.  top16_merge.cpp.
  walk(scores, sel, bonus, slots) -> drafts [U, 8] uint32 row-major: the selector's greedy walk.  walk.cpp.
"""

from __future__ import annotations

import ttnn

from .common import COLS, ROWS
from .common import core_rects as _rects

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/dflash2/"
FP32_TILE = 4096
TOPN = 16


def _grid(mesh):
    g = mesh.compute_with_storage_grid_size()
    return g.x, g.y


def _cores(mesh, n: int, start: int = 0):
    gx, _ = _grid(mesh)
    coords = [ttnn.CoreCoord(i % gx, i // gx) for i in range(start, start + n)]
    return coords, _rects(coords)


def _scratch(cores, size: int) -> ttnn.CBDescriptor:
    """One circular buffer used as plain L1 scratch."""
    size = -(-size // 64) * 64
    return ttnn.CBDescriptor(
        total_size=size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=size)],
    )


def _whole(program) -> ttnn.MeshProgramDescriptor:
    mesh_program = ttnn.MeshProgramDescriptor()
    whole = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(ROWS - 1, COLS - 1))
    mesh_program[whole] = program
    return mesh_program


def _alloc(mesh, shape, dtype, layout):
    spec = ttnn.TensorSpec(ttnn.Shape(list(shape)), dtype, layout, buffer_type=ttnn.BufferType.DRAM)
    return ttnn.allocate_tensor_on_device(spec, mesh)


def _split(n: int, parts: int) -> list[tuple[int, int]]:
    parts = max(1, min(parts, n))
    per = -(-n // parts)
    return [(a, min(n, a + per)) for a in range(0, n, per)]


def rope_rows(
    mesh,
    idx,
    cos_tab,
    sin_tab,
    rows: int,
    div: int,
    mod: int = 0,
    cur: tuple | None = None,
    max_cores=120,
):
    """idx [.., n] int32 row-major (one page), the tables [T, 128] fp32 row-major -> cos, sin [1, 1, rows, 128] fp32
    tiled (rows a multiple of 32) and, with cur = (n_cur, cur_div, cur_add), cur_pos [n_cur] int32 row-major.

    sin_tab None: cos_tab is one table of the two halves (cos's columns 0-63, sin's 64-127; rope_rows.cpp fmt 1).
    Each output row reads the table's row of its position (rope_rows.cpp wmode 0; the host-made rows of wmode 1 / 2
    were removed: the kernel's base tensor is the index, unread)."""
    assert rows % 32 == 0 and idx.dtype == ttnn.int32 and idx.layout == ttnn.ROW_MAJOR_LAYOUT, (rows, idx)
    assert cos_tab.element_size() == 4 and cos_tab.shape[-1] == 128, cos_tab  # (fp32 values; the step's: their bits)
    fmt = 1 if sin_tab is None else 0
    sin_tab = cos_tab if sin_tab is None else sin_tab
    n_idx = idx.shape[-1]
    wmode, wrows, udiv, umod, woff, n_base = 0, 1, 1, 1, 0, 0
    cos = _alloc(mesh, [1, 1, rows, 128], ttnn.float32, ttnn.TILE_LAYOUT)
    sin = _alloc(mesh, [1, 1, rows, 128], ttnn.float32, ttnn.TILE_LAYOUT)
    n_cur, cur_div, cur_add = cur if cur else (0, 1, 0)
    cur_pos = _alloc(mesh, [max(n_cur, 1)], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    parts = _split(rows // 32 * 4, max_cores)  # items (tile row, column tile)
    coords, cores = _cores(mesh, len(parts))
    rt = ttnn.RuntimeArgs()
    for i, (c, (a, b)) in enumerate(zip(coords, parts)):
        rt[c.x][c.y] = [
            idx.buffer_address(),
            cos_tab.buffer_address(),
            sin_tab.buffer_address(),
            cos.buffer_address(),
            sin.buffer_address(),
            cur_pos.buffer_address(),
            a,
            b,
            int(i == 0),
            idx.buffer_address(),
        ]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KDIR + "rope_rows.cpp",
        core_ranges=cores,
        compile_time_args=[n_idx, div, mod, n_cur, cur_div, cur_add, cos_tab.shape[-2], 0]
        + [fmt, wmode, wrows, udiv, umod, woff, n_base]
        + ttnn.TensorAccessorArgs(idx).get_compile_time_args()
        + ttnn.TensorAccessorArgs(cos_tab).get_compile_time_args()
        + ttnn.TensorAccessorArgs(cos).get_compile_time_args()
        + ttnn.TensorAccessorArgs(cur_pos).get_compile_time_args()
        + ttnn.TensorAccessorArgs(idx).get_compile_time_args(),
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    slot = lambda n: -(-n // 64) * 64  # noqa: E731
    size = slot(n_idx * 4) + 2 * FP32_TILE + slot(max(n_cur * 4, 64)) + slot(max(n_base * 4, 64))
    program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_scratch(cores, size)])
    tensors = [idx, cos_tab] + ([] if sin_tab is cos_tab else [sin_tab])
    ttnn.generic_op(tensors + [cos, sin, cur_pos], _whole(program))
    if cur:
        return cos, sin, cur_pos
    ttnn.deallocate(cur_pos)
    return cos, sin


def _kv_kernel(
    src,
    cache,
    idx,
    other,
    mode,
    n_idx,
    coords,
    cores,
    parts,
    *,
    tokens,
    block,
    layers,
    heads,
    n_users,
    rows,
    cols,
    pages,
    layer_pages,
    page_base,
    batch,
    ring,
):
    rt = ttnn.RuntimeArgs()
    for c, (a, b) in zip(coords, parts):
        rt[c.x][c.y] = [
            src.buffer_address(),
            cache.buffer_address(),
            idx.buffer_address(),
            other.buffer_address(),
            a,
            b,
            page_base,
        ]
    return ttnn.KernelDescriptor(
        kernel_source=KDIR + "kv_write.cpp",
        core_ranges=cores,
        compile_time_args=[mode, block, n_idx, n_users, layers, heads, src.shape[-1] // 32, *rows, *cols]
        + [pages, layer_pages, batch, 0, KV_BITS[cache.dtype], int(ring)]
        + ttnn.TensorAccessorArgs(src).get_compile_time_args()
        + ttnn.TensorAccessorArgs(cache).get_compile_time_args()
        + ttnn.TensorAccessorArgs(idx).get_compile_time_args()
        + ttnn.TensorAccessorArgs(other).get_compile_time_args()
        + [-(-tokens // len(parts))],  # (the split in the compile-time args)
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )


# kv_write's cache formats (the drafter's K / V cache in BFP8_b / BFP4_b, GLM53_DFLASH2_KV; the source the same)
KV_BITS = {ttnn.bfloat16: 0, ttnn.bfloat8_b: 8, ttnn.bfloat4_b: 4}


def kv_write(
    mesh,
    writes,
    idx,
    users=None,
    *,
    tokens: int,
    block: int,
    layers: int,
    heads: int,
    n_users: int,
    pages: int,
    layer_pages: int,
    page_base: int = 0,
    max_cores: int = 60,
    ring: bool = False,
):
    """Head rows of sources (tiled bf16) into paged caches [pages, 1, 32, 128] bf16 (tiled), in one op: writes =
    [(src, cache, rows, cols), ..] (each on its own cores), for `tokens` tokens; rows = (sU, sP, sL, sH), cols =
    (c0, cL, cH) (128-column blocks) as kv_write.cpp maps them, from page `page_base`.  users None: mode 0 (idx [U]
    each user's first position, user r // block, position idx[u] + r % block), else mode 1 (idx and users [tokens]
    int32: each token's position and user).  ring: positions taken mod pages x 32 (the drafter's last positions only,
    dflash2.RING)."""
    mode = 0 if users is None else 1
    n_idx = idx.shape[-1]
    if mode == 1:
        assert users.shape[-1] == n_idx == tokens, (users.shape, n_idx, tokens)
    other = users if users is not None else idx
    batch = 32
    parts = _split(tokens, max_cores // len(writes))
    kernels, all_coords, tensors = [], [], [idx, other]
    for w, (src, cache, rows, cols) in enumerate(writes):
        assert src.dtype == cache.dtype and src.layout == ttnn.TILE_LAYOUT and src.shape[-1] % 128 == 0, src
        assert cache.dtype in KV_BITS and tuple(cache.shape)[-3:] == (1, 32, 128), cache
        coords, cores = _cores(mesh, len(parts), start=w * len(parts))
        all_coords += coords
        kernels.append(
            _kv_kernel(src, cache, idx, other, mode, n_idx, coords, cores, parts, tokens=tokens, block=block,
                       layers=layers, heads=heads, n_users=n_users, rows=rows, cols=cols, pages=pages,
                       layer_pages=layer_pages, page_base=page_base, batch=batch, ring=ring)  # fmt: skip
        )
        tensors += [src, cache]
    every = _rects(all_coords)
    size = 2 * (-(-n_idx * 4 // 64) * 64) + batch * 4 * 2 * 64 * (1 if writes[0][1].dtype == ttnn.bfloat16 else 2)
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[_scratch(every, size)])
    ttnn.generic_op(tensors, _whole(program))


class Top16:
    """top16 for logits of `rows` x `width`; the rows g with g % period >= real (a mesh row's padding) not live (not
    written).  The live rows of each half of a tile row (a prefix of it) split into groups of at most G rows, the
    columns into runs; items (group, run) over both RISCs of the cores, G and the runs chosen by a cycle model of the
    scan, the merge and the reads."""

    SCAN = 25  # cycles a value scanned (the RISC's loads from L1, measured)
    MERGE = 60  # .. a row and run merged at the root
    READ = 40  # .. a tile's two face-row reads issued
    CB_MAX = 512 * 1024

    def __init__(self, mesh, rows: int, width: int, valid: list[int], id_width: int, period: int, real: int):
        assert width % 32 == 0 and len(valid) == ROWS * COLS, (width, len(valid), rows)
        self.mesh, self.rows, self.width, self.valid, self.id_width = mesh, rows, width, valid, id_width
        self.period, self.real = period, real
        gx, gy = _grid(mesh)
        ncores = gx * gy
        wt = width // 32
        halves = []  # (tile row, half, live rows: a prefix of the half)
        for t in range(-(-rows // 32)):
            for h in (0, 1):
                live = [r for r in range(16) if t * 32 + h * 16 + r < rows and (t * 32 + h * 16 + r) % period < real]
                assert live == list(range(len(live))), (rows, period, real, t, h)
                if live:
                    halves.append((t, h, len(live)))
        best = None
        for G in range(1, 17):
            groups = [(t, h, a, b) for t, h, n in halves for a, b in _split(n, -(-n // G))]
            g_max = max(b - a for *_, a, b in groups)
            runs = min(wt, (2 * ncores) // len(groups))
            if runs < 1:
                continue
            per = -(-wt // runs)
            runs = -(-wt // per)
            cb = 2 * per * 2 * g_max * 64 + 2 * runs * g_max * 128 + 2 * g_max * 128
            if cb > self.CB_MAX:
                continue
            cost = g_max * per * 32 * self.SCAN + per * self.READ + g_max * runs * self.MERGE
            if best is None or cost < best[0]:
                best = (cost, g_max, groups, runs, per, cb)
        assert best is not None, (rows, period, real)
        _, self.G, groups, self.runs, self.per, self.cb_bytes = best
        self.wt = wt
        self.items = []  # (core, risc, args without the addresses and the chip's), root first in a group
        n = 0
        for t, h, a, b in groups:
            root = None
            for run in range(self.runs):
                core, risc = n % ncores, n // ncores
                n += 1
                coord = ttnn.CoreCoord(core % gx, core // gx)
                if root is None:
                    root = (coord, risc)
                ct0, ct1 = run * self.per, min(wt, (run + 1) * self.per)
                self.items.append((coord, risc, [t, h, a, b, ct0, ct1, run], root))
        assert n <= 2 * ncores, (n, ncores)
        used = sorted({(c.x, c.y) for c, *_ in self.items}, key=lambda v: (v[1], v[0]))
        self.cores = _rects([ttnn.CoreCoord(*v) for v in used])
        self.risc_cores = [[c for c, risc, *_ in self.items if risc == k] for k in (0, 1)]
        self.risc_cores = [_rects(r) if r else None for r in self.risc_cores]

    def _program(self, k: int, logits, out) -> ttnn.ProgramDescriptor:
        rts = [ttnn.RuntimeArgs(), ttnn.RuntimeArgs()]
        for c, risc, args, (root, root_risc) in self.items:
            rc = self.mesh.worker_core_from_logical_core(root)
            rts[risc][c.x][c.y] = [logits.buffer_address(), out.buffer_address(), *args, rc.x, rc.y, root_risc] + [
                self.valid[k],
                k * self.id_width,
            ]
        kernels = []
        for risc, config in ((0, ttnn.ReaderConfigDescriptor()), (1, ttnn.WriterConfigDescriptor())):
            if self.risc_cores[risc] is None:
                continue
            ct = [risc, self.wt, self.runs, self.per, self.G, 0, 0, self.rows, self.period, self.real]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=KDIR + "top16.cpp",
                    core_ranges=self.risc_cores[risc],
                    compile_time_args=ct
                    + ttnn.TensorAccessorArgs(logits).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                    runtime_args=rts[risc],
                    config=config,
                )
            )
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.cores, initial_value=0) for i in (0, 1)]  # a RISC's
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=[_scratch(self.cores, self.cb_bytes)])

    def __call__(self, logits):
        assert logits.dtype == ttnn.float32 and logits.layout == ttnn.TILE_LAYOUT, logits
        assert logits.shape[-1] == self.width and logits.shape[-2] >= self.rows, (logits.shape, self.rows)
        out = _alloc(self.mesh, [1, 1, self.rows, 2 * TOPN], ttnn.float32, ttnn.TILE_LAYOUT)
        mesh_program = ttnn.MeshProgramDescriptor()
        for r in range(ROWS):
            for c in range(COLS):
                coord = ttnn.MeshCoordinate(r, c)
                mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = self._program(COLS * r + c, logits, out)
        ttnn.generic_op([logits, out], mesh_program)
        return out


_TOP16: dict = {}


def top16(mesh, logits, rows: int, valid: list[int], id_width: int, period: int | None = None, real: int | None = None):
    """-> [1, 1, rows, 32] fp32 tiled: per row the 16 values (columns 0-15), then the 16 ids (16-31); rows g with
    g % period >= real not written."""
    period = rows if period is None else period
    real = period if real is None else real
    key = (id(mesh), rows, logits.shape[-1], tuple(valid), id_width, period, real)
    if key not in _TOP16 or _TOP16[key].mesh is not mesh:
        _TOP16[key] = Top16(mesh, rows, logits.shape[-1], valid, id_width, period, real)
    return _TOP16[key](logits)


def top16_merge(
    mesh, gathered, bonus, users: int, slots: int, period: int, block_rows: int | None = None, max_cores: int = 112
):
    """gathered [chips, rows, 32] fp32 tiled (top16's tiles of every chip, chip k = 4 r + c at block k of block_rows
    rows, whole tiles: 8 period by default; mesh row r's slot rows at r period + u slots + s), bonus [1, users] uint32
    -> (sel [users slots, 32] uint32, cand and pred [1, users slots 32] uint32), per chip its mesh row's."""
    assert gathered.dtype == ttnn.float32 and gathered.layout == ttnn.TILE_LAYOUT, gathered
    chips = ROWS * COLS
    n_rows = ROWS * period if block_rows is None else block_rows
    assert n_rows % 32 == 0 and users * slots <= period and ROWS * period <= n_rows, (users, slots, period, n_rows)
    sel = _alloc(mesh, [users * slots, 2 * TOPN], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    cand = _alloc(mesh, [1, users * slots * 2 * TOPN], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    pred = _alloc(mesh, [1, users * slots * 2 * TOPN], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    parts = _split(users * slots, max_cores)  # items (user, slot)
    coords, cores = _cores(mesh, len(parts))
    ct = [n_rows, slots, users, chips, 0]
    ct += ttnn.TensorAccessorArgs(gathered).get_compile_time_args()
    ct += ttnn.TensorAccessorArgs(bonus).get_compile_time_args()
    ct += ttnn.TensorAccessorArgs(sel).get_compile_time_args()
    ct += ttnn.TensorAccessorArgs(cand).get_compile_time_args()
    ct += [-(-users * slots // len(parts))]  # (the split in the compile-time args)
    size = -(-users * 4 // 64) * 64 + chips * 128 + 3 * 128
    mesh_program = ttnn.MeshProgramDescriptor()
    for r in range(ROWS):
        rt = ttnn.RuntimeArgs()
        for c, (a, b) in zip(coords, parts):
            rt[c.x][c.y] = [
                gathered.buffer_address(),
                bonus.buffer_address(),
                sel.buffer_address(),
                cand.buffer_address(),
                pred.buffer_address(),
                a,
                b,
                r * period,
            ]
        kernel = ttnn.KernelDescriptor(
            kernel_source=KDIR + "top16_merge.cpp",
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_scratch(cores, size)])
        span = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, 0), ttnn.MeshCoordinate(r, COLS - 1))
        mesh_program[span] = program
    ttnn.generic_op([gathered, bonus, sel, cand, pred], mesh_program)
    return sel, cand, pred


def walk(mesh, scores, sel, bonus, users: int, slots: int, max_cores: int = 16):
    """scores [1, users slots, 32, 32] fp32 tiled, sel (top16_merge's), bonus [1, users] uint32 -> drafts
    [users, 8] uint32 row-major (the bonus token, the drafts, zeros)."""
    assert scores.dtype == ttnn.float32 and scores.layout == ttnn.TILE_LAYOUT, scores
    assert slots <= 7, slots
    out = _alloc(mesh, [users, 8], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    parts = _split(users, max_cores)
    coords, cores = _cores(mesh, len(parts))
    rt = ttnn.RuntimeArgs()
    for c, (a, b) in zip(coords, parts):
        rt[c.x][c.y] = [
            scores.buffer_address(),
            sel.buffer_address(),
            bonus.buffer_address(),
            out.buffer_address(),
            a,
            b,
        ]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KDIR + "walk.cpp",
        core_ranges=cores,
        compile_time_args=[slots, users, 0]
        + ttnn.TensorAccessorArgs(scores).get_compile_time_args()
        + ttnn.TensorAccessorArgs(sel).get_compile_time_args()
        + ttnn.TensorAccessorArgs(bonus).get_compile_time_args()
        + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    size = -(-users * 4 // 64) * 64 + slots * 1024 + slots * 128 + 64
    program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_scratch(cores, size)])
    ttnn.generic_op([scores, sel, bonus, out], _whole(program))
    return out


def heads_split(
    mesh, qkv, users: int, block: int, group: int, kv_heads: int, k0: int, max_cores: int = 64, combined: bool = False
):
    """The block's projections [1, 1, U B, W] bf16 (q heads at 128-column blocks 0 .., k heads at k0 ..) -> q, k
    [1, kv_heads U, 32, 128] bf16 (virtual user (u, j): q rows h B + pos, k rows pos; the rest zero).  heads_split.cpp.
    combined: the two as one [1, 2 kv_heads U, 32, 128] (q's virtual users, then k's)."""
    assert qkv.dtype == ttnn.bfloat16 and qkv.layout == ttnn.TILE_LAYOUT and 32 % block == 0, (qkv, block)
    vu = users * kv_heads
    if combined:
        q = k = _alloc(mesh, [1, 2 * vu, 32, 128], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    else:
        q = _alloc(mesh, [1, vu, 32, 128], ttnn.bfloat16, ttnn.TILE_LAYOUT)
        k = _alloc(mesh, [1, vu, 32, 128], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    k_tile0 = vu * 4 if combined else 0
    parts = _split(vu * 4, max_cores)  # items (virtual user, column tile)
    coords, cores = _cores(mesh, len(parts))
    rt = ttnn.RuntimeArgs()
    for c, (a, b) in zip(coords, parts):
        rt[c.x][c.y] = [qkv.buffer_address(), q.buffer_address(), k.buffer_address(), a, b, k_tile0]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KDIR + "heads_split.cpp",
        core_ranges=cores,
        compile_time_args=[block, group, kv_heads, qkv.shape[-1] // 32, k0, 0]
        + ttnn.TensorAccessorArgs(qkv).get_compile_time_args()
        + ttnn.TensorAccessorArgs(q).get_compile_time_args(),
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_scratch(cores, (group + 2) * 2048)])
    ttnn.generic_op([qkv, q] if combined else [qkv, q, k], _whole(program))
    return q if combined else (q, k)


def heads_merge(mesh, attn, users: int, block: int, group: int, kv_heads: int, max_cores: int = 64):
    """The paged sdpa decode's output [1, kv_heads U, 32, 128] bf16 -> [1, 1, U B, kv_heads group 128] bf16 (the
    block's rows, the chip's q heads in order).  heads_merge.cpp."""
    assert attn.dtype == ttnn.bfloat16 and attn.layout == ttnn.TILE_LAYOUT, attn
    rows = users * block
    out = _alloc(mesh, [1, 1, rows, kv_heads * group * 128], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    tiles = -(-rows // 32) * kv_heads * group * 4
    parts = _split(tiles, max_cores)
    coords, cores = _cores(mesh, len(parts))
    rt = ttnn.RuntimeArgs()
    for c, (a, b) in zip(coords, parts):
        rt[c.x][c.y] = [attn.buffer_address(), out.buffer_address(), a, b]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KDIR + "heads_merge.cpp",
        core_ranges=cores,
        compile_time_args=[block, group, kv_heads, users, 0]
        + ttnn.TensorAccessorArgs(attn).get_compile_time_args()
        + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_scratch(cores, (32 // block + 1) * 2048)])
    ttnn.generic_op([attn, out], _whole(program))
    return out


def conv_constants(mesh, block: int, conv_group: int = 16):
    """The conv op's fixed operands: E [1, 1, 16 x 32, 32] bf16 (tile p: group rows 2 p, 2 p + 1 spread over the 16
    channels each) and S [1, 1, 32, 32] bf16 (row i takes row i - 1, none at a block's start: B divides 32)."""
    import torch

    from .common import put

    assert 32 % block == 0 and conv_group == 16, (block, conv_group)
    e = torch.zeros(16, 32, 32)
    for p in range(16):
        e[p, 2 * p, :16] = 1.0
        e[p, 2 * p + 1, 16:] = 1.0
    s = torch.zeros(32, 32)
    for i in range(32):
        if i % block:
            s[i, i - 1] = 1.0
    rep = dict(dims=(None, None))
    return put(mesh, e.view(1, 1, 512, 32).bfloat16(), **rep), put(mesh, s.view(1, 1, 32, 32).bfloat16(), **rep)


def conv_base(mesh, base):
    """A side's base kernel [taps, hidden] -> [1, taps, 32, hidden] fp32 (the row repeated: the conv op's tiles)."""
    from .common import put

    b = base.float().reshape(1, base.shape[0], 1, base.shape[1]).expand(1, base.shape[0], 32, base.shape[1])
    return put(mesh, b.contiguous(), dims=(None, None), dtype=ttnn.float32)


def conv(mesh, x, delta, base, consts, side: int, max_cores: int = 120):
    """The drafter's grouped dynamic 2-tap convolution over the block (tt/dflash2.py `_conv`): x [1, 1, M, 4096] bf16,
    delta [1, 1, M, 1024] bf16 (side s, tap t, group g at column 512 s + 256 t + g), base (`conv_base`) of the side,
    consts (`conv_constants`) -> bf16(c0 x + c1 x[i - 1]), c_t = base_t + delta_t spread over its 16 channels.
    conv_rows_reader / compute / writer: the reader pushes a tile row at a time (the compute works on it while the next
    ones are read) and reads each delta group's two tiles once a row."""
    assert x.dtype == ttnn.bfloat16 and delta.dtype == ttnn.bfloat16 and x.shape[-1] == 4096, (x, delta)
    e, s = consts
    m = x.shape[-2]
    r_tiles, xw, dw = -(-m // 32), x.shape[-1] // 32, delta.shape[-1] // 32
    out = _alloc(mesh, [1, 1, m, x.shape[-1]], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    parts = _split(xw, max_cores)
    coords, cores = _cores(mesh, len(parts))
    cb_x, cb_d, cb_e, cb_s, cb_b, cb_o = 0, 1, 2, 3, 4, 16
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (a, b) in zip(coords, parts):
        rd[c.x][c.y] = [x.buffer_address(), delta.buffer_address(), base.buffer_address(), e.buffer_address(),
                        s.buffer_address(), a, b]  # fmt: skip
        cp[c.x][c.y] = [a, b]
        wr[c.x][c.y] = [out.buffer_address(), a, b]
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[cb_b] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    cfg.unpack_to_dest_mode = modes
    key = [r_tiles, xw, -(-xw // len(parts))]  # (the split in the compile-time args)
    form = "conv_rows_"
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KDIR + form + "reader.cpp",
            core_ranges=cores,
            compile_time_args=[r_tiles, xw, dw, side, cb_x, cb_d, cb_e, cb_s, cb_b]
            + ttnn.TensorAccessorArgs(x).get_compile_time_args()
            + ttnn.TensorAccessorArgs(delta).get_compile_time_args()
            + ttnn.TensorAccessorArgs(base).get_compile_time_args()
            + ttnn.TensorAccessorArgs(e).get_compile_time_args()
            + ttnn.TensorAccessorArgs(s).get_compile_time_args()
            + key,
            runtime_args=rd,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + form + "compute.cpp",
            core_ranges=cores,
            compile_time_args=[r_tiles, cb_x, cb_d, cb_e, cb_s, cb_b, cb_o] + key,
            runtime_args=cp,
            config=cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + form + "writer.cpp",
            core_ranges=cores,
            compile_time_args=[r_tiles, xw, cb_o] + ttnn.TensorAccessorArgs(out).get_compile_time_args() + key,
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]

    def cb(index, fmt, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
        )

    nj = key[2]  # a core's column tiles (the reader takes them all at once)
    ng = max((b - 1) // 16 - a // 16 + 1 for a, b in parts)  # (a core's delta groups)
    cbs = [
        cb(cb_x, ttnn.bfloat16, 2048, nj * r_tiles),
        cb(cb_d, ttnn.bfloat16, 2048, 2 * ng * r_tiles),
        cb(cb_e, ttnn.bfloat16, 2048, nj),
        cb(cb_s, ttnn.bfloat16, 2048, 1),
        cb(cb_b, ttnn.float32, 4096, 2 * nj),
        cb(cb_o, ttnn.bfloat16, 2048, 2),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op([x, delta, base, e, s, out], _whole(program))
    return out


def rope(mesh, n, w, cos, sin, max_cores: int = 120, halves: int = 0):
    """The heads' norm weight and RoPE (tt/dflash2.py: weight * bf16(normalized), then `_rope` in fp32): n [.., R, 128]
    bf16 (rows normalized, rounded), w [1, 1, 32, 128] bf16 (one weight for every row, repeated over the 32 rows) or of
    n's shape (a weight per row), cos / sin of n's shape fp32 -> [.., R, 128] bf16.  rope_reader / compute / writer.
    halves: n [1, 2 h, 32, 128] (q's h tile rows, then k's), w [1, 1, 64, 128] (q's weight rows, then k's), cos / sin
    [1, h, 32, 128] (the same for both) -> (q's, k's) [1, h, 32, 128] each."""
    assert n.dtype == ttnn.bfloat16 and n.shape[-1] == 128 and cos.dtype == ttnn.float32, (n, cos)
    rows = 1
    for d in tuple(n.shape)[:-1]:
        rows *= int(d)
    assert rows % 32 == 0, n.shape
    tiles = rows // 32
    w_total = 1
    for d in tuple(w.shape)[:-1]:
        w_total *= int(d)
    if halves:
        assert w_total == 64 and tiles == 2 * halves and tuple(n.shape)[:2] == (1, 2 * halves), (w.shape, n.shape)
    else:
        assert w_total in (32, rows), (w.shape, n.shape)
    w_rows = 1 if w_total in (32, 64) else tiles  # (the reader: a weight tile row for all (q's, k's), or a row's)
    half_shape = [1, halves, 32, 128]
    out = _alloc(mesh, half_shape if halves else list(n.shape), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    out2 = _alloc(mesh, half_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT) if halves else out
    parts = _split(tiles * 4, max_cores)  # items (tile row, column tile)
    coords, cores = _cores(mesh, len(parts))
    cb_n, cb_w, cb_c, cb_s, cb_o = 0, 1, 2, 3, 16
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (a, b) in zip(coords, parts):
        rd[c.x][c.y] = [n.buffer_address(), w.buffer_address(), cos.buffer_address(), sin.buffer_address(), a, b]
        cp[c.x][c.y] = [a, b]
        wr[c.x][c.y] = [out.buffer_address(), a, b, out2.buffer_address()]
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[cb_c] = ttnn.UnpackToDestMode.UnpackToDestFp32
    modes[cb_s] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    cfg.unpack_to_dest_mode = modes
    key = [tiles, -(-tiles * 4 // len(parts))]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "rope_reader.cpp",
            core_ranges=cores,
            compile_time_args=[w_rows, cb_n, cb_w, cb_c, cb_s]
            + ttnn.TensorAccessorArgs(n).get_compile_time_args()
            + ttnn.TensorAccessorArgs(w).get_compile_time_args()
            + ttnn.TensorAccessorArgs(cos).get_compile_time_args()
            + key
            + [halves],
            runtime_args=rd,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "rope_compute.cpp",
            core_ranges=cores,
            compile_time_args=[w_rows, cb_n, cb_w, cb_c, cb_s, cb_o] + key + [halves],
            runtime_args=cp,
            config=cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "rope_writer.cpp",
            core_ranges=cores,
            compile_time_args=[cb_o] + ttnn.TensorAccessorArgs(out).get_compile_time_args() + key + [halves],
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]

    def cb(index, fmt, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
        )

    cbs = [
        cb(cb_n, ttnn.bfloat16, 2048, 4),
        cb(cb_w, ttnn.bfloat16, 2048, 8 if halves else 4),
        cb(cb_c, ttnn.float32, 4096, 2),
        cb(cb_s, ttnn.float32, 4096, 2),
        cb(cb_o, ttnn.bfloat16, 2048, 2),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op([n, w, cos, sin, out] + ([out2] if halves else []), _whole(program))
    return (out, out2) if halves else out


def silu_mul(mesh, gu, inter: int, max_cores: int = 120):
    """The MLP's gate: gate | up [1, 1, M, 2 inter] bf16 tiled -> SiLU(gate) * up [1, 1, M, inter] bf16, in one op (in
    place of two slices and ttnn.multiply with the SiLU on its first input: the same values, its steps)."""
    assert gu.dtype == ttnn.bfloat16 and gu.layout == ttnn.TILE_LAYOUT and gu.shape[-1] == 2 * inter, gu
    m = gu.shape[-2]
    it = inter // 32
    tiles = -(-m // 32) * it
    out = _alloc(mesh, [1, 1, m, inter], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    parts = _split(tiles, max_cores)
    coords, cores = _cores(mesh, len(parts))
    cb_g, cb_u, cb_s, cb_o = 0, 1, 2, 16
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (a, b) in zip(coords, parts):
        rd[c.x][c.y] = [gu.buffer_address(), a, b]
        cp[c.x][c.y] = [a, b]
        wr[c.x][c.y] = [out.buffer_address(), a, b]
    key = [tiles, -(-tiles // len(parts))]  # (the split in the compile-time args)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "silu_mul_reader.cpp",
            core_ranges=cores,
            compile_time_args=[it, cb_g, cb_u] + ttnn.TensorAccessorArgs(gu).get_compile_time_args() + key,
            runtime_args=rd,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "silu_mul_compute.cpp",
            core_ranges=cores,
            compile_time_args=[cb_g, cb_u, cb_s, cb_o] + key,
            runtime_args=cp,
            config=cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "silu_mul_writer.cpp",
            core_ranges=cores,
            compile_time_args=[cb_o] + ttnn.TensorAccessorArgs(out).get_compile_time_args() + key,
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]

    def cb(index, pages):
        return ttnn.CBDescriptor(
            total_size=2048 * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=2048)],
        )

    cbs = [cb(cb_g, 2), cb(cb_u, 2), cb(cb_s, 2), cb(cb_o, 2)]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op([gu, out], _whole(program))
    return out


def sel_scores(mesh, pred_rows, proj_rows, succ_rows, items: int, max_cores: int = 120, terms: int = 3):
    """The selector's pairwise terms in one op: pred_rows, proj_rows, succ_rows [1, 1, items 32, rank] bf16 (tiled; an
    item's 32 rows: its 16 predecessors' / the slot's proj spread / its 16 candidates' rows, 16 padding) -> S
    [1, items, 32, 32] fp32: S[p][c] = sum_r pred[p][r] proj[r] succ[c][r], the products exact (P = pred proj in fp32,
    split into two bf16 parts), the sums fp32 (terms 1 / 2: the hi / lo part alone, tests).  sel_reader / sel_compute /
    sel_writer."""
    for t in (pred_rows, proj_rows, succ_rows):
        assert t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT and t.shape[-2] == items * 32, t
    kt = pred_rows.shape[-1] // 32
    out = _alloc(mesh, [1, items, 32, 32], ttnn.float32, ttnn.TILE_LAYOUT)
    parts = _split(items, max_cores)
    coords, cores = _cores(mesh, len(parts))
    cb_pr, cb_pj, cb_su, cb_hi, cb_lo, cb_o = 0, 1, 2, 3, 4, 16
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (a, b) in zip(coords, parts):
        rd[c.x][c.y] = [pred_rows.buffer_address(), proj_rows.buffer_address(), succ_rows.buffer_address(), a, b]
        cp[c.x][c.y] = [a, b]
        wr[c.x][c.y] = [out.buffer_address(), a, b]
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    key = [items, -(-items // len(parts))]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "sel_reader.cpp",
            core_ranges=cores,
            compile_time_args=[kt, cb_pr, cb_pj, cb_su]
            + ttnn.TensorAccessorArgs(pred_rows).get_compile_time_args()
            + key,
            runtime_args=rd,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "sel_compute.cpp",
            core_ranges=cores,
            compile_time_args=[kt, cb_pr, cb_pj, cb_su, cb_hi, cb_lo, cb_o, terms] + key,
            runtime_args=cp,
            config=cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "sel_writer.cpp",
            core_ranges=cores,
            compile_time_args=[cb_o] + ttnn.TensorAccessorArgs(out).get_compile_time_args() + key,
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]

    def cb(index, fmt, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
        )

    cbs = [
        cb(cb_pr, ttnn.bfloat16, 2048, kt),
        cb(cb_pj, ttnn.bfloat16, 2048, kt),
        cb(cb_su, ttnn.bfloat16, 2048, kt),
        cb(cb_hi, ttnn.bfloat16, 2048, kt),
        cb(cb_lo, ttnn.bfloat16, 2048, kt),
        cb(cb_o, ttnn.float32, 4096, 2),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op([pred_rows, proj_rows, succ_rows, out], _whole(program))
    return out
