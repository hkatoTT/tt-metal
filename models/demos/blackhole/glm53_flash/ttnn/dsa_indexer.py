# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The DSA indexer's small device steps, through ttnn.generic_op, the same program on every chip
(each chip its own row's users):

  rows_write(src, dst, rows): each row i of src [1, 1, n, D] row-major (a user's new token, or a prefill chunk's tokens)
      into row rows[i] of the row-major cache dst [1, 1, N, D] (rows [1, 1, 1, >= n] uint32, 0xFFFFFFFF: none);
  pages_gather(cache, table): the pages table names [1, n] int32 of a paged cache [P, 1, 32, D] (tiled) -> [1, 1, 32 n,
      D] (a prefill row's user's pool keys);
  pool_scores(qi, cache, hw, params, n_pools, shared): the pool scores of N queries over their users' pages of the pools
      cache, masked, bf16 row-major for topk_large_indices (three kernels: kernels/dsa/scores_*.cpp);
  select_tokens(pools, params, width): topk_large_indices' pools [1, 1, U, k] (best first, 0xFFFFFFFF after the
      visible ones) and params [1, 1, U, 16] uint32 (pos, base; pos 0xFFFFFFFF: no query) -> [1, 1, U, width] uint32,
      the token rows base + 4 p + j of the pools, then the tail (the visible tokens after the last complete pool), then
      0xFFFFFFFF (sparse_sdpa reads up to the first one).
  sparse_split(q32, cache, tokens, splits, scale) and sparse_combine(parts, tokens, splits, scale): decode's sparse
      attention with each query's keys split over cores (sparse_sdpa computes a query on one core), then the splits'
      partials combined -> [1, S, 16, 512] tiles.
"""

from __future__ import annotations

import os

import ttnn

from .common import COLS, ROWS, core_rects

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/dsa/"
SENT = 0xFFFFFFFF
PARAMS = 16  # uint32 a params row (64 bytes: a DRAM-aligned page)


def _cb(index: int, size: int, cores) -> ttnn.CBDescriptor:
    return ttnn.CBDescriptor(
        total_size=size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint32, page_size=size)],
    )


def _run(mesh, tensors: list, program: ttnn.ProgramDescriptor) -> None:
    mesh_program = ttnn.MeshProgramDescriptor()
    whole = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(ROWS - 1, COLS - 1))
    mesh_program[whole] = program
    ttnn.generic_op(tensors, mesh_program)


def _run_cols(mesh, tensors: list, make) -> None:
    """generic_op with mesh column c's program make(c) (its compile-time args name c: a sharded cache's column)."""
    mesh_program = ttnn.MeshProgramDescriptor()
    for col in range(COLS):
        rng = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, col), ttnn.MeshCoordinate(ROWS - 1, col))
        mesh_program[rng] = make(col)
    ttnn.generic_op(tensors, mesh_program)


def _split(n: int, grid, per_min: int) -> list[tuple[int, int]]:
    """n items over up to the grid's cores, at least per_min a core."""
    cores = max(1, min(grid.x * grid.y, -(-n // per_min)))
    per = -(-n // cores)
    return [(i, min(n, i + per)) for i in range(0, n, per)]


def _unit_ranges(units: int, grid) -> list[tuple[int, int]]:
    """units over up to the grid's cores, a contiguous range a core: [(first unit, count)] (one each when they fit)."""
    n = min(units, grid.x * grid.y)
    per = -(-units // n)
    return [(u, min(per, units - u)) for u in range(0, units, per)]


def _cores(parts, grid):
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(parts))]
    return coords, core_rects(coords)


def rows_write(mesh, src: ttnn.Tensor, dst: ttnn.Tensor, rows, params=None, shard: bool = False) -> None:
    """src [1, 1, n, D] bf16 row-major -> rows rows[i] of dst [1, 1, N, D] bf16 row-major, in place; or with `params`
    ([1, 1, n, PARAMS] uint32 row-major, rows None) row base + position of params[i] (0xFFFFFFFF position: none).
    `shard` (the split latents: a long cache's rows over the row's 4 chips by token): mesh column c's dst holds only the
    rows r = c mod COLS, at r / COLS."""
    n, d = src.shape[-2], src.shape[-1]
    row_bytes = d * src.element_size()
    from_params = params is not None
    index = params if from_params else rows
    page = PARAMS * 4 if from_params else rows.shape[-1] * 4
    block = 32
    parts = _split(n, mesh.compute_with_storage_grid_size(), block)
    coords, cores = _cores(parts, mesh.compute_with_storage_grid_size())
    rt = ttnn.RuntimeArgs()
    for c, (r0, r1) in zip(coords, parts):
        rt[c.x][c.y] = [src.buffer_address(), dst.buffer_address(), index.buffer_address(), r0, r1]
    index_bytes = block * page if from_params else max(page, 64)

    def program(mul: int, add: int):
        kernel = ttnn.KernelDescriptor(
            kernel_source=KDIR + "rm_rows_write.cpp",
            core_ranges=cores,
            compile_time_args=[block, row_bytes, page, 0, int(from_params)]
            + ttnn.TensorAccessorArgs(src).get_compile_time_args()
            + ttnn.TensorAccessorArgs(dst).get_compile_time_args()
            + ttnn.TensorAccessorArgs(index).get_compile_time_args()
            + [mul, add],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        return ttnn.ProgramDescriptor(
            kernels=[kernel], semaphores=[], cbs=[_cb(0, block * row_bytes + index_bytes, cores)]
        )

    if shard:
        _run_cols(mesh, [src, index, dst], lambda c: program(COLS, c))
    else:
        _run(mesh, [src, index, dst], program(1, 0))


def prefix_rows(mesh, cache: ttnn.Tensor, params: ttnn.Tensor, n_rows: int, mul: int = 4) -> ttnn.Tensor:
    """The gathered prefix (kernels/dsa/rm_rows_gather.cpp; a prefill chunk over the split latents): this chip's n_rows
    rows of a mesh row's user, from row base / mul of the row-major cache [1, 1, R, D] (base: the params' row 0, word 1,
    [1, 1, *, PARAMS] uint32) -> [1, 1, n_rows, D] in the cache's dtype; gathered over the row's chips (all_gather dim
    2) they are its user's first 4 n_rows tokens, token t at (t mod 4) n_rows + t / 4
    (`select_tokens(prefix_q=n_rows)`)."""
    d = cache.shape[-1]
    row_bytes = d * cache.element_size()
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            ttnn.Shape([1, 1, n_rows, d]), cache.dtype, ttnn.ROW_MAJOR_LAYOUT, buffer_type=ttnn.BufferType.DRAM
        ),
        mesh,
    )
    grid = mesh.compute_with_storage_grid_size()
    parts = _split(n_rows, grid, 16)
    coords, cores = _cores(parts, grid)
    rt = ttnn.RuntimeArgs()
    for c, (r0, r1) in zip(coords, parts):
        rt[c.x][c.y] = [cache.buffer_address(), params.buffer_address(), out.buffer_address(), r0, r1]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KDIR + "rm_rows_gather.cpp",
        core_ranges=cores,
        compile_time_args=[row_bytes, n_rows, int(cache.shape[-2]), mul, 0]
        + ttnn.TensorAccessorArgs(cache).get_compile_time_args()
        + ttnn.TensorAccessorArgs(params).get_compile_time_args()
        + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        + [len(parts)],
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_cb(0, 16 * row_bytes + 64, cores)])
    _run(mesh, [cache, params, out], program)
    return out


def pages_gather(mesh, cache: ttnn.Tensor, table: ttnn.Tensor) -> ttnn.Tensor:
    """cache [P, 1, 32, D] tiled, table [1, n] int32 row-major -> [1, 1, 32 n, D] (pages table[0], table[1], ..)."""
    n, d = table.shape[-1], cache.shape[-1]
    tpp = d // 32
    tile = cache.buffer_page_size()
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape([1, 1, 32 * n, d]), cache.dtype, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM),
        mesh,
    )
    block = 8 * tpp
    grid = mesh.compute_with_storage_grid_size()
    parts = _split(n, grid, 2)
    coords, cores = _cores(parts, grid)
    rt = ttnn.RuntimeArgs()
    for c, (i0, i1) in zip(coords, parts):
        rt[c.x][c.y] = [cache.buffer_address(), table.buffer_address(), out.buffer_address(), i0, i1]
    table_page = max(n * 4, 64)
    kernel = ttnn.KernelDescriptor(
        kernel_source=KDIR + "pages_gather.cpp",
        core_ranges=cores,
        compile_time_args=[tpp, tile, n * 4, block, 0]
        + ttnn.TensorAccessorArgs(cache).get_compile_time_args()
        + ttnn.TensorAccessorArgs(table).get_compile_time_args()
        + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_cb(0, block * tile + table_page, cores)])
    _run(mesh, [cache, table, out], program)
    return out


def _cb_fmt(index: int, fmt, page: int, pages: int, cores) -> ttnn.CBDescriptor:
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
    )


SCORES_SHARED_QUERIES = 32  # the queries a core keeps in L1 when they share the pools (10 KB each)
# decode's pool scores with a user's queries (MTP's / DFlash2's verify step: its T tokens, virtual users u, u + U, ..)
# on the same cores, each pool page read once for all of them (the prefill chunk's `shared` order) instead of once a
# query: the reads 1 / T, the same values (each query's unit the same matmuls on the same tiles).  0: a page a query
# (the model bit for bit with both on; the DFlash2 step at a 32K cache 8 / 32 / 128 users 42.59 / 53.50 / 78.06 ->
# 42.82 / 52.52 / 74.53 ms; 2,000 steps each at 8 / 32 / 128 users, positions 0 .. 16K, clean)
SCORES_GROUP = True  # (GLM53_DSA_SCORES_GROUP=1 at the call: on whatever the size, 0: off)
# ... by default (the module's SCORES_GROUP, the environment's unset) only from this many pool pages a step (users x
# blocks of 32 pools): below it the grouped cores' compute (a core's T queries a page) outweighs the reads saved
# (1 user x 8 tokens at 32K 55.9 -> 58.5 us, at 128K 183 -> 104 us; 4 x 8 at 32K 186 -> 97 us)
SCORES_GROUP_MIN = 1024
# the pool scores read and compute only the blocks of 32 pools with a pool the core's queries see (from their
# positions in params, read on the device: the cache past the context neither read nor computed), the writer's -inf for
# the rest as when they were computed (it masked them): the same values.  0: every block of the range (the cache's
# length)
SCORES_VISIBLE = True  # (GLM53_DSA_SCORES_VISIBLE=0 at the call: off)
TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}


def _switch(name: str, value: bool) -> bool:
    """A pool scores switch: the environment's value at the call when set (a speed test's arms on one load), else
    the module's."""
    env = os.environ.get(name)
    return value if env is None else env == "1"


def pool_scores(
    mesh,
    qi,
    cache,
    hw,
    params,
    n_pools: int,
    shared: bool,
    computed: int | None = None,
    kpool: int = 4,
    group: tuple[int, int] | None = None,
    visible: bool | None = None,
    shard: bool = False,
):
    """The indexer's pool scores in one op (kernels/dsa/scores_*.cpp): qi [N, 1, 32, 128] bf16 tiled (each query's 32
    heads), the pools cache [*, 1, 32, 128] tiled (bf16, or BFP8_b / BFP4_b: dsa.POOLS_DTYPE; query n's pool p at
    row p mod 32 of page params[n][2] + p / 32), hw [1, 1, N, 32] bf16 tiled (the head weights, scaled), params [1, 1, N, 16] uint32
    row-major (position, row base, pool page base) -> [1, 1, N, n_pools] bf16 row-major: sum_h hw_h relu(q_h . k_p),
    -inf for p >= (position + 1) / 4.  `shared`: every query reads the same pages (one user's prefill chunk; the cores
    keep their queries and stream the pages once); else each its own (decode).  Pools from `computed` on (rounded up
    to 32) are -inf without being computed.  `group` (users U, tokens T; decode, SCORES_GROUP): query n's user is n mod
    U, and a user's T queries share its pages as `shared` does.  `visible` (default SCORES_VISIBLE): only the blocks
    with a pool the core's queries see are read and computed.  `shard` (decode; the split scores over the row's 4
    chips): the chip of mesh column c computes only the blocks 4 j + c (j < n_pools / 128) of its queries' pools -> [1,
    1, N, n_pools / 4], column 32 j + r its block 4 j + c's pool r (`shard_candidates` and `shard_merge` take the top
    512 of the row's 4 chips)."""
    q_rm = qi.layout == ttnn.ROW_MAJOR_LAYOUT  # decode: [1, 1, N, 4096] row-major (no reshape; compute tilizes)
    if q_rm and shared:
        raise ValueError("pool_scores: row-major queries in decode only")
    visible = _switch("GLM53_DSA_SCORES_VISIBLE", SCORES_VISIBLE) if visible is None else visible
    n, blocks = (qi.shape[2] if q_rm else qi.shape[0]), n_pools // 32
    if shard:
        if shared or computed is not None or blocks % COLS:
            raise ValueError(f"pool_scores: shard is decode's, {n_pools} pools in blocks of 32 a multiple of {COLS}")
        blocks //= COLS  # this chip's: global block 4 j + c
    width = blocks * 32
    jc_all = blocks if computed is None else min(blocks, -(-computed // 32))
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            ttnn.Shape([1, 1, n, width]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, buffer_type=ttnn.BufferType.DRAM
        ),
        mesh,
    )
    grid = mesh.compute_with_storage_grid_size()
    cores = grid.x * grid.y
    work = []  # (first query, queries, query stride, ja, jb, jc) a core
    env_group = os.environ.get("GLM53_DSA_SCORES_GROUP")
    grouped = (
        not shared
        and group is not None
        and group[1] > 1
        and (env_group == "1" if env_group is not None else SCORES_GROUP and group[0] * blocks >= SCORES_GROUP_MIN)
    )
    if shared:
        qc = min(SCORES_SHARED_QUERIES, n)
        groups = [(a, min(n, a + qc)) for a in range(0, n, qc)]
        per = -(-blocks // max(1, cores // len(groups)))
        for a, b in groups:
            for j0 in range(0, blocks, per):
                j1 = min(blocks, j0 + per)
                work.append((a, b - a, 1, j0, j1, max(j0, min(j1, jc_all))))
    elif grouped:
        users, t = group
        if users * t != n or t > SCORES_SHARED_QUERIES or users > cores:
            raise ValueError(f"pool_scores: {n} queries as {users} users x {t} ({cores} cores)")
        qc = t
        per = -(-blocks // max(1, cores // users))
        for u in range(users):
            for j0 in range(0, blocks, per):
                j1 = min(blocks, j0 + per)
                work.append((u, t, users, j0, j1, max(j0, min(j1, jc_all))))
    elif n >= cores:
        qc = -(-n // cores)
        work = [(a, min(n, a + qc) - a, 1, 0, blocks, jc_all) for a in range(0, n, qc)]
    else:
        qc = 1
        per = -(-blocks // (cores // n))
        for a in range(n):
            for j0 in range(0, blocks, per):
                j1 = min(blocks, j0 + per)
                work.append((a, 1, 1, j0, j1, max(j0, min(j1, jc_all))))
    together = shared or grouped  # the core's queries share its pages
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(work))]
    core_set = core_rects(coords)
    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (na, nq, stride, ja, jb, jc) in zip(coords, work):
        reader_rt[c.x][c.y] = [
            qi.buffer_address(),
            cache.buffer_address(),
            hw.buffer_address(),
            params.buffer_address(),
            na,
            nq,
            stride,
            ja,
            jc,
        ]
        compute_rt[c.x][c.y] = [nq, jc - ja]
        writer_rt[c.x][c.y] = [out.buffer_address(), params.buffer_address(), na, nq, stride, ja, jb, jc]
    tile = 2048
    k_tile = TILE_BYTES[cache.dtype]
    kdir = KDIR + "scores_"

    def program(mul: int, add: int):  # local block j -> global block j mul + add (1, 0: every block)
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=kdir + "reader.cpp",
                core_ranges=core_set,
                compile_time_args=[int(together), tile, int(q_rm), int(visible), kpool, k_tile]
                + ttnn.TensorAccessorArgs(qi).get_compile_time_args()
                + ttnn.TensorAccessorArgs(cache).get_compile_time_args()
                + ttnn.TensorAccessorArgs(hw).get_compile_time_args()
                + ttnn.TensorAccessorArgs(params).get_compile_time_args()
                + [mul, add],
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=kdir + "compute.cpp",
                core_ranges=core_set,
                compile_time_args=[int(together), int(q_rm), int(visible)],
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
            ),
            ttnn.KernelDescriptor(
                kernel_source=kdir + "writer.cpp",
                core_ranges=core_set,
                compile_time_args=[int(together), width * 2, kpool, int(visible)]
                + ttnn.TensorAccessorArgs(out).get_compile_time_args()
                + ttnn.TensorAccessorArgs(params).get_compile_time_args()
                + [mul, add],
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    q_slots = qc if together else 2
    cbs = [
        _cb_fmt(0, ttnn.bfloat16, tile, 4 * q_slots, core_set),  # Q
        _cb_fmt(1, ttnn.bfloat16, tile, q_slots, core_set),  # W
        _cb_fmt(2, cache.dtype, k_tile, 8, core_set),  # K (the pools cache's format)
        _cb_fmt(3, ttnn.float32, 2 * tile, 2, core_set),  # S
        _cb_fmt(4, ttnn.bfloat16, tile, 4, core_set),  # out
        _cb(5, qc * PARAMS * 4 + tile, core_set),  # the reader's scratch
        _cb(6, qc * PARAMS * 4 + 64, core_set),  # the writer's scratch
        _cb_fmt(7, ttnn.bfloat16, 256, 32, core_set),  # a query's rows (q_rm)
    ]
    if visible:
        cbs.append(_cb_fmt(8, ttnn.uint32, 16, 2, core_set))  # the blocks to compute (the reader's count)
    tensors = [qi, cache, hw, params, out]
    if not shard:
        _run(mesh, tensors, program(1, 0))
        return out
    _run_cols(mesh, tensors, lambda c: program(COLS, c))
    return out


def _query_cores(mesh, n: int):
    """n queries over the grid's cores, a contiguous range a core -> (coords, core set, [(q0, q1)])."""
    grid = mesh.compute_with_storage_grid_size()
    cores = min(n, grid.x * grid.y)
    per = -(-n // cores)
    parts = [(a, min(n, a + per)) for a in range(0, n, per)]
    coords, core_set = _cores(parts, grid)
    return coords, core_set, parts


def shard_candidates(mesh, rows, top) -> ttnn.Tensor:
    """The split scores' candidates (kernels/dsa/shard_cand.cpp): this chip's scores `rows` [1, 1, N, W]
    bf16 row-major (`pool_scores(shard=True)`, W <= 65,536) and their top k `top` [1, 1, N, k] uint32
    (topk_large_indices) -> [1, 2, N, k] row-major: plane 0 the k scores (bf16; -inf for none), plane 1 their columns
    (uint16 in a bf16 container).  Gathered over the row (ttnn.all_gather dim 3: chip c's at c k), plane 0 is the row's
    candidates for topk_large_indices and plane 1 maps its picks to pools (`select_tokens(cand=...)`)."""
    n, width, k = rows.shape[-2], rows.shape[-1], top.shape[-1]
    if width > 65536 or rows.dtype != ttnn.bfloat16:
        raise ValueError(f"shard_candidates: bf16 rows of at most 65,536 columns (got {rows.dtype} {width})")
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            ttnn.Shape([1, 2, n, k]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, buffer_type=ttnn.BufferType.DRAM
        ),
        mesh,
    )
    coords, cores, parts = _query_cores(mesh, n)
    rt = ttnn.RuntimeArgs()
    for c, (q0, q1) in zip(coords, parts):
        rt[c.x][c.y] = [rows.buffer_address(), top.buffer_address(), out.buffer_address(), q0, q1]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KDIR + "shard_cand.cpp",
        core_ranges=cores,
        compile_time_args=[width, k, 0]
        + ttnn.TensorAccessorArgs(rows).get_compile_time_args()
        + ttnn.TensorAccessorArgs(top).get_compile_time_args()
        + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        + [n, len(parts)],  # (N, and the split in the compile-time args: the program cache does not see runtime values)
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_cb(0, width * 2 + 8 * k, cores)])
    _run(mesh, [rows, top, out], program)
    return out


def shard_pick(mesh, every, k: int = 512) -> ttnn.Tensor:
    """The split scores' choice: the row's candidates `every` [1, 2, N, COLS k] (`shard_candidates` gathered over the
    row) -> the top k of plane 0 [1, 1, N, k] uint32 (topk_large_indices: best first, 0xFFFFFFFF for -inf): positions in
    the row, `select_tokens(cand=every)` maps them to pools."""
    n, w = every.shape[-2], every.shape[-1]
    scores = ttnn.slice(every, [0, 0, 0, 0], [1, 1, n, w])  # plane 0 (row-major: the first N rows)
    top = ttnn.experimental.topk_large_indices(scores, k=k)
    if scores.is_allocated() and scores.buffer_address() != every.buffer_address():
        ttnn.deallocate(scores)
    return top


# True: a user's row over cores (the same rows).  False (a test's): a row on one core
SELECT_PARTS = True
# ... up to this many users (a row over 2 to 8 cores: 1 / 2 users 18.8 / 18.2 -> 14.6 / 16.9 us, 4 / 16 users slower:
# 16.7 / 26.7 -> 20.6 / 34.3 us)
SELECT_PARTS_MAX = 2


def select_tokens(
    mesh,
    pools: ttnn.Tensor,
    params: ttnn.Tensor,
    width: int,
    kpool: int = 4,
    cand: ttnn.Tensor | None = None,
    shard: bool = False,
    prefix_q: int = 0,
) -> ttnn.Tensor:
    """pools [1, 1, U, k] uint32 row-major, params [1, 1, U, 16] uint32 row-major -> [1, 1, U, width] uint32.  `cand`
    (the split scores): `pools` are picks among the row's candidates (`shard_pick`), mapped to pools by cand's plane 1.
    `shard` (the split latents): mesh column c's rows of its tokens t = c mod 4 (rm_rows_write's shard): a pool's one,
    the tail's of c.  `prefix_q` (the gathered prefix): rows of its user's prefix gathered over the row (`prefix_rows`
    of prefix_q rows a chip, then all_gather): token t at (t mod 4) prefix_q + t / 4, in the unsplit cache's token
    order."""
    if prefix_q and shard:
        raise ValueError("select_tokens: prefix_q (the gathered prefix) or shard (the split latents), not both")
    users, k = pools.shape[-2], pools.shape[-1]
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            ttnn.Shape([1, 1, users, width]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, buffer_type=ttnn.BufferType.DRAM
        ),
        mesh,
    )
    grid = mesh.compute_with_storage_grid_size()
    n_cores = grid.x * grid.y
    # a user's row over up to 8 cores when the step's users leave them free (a row on one core: 17 us)
    parts = 1
    while (
        SELECT_PARTS
        and not shard
        and not prefix_q
        and users <= SELECT_PARTS_MAX
        and parts < 8
        and users * parts * 2 <= n_cores
        and (k // (parts * 2)) % 4 == 0
    ):
        parts *= 2
    if parts > 1:
        units = [(u, u + 1, j) for u in range(users) for j in range(parts)]
    else:
        ncores = min(users, n_cores)
        per = -(-users // ncores)
        units = [(i * per, min(users, (i + 1) * per), 0) for i in range(ncores)]
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(units))]
    cores = core_rects(coords)
    rt = ttnn.RuntimeArgs()
    mapped = cand is not None
    for (u0, u1, j), c in zip(units, coords):
        rt[c.x][c.y] = [pools.buffer_address(), params.buffer_address(), out.buffer_address(), u0, u1, j] + (
            [cand.buffer_address(), users] if mapped else []
        )
    params_page = PARAMS * 4
    cols = 4 * k * 2 if mapped else 0  # the candidates' columns row (COLS k uint16)

    def program(mul: int, add: int):
        kernel = ttnn.KernelDescriptor(
            kernel_source=KDIR + "select_tokens.cpp",
            core_ranges=cores,
            compile_time_args=[k, width, kpool, params_page, 0, parts]
            + ttnn.TensorAccessorArgs(pools).get_compile_time_args()
            + ttnn.TensorAccessorArgs(params).get_compile_time_args()
            + ttnn.TensorAccessorArgs(out).get_compile_time_args()
            + [int(mapped)]
            + ttnn.TensorAccessorArgs(cand if mapped else pools).get_compile_time_args()  # (unused when not mapped)
            + [mul, add, prefix_q],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        return ttnn.ProgramDescriptor(
            kernels=[kernel], semaphores=[], cbs=[_cb(0, k * 4 + params_page + width * 4 + cols, cores)]
        )

    tensors = [pools, params] + ([cand] if mapped else []) + [out]  # (generic_op: the output last)
    if shard:
        _run_cols(mesh, tensors, lambda c: program(kpool, c))
    else:
        _run(mesh, tensors, program(1, 0))
    return out


def _fp32_bits(x: float) -> int:
    import struct

    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


SPLIT_K_CHUNK = 128  # keys a chunk (sparse_sdpa's k_chunk_size: the selected widths are multiples of it)
# an FP8 latent cache's K tiles in sparse_split: "bfp8" (sparse_sdpa's for an fp8 kv, the prefill's) or "bf16"
FP8_K_IN = os.environ.get("GLM53_DSA_FP8_K_IN", "bfp8")
PART_TILES = 18  # a unit's partial: 16 out tiles, the row sums, the row maxima (kernels/dsa/split_common.hpp)


def sparse_split(
    mesh,
    q32,
    cache,
    tokens,
    splits: int,
    scale: float,
    upq: int = 1,
    write_empty: bool = False,
    out=None,
    base_unit: int = 0,
) -> ttnn.Tensor:
    """Decode's sparse attention with each query's keys split over `splits` cores (kernels/dsa/split_*.cpp, from
    ttnn's sparse_sdpa: its gather over both NoCs and its flash softmax over chunks of 128 keys): q32 [1, 32, S, 512]
    bf16 row-major (the 16 heads, zeros below), or [1, S, 16, 512] bf16 tiles (a token's tile row its heads, the
    padding rows' values only reach their own rows; the split latents: [1, 2 S, 32, 512] tiles or [1, 64, S, 512]
    row-major, a query's 64 heads), cache [1, 1, T, 512] bf16 row-major (the latents), tokens [1, 1, S,
    W] uint32 (`select_tokens`) -> the partials [1, 1, S G 32, 18 * 32] bf16 tiles: unit (s, j) = s G + j its
    split's unnormalized out [32, 512], row sums and row maxima (an empty share writes nothing), for
    `sparse_combine`.  S G <= the grid's cores.  `upq` (the split latents): units a query (q's tokens u share tokens'
    row u / upq: a query's 64 heads as two 32-head tile rows); `write_empty`: an empty share writes a neutral partial
    (out and sums 0, maxima -inf) instead of none (`sparse_combine(xchip=...)` reads every one); `out` / `base_unit`:
    the partials written into `out` from its unit `base_unit` (the split latents: a block of the queries' units into the
    exchange block's partials).  A row-major q of 64 heads with upq 1 (the split latents' 64-head unit): a query's two
    tile rows over its keys read once."""
    q_tiled = q32.layout == ttnn.TILE_LAYOUT
    # (row-major: [1, upq 32, S / upq, 512], the split latents' q~ of a query's 64 heads as gathered (no transpose):
    # unit-token u = query u / upq's heads 32 (u % upq) ..)
    heads, s, k_dim = (
        (32, q32.shape[1], q32.shape[3]) if q_tiled else (q32.shape[1] // upq, q32.shape[2] * upq, q32.shape[3])
    )
    width = tokens.shape[-1]
    grid = mesh.compute_with_storage_grid_size()
    units = s * splits
    # the split latents' 64-head unit (row-major [1, 64, S, 512], upq 1): a query's two tile rows over its keys read
    # once, a partial each (written as the two 32-head units' would be: the partials' unit (2 s + r) G + j)
    sqt = heads // 32
    if not (heads == 32 or (heads == 64 and not q_tiled and upq == 1)) or k_dim != 512 or width % SPLIT_K_CHUNK:
        raise ValueError(
            f"sparse_split: 32 heads (or 64 row-major) of 512 and widths of 128 (got {q32.shape}, {width})"
        )
    out = (
        out
        if out is not None
        else ttnn.allocate_tensor_on_device(
            ttnn.TensorSpec(
                ttnn.Shape([1, 1, units * sqt * 32, PART_TILES * 32]),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                buffer_type=ttnn.BufferType.DRAM,
            ),
            mesh,
        )
    )
    work = _unit_ranges(units, grid)  # (first unit, count) a core: more units than cores go in turn
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(work))]
    cores = core_rects(coords)
    tile, row = 2048, k_dim * 2
    # an FP8 e4m3 latent cache (GLM53_KV_LATENT_FP8): its rows 1 byte a value, tilized from fp8 through a 32-bit Dest
    # into K tiles of FP8_K_IN (bfp8_b: as sparse_sdpa does for an fp8 kv, the prefill's; bf16: no second rounding)
    fp8 = cache.dtype == ttnn.fp8_e4m3
    kv_bytes = 1 if fp8 else 2
    k_in_fmt = (ttnn.bfloat8_b if FP8_K_IN == "bfp8" else ttnn.bfloat16) if fp8 else ttnn.bfloat16
    k_in_tile = 1088 if k_in_fmt == ttnn.bfloat8_b else tile
    skt, dht, vdht = SPLIT_K_CHUNK // 32, k_dim // 32, 512 // 32
    (
        cb_q_rm,
        cb_q_in,
        cb_k_rm,
        cb_k_in,
        cb_neginf,
        cb_mask,
        cb_scale,
        cb_qk,
        cb_max_a,
        cb_max_b,
        cb_sum_a,
        cb_sum_b,
        cb_out_a,
        cb_out_b,
        cb_corr,
        cb_part,
        cb_idx,
        cb_ctrl,
        cb_ident,
        cb_kreq,
        cb_kack,
    ) = range(21)
    bf = ttnn.bfloat16
    cbs = [
        _cb_fmt(cb_q_rm, bf, row, 1 if q_tiled else heads, cores),
        _cb_fmt(cb_q_in, bf, tile, sqt * dht, cores),
        _cb_fmt(cb_k_rm, cache.dtype, k_dim * kv_bytes, SPLIT_K_CHUNK, cores),
        _cb_fmt(cb_k_in, k_in_fmt, k_in_tile, skt * dht, cores),
        _cb_fmt(cb_neginf, bf, tile, 1, cores),
        _cb_fmt(cb_mask, bf, tile, 1, cores),
        _cb_fmt(cb_scale, bf, tile, 1, cores),
        _cb_fmt(cb_qk, bf, tile, sqt * skt, cores),
        _cb_fmt(cb_max_a, bf, tile, sqt, cores),
        _cb_fmt(cb_max_b, bf, tile, sqt, cores),
        _cb_fmt(cb_sum_a, bf, tile, sqt, cores),
        _cb_fmt(cb_sum_b, bf, tile, sqt, cores),
        _cb_fmt(cb_out_a, bf, tile, sqt * vdht, cores),
        _cb_fmt(cb_out_b, bf, tile, sqt * vdht, cores),
        _cb_fmt(cb_corr, bf, tile, 1, cores),
        _cb_fmt(cb_part, bf, tile, PART_TILES, cores),
        _cb_fmt(cb_idx, bf, width * 4, 1, cores),
        _cb_fmt(cb_ctrl, bf, 16, 2, cores),
        _cb_fmt(cb_ident, bf, tile, 1, cores),
        _cb_fmt(cb_kreq, bf, 32, 2, cores),
        _cb_fmt(cb_kack, bf, 16, 2, cores),
    ]
    cb_fill = 21
    if write_empty:
        cbs.append(_cb_fmt(cb_fill, bf, tile, 2, cores))
    compute_cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    if fp8:  # (fp8 unpacks into a 32-bit Dest: sparse_sdpa's unpack_to_dest_mode for its cb_k_rm)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[cb_k_rm] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_cfg.unpack_to_dest_mode = modes
    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for (u0, n), c in zip(work, coords):
        reader_rt[c.x][c.y] = [q32.buffer_address(), cache.buffer_address(), tokens.buffer_address(), u0, n]
        writer_rt[c.x][c.y] = [out.buffer_address(), cache.buffer_address(), u0, n, base_unit]
        compute_rt[c.x][c.y] = [n]
    kdir = KDIR + "split_"
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=kdir + "reader.cpp",
            core_ranges=cores,
            compile_time_args=[
                heads,
                s,
                width,
                SPLIT_K_CHUNK,
                k_dim,
                splits,
                cb_q_rm,
                cb_k_rm,
                cb_mask,
                cb_idx,
                cb_ctrl,
                cb_kreq,
                cb_kack,
                int(q_tiled),
                cb_q_in,
                kv_bytes,
            ]
            + ttnn.TensorAccessorArgs(q32).get_compile_time_args()
            + ttnn.TensorAccessorArgs(cache).get_compile_time_args()
            + ttnn.TensorAccessorArgs(tokens).get_compile_time_args()
            + [upq, int(write_empty)],
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=kdir + "writer.cpp",
            core_ranges=cores,
            compile_time_args=[vdht, k_dim, splits, s, cb_part, cb_scale, cb_ident, cb_neginf, cb_idx, cb_kreq, cb_kack]
            + [kv_bytes]
            + ttnn.TensorAccessorArgs(out).get_compile_time_args()
            + ttnn.TensorAccessorArgs(cache).get_compile_time_args()
            + [int(write_empty), cb_fill, sqt],
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=kdir + "compute.cpp",
            core_ranges=cores,
            compile_time_args=[
                heads,
                dht,
                vdht,
                skt,
                _fp32_bits(scale),
                cb_q_rm,
                cb_q_in,
                cb_k_rm,
                cb_k_in,
                cb_neginf,
                cb_mask,
                cb_scale,
                cb_qk,
                cb_max_a,
                cb_max_b,
                cb_sum_a,
                cb_sum_b,
                cb_out_a,
                cb_out_b,
                cb_corr,
                cb_part,
                cb_ctrl,
                cb_ident,
                0,
                1,
                int(q_tiled),
            ],
            runtime_args=compute_rt,
            config=compute_cfg,
        ),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    _run(mesh, [q32, cache, tokens, out], program)
    return out


def shard_repack(mesh, parts, q: int, g: int) -> ttnn.Tensor:
    """The split latents' partials for the row's exchange (kernels/dsa/shard_repack.cpp): `sparse_split`'s partials of
    the 2 q 32-head units of q queries, g splits each ([1, 1, 2 q g 32, 18 x 32]) -> [1, 1, 4 R, 18 x 32] (R = q g 16),
    chip d's chunk its heads 16 d .. of every (query, split), two a tile; all_to_all_async_generic (in_dim 1, out_dim 2)
    sends chunk d to chip d."""
    units = 2 * q * g
    if (q * g) % 2 or parts.shape[-2] != units * 32:
        raise ValueError(f"shard_repack: {parts.shape} for {q} queries x {g} splits (q g even)")
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            ttnn.Shape([1, 1, COLS * q * g * 16, PART_TILES * 32]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            buffer_type=ttnn.BufferType.DRAM,
        ),
        mesh,
    )
    grid = mesh.compute_with_storage_grid_size()
    split = _split(units, grid, 1)
    coords, cores = _cores(split, grid)
    rt = ttnn.RuntimeArgs()
    for c, (u0, u1) in zip(coords, split):
        rt[c.x][c.y] = [parts.buffer_address(), out.buffer_address(), u0, u1]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KDIR + "shard_repack.cpp",
        core_ranges=cores,
        compile_time_args=[g, q, PART_TILES, 0]
        + ttnn.TensorAccessorArgs(parts).get_compile_time_args()
        + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        + [len(split)],
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_cb(0, PART_TILES * 2048, cores)])
    _run(mesh, [parts, out], program)
    return out


def sparse_combine(mesh, parts, tokens, splits: int, scale: float, xchip: tuple[int, int] | None = None) -> ttnn.Tensor:
    """`sparse_split`'s partials -> the attention's out [1, S, 16, 512] bf16 tiles (a token's heads in rows 0-15 of
    its tile row): per token and row, the splits' outs weighted by exp((m_j - M) scale) / sum_j l_j exp((m_j - M)
    scale) (kernels/dsa/combine_*.cpp; the splits that wrote one, from the token's valid keys; M by the reader, the
    sums by the packer's L1 accumulation in fp32).  `xchip` (Q, G) (the split latents): the partials are
    the row's exchange (`shard_repack`, then all_to_all: [1, 4, Q G 16, 18 x 32], chip c's G splits of each of the Q
    tokens, 16 heads each), every one written: splits = 4 G, tokens only for its shape."""
    s, width = tokens.shape[-2], tokens.shape[-1]
    xg = 1
    if xchip is not None:
        if xchip[0] != s or splits != COLS * xchip[1]:
            raise ValueError(f"sparse_combine: xchip {xchip} with {s} tokens and {splits} splits")
        xg = xchip[1]
    grid = mesh.compute_with_storage_grid_size()
    groups = next((q for q in (4, 2, 1) if s * q <= grid.x * grid.y), 1)  # (more tokens than cores: units in turn)
    vdht = 16
    per = vdht // groups
    units = s * groups
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape([1, s, 16, 512]), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM),
        mesh,
    )
    work = _unit_ranges(units, grid)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(work))]
    cores = core_rects(coords)
    tile = 2048
    cb_m, cb_l, cb_o, cb_ctrl, cb_idx, cb_max, cb_w, cb_out, cb_w2, cb_sum, cb_recip, cb_acc = range(12)
    bf, f32 = ttnn.bfloat16, ttnn.float32
    cbs = [
        _cb_fmt(cb_m, bf, tile, splits, cores),
        _cb_fmt(cb_l, bf, tile, splits, cores),
        _cb_fmt(cb_o, bf, tile, splits * per, cores),
        _cb_fmt(cb_ctrl, bf, 16, 1, cores),
        _cb_fmt(cb_idx, bf, width * 4, 1, cores),
        _cb_fmt(cb_max, bf, tile, 1, cores),
        _cb_fmt(cb_w, bf, tile, splits, cores),
        _cb_fmt(cb_out, bf, tile, 2, cores),
        _cb_fmt(cb_w2, bf, tile, splits, cores),
        _cb_fmt(cb_sum, f32, 2 * tile, 1, cores),
        _cb_fmt(cb_recip, bf, tile, 1, cores),
        _cb_fmt(cb_acc, f32, 2 * tile, 1, cores),
    ]
    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for (u0, n), c in zip(work, coords):
        reader_rt[c.x][c.y] = [parts.buffer_address(), tokens.buffer_address(), u0, n]
        writer_rt[c.x][c.y] = [out.buffer_address(), u0, n]
        compute_rt[c.x][c.y] = [n]
    kdir = KDIR + "combine_"
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=kdir + "reader.cpp",
            core_ranges=cores,
            compile_time_args=[
                s,
                splits,
                vdht,
                per,
                groups,
                width,
                SPLIT_K_CHUNK,
                cb_m,
                cb_l,
                cb_o,
                cb_ctrl,
                cb_idx,
                cb_max,
            ]
            + ttnn.TensorAccessorArgs(parts).get_compile_time_args()
            + ttnn.TensorAccessorArgs(tokens).get_compile_time_args()
            + ([1, xg, s * xg // 2] if xchip else [0, 1, 1]),
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=kdir + "writer.cpp",
            core_ranges=cores,
            compile_time_args=[s, vdht, per, groups, cb_out] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=kdir + "compute.cpp",
            core_ranges=cores,
            compile_time_args=[
                per,
                _fp32_bits(scale),
                cb_m,
                cb_l,
                cb_o,
                cb_ctrl,
                cb_max,
                cb_w,
                cb_out,
                cb_w2,
                cb_sum,
                cb_recip,
                cb_acc,
            ],
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
        ),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    _run(mesh, [parts, tokens, out], program)
    return out


# the recent pages' rows: position pos's indexer key and gate at row pos mod RECENT_RING of its user's page (a ring of 4
# pools).  A verify step writes T positions p .. p + T - 1 and the next one starts at p + n + 1, so the rejected drafts'
# rows stay behind; with rows pos mod 4 a draft in the next pool (or the one after) would overwrite the row of a
# committed token of the next step's pool, whose key then took the draft's (the long cache's verify step parted from
# the plain decode at T >= 3).  The ring keeps them for up to RECENT_RING - 2 tokens a user a step
RECENT_RING = 16


def recent_row(pos: int) -> int:
    """Position pos's row in its user's recent pages."""
    return pos % RECENT_RING


# `index_update`'s defaults: the split form, its cores a user (0: the step's segments), keys made only where the step
# leaves them, and the column tiles an op
INDEX_SPLIT = True
INDEX_PARTS = 0
INDEX_SKIP = True
INDEX_BATCH = 4


def index_update(
    mesh,
    ik,
    ig,
    params,
    rkey,
    rgate,
    pools,
    bias,
    users: int,
    segs: int,
    split: bool | None = None,
    parts: int | None = None,
    skip: bool | None = None,
    batch: int | None = None,
) -> None:
    """Decode's indexer update in one op (`TtDsa._step_indexer`; it replaced ~20 ops a segment: the per-segment
    writes, the pool keys and their writes), as a core per user doing its `segs` segments in turn would (virtual user
    v = s users + u): row `recent_row(pos)` of its recent pages (rkey / rgate page u) <- row v of ik / ig [1, 1, B,
    128], then the pool's key from its 4 rows (as `_compress`) into row (pos / 4) mod 32 of its pool page (params row
    v: [pos, base, pool page base]), the caches updated in place (whole tiles read and written).

    The split form (`split`, default INDEX_SPLIT, on; kernels/dsa/index_update_split_*.cpp) leaves the caches the
    same bit for bit: a user's segments over `parts` cores (default INDEX_PARTS or `segs`, as many as the grid holds),
    only the keys the step leaves made (`skip`, default INDEX_SKIP: False makes every key, a timing control), the 4
    column tiles an op (`batch`, default INDEX_BATCH 4; 1: a tile an op), a user's last core writing its pages.
    Off: kernels/dsa/index_update_*.cpp, a core per user (the reference form)."""
    if split is None:
        split = INDEX_SPLIT
    if split:
        if parts is None:
            parts = INDEX_PARTS or segs
        if skip is None:
            skip = INDEX_SKIP
        if batch is None:
            batch = INDEX_BATCH
        _index_update_split(mesh, ik, ig, params, rkey, rgate, pools, bias, users, segs, parts, skip, batch)
        return
    if pools.dtype != ttnn.bfloat16:
        raise ValueError("index_update: a BFP pools cache takes the split form (INDEX_SPLIT)")
    grid = mesh.compute_with_storage_grid_size()
    if users > grid.x * grid.y:
        raise ValueError(f"index_update: {users} users over {grid.x * grid.y} cores")
    if segs > RECENT_RING - 2:
        raise ValueError(f"index_update: {segs} tokens a user, the recent pages' ring keeps {RECENT_RING - 2}")
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(users)]
    cores = core_rects(coords)
    tile = 2048
    (
        cb_rk,
        cb_rg,
        cb_key,
        cb_pool,
        cb_ik,
        cb_ig,
        cb_bias,
        cb_scaler,
        cb_params,
        cb_t,
        cb_m,
        cb_e,
        cb_rs,
        cb_w,
        cb_p,
        cb_ring,
    ) = range(16)
    bf, f32 = ttnn.bfloat16, ttnn.float32
    cbs = [
        _cb_fmt(cb_rk, bf, tile, 4, cores),
        _cb_fmt(cb_rg, bf, tile, 4, cores),
        _cb_fmt(cb_key, bf, tile, 4, cores),
        _cb_fmt(cb_pool, bf, tile, 4, cores),
        _cb_fmt(cb_ik, bf, tile, 4, cores),
        _cb_fmt(cb_ig, bf, tile, 4, cores),
        _cb_fmt(cb_bias, f32, 2 * tile, 4, cores),
        _cb_fmt(cb_scaler, bf, tile, 1, cores),
        _cb(cb_params, 64, cores),
        _cb_fmt(cb_t, f32, 2 * tile, 1, cores),
        _cb_fmt(cb_m, f32, 2 * tile, 1, cores),
        _cb_fmt(cb_e, f32, 2 * tile, 1, cores),
        _cb_fmt(cb_rs, f32, 2 * tile, 1, cores),
        _cb_fmt(cb_w, bf, tile, 1, cores),
        _cb_fmt(cb_p, bf, tile, 1, cores),
        _cb_fmt(cb_ring, bf, tile, 8, cores),  # a user's recent pages (rkey, rgate), kept over the segments
    ]
    tensors = [ik, ig, params, rkey, rgate, pools, bias]
    reader_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for u, c in enumerate(coords):
        reader_rt[c.x][c.y] = [t.buffer_address() for t in tensors] + [u]
        compute_rt[c.x][c.y] = [1]
    ct = []
    for t in tensors:
        ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "index_update_reader.cpp",
            core_ranges=cores,
            compile_time_args=[users, segs, cb_rk, cb_rg, cb_key, cb_pool, cb_ik, cb_ig, cb_bias, cb_scaler, cb_params]
            + [cb_ring, RECENT_RING]
            + ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "index_update_compute.cpp",
            core_ranges=cores,
            compile_time_args=[segs, cb_rk, cb_rg, cb_key, cb_bias, cb_scaler, cb_t, cb_m, cb_e, cb_rs, cb_w, cb_p],
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
        ),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    _run(mesh, tensors, program)


INDEX_SPLIT_SEM = 0
INDEX_SPLIT_PAGES_AHEAD = 2  # pool pages the owner reads before the keys come (a step's consecutive positions: <= 2)


def _index_update_split(mesh, ik, ig, params, rkey, rgate, pools, bias, users, segs, parts, skip, batch) -> None:
    """`index_update`'s split form: users x parts cores (user u's part k at core u parts + k: segments k segs / parts ..
    (k + 1) segs / parts - 1; the last part, the owner, writes u's pages; kernels/dsa/index_update_split_*.cpp
    say why the caches come out the same)."""
    grid = mesh.compute_with_storage_grid_size()
    n_cores = grid.x * grid.y
    if users > n_cores:
        raise ValueError(f"index_update: {users} users over {n_cores} cores")
    if segs > RECENT_RING - 2:
        raise ValueError(f"index_update: {segs} tokens a user, the recent pages' ring keeps {RECENT_RING - 2}")
    if batch not in (1, 4):
        raise ValueError(f"index_update: batch {batch} (1 or 4)")
    parts = max(1, min(parts, segs, n_cores // users))
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(users * parts)]
    cores = core_rects(coords)
    tile = 2048
    nblk = max(((segs - 1) * users + u) // 32 - u // 32 + 1 for u in range(users))
    npb = INDEX_SPLIT_PAGES_AHEAD
    # the pools' format (dsa.POOLS_DTYPE): a BFP cache's pages read and written in its tiles, the kept keys' rows
    # rounded in by the owner (kernels/dsa/bfp_rows.hpp)
    pfmt = pools.dtype
    if pfmt not in TILE_BYTES:
        raise ValueError(f"index_update: a pools cache of {pfmt}")
    pbits = {ttnn.bfloat16: 0, ttnn.bfloat8_b: 8, ttnn.bfloat4_b: 4}[pfmt]
    (
        cb_rk,
        cb_rg,
        cb_key,
        cb_pool,
        cb_ik,
        cb_ig,
        cb_bias,
        cb_scaler,
        cb_params,
        cb_t,
        cb_m,
        cb_e,
        cb_rs,
        cb_w,
        cb_p,
        cb_ring,
        cb_ctl,
        cb_slot,
    ) = range(18)
    bf, f32 = ttnn.bfloat16, ttnn.float32
    cbs = [
        _cb_fmt(cb_rk, bf, tile, 8, cores),  # two keys' rows (the reserves alternate)
        _cb_fmt(cb_rg, bf, tile, 8, cores),
        _cb_fmt(cb_key, bf, tile, 4, cores),
        _cb_fmt(cb_pool, pfmt, TILE_BYTES[pfmt], 4 * npb, cores),
        _cb_fmt(cb_ik, bf, tile, 4 * nblk, cores),
        _cb_fmt(cb_ig, bf, tile, 4 * nblk, cores),
        _cb_fmt(cb_bias, f32, 2 * tile, 4, cores),
        _cb_fmt(cb_scaler, bf, tile, 1, cores),
        _cb(cb_params, segs * PARAMS * 4, cores),
        _cb_fmt(cb_t, f32, 2 * tile, batch, cores),
        _cb_fmt(cb_m, f32, 2 * tile, batch, cores),
        _cb_fmt(cb_e, f32, 2 * tile, batch, cores),
        _cb_fmt(cb_rs, f32, 2 * tile, batch, cores),
        _cb_fmt(cb_w, bf, tile, batch, cores),
        _cb_fmt(cb_p, bf, tile, batch, cores),
        _cb_fmt(cb_ring, bf, tile, 8, cores),  # a user's recent pages (rkey, rgate)
        _cb(cb_ctl, 64, cores),  # the count of keys, the reader's to compute
        _cb(cb_slot, segs * 256, cores),  # the owner's: a kept key's row a segment
    ]
    tensors = [ik, ig, params, rkey, rgate, pools, bias]
    addrs = [t.buffer_address() for t in tensors]
    reader_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for u in range(users):
        own = mesh.worker_core_from_logical_core(coords[u * parts + parts - 1])
        for k in range(parts):
            c = coords[u * parts + k]
            s0, s1 = k * segs // parts, (k + 1) * segs // parts
            reader_rt[c.x][c.y] = addrs + [u, s0, s1, int(k == parts - 1), own.x, own.y]
            compute_rt[c.x][c.y] = [1]
    ct = []
    for t in tensors:
        ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    reader_ct = [users, segs, cb_rk, cb_rg, cb_key, cb_pool, cb_ik, cb_ig, cb_bias, cb_scaler, cb_params, cb_ring]
    reader_ct += [cb_ctl, cb_slot, RECENT_RING, int(skip), INDEX_SPLIT_SEM, nblk, npb, pbits]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "index_update_split_reader.cpp",
            core_ranges=cores,
            compile_time_args=reader_ct + ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "index_update_split_compute.cpp",
            core_ranges=cores,
            compile_time_args=[cb_rk, cb_rg, cb_key, cb_bias, cb_scaler, cb_t, cb_m, cb_e, cb_rs, cb_w, cb_p]
            + [cb_ctl, batch],
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
        ),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=INDEX_SPLIT_SEM, core_ranges=cores, initial_value=0)]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    _run(mesh, tensors, program)


def zeros_bytes(mesh, shape, dtype, layout) -> ttnn.Tensor:
    """A DRAM tensor of a per-chip shape whose bytes are all zero (0.0 in FP8 e4m3 and in BFP8_b / BFP4_b: the
    low-precision KV caches, which ttnn.zeros cannot make on the device: FP8 not at all, BFP from the host), by one
    op (kernels/dsa/zero_pages.cpp) over its pages."""
    import math

    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape([int(d) for d in shape]), dtype, layout, buffer_type=ttnn.BufferType.DRAM), mesh
    )
    if layout == ttnn.TILE_LAYOUT:
        page = TILE_BYTES[dtype]
        pages = math.prod(int(d) for d in out.padded_shape) // 1024
    else:  # (row-major: a row a page)
        page = int(shape[-1]) * out.element_size()
        pages = math.prod(int(d) for d in shape[:-1])
    page = -(-page // 64) * 64  # (DRAM pages are aligned)
    grid = mesh.compute_with_storage_grid_size()
    parts = _split(pages, grid, 64)
    coords, cores = _cores(parts, grid)
    rt = ttnn.RuntimeArgs()
    for c, (p0, p1) in zip(coords, parts):
        rt[c.x][c.y] = [out.buffer_address(), p0, p1]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KDIR + "zero_pages.cpp",
        core_ranges=cores,
        compile_time_args=[0, page] + ttnn.TensorAccessorArgs(out).get_compile_time_args() + [pages],
        runtime_args=rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    # (generic_op takes its inputs and then its output: the tensor stands for both)
    _run(mesh, [out, out], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_cb(0, page, cores)]))
    return out


def bfp_cast(mesh, x, dtype, precise: bool = True, fp32_dest: bool = False) -> ttnn.Tensor:
    """x [.., H, W] bf16 tiled -> the same values in BFP8_b / BFP4_b tiles (the BFP KV caches), packed
    by the packer from Float16_b (`precise`, the ComputeConfigDescriptor's bfp8_pack_precise: ttnn.typecast sets it
    for BFP8 only, and its BFP4 truncates) or from its BFP8 source (precise False, as typecast's BFP4).  One op
    (kernels/dsa/bfp_cast_*.cpp), the tiles over the cores."""
    import math

    tiles = math.prod(int(d) for d in x.padded_shape) // 1024  # (the padded tiles: a tile row of 8 rows is a tile)
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape(list(x.shape)), dtype, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM), mesh
    )
    grid = mesh.compute_with_storage_grid_size()
    parts = _split(tiles, grid, 1)
    coords, cores = _cores(parts, grid)
    cb_in, cb_out = 0, 1
    cbs = [_cb_fmt(cb_in, ttnn.bfloat16, 2048, 2, cores), _cb_fmt(cb_out, dtype, TILE_BYTES[dtype], 2, cores)]
    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (t0, t1) in zip(coords, parts):
        reader_rt[c.x][c.y] = [x.buffer_address(), t0, t1]
        compute_rt[c.x][c.y] = [t1 - t0]
        writer_rt[c.x][c.y] = [out.buffer_address(), t0, t1]
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_dest, math_approx_mode=False
    )
    cfg.bfp8_pack_precise = precise
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "bfp_cast_reader.cpp",
            core_ranges=cores,
            compile_time_args=[cb_in] + ttnn.TensorAccessorArgs(x).get_compile_time_args(),
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "bfp_cast_compute.cpp",
            core_ranges=cores,
            compile_time_args=[cb_in, cb_out, tiles],  # (tiles: the per-core split follows them, gv)
            runtime_args=compute_rt,
            config=cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "bfp_cast_writer.cpp",
            core_ranges=cores,
            compile_time_args=[cb_out] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    _run(mesh, [x, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out


def bfp_pack(mesh, x, dtype) -> ttnn.Tensor:
    """x [.., H, W] bf16 tiled -> the same shape in BFP8_b / BFP4_b tiles, each face row of 16 values rounded to nearest
    (ties away from zero) on its largest exponent by the data movement cores (kernels/dsa/bfp_rows.hpp:
    tools/kv_round.bfp_rna; the packer's BFP4, bfp_cast's, truncates its BFP8 value): the pools' keys a prefill chunk
    writes into a BFP pools cache (dsa.POOLS_DTYPE), as decode's index_update rounds them.  The tiles over the
    cores, each core's on both of its data movement cores."""
    import math

    if dtype not in (ttnn.bfloat8_b, ttnn.bfloat4_b) or x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"bfp_pack: bf16 tiles to BFP8_b / BFP4_b, not {x.dtype} {x.layout} to {dtype}")
    tiles = math.prod(int(d) for d in x.padded_shape) // 1024
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape(list(x.shape)), dtype, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM), mesh
    )
    bits = 8 if dtype == ttnn.bfloat8_b else 4
    grid = mesh.compute_with_storage_grid_size()
    parts = _split(tiles, grid, 2)  # (a core's range: half on each data movement core)
    coords, cores = _cores(parts, grid)
    rts = [ttnn.RuntimeArgs(), ttnn.RuntimeArgs()]
    for c, (t0, t1) in zip(coords, parts):
        mid = t0 + (t1 - t0 + 1) // 2
        rts[0][c.x][c.y] = [x.buffer_address(), out.buffer_address(), t0, mid]
        rts[1][c.x][c.y] = [x.buffer_address(), out.buffer_address(), mid, t1]
    slots = 2 * 2048 + 2 * TILE_BYTES[dtype]
    acc = ttnn.TensorAccessorArgs(x).get_compile_time_args() + ttnn.TensorAccessorArgs(out).get_compile_time_args()
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "bfp_pack.cpp",
            core_ranges=cores,
            compile_time_args=[bits, cb] + acc + [tiles],  # (tiles: the per-core split follows them)
            runtime_args=rts[cb],
            config=ttnn.ReaderConfigDescriptor() if cb == 0 else ttnn.WriterConfigDescriptor(),
        )
        for cb in (0, 1)
    ]
    cbs = [_cb(0, slots, cores), _cb(1, slots, cores)]
    _run(mesh, [x, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out


def pool_keys_bias(mesh, pool_bias):
    """The slot biases [4, 128] (None: 0) as pool_keys' operand: [1, 1, 32, 128] fp32, row r = bias[r mod 4]."""
    import torch

    from .common import put

    rows = torch.zeros(1, 1, 32, 128)
    if pool_bias is not None:
        rows[0, 0] = pool_bias.float().reshape(4, 128).repeat(8, 1)
    return put(mesh, rows, dims=(None, None), dtype=ttnn.float32)


def pool_keys(mesh, ik, ig, bias) -> ttnn.Tensor:
    """A prefill chunk's pools' keys in one op (kernels/dsa/pool_keys_*.cpp, `TtDsa._chunk_pool_keys`, in place of
    ~19 ops: the two pads to a pool a tile, the 10 of `_compress` and the reshape): ik, ig [1, 1, T, 128] bf16 tiled
    (T a multiple of 128), bias from `pool_keys_bias` -> [1, 1, T / 4, 128] bf16 tiled, the same values (each step as
    the ttnn op it replaces, on 8 pools a tile)."""
    t = ik.shape[-2]
    if t % 128 or ik.shape[-1] != 128 or tuple(ig.shape) != tuple(ik.shape):
        raise ValueError(f"pool_keys: [1, 1, T, 128] with T a multiple of 128, not {ik.shape} / {ig.shape}")
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            ttnn.Shape([1, 1, t // 4, 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM
        ),
        mesh,
    )
    grid = mesh.compute_with_storage_grid_size()
    parts = _split(t // 32 * 4, grid, 1)
    coords, cores = _cores(parts, grid)
    cb_ig, cb_ik, cb_bias, cb_scaler, cb_prod, cb_pad, cb_key, cb_stage = range(8)
    pad_pages = 16
    bf, f32, tile = ttnn.bfloat16, ttnn.float32, 2048
    cbs = [
        _cb_fmt(cb_ig, bf, tile, 2, cores),
        _cb_fmt(cb_ik, bf, tile, 2, cores),
        _cb_fmt(cb_bias, f32, 2 * tile, 4, cores),
        _cb_fmt(cb_scaler, bf, tile, 1, cores),
        _cb_fmt(cb_prod, bf, tile, 2, cores),
        _cb_fmt(cb_pad, bf, tile, pad_pages, cores),
        _cb_fmt(cb_key, bf, tile, 16, cores),
        _cb(cb_stage, 512, cores),
    ]
    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (i0, i1) in zip(coords, parts):
        reader_rt[c.x][c.y] = [ig.buffer_address(), ik.buffer_address(), bias.buffer_address(), i0, i1]
        compute_rt[c.x][c.y] = [i0, i1]
        writer_rt[c.x][c.y] = [out.buffer_address(), i0, i1]
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[cb_bias] = ttnn.UnpackToDestMode.UnpackToDestFp32  # fp32 into Dest whole (SrcA would keep tf32)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    cfg.unpack_to_dest_mode = modes
    key = [t, len(parts)]  # (the per-core split follows T: the program cache does not see runtime values)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "pool_keys_reader.cpp",
            core_ranges=cores,
            compile_time_args=[cb_ig, cb_ik, cb_bias, cb_scaler, cb_prod, cb_pad, pad_pages]
            + ttnn.TensorAccessorArgs(ig).get_compile_time_args()
            + ttnn.TensorAccessorArgs(ik).get_compile_time_args()
            + ttnn.TensorAccessorArgs(bias).get_compile_time_args()
            + key,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "pool_keys_compute.cpp",
            core_ranges=cores,
            compile_time_args=[cb_ig, cb_ik, cb_bias, cb_scaler, cb_prod, cb_pad, cb_key] + key,
            runtime_args=compute_rt,
            config=cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "pool_keys_writer.cpp",
            core_ranges=cores,
            compile_time_args=[cb_key, cb_stage] + ttnn.TensorAccessorArgs(out).get_compile_time_args() + key,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    _run(mesh, [ig, ik, bias, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out


__all__ = [
    "index_update",
    "shard_candidates",
    "shard_pick",
    "shard_repack",
    "prefix_rows",
    "pool_keys",
    "pool_keys_bias",
    "RECENT_RING",
    "recent_row",
    "PARAMS",
    "SENT",
    "pages_gather",
    "pool_scores",
    "rows_write",
    "select_tokens",
    "sparse_combine",
    "sparse_split",
]
