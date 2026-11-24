"""Accepting a verify step's drafts (`TtGlm53Model.forward_verify`): per user, the drafts accepted n
(the leading drafts each equal to the model's greedy id one position before: draft j at p + j against the greedy id at
p + j - 1) and the bonus token (the greedy id at p + n), from which the next step goes on: KDA's keep code n, the next
token the bonus, the position p + n + 1.

`accept_host` on the host's tokens and greedy ids; `accept_device` on the device from forward_verify's ids and the
step's own token buffer, the result of every row on every chip, read back once ([ROWS, 2 U]: n, then the bonus)."""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.blackhole.glm53_flash.ttnn.common import COLS, ROW_AXIS, ROWS, put


def accept_host(tokens: torch.Tensor, greedy: torch.Tensor, drafts: int) -> tuple[torch.Tensor, torch.Tensor]:
    """tokens, greedy [ROWS, (K + 1) U] (virtual user j U + u: user u's token j, j = 0 the given token, 1 .. K its
    drafts; greedy the model's ids there) -> (n, bonus), each [ROWS, U] int64."""
    rows, b = tokens.shape
    t, g = tokens.reshape(rows, drafts + 1, -1), greedy.reshape(rows, drafts + 1, -1)
    n = (t[:, 1:] == g[:, :-1]).to(torch.int64).cumprod(dim=1).sum(dim=1)
    return n, g.gather(1, n[:, None, :]).squeeze(1)


_STEPS: dict = {}  # (mesh id, K + 1) -> [1, 1, K + 1, 1] fp32: 0 .. K (the bonus's pick)


def _own(ids, io, b: int):
    """Every row's ids [1, 1, 8 b, 1] fp32 -> this row's [1, 1, 1, b]."""
    grid = ttnn.reshape(ids, [1, 1, ROWS, b])
    own = ttnn.min(ttnn.add(grid, io.others), dim=2, keepdim=True)
    ttnn.deallocate(grid)
    return own


def _frac(x):
    """x - floor(x) (fp32); x deallocated."""
    whole = ttnn.floor(x)
    out = ttnn.subtract(x, whole)
    for t in (x, whole):
        ttnn.deallocate(t)
    return out


def forced_accept(base1, seats, drafts: int):
    """A test's acceptance (GLM53_TEST_FORCE_ACCEPT, `DFlash2Step`; the synthetic weights' drafts are never accepted):
    each user's n from its position and seat alone, its drafts taken whatever the greedy ids, so that the runs a test
    compares keep 0 .. K drafts alike: a = frac(0.0137 (p + 1) + 0.755 seat), b = frac(251.17 a^2 + 311.9 a),
    n = floor((K + 1) b) (a quadratic mix: frac(c p) alone locks into a few n as p steps by n + 1; this one spreads
    0 .. K evenly over 32 seats' runs in a host fp32 model).  base1 [1, 1, 1, U] int32 row-major (p + 1), seats
    [1, 1, 1, U] fp32 -> n [1, 1, 1, U] fp32."""
    b1 = ttnn.to_layout(base1, ttnn.TILE_LAYOUT)
    x = ttnn.typecast(b1, ttnn.float32)
    ttnn.deallocate(b1)
    xs, ss = ttnn.multiply(x, 0.0137), ttnn.multiply(seats, 0.7548776662)
    ttnn.deallocate(x)
    a = _frac(ttnn.add(xs, ss))
    for t in (xs, ss):
        ttnn.deallocate(t)
    sq = ttnn.multiply(a, a)
    hi, lo = ttnn.multiply(sq, 251.17), ttnn.multiply(a, 311.9)
    for t in (a, sq):
        ttnn.deallocate(t)
    b = _frac(ttnn.add(hi, lo))
    for t in (hi, lo):
        ttnn.deallocate(t)
    scaled = ttnn.multiply(b, float(drafts + 1))
    ttnn.deallocate(b)
    n = ttnn.floor(scaled)
    ttnn.deallocate(scaled)
    out = ttnn.clamp(n, min=0.0, max=float(drafts))
    ttnn.deallocate(n)
    return out


def accept_device(mesh, ids, io, raw=None, forced=None) -> object:
    """forward_verify's greedy ids [1, 1, 8 (K + 1) U, 1] fp32 (every row's on every chip) and `io` (SpecIO: its own
    row's tokens `ids` [1, (K + 1) U] uint32 row-major, `others`) -> [1, 1, 8, 2 U] fp32 row-major on every chip: row
    r's n (U), then its bonus tokens (U).  `raw` (the looser acceptance, GLM53_ACCEPT_RELAX: the plain argmax ids, as
    `ids`): [1, 1, 8, 3 U], each user's accepted drafts that were not the argmax after them.  `forced` (a test's,
    (base1, seats): `forced_accept`): n from the positions and seats in place of the drafts' match."""
    n_tok, u = io.drafts + 1, io.users
    b = n_tok * u
    own = _own(ids, io, b)  # [1, 1, 1, b]: this row's greedy ids
    tok = ttnn.typecast(ttnn.to_layout(ttnn.reshape(io.ids, [1, 1, 1, b]), ttnn.TILE_LAYOUT), ttnn.float32)
    g4 = ttnn.reshape(own, [1, 1, n_tok, u])
    t4 = ttnn.reshape(tok, [1, 1, n_tok, u])
    for x in (own, tok):
        ttnn.deallocate(x)
    drafts = ttnn.slice(t4, [0, 0, 1, 0], [1, 1, n_tok, u])
    match = ttnn.eq(drafts, ttnn.slice(g4, [0, 0, 0, 0], [1, 1, n_tok - 1, u]))
    ttnn.deallocate(t4)
    run = ttnn.cumprod(match, dim=2)
    ttnn.deallocate(match)
    n = ttnn.sum(run, dim=2, keepdim=True)  # [1, 1, 1, U]
    if forced is not None:  # (GLM53_TEST_FORCE_ACCEPT: no looser acceptance with it)
        assert raw is None, "GLM53_TEST_FORCE_ACCEPT with the looser acceptance"
        ttnn.deallocate(n)
        n = forced_accept(*forced, io.drafts)
    changed = None
    if raw is not None:  # the accepted drafts that differ from the plain argmax: n - sum(run x exact)
        r4 = ttnn.reshape(_own(raw, io, b), [1, 1, n_tok, u])
        exact = ttnn.eq(drafts, ttnn.slice(r4, [0, 0, 0, 0], [1, 1, n_tok - 1, u]))
        ttnn.deallocate(r4)
        kept = ttnn.sum(ttnn.mul(run, exact), dim=2, keepdim=True)
        ttnn.deallocate(exact)
        changed = ttnn.sub(n, kept)
        ttnn.deallocate(kept)
    ttnn.deallocate(drafts)
    ttnn.deallocate(run)
    key = (id(mesh), n_tok)
    if key not in _STEPS:
        steps = torch.arange(n_tok, dtype=torch.float32).reshape(1, 1, n_tok, 1)
        _STEPS[key] = put(mesh, steps, dims=(None, None), dtype=ttnn.float32)
    pick = ttnn.eq(_STEPS[key], n)  # [1, 1, K + 1, U]: 1 at row n
    chosen = ttnn.mul(g4, pick)
    for x in (g4, pick):
        ttnn.deallocate(x)
    bonus = ttnn.sum(chosen, dim=2, keepdim=True)  # [1, 1, 1, U]
    ttnn.deallocate(chosen)
    parts = [n, bonus] + ([changed] if changed is not None else [])
    both = ttnn.to_layout(ttnn.concat(parts, dim=3), ttnn.ROW_MAJOR_LAYOUT)  # [1, 1, 1, 2 U] (3 U)
    for x in parts:
        ttnn.deallocate(x)
    out = ttnn.all_gather(both, dim=2, cluster_axis=ROW_AXIS)  # [1, 1, 8, 2 U]
    ttnn.deallocate(both)
    return out


def accepted_to_host(out, users: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`accept_device`'s tensor -> (n, bonus), each [ROWS, U] int64 (the first chip's)."""
    flat = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(ROWS, -1).round().to(torch.int64)
    return flat[:, :users], flat[:, users : 2 * users]


# The acceptance and the drafter's inputs in one op: accept_device's and dflash2_step.drafter_inputs's ~40 small ttnn
# ops (the reshape of the ids, min, cumprod, sums, typecasts, tilize / untilize: ~0.2 ms a step at every point, most of
# it single-core ops and their gaps) as one single-core program a chip (`kernels/dflash2/accept_inputs.cpp`) and the
# all-gather of `both` over the rows; the same values (whole numbers: the fp32 compares, cumprod and sums are exact on
# them).  GLM53_DFLASH2_ACCEPT_FUSED (read at each call; 1 by default; =0: the separate ops).  Not with the looser
# acceptance's raw ids (those keep the ops), nor ids other than fp32 TILE [1, 1, 8 B, 1]; a test's forced acceptance
# keeps `forced_accept`'s ops for n and hands its n to the op.
KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/dflash2/"


def accept_fused_on() -> bool:
    return os.environ.get("GLM53_DFLASH2_ACCEPT_FUSED", "1") == "1"


def accept_fused_takes(ids, io) -> bool:
    b = (io.drafts + 1) * io.users
    return (
        ids.dtype == ttnn.float32
        and ids.layout == ttnn.TILE_LAYOUT
        and tuple(ids.shape) == (1, 1, ROWS * b, 1)
        and io.ids.dtype == ttnn.uint32
        and io.ids.layout == ttnn.ROW_MAJOR_LAYOUT
        and all(_dram(t) for t in (ids, io.ids))
    )


def _dram(t) -> bool:
    return t.memory_config().buffer_type == ttnn.BufferType.DRAM and not t.is_sharded()


def _rm(mesh, shape, dtype):
    spec = ttnn.TensorSpec(ttnn.Shape(shape), dtype, ttnn.ROW_MAJOR_LAYOUT, buffer_type=ttnn.BufferType.DRAM)
    return ttnn.allocate_tensor_on_device(spec, mesh)


def accept_inputs(mesh, ids, io, base1, forced=None, gather: bool = True) -> tuple:
    """`accept_device` (no raw, no forced) and `dflash2_step.drafter_inputs` in one op: forward_verify's ids, `io`
    (SpecIO), base1 [1, 1, 1, U] int32 row-major (each user's p + 1) -> (accepted [1, 1, 8, 2 U] fp32 row-major on every
    chip, bonus [1, U] uint32 row-major, starts [1, U] int32 row-major), the same tensors (shape, layout, values) as
    the ops'.  `forced` (base1, seats): a test's forced acceptance, n from `forced_accept` (its ops).  gather=False: the
    own row [1, 1, 1, 2 U] in place of `accepted` (for `results_row`, which gathers it with the drafter's block)."""
    u, k = io.users, io.drafts
    n_forced = forced_accept(*forced, k) if forced is not None else None
    b = (k + 1) * u
    both = _rm(mesh, [1, 1, 1, 2 * u], ttnn.float32)
    bonus = _rm(mesh, [1, u], ttnn.uint32)
    starts = _rm(mesh, [1, u], ttnn.int32)
    core = ttnn.CoreCoord(0, 0)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    tile_bytes = 4096
    # (one CB size for every row: the program's cache key takes the CBs, not the runtime arguments)
    cb_bytes = ((b + 31) // 32 + 1) * tile_bytes + 7 * 64 + io.ids.buffer_page_size() + base1.buffer_page_size()
    cb_bytes += 16 * u + ((u + 31) // 32) * tile_bytes + 64  # (forced: n's tiles)
    mesh_program = ttnn.MeshProgramDescriptor()
    for r in range(ROWS):
        t0, t1 = (r * b) // 32, (r * b + b - 1) // 32
        tiles = t1 - t0 + 1
        args = [ids.buffer_address(), t0, tiles, io.ids.buffer_address(), io.ids.buffer_page_size()]
        args += [base1.buffer_address(), base1.buffer_page_size(), r]
        for t in (both, bonus, starts):
            args += [t.buffer_address(), t.buffer_page_size()]
        if n_forced is not None:
            args += [n_forced.buffer_address()]
        rt = ttnn.RuntimeArgs()
        rt[0][0] = args
        kernel = ttnn.KernelDescriptor(
            kernel_source=KDIR + "accept_inputs.cpp",
            core_ranges=cores,
            compile_time_args=[u, k, 0, int(n_forced is not None)],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        cb = ttnn.CBDescriptor(
            total_size=cb_bytes,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=cb_bytes)],
        )
        span = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, 0), ttnn.MeshCoordinate(r, COLS - 1))
        mesh_program[span] = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])
    ins = [ids, io.ids, base1] + ([n_forced] if n_forced is not None else [])
    ttnn.generic_op(ins + [both, bonus, starts], mesh_program)
    if n_forced is not None:
        ttnn.deallocate(n_forced)
    if not gather:
        return both, bonus, starts
    out = ttnn.all_gather(both, dim=2, cluster_axis=ROW_AXIS)  # [1, 1, 8, 2 U]
    ttnn.deallocate(both)
    return out, bonus, starts


def results_row(mesh, both, drafts, users: int):
    """`accept_inputs(.., gather=False)`'s own row [1, 1, 1, 2 U] fp32 and the drafter's block [U, 8] uint32 row-major
    -> `dflash2_step.gathered_results`'s [1, 1, 8, 10 U] fp32 row-major on every chip, byte for byte: the row's n,
    bonus and 8 draft words a user as fp32 by one single-core op (`kernels/dflash2/results_row.cpp`), then one
    all-gather over the rows (in place of two all-gathers, the drafts' four layout / type ops and the concat)."""
    u = users
    assert tuple(drafts.shape) == (u, 8) and drafts.layout == ttnn.ROW_MAJOR_LAYOUT, drafts
    row = _rm(mesh, [1, 1, 1, 10 * u], ttnn.float32)
    core = ttnn.CoreCoord(0, 0)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    args = [both.buffer_address(), both.buffer_page_size(), drafts.buffer_address(), drafts.buffer_page_size()]
    args += [row.buffer_address(), row.buffer_page_size()]
    rt = ttnn.RuntimeArgs()
    rt[0][0] = args
    cb_bytes = both.buffer_page_size() + 64 * u + row.buffer_page_size() + 3 * 64
    kernel = ttnn.KernelDescriptor(
        kernel_source=KDIR + "results_row.cpp",
        core_ranges=cores,
        compile_time_args=[u, 0],
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    cb = ttnn.CBDescriptor(
        total_size=cb_bytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=cb_bytes)],
    )
    mesh_program = ttnn.MeshProgramDescriptor()
    whole = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(ROWS - 1, COLS - 1))
    mesh_program[whole] = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])
    ttnn.generic_op([both, drafts, row], mesh_program)
    if os.environ.get("GLM53_DFLASH2_SMALL_GATHER", "0") == "1":  # (as gathered_results's switch)
        from models.demos.blackhole.glm53_flash.ttnn.small_gather import small_gather

        out = small_gather(mesh, row)
    else:
        out = ttnn.all_gather(row, dim=2, cluster_axis=ROW_AXIS)  # [1, 1, 8, 10 U]
    ttnn.deallocate(row)
    return out
