# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The prefill MoE's way back as one op: `ttnn.experimental.col_row_reduce_scatter`
runs the column's ring reduce-scatter of the experts' partial sums one traversal a batch (the partials batch-major:
`moe_prefill.combine(batches=B)`), adds the shared expert's partial to each batch as it is written (ELWADD, ttnn.add's
bits) and runs the row all-reduce's ring reduce-scatter on it, batch b's as soon as the column's batch b is final; then
the row's all-gather in the op too (AG_CORE: link 1, the row's reduce-scatter on link 0).  The same bits as `TtMoe._finish`'s
reduce_scatter -> ttnn.add -> tp_all_reduce (the ring's chunk parity hangs on the tile column only, so the batches do
not change any element's adds); the row's reduce-scatter and the add behind the column's.

    run(mesh, partial, part, t) -> [1, 1, t, D]: partial [1, 1, 8 t, D] batch-major, part [1, 1, t, D] (or None: no add)

The buffers (the intermediates, outputs, the add's total) and the global semaphores are persistent per size (t tokens
a row: 11.25 t D bf16 values a chip, 90 MiB at 1,024, 45 MiB at 512), taken when the generator is made for the sizes
its chunks give (`ts_for`, `prealloc`) so that no allocation happens inside a trace; a size without them takes the
usual path (`batches_for` 1).  Cores: the column's
reduce-scatter from (0, 0), the row's from ROW_OFF, the add stage's ADD_CORES cores in a row from ADD_OFF, the relay
core RELAY, the row's all-gather AG_CORE (logical).  (The row's all-gather by ttnn after the op, GLM53_CRR_AG=0, and the
GLM53_CRR_* knobs were removed.)
"""

from __future__ import annotations

import os

import ttnn

from . import moe_prefill
from .common import COLS, ROWS, TP_AXIS

D = 4096
# the chunk sizes a mesh row it takes, and its batches; 128 too by default (the 1 x 1K and 8 x 128 waves, whose way
# back was the ttnn RS -> add -> RS + AG: the same values; op alone 192.7 -> 174.5 us; the 1 x 1K wave's MoE way back
# 15.80 -> 14.94 ms); GLM53_CRR_128=0: 512 and 1,024 only, as before
TS = (512, 1024) if os.environ.get("GLM53_CRR_128", "1") == "0" else (128, 512, 1024)
BATCHES = 4
# a size's batches other than BATCHES: 128 tokens a row in 2 (in 4, one tile row a batch, the op's outputs come out
# wrong; 2 and 1 bit for bit with the watcher; 2 the faster)
BATCHES_AT = {128: 2}
# GLM53_CRR_COL_WORKERS "t:w,t:w" (read at each call): the column's reduce-scatter on w workers a direction at t tokens
# a row, not the op's choice (8 at 512 and 1,024): op alone at 512 517.6 -> 473.9 us with 4 (the separate ttnn ops
# 473.3), 1,024 best at 8 (821.0, 4: 912.8); the same values.  Unset: the op's choice
COL_WORKERS = "GLM53_CRR_COL_WORKERS"
# GLM53_CRR_CARRY (read at each call): 1, the default, the MoE's combine
# carried in the op's program on the cores it leaves (`moe_prefill.combine(carry=)`), each combine core telling
# CARRY_RELAY once a batch of its tiles is written and the column's reduce-scatter starting each batch as soon as all
# have (a fused-op receiver, as the row's), so that combine's later batches run beside the column's first ones; the
# same values (the same items, sums, tiles and reduce-scatters).  The op's workers then explicit (`_workers`: the op's
# own choice unless COL_WORKERS), so that the cores it takes are known here (`busy`; the op checks that none of the
# carried program's is one of them).  The 4 x 1K wave 192.92 -> 186.39 ms, 8 x 2,048 0.5763 -> 0.5594 s (synthetic
# weights).  0: the combine, then the op, as before
CARRY = "GLM53_CRR_CARRY"
CARRY_RELAY = ttnn.CoreCoord(9, 9)
CARRY_SEM = 0  # (the relay core's semaphore ids CARRY_SEM + b, one a batch: on that core only)
LINKS = 2  # (the column's and row's links: ttnn's get_num_links on this mesh)
ROW_OFF = ttnn.CoreCoord(0, 4)
ADD_OFF = ttnn.CoreCoord(0, 8)
ADD_CORES = 8
RELAY = ttnn.CoreCoord(11, 9)
AG_CORE = ttnn.CoreCoord(10, 9)  # (the row's all-gather in the op: the row's link 1, its reduce-scatter link 0)


class _State:
    def __init__(self, mesh, t: int, col8: bool = False):
        b = BATCHES_AT.get(t, BATCHES)
        n = ROWS * t
        self.t, self.b = t, b

        def alloc(shape, dtype=ttnn.bfloat16):
            spec = ttnn.TensorSpec(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM)
            return ttnn.allocate_tensor_on_device(spec, mesh)

        # (col8, moe_prefill.CRR_BFP8: the column's ring in BFP8; its intermediate and output BFP8)
        self.col_interm = alloc([b, 1, n // b, D], ttnn.bfloat8_b if col8 else ttnn.bfloat16)
        self.col_out = alloc([b, 1, t // b, D], ttnn.bfloat8_b if col8 else ttnn.bfloat16)
        self.row_interm = alloc([b, 1, t // b, D])
        self.row_out = alloc([b, 1, t // b, D // COLS])
        self.total = alloc([b, 1, t // b, D])
        grid = mesh.compute_with_storage_grid_size()
        full = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        self.col_sems = [ttnn.create_global_semaphore(mesh, full, 0) for _ in range(3)]
        self.row_sems = [ttnn.create_global_semaphore(mesh, full, 0) for _ in range(3)]
        self.ag_counter = ttnn.create_global_semaphore(mesh, full, 0)
        self.out_spec = ttnn.TensorSpec(
            ttnn.Shape([1, 1, t, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM
        )


_STATE: dict = {}


def _key(mesh, t: int, col8: bool | None = None) -> tuple:
    """The buffers' key: the BFP8 ring's (moe_prefill.CRR_BFP8, read at each call) apart from the bf16 ones."""
    col8 = moe_prefill.CRR_BFP8 if col8 is None else col8
    return (id(mesh), t, "bfp8") if col8 else (id(mesh), t)


def _state(mesh, t: int, col8: bool | None = None) -> _State:
    col8 = moe_prefill.CRR_BFP8 if col8 is None else col8
    key = _key(mesh, t, col8)
    if key not in _STATE:
        _STATE[key] = _State(mesh, t, col8)
    return _STATE[key]


def _traced(mesh) -> bool:
    """Whether a captured trace holds the trace region now (a replay would write where it allocated)."""
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
    return int(view.total_bytes_allocated_per_bank) > 0


def prealloc(mesh, ts=None) -> None:
    """The persistent buffers and semaphores for the chunk sizes `ts` (TS by default), before any trace: a size new
    after a capture (a second generator in the process asking for one the first did not take) stops here, as such a
    buffer can sit where a captured trace keeps its scratch and a replay writes over it."""
    new = [t for t in (TS if ts is None else ts) if _key(mesh, t) not in _STATE]
    assert not (new and _traced(mesh)), (
        f"col_row_rs: buffers for {new} tokens a row asked for after a trace was captured (a replay could write over "
        "them): add those chunk sizes to the first generator's chunk set (Glm53Generator(chunk=...), or call "
        "col_row_rs.prealloc for them before any capture)"
    )
    for t in new:
        _state(mesh, t)


def ts_for(chunks, sp_max=None) -> list:
    """The sizes of TS a prefill with these chunk lengths takes: each chunk's (a wave of it a row), and 2 sp_max when
    sp_max is one of them (`Glm53Generator._sp_pair`: two sequence-parallel waves of sp_max as one; None: no pairs)."""
    rows = set(chunks) | ({2 * sp_max} if sp_max is not None and sp_max in chunks else set())
    return sorted(t for t in rows if t in TS)


def _lookup(mesh, t: int):
    """The buffers for t tokens a row at the current setting (moe_prefill.CRR_BFP8), or None when t is not taken; the
    BFP8 ring's made here when the bf16 ones are (an in-process switch of a quality test, eager: not after a capture,
    `prealloc`)."""
    st = _STATE.get(_key(mesh, t))
    if st is None and moe_prefill.CRR_BFP8 and (id(mesh), t) in _STATE:
        prealloc(mesh, [t])
        st = _STATE.get(_key(mesh, t))
    return st


def batches_for(mesh, t: int) -> int:
    """The batches `run` takes t tokens a row in (1: it does not take them: the usual path)."""
    st = _lookup(mesh, t)
    return st.b if st is not None else 1


def carrying() -> bool:
    return os.environ.get(CARRY, "1") in ("1", "2")


def _ring_workers(nbytes: int, ring: int, links: int) -> int:
    """ttnn's reduce_scatter_default_workers for a ring on Blackhole (the cores fitting)."""
    per_link = nbytes * (ring - 1) / ring / links / 2
    return 8 if per_link > 1024 * 1024 else 1 if per_link <= 4096 else 2


def _workers(t: int, ag: bool) -> tuple:
    """The column's and row's reduce-scatter workers a direction at t tokens a row: COL_WORKERS's or the op's choice."""
    col = {int(a): int(w) for a, w in (kv.split(":") for kv in os.environ.get(COL_WORKERS, "").split(",") if kv)}
    col_w = col.get(t) or _ring_workers(ROWS * t * D * 2, ROWS, LINKS)
    row_w = _ring_workers(t * D * 2, COLS, 1 if ag else LINKS)
    return col_w, row_w


def busy(mesh, t: int, ag: bool, add: bool = True) -> set:
    """The (x, y) cores the op takes at t tokens a row with `_workers` (as ttnn's choose_worker_cores: links x 2
    directions x (workers + a mux) cores row-major from the offset), the add stage's, the relay and all-gather cores."""
    gx = mesh.compute_with_storage_grid_size().x
    col_w, row_w = _workers(t, ag)

    def rows_from(off, n):
        return {(off.x + i % gx, off.y + i // gx) for i in range(n)}

    taken = rows_from(ttnn.CoreCoord(0, 0), LINKS * 2 * (col_w + (col_w > 1)))
    taken |= rows_from(ROW_OFF, (1 if ag else LINKS) * 2 * (row_w + (row_w > 1)))
    if add:
        taken |= {(ADD_OFF.x + k, ADD_OFF.y) for k in range(ADD_CORES)}
    taken |= {(RELAY.x, RELAY.y)} | ({(AG_CORE.x, AG_CORE.y)} if ag else set())
    return taken


def carry_for(mesh, t: int, ag: bool, add: bool = True):
    """`moe_prefill.combine`'s `carry` for the op at t tokens a row (or None: CARRY off or no buffers for t)."""
    if not carrying() or batches_for(mesh, t) == 1:
        return None
    # (CARRY 2, for measurements: the combine on the carried layout, items batch by batch on the cores the op leaves,
    # but as its own program before the op: `moe_prefill.combine` then runs it and gives its out)
    relay = CARRY_RELAY if os.environ.get(CARRY, "1") == "1" else None
    return busy(mesh, t, ag, add), relay, CARRY_SEM


def run(mesh, partial, part, t: int, ag: bool = True):
    """partial [1, 1, 8 t, D] bf16 tiled, batch-major (`moe_prefill.combine(batches=batches_for(mesh, t))`; or its
    `Carried` form, `carry_for`: the combine then runs in the op's program, its tensors the caller's to free), part [1, 1,
    t, D] (the shared expert's partial) or None -> the row's sum [1, 1, t, D], as reduce_scatter -> ttnn.add ->
    tp_all_reduce.  `ag` False (mHC over the row by tokens, common.ROW_SPLIT): the op ends with the row's
    reduce-scatter (no all-gather), and this chip's quarter of the tokens [1, 1, t / 4, D] comes by an all-to-all over
    the row from the reduce-scatter's output (its batches are the row's token quarters when there are 4)."""
    st = _lookup(mesh, t)
    b = st.b
    carried = partial if isinstance(partial, moe_prefill.Carried) else None
    if carried is not None:  # (the combine in the op's program: its out is the op's input)
        col_w, row_w = _workers(t, ag)
        extra = dict(
            col_workers_per_direction=col_w,
            row_workers_per_direction=row_w,
            extra_program=carried.program,
            extra_inputs=list(carried.tensors),
            extra_slots=carried.slots,
            extra_relay_core=CARRY_RELAY,
            extra_relay_sem=CARRY_SEM,
            extra_producers=carried.producers,
        )
        partial = carried.out
    else:
        workers = {
            int(a): int(w) for a, w in (kv.split(":") for kv in os.environ.get(COL_WORKERS, "").split(",") if kv)
        }.get(t)
        extra = {"col_workers_per_direction": workers} if workers else {}
    x = ttnn.reshape(partial, [b, 1, ROWS * t // b, D])
    shared = ttnn.reshape(part, [b, 1, t // b, D]) if part is not None else None
    if not ag:
        ttnn.experimental.col_row_reduce_scatter(
            x,
            st.col_interm,
            st.col_out,
            st.row_interm,
            st.row_out,
            st.col_sems,
            st.row_sems,
            ttnn.CoreCoord(0, 0),
            ROW_OFF,
            RELAY,
            shared=shared,
            total=st.total if part is not None else None,
            add_core_grid_offset=ADD_OFF,
            add_cores=ADD_CORES,
            **extra,
        )
        rows = ttnn.reshape(st.row_out, [1, 1, t, D // COLS])  # (batch-major: the row's tokens in order)
        return ttnn.experimental.all_to_all_async_generic(rows, in_dim=3, out_dim=2, cluster_axis=TP_AXIS)
    out = ttnn.allocate_tensor_on_device(st.out_spec, mesh)  # (an ordinary output: the caller's)
    ttnn.experimental.col_row_reduce_scatter(
        x,
        st.col_interm,
        st.col_out,
        st.row_interm,
        st.row_out,
        st.col_sems,
        st.row_sems,
        ttnn.CoreCoord(0, 0),
        ROW_OFF,
        RELAY,
        shared=shared,
        total=st.total if part is not None else None,
        add_core_grid_offset=ADD_OFF,
        add_cores=ADD_CORES,
        ag_output=out,
        ag_counter=st.ag_counter,
        ag_core=AG_CORE,
        **extra,
    )
    return out


__all__ = ["batches_for", "busy", "carry_for", "carrying", "prealloc", "run", "ts_for"]
