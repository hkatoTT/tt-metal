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

import ttnn

from .common import COLS, ROWS

D = 4096
# the chunk sizes a mesh row it takes, and its batches
TS = (512, 1024)
BATCHES = 4
ROW_OFF = ttnn.CoreCoord(0, 4)
ADD_OFF = ttnn.CoreCoord(0, 8)
ADD_CORES = 8
RELAY = ttnn.CoreCoord(11, 9)
AG_CORE = ttnn.CoreCoord(10, 9)  # (the row's all-gather in the op: the row's link 1, its reduce-scatter link 0)


class _State:
    def __init__(self, mesh, t: int):
        b = BATCHES
        n = ROWS * t
        self.t, self.b = t, b

        def alloc(shape):
            spec = ttnn.TensorSpec(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM)
            return ttnn.allocate_tensor_on_device(spec, mesh)

        self.col_interm = alloc([b, 1, n // b, D])
        self.col_out = alloc([b, 1, t // b, D])
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


def _state(mesh, t: int) -> _State:
    key = (id(mesh), t)
    if key not in _STATE:
        _STATE[key] = _State(mesh, t)
    return _STATE[key]


def _traced(mesh) -> bool:
    """Whether a captured trace holds the trace region now (a replay would write where it allocated)."""
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
    return int(view.total_bytes_allocated_per_bank) > 0


def prealloc(mesh, ts=None) -> None:
    """The persistent buffers and semaphores for the chunk sizes `ts` (TS by default), before any trace: a size new
    after a capture (a second generator in the process asking for one the first did not take) stops here, as such a
    buffer can sit where a captured trace keeps its scratch and a replay writes over it."""
    new = [t for t in (TS if ts is None else ts) if (id(mesh), t) not in _STATE]
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


def batches_for(mesh, t: int) -> int:
    """The batches `run` takes t tokens a row in (1: it does not take them: the usual path)."""
    st = _STATE.get((id(mesh), t))
    return st.b if st is not None else 1


def run(mesh, partial, part, t: int):
    """partial [1, 1, 8 t, D] bf16 tiled, batch-major (`moe_prefill.combine(batches=batches_for(mesh, t))`), part [1, 1,
    t, D] (the shared expert's partial) or None -> the row's sum [1, 1, t, D], as reduce_scatter -> ttnn.add ->
    tp_all_reduce."""
    st = _STATE[(id(mesh), t)]
    b = st.b
    x = ttnn.reshape(partial, [b, 1, ROWS * t // b, D])
    shared = ttnn.reshape(part, [b, 1, t // b, D]) if part is not None else None
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
    )
    return out


__all__ = ["batches_for", "prealloc", "run", "ts_for"]
