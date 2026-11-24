"""mHC over the row's chips by tokens (common.ROW_SPLIT, GLM53_MHC_ROW_SPLIT): its pieces alone at T tokens
a row (GLM53_MRS_T, default 1024,512), on layer 20's mHC weights and random streams, each chip its own partial sum of a
sub-layer's output (as o_proj leaves it), each mesh row its own tokens.

  values, every chip, bit for bit:
    (a) the row all-reduce of T rows under `row_quarter(T)` (its reduce-scatter, then an all-to-all over the row)
        = this chip's quarter of the tokens of `tp_all_reduce`'s output;
    (b) `row_quarter_of` (take) of that output = the same quarter;
    (c) the read of T / 4 tokens on the quarter's streams = the quarter of the read of T (post, comb, normed input);
    (d) write_read of T / 4 on the quarter = the quarter of write_read of T (new streams, post, comb, normed input);
    (e) the all-gather over the row on the tokens of (d)'s normed input = write_read of T's normed input.
  times (GLM53_MRS_TIME=1, default; trace, REPS calls a replay, GLM53_MRS_ROUNDS rounds, the median), µs a call:
    the all-reduce, its reduce-scatter + all-to-all, the take, the all-gather on the tokens, write_read at T and at
    T / 4, and a sub-layer's end as the model runs it: "now" = all-reduce + write_read(T); "split" = reduce-scatter +
    all-to-all + write_read(T / 4) + all-gather (the attention's and the dense MLP's ends); "split_take" = all-reduce +
    take + write_read(T / 4) + all-gather (the MoE's end, its one op ending with the row's all-gather).
Prints "MRS ..." lines; a study: an exception is printed (a device timeout raised), GLM53_TEST_STRICT=1 fails on a
difference.  Run (a device test):
    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_mhc_row_split.py -s
"""

from __future__ import annotations

import os
import statistics
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.config import Glm53Config  # noqa: E402
from models.demos.blackhole.glm53_flash.tests._trace import is_device_timeout  # noqa: E402
from models.demos.blackhole.glm53_flash.tests._trace import traced as trace_capture  # noqa: E402
from models.demos.blackhole.glm53_flash.tt.mhc import HyperConnectionWeights  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import common  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    COLS,
    DEVICE_PARAMS_TRACE,
    MESH_SHAPE,
    ROWS,
    put,
    row_quarter,
    row_quarter_of,
    row_tokens_gather,
    tp_all_reduce,
)
from models.demos.blackhole.glm53_flash.ttnn.head import TtRmsNorm  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.mhc import TtHyperConnection, streams_to_device  # noqa: E402

REPS = int(os.environ.get("GLM53_MRS_REPS", "10"))


def chips(mesh, t) -> torch.Tensor:
    """Every chip's [.., R, W] (mHC's post and comb are [T, ..]) -> [ROWS, COLS, R, W] on the host."""
    if len(t.shape) != 4:
        t = ttnn.reshape(t, [1, 1, t.shape[-2], t.shape[-1]])
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=MESH_SHAPE, dims=(0, 1)))


def bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    width = {2: torch.int16, 4: torch.int32}[a.element_size()]
    return torch.equal(a.contiguous().view(width), b.contiguous().view(width))


def quarter_same(full: torch.Tensor, part: torch.Tensor, rows_a_token: int = 1) -> bool:
    """part [ROWS, COLS, R / 4, W] holds, on chip (r, c), the rows c R / 4 .. of chip (r, c)'s full [.., R, W]."""
    q = full.shape[-2] // COLS
    assert part.shape[-2] == q, (full.shape, part.shape)
    return all(bits(full[:, c, c * q : (c + 1) * q], part[:, c]) for c in range(COLS))


def flat(out):
    if isinstance(out, (tuple, list)):
        return [x for o in out for x in flat(o)]
    return [out]


def timed(mesh, fn) -> float:
    """Seconds a call of fn, REPS calls captured in one trace and replayed once (the outputs freed after)."""
    for o in flat(fn()):
        ttnn.deallocate(o)
    ttnn.synchronize_device(mesh)
    with trace_capture(mesh) as trace:
        outs = [fn() for _ in range(REPS)]
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    seconds = (time.perf_counter() - t0) / REPS
    ttnn.release_trace(mesh, trace)
    for o in flat(outs):
        ttnn.deallocate(o)
    return seconds


def _study(mesh, t: int) -> bool:
    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    kw = dict(
        hc_mult=config.hc_mult,
        sinkhorn_iters=config.hc_sinkhorn_iters,
        hc_eps=config.hc_eps,
        norm_eps=config.rms_norm_eps,
    )
    attn = TtHyperConnection(mesh, HyperConnectionWeights.from_checkpoint(checkpoint, 20, "attn"), **kw)
    ffn = TtHyperConnection(mesh, HyperConnectionWeights.from_checkpoint(checkpoint, 20, "ffn"), **kw)
    p = "model.language_model.layers.20."
    norm = TtRmsNorm(mesh, checkpoint.tensor(p + "post_attention_layernorm.weight"), config.rms_norm_eps)
    in_norm = TtRmsNorm(mesh, checkpoint.tensor(p + "input_layernorm.weight"), config.rms_norm_eps)
    n, d, q = config.hc_mult, config.hidden_size, t // COLS
    g = torch.Generator().manual_seed(t)
    base = torch.randn(ROWS, t, 1, d, generator=g) * 0.02
    streams = (base + torch.randn(ROWS, t, n, d, generator=g) * 0.02).to(torch.bfloat16)
    partial = (torch.randn(ROWS, COLS, t, d, generator=g) * 0.25).to(torch.bfloat16)
    dev_streams = streams_to_device(mesh, streams)  # [1, 1, 4 T, d], the same on a row's chips
    dev_partial = put(mesh, partial, dims=(0, 1))  # chip (r, c): its own [1, 1, T, d]
    mine_streams = row_quarter_of(mesh, dev_streams)  # [1, 1, 4 T / 4, d]: chip c's tokens c T / 4 ..

    # the whole wave's form
    post_f, comb_f, x_f = attn.read(dev_streams, t, in_norm)
    y_f = tp_all_reduce(mesh, dev_partial)
    new_f, (p2_f, c2_f, h_f) = attn.write_read(y_f, dev_streams, post_f, comb_f, ffn, t, norm, prefill=True)
    # this chip's quarter
    post_q, comb_q, x_q = attn.read(mine_streams, q, in_norm)
    with row_quarter(t):
        y_q = tp_all_reduce(mesh, dev_partial)
    y_take = row_quarter_of(mesh, y_f)
    new_q, (p2_q, c2_q, h_q) = attn.write_read(y_q, mine_streams, post_q, comb_q, ffn, q, norm, prefill=True)
    h_all = row_tokens_gather(h_q)
    ttnn.synchronize_device(mesh)

    checks = {
        "(a) reduce-scatter + all-to-all = the all-reduce's quarter": quarter_same(chips(mesh, y_f), chips(mesh, y_q)),
        "(b) take = the all-reduce's quarter": quarter_same(chips(mesh, y_f), chips(mesh, y_take)),
        "(c) read post": quarter_same(chips(mesh, post_f), chips(mesh, post_q)),
        "(c) read comb": quarter_same(chips(mesh, comb_f), chips(mesh, comb_q)),
        "(c) read normed input": quarter_same(chips(mesh, x_f), chips(mesh, x_q)),
        "(d) write_read new streams": quarter_same(chips(mesh, new_f), chips(mesh, new_q)),
        "(d) write_read post": quarter_same(chips(mesh, p2_f), chips(mesh, p2_q)),
        "(d) write_read comb": quarter_same(chips(mesh, c2_f), chips(mesh, c2_q)),
        "(d) write_read normed input": quarter_same(chips(mesh, h_f), chips(mesh, h_q)),
        "(e) all-gather of the normed input": bits(chips(mesh, h_f), chips(mesh, h_all)),
    }
    for name, ok in checks.items():
        print(f"MRS T={t} {name}: {'same' if ok else 'DIFFERENT'}", flush=True)
    same = all(checks.values())
    tally = f"{sum(checks.values())} of {len(checks)}"
    print(f"MRS T={t} values: {'ALL SAME' if same else 'NOT ALL SAME'} ({tally})", flush=True)
    for x in (x_f, x_q, y_take, new_f, p2_f, c2_f, h_f, new_q, p2_q, c2_q, h_q, h_all):
        ttnn.deallocate(x)

    if os.environ.get("GLM53_MRS_TIME", "1") == "1":
        kw_ar = common._tp_ar_split(t) or {}

        def rs_a2a():
            with row_quarter(t):
                return tp_all_reduce(mesh, dev_partial)

        def rs_only():
            return ttnn.reduce_scatter(dev_partial, dim=3, cluster_axis=common.TP_AXIS, **kw_ar)

        def now():
            y = tp_all_reduce(mesh, dev_partial)
            out = attn.write_read(y, dev_streams, post_f, comb_f, ffn, t, norm, prefill=True)
            ttnn.deallocate(y)
            return out

        def split():
            with row_quarter(t):
                y = tp_all_reduce(mesh, dev_partial)
            new, (p2, c2, h) = attn.write_read(y, mine_streams, post_q, comb_q, ffn, q, norm, prefill=True)
            ttnn.deallocate(y)
            full = row_tokens_gather(h)
            ttnn.deallocate(h)
            return new, p2, c2, full

        def split_take():
            y = tp_all_reduce(mesh, dev_partial)
            mine = row_quarter_of(mesh, y)
            ttnn.deallocate(y)
            new, (p2, c2, h) = attn.write_read(mine, mine_streams, post_q, comb_q, ffn, q, norm, prefill=True)
            ttnn.deallocate(mine)
            full = row_tokens_gather(h)
            ttnn.deallocate(h)
            return new, p2, c2, full

        arms = {
            "all-reduce": lambda: tp_all_reduce(mesh, dev_partial),
            "reduce-scatter": rs_only,
            "reduce-scatter + all-to-all": rs_a2a,
            "take": lambda: row_quarter_of(mesh, y_f),
            "all-gather (tokens)": lambda: row_tokens_gather(y_q),  # (a live [T / 4, D]: x_q is freed above)
            "write_read T": lambda: attn.write_read(y_f, dev_streams, post_f, comb_f, ffn, t, norm, prefill=True),
            "write_read T/4": lambda: attn.write_read(y_q, mine_streams, post_q, comb_q, ffn, q, norm, prefill=True),
            "now": now,
            "split": split,
            "split_take": split_take,
        }
        rounds = int(os.environ.get("GLM53_MRS_ROUNDS", "3"))
        got = {name: [] for name in arms}
        for _ in range(rounds):
            for name, fn in arms.items():
                got[name].append(timed(mesh, fn) * 1e6)
        med = {name: statistics.median(v) for name, v in got.items()}
        for name, v in got.items():
            print(f"MRS TIME T={t} {name}: {med[name]:.1f} us ({' / '.join(f'{x:.1f}' for x in v)})", flush=True)
        print(
            f"MRS TIME T={t} a sub-layer's end: now {med['now']:.1f}, split {med['split']:.1f} "
            f"({med['split'] - med['now']:+.1f}), split_take {med['split_take']:.1f} "
            f"({med['split_take'] - med['now']:+.1f}) us",
            flush=True,
        )
    for x in (post_f, comb_f, y_f, post_q, comb_q, y_q, dev_streams, dev_partial, mine_streams):
        ttnn.deallocate(x)
    return same


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_mhc_row_split(mesh_device):
    strict = os.environ.get("GLM53_TEST_STRICT") == "1"
    results = []
    for t in [int(x) for x in os.environ.get("GLM53_MRS_T", "1024,512").split(",")]:
        try:
            results.append(_study(mesh_device, t))
        except Exception as e:  # noqa: BLE001  (a study: printed; a device timeout raised: the run ends non-zero)
            if is_device_timeout(e) or strict:
                raise
            import traceback

            traceback.print_exc()
            print(f"MRS T={t}: STUDY FAILED (printed, not raised): {type(e).__name__}: {e}", flush=True)
            results.append(False)
    print(f"MRS ALL {'SAME' if all(results) else 'NOT SAME'} ({sum(results)} of {len(results)} sizes)", flush=True)
    if strict:
        assert all(results)
