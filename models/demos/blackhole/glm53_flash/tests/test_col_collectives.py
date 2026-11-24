"""The MoE's column collectives for a few tokens a row (`ttnn/col_collectives.py`: one-shot over the fabric,
generic_op) against torch and against the ttnn forms the model used before them (kept here: the gather as untilize,
row-major all_gather, tilize; the scatter as all_reduce over the column and a 0/1 pick matmul; the padded scatter as
ttnn.reduce_scatter).

One pytest node per case (a hang costs one), GLM53_CA_CASES = comma list of
kind:op:t[:workers:links][:today][:nodirect][:s8] (op: gather, scatter, or pscatter: the padded scatter of the experts
op's `row_tokens` y [8 x 32, D], against ttnn.reduce_scatter as `TtMoe._scatter_rows` takes it today) (nodirect: the gather's t = 1 row untilized, not read by 64-byte
pieces; s8: the scatter's rows all staged by the reader; the gather above 4 tokens with 32 workers, e.g.
values:gather:16:32)
(default values:gather:1,values:gather:4,values:scatter:1,values:scatter:4):
    values  eagerly, four calls (both parities twice): every chip against torch and today's; the gather's zero rows
            zero
    time    trace-timed (REPS calls a trace, GLM53_CA_REPS, 100); with `today`, today's beside it
    mixed   the gather: one op called at t = 4 twice, then at t four times (its zero rows after the 32-row calls)
Run (a device test): GLM53_DEVICE_TEST=1 pytest ... -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.tests._trace import traced as trace_capture  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.col_collectives import (  # noqa: E402
    GATHER,
    SCATTER,
    SCATTER_PADDED,
    SCATTER_WIDE,
    ColCollective,
)
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    COLS,
    DEVICE_PARAMS_TRACE,
    HIFI4_FP32,
    MESH_SHAPE,
    ROW_AXIS,
    ROWS,
    put,
)

REPS = int(os.environ.get("GLM53_CA_REPS", "100"))
HIDDEN = 4096
CASES = os.environ.get("GLM53_CA_CASES", "values:gather:1,values:gather:4,values:scatter:1,values:scatter:4").split(",")


def chips(t) -> list:
    return [ttnn.to_torch(d).float().reshape(-1, HIDDEN) for d in ttnn.get_device_tensors(t)]


def padded(t) -> list:
    return [d.cpu().to_torch_with_padded_shape().float().reshape(-1, HIDDEN) for d in ttnn.get_device_tensors(t)]


def timed(mesh, fn) -> float:
    fn()
    ttnn.synchronize_device(mesh)
    with trace_capture(mesh) as trace:
        outs = [fn() for _ in range(REPS)]
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    seconds = (time.perf_counter() - t0) / REPS
    ttnn.release_trace(mesh, trace)
    for o in outs:
        ttnn.deallocate(o)
    return seconds


def today_gather(mesh, x):
    """The former `TtMoe._gather_rows` for t <= 4 (CCL_RM_GATHER_MAX): untilize, row-major all_gather, tilize."""
    t, d = x.shape[-2], x.shape[-1]
    flat = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    gathered = ttnn.all_gather(flat, dim=2, cluster_axis=ROW_AXIS)
    ttnn.deallocate(flat)
    x_all = ttnn.tilize_with_zero_padding(gathered)
    ttnn.deallocate(gathered)
    n = -(-ROWS * t // 32) * 32
    if ROWS * t != n:
        x_all = ttnn.reshape(x_all, ttnn.Shape([1, 1, n, d]), ttnn.Shape([1, 1, n, d]))
    return x_all


def today_pscatter(mesh, partial, t):
    """`TtMoe._scatter_rows` for the experts op's padded y (`row_tokens`): ttnn.reduce_scatter, the first t rows."""
    rows = ttnn.reduce_scatter(partial, dim=2, cluster_axis=ROW_AXIS)
    return ttnn.slice(rows, [0, 0, 0, 0], [1, 1, t, partial.shape[-1]])


_COL_PICKS: dict = {}


def _col_picks(mesh, t: int):
    """Chip (r, c)'s [1, 1, t, 32] 0/1 matrix that takes row r's t tokens (r t + i) from the column's 32 summed rows."""
    if t not in _COL_PICKS:
        picks = torch.zeros(ROWS, 4, t, 32)
        for r in range(ROWS):
            for i in range(t):
                picks[r, :, i, r * t + i] = 1.0
        _COL_PICKS[t] = put(mesh, picks, dims=(0, 1))
    return _COL_PICKS[t]


def today_scatter(mesh, partial, t):
    """The former `TtMoe._scatter_rows` with CCL_COL_AR (ttnn.all_reduce over the column, the 0/1 pick)."""
    summed = ttnn.all_reduce(partial, cluster_axis=ROW_AXIS)
    rows = ttnn.matmul(_col_picks(mesh, t), summed, compute_kernel_config=HIFI4_FP32)
    ttnn.deallocate(summed)
    return rows


@pytest.mark.timeout(300)
@pytest.mark.parametrize("case", CASES, ids=[c.replace(":", "-") for c in CASES])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_col_collectives(mesh_device, case):
    f = case.split(":")
    kind, which, t = f[0], f[1], int(f[2])
    cfg = dict(zip(("workers", "links"), [int(v) for v in f[3:] if v.isdigit()]))
    if "nodirect" in f:
        cfg["direct"] = False
    if "s8" in f:
        cfg["split_stage"] = 8
    mode = {"gather": GATHER, "scatter": SCATTER, "pscatter": SCATTER_PADDED}[which]
    if mode == SCATTER_PADDED:
        cfg.setdefault("workers", SCATTER_WIDE)
    op = ColCollective(mesh_device, HIDDEN, mode, **cfg)
    name = f"{which} t {t} (workers {op.workers}, links {op.links})"
    print(f"\n{case}: {name}; fabric max payload {op.max_payload} B", flush=True)
    g = torch.Generator().manual_seed(t + 10 * mode)
    if mode == GATHER:
        host = (torch.randn(ROWS, COLS, t, HIDDEN, generator=g) * 0.1).to(torch.bfloat16)
        x = put(mesh_device, host, dims=(0, 1))  # chip (r, c): host[r, c]
        run = lambda: op(x, t)  # noqa: E731
        today = lambda: today_gather(mesh_device, x)  # noqa: E731
    elif mode == SCATTER:
        host = (torch.randn(ROWS, COLS, 32, HIDDEN, generator=g) * 0.1).to(torch.bfloat16)
        x = put(mesh_device, host, dims=(0, 1))
        run = lambda: op(x, t)  # noqa: E731
        today = lambda: today_scatter(mesh_device, x, t)  # noqa: E731
    else:  # every chip's [8 x 32, D]: mesh row r's tokens in rows 32 r .. 32 r + t - 1 (the rest zero, as the op's y)
        host = torch.zeros(ROWS, COLS, ROWS * 32, HIDDEN)
        for r in range(ROWS):
            host[:, :, 32 * r : 32 * r + t] = torch.randn(ROWS, COLS, t, HIDDEN, generator=g) * 0.1
        host = host.to(torch.bfloat16)
        x = put(mesh_device, host, dims=(0, 1))
        run = lambda: op(x, t)  # noqa: E731
        today = lambda: today_pscatter(mesh_device, x, t)  # noqa: E731
    if kind == "mixed":  # one op, t = 4 then t (the gather's shared block: its zero rows after the 32-row calls)
        assert mode == GATHER
        ok = True
        for tt in (4, 4, t, t, t, t):
            hh = (torch.randn(ROWS, COLS, tt, HIDDEN, generator=g) * 0.1).to(torch.bfloat16)
            xx = put(mesh_device, hh, dims=(0, 1))
            out = op(xx, tt)
            got, pad = chips(out), padded(out)
            n_diff, pad_max = 0, 0.0
            for dev, gg in enumerate(got):
                want = torch.cat([hh[r, dev % COLS].float() for r in range(ROWS)], 0)
                n_diff += int((gg[: ROWS * tt] != want).sum().item())
                pad_max = max(pad_max, pad[dev][ROWS * tt :].abs().max().item() if ROWS * tt < 32 else 0.0)
            print(f"{name} mixed, t {tt}: != torch {n_diff}; zero rows max |x| {pad_max:.3g}", flush=True)
            ok &= n_diff == 0 and pad_max == 0.0
            ttnn.deallocate(out)
            ttnn.deallocate(xx)
        assert ok
    elif kind == "values":
        ref_today = chips(today())
        ok = True
        for call in range(4):
            out = run()
            got = chips(out)
            n_diff = n_today = 0
            worst = worst_today_vs_exact = worst_new_vs_exact = 0.0
            pad_max = 0.0
            if mode == GATHER:
                pad = padded(out)
                for dev, gg in enumerate(got):
                    c = dev % COLS
                    want = torch.cat([host[r, c].float() for r in range(ROWS)], 0)  # [8 t, D]
                    n_diff += int((gg[: ROWS * t] != want).sum().item())
                    n_today += int((gg != ref_today[dev]).sum().item())
                    pad_max = max(
                        pad_max, pad[dev][ROWS * t :].abs().max().item() if ROWS * t < pad[dev].shape[0] else 0.0
                    )
                print(
                    f"{name} call {call}: != torch {n_diff}, != today's {n_today}; zero rows max |x| {pad_max:.3g}",
                    flush=True,
                )
                ok &= n_diff == 0 and pad_max == 0.0
            else:
                h = host.float()
                pad = padded(out)
                errs = {"op": [], "today": []}
                ulps, mags = [], []
                for dev, gg in enumerate(got):
                    r, c = dev // COLS, dev % COLS
                    first = 32 * r if mode == SCATTER_PADDED else r * t
                    part = h[:, c, first : first + t]  # [8, t, D]
                    exact = part.sum(0, dtype=torch.float64).float()
                    pairs = ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7]))
                    gg = gg[:t]
                    n_diff += int((gg != pairs.to(torch.bfloat16).float()).sum().item())
                    n_today += int((gg != ref_today[dev][:t]).sum().item())
                    scale = exact.abs().max().item()
                    worst = max(worst, (gg - pairs).abs().max().item() / scale)
                    worst_new_vs_exact = max(worst_new_vs_exact, (gg - exact).abs().max().item())
                    worst_today_vs_exact = max(worst_today_vs_exact, (ref_today[dev][:t] - exact).abs().max().item())
                    pad_max = max(pad_max, pad[dev][t:32].abs().max().item())
                    # against the float64 sum: signed errors (a bias would show in their mean) and in bf16 ulps of the
                    # exact value (correctly rounded: at most half an ulp)
                    exact64 = part.double().sum(0)
                    errs["op"].append((gg.double() - exact64).flatten())
                    errs["today"].append((ref_today[dev][:t].double() - exact64).flatten())
                    ulps.append(torch.exp2(torch.floor(torch.log2(exact64.abs().clamp_min(1e-30)))) / 128)
                    mags.append(exact64.abs().flatten())
                ulp, mag = torch.cat(ulps).flatten(), torch.cat(mags)
                big = mag >= 1e-3 * mag.max()  # (the ulps of the values near 0 are tiny: left out of the ulp figures)
                for key in ("op", "today"):
                    e = torch.cat(errs[key])
                    print(
                        f"{name} call {call} {key} vs float64: mean {e.mean().item():+.3e}, mean |d| "
                        f"{e.abs().mean().item():.3e}, within half an ulp "
                        f"{(e.abs() <= ulp / 2).double().mean() * 100:.2f} %; of |x| >= 1e-3 max: max "
                        f"{(e.abs() / ulp)[big].max().item():.2f} ulp, mean signed "
                        f"{(e / ulp)[big].mean().item():+.4f} ulp",
                        flush=True,
                    )
                print(f"{name} call {call}: padding rows t .. 31 max |x| {pad_max:.3g}", flush=True)
                ok &= pad_max == 0.0
                n = ROWS * COLS * t * HIDDEN
                print(
                    f"{name} call {call}: != torch (fp32, the op's order, bf16) {n_diff} of {n}; != today's {n_today};"
                    f" max |d| vs the exact sum: the op {worst_new_vs_exact:.3g}, today's {worst_today_vs_exact:.3g}",
                    flush=True,
                )
                ok &= worst < 1e-2
            ttnn.deallocate(out)
        assert ok
    else:
        sec = timed(mesh_device, run)
        print(f"{name}: one-shot {sec * 1e6:7.1f} us", flush=True)
        if "today" in f:
            sec_today = timed(mesh_device, today)
            print(f"{name}: today's {sec_today * 1e6:7.1f} us", flush=True)


AR_CASES = os.environ.get("GLM53_CA_AR_CASES", "1,2,4,8").split(",")


@pytest.mark.timeout(300)
@pytest.mark.parametrize("t", [int(v) for v in AR_CASES], ids=[f"t{v}" for v in AR_CASES])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_col_scatter_ar(mesh_device, t):
    """col_scatter_ar (the column's scatter and the row's all-reduce in one op: t <= 4 from [1, 1, 32, D], t = 8 from
    the padded [1, 1, 8 x 32, D]) against col_scatter(_padded) and then row_all_reduce, bit for bit on every chip, four
    calls (both parities twice); against the float64 sum; at t = 8 also against all_reduce_async (the default
    there).  Then the times (trace): the one op, the two, and (t = 8) the scatter with all_reduce_async."""
    from models.demos.blackhole.glm53_flash.ttnn.common import TP_AXIS, all_reduce_async
    from models.demos.blackhole.glm53_flash.ttnn.row_all_reduce import row_all_reduce

    padded_mode = t > 4
    mode = SCATTER_PADDED if padded_mode else SCATTER
    workers = SCATTER_WIDE if padded_mode else None
    fused = ColCollective(mesh_device, HIDDEN, mode, workers, ar=True)
    plain = ColCollective(mesh_device, HIDDEN, mode, workers)
    g = torch.Generator().manual_seed(100 + t)
    if padded_mode:
        host = torch.zeros(ROWS, COLS, ROWS * 32, HIDDEN)
        for r in range(ROWS):
            host[:, :, 32 * r : 32 * r + t] = torch.randn(ROWS, COLS, t, HIDDEN, generator=g) * 0.1
    else:
        host = torch.randn(ROWS, COLS, 32, HIDDEN, generator=g) * 0.1
    host = host.to(torch.bfloat16)
    x = put(mesh_device, host, dims=(0, 1))
    print(f"\nscatter_ar t {t}: {'padded, ' if padded_mode else ''}workers {fused.workers}", flush=True)

    def two():
        y = plain(x, t)
        out = row_all_reduce(mesh_device, y)
        ttnn.deallocate(y)
        return out

    def two_async():
        y = plain(x, t)
        out = all_reduce_async(mesh_device, y, TP_AXIS)
        ttnn.deallocate(y)
        return out

    h = host.double()
    ok = True
    for call in range(4):
        a, b = fused(x, t), two()
        ga, gb = chips(a), chips(b)
        n_diff, worst = 0, 0.0
        for dev in range(ROWS * COLS):
            r = dev // COLS
            first = 32 * r if padded_mode else r * t
            exact = h[:, :, first : first + t].sum((0, 1))  # [t, D] over all 32 chips
            n_diff += int((ga[dev][:t] != gb[dev][:t]).sum().item())
            worst = max(worst, (ga[dev][:t].double() - exact).abs().max().item())
        print(f"call {call}: != the two ops {n_diff} of {ROWS * COLS * t * HIDDEN}; max |d| vs float64 {worst:.3g}")
        ok &= n_diff == 0
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    if padded_mode:
        a, b = fused(x, t), two_async()
        ga, gb = chips(a), chips(b)
        n_diff, wa, wb = 0, 0.0, 0.0
        for dev in range(ROWS * COLS):
            r = dev // COLS
            exact = h[:, :, 32 * r : 32 * r + t].sum((0, 1))
            n_diff += int((ga[dev][:t] != gb[dev][:t]).sum().item())
            wa = max(wa, (ga[dev][:t].double() - exact).abs().mean().item())
            wb = max(wb, (gb[dev][:t].double() - exact).abs().mean().item())
        print(f"vs the scatter and all_reduce_async: != {n_diff}; mean |d| vs float64 one op {wa:.3g}, async {wb:.3g}")
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    # (an even number of calls of each family between the traces: the parities as the model has them)
    for name, fn in [("one op", lambda: fused(x, t)), ("two ops", two)] + (
        [("scatter + all_reduce_async", two_async)] if padded_mode else []
    ):
        sec = timed(mesh_device, fn)
        print(f"t {t}, {name}: {sec * 1e6:7.1f} us", flush=True)
    assert ok
