"""The prefill MoE that reads each expert's weights once (`ttnn/moe_prefill.py`) on the
column-sliced layout (each chip 256 columns of each of its mesh column's 72 experts), random weights, no checkpoint:
the plan against the route op's rows read back, the result against torch (`test_moe_experts.sliced_reference`, up to
512 tokens) and against today's path (the experts op, 512 tokens a call), and the stages' times.

    GLM53_MOE_PREFILL_TOKENS (512,4096: the column's tokens)
Run (a device test):
    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_moe_prefill.py -s
"""

from __future__ import annotations

import os
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

# the weights' precision (bfp8, or bfp4: the low-precision patterns)
WDT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[os.environ.get("GLM53_TEST_WDTYPE", "bfp8")]

from models.demos.blackhole.glm53_flash.tests._trace import traced as trace_capture  # noqa: E402
from models.demos.blackhole.glm53_flash.tests.test_moe_experts import (  # noqa: E402
    LIMIT,
    D,
    Q,
    row_sharded,
    sliced_reference,
)
from models.demos.blackhole.glm53_flash.ttnn import moe_prefill  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, put  # noqa: E402

K = 8


def timed(mesh, fn, reps: int = 3):
    out = fn()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.deallocate(out)
        out = fn()
    ttnn.synchronize_device(mesh)
    return out, (time.perf_counter() - t0) / reps


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_moe_prefill(mesh_device):
    import torch

    e_all, e_col, s_col, base = 288, 72, 256, 144
    t_col = s_col // 32
    rand = lambda shape, dt: ttnn.rand(  # noqa: E731
        shape, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, low=-0.05, high=0.05
    )
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])  # noqa: E731
    # GLM53_MOE_PROG_F (moe_prefill.PROG, the progressive form): each slot a base and a correction (both random here),
    # the corrections after the bases; the prefill reads both for every expert, the reference adds them (and the
    # decode op "today's path" reads them for the experts its F marks: F = 2 for every one)
    prog = moe_prefill.PROG
    sets = 2 if prog else 1
    w1 = rand([1, 1, t_col * sets * e_col * D, 64], WDT)
    w2 = rand([1, 1, Q * sets * e_col * s_col, D // Q], WDT)
    host_w1, host_w2 = first(w1).float()[0, 0], first(w2).float()[0, 0]
    if prog:  # (the reference's weights: base + correction)
        v1 = host_w1.reshape(2 * e_col, t_col, D, 64)
        v2 = host_w2.reshape(Q, 2 * e_col, s_col, D // Q)
        host_w1 = (v1[:e_col] + v1[e_col:]).reshape(-1, 64)
        host_w2 = (v2[:, :e_col] + v2[:, e_col:]).reshape(-1, D // Q)
    nd1 = ttnn.to_memory_config(w1, row_sharded(mesh_device, 2, 16 * 32))
    nd2 = ttnn.to_memory_config(w2, row_sharded(mesh_device, D // Q // 32, sets * e_col * s_col))
    ttnn.deallocate(w1)
    ttnn.deallocate(w2)
    rep = dict(dims=(None, None), dtype=ttnn.float32)
    d_bias = put(mesh_device, torch.zeros(1, 1, 1, e_all), **rep)
    bases = torch.zeros(1, 1, 1, 8, dtype=torch.int32)
    bases[..., 0] = base
    d_base = put(mesh_device, bases, dims=(None, None), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def y8_check(n, w, gf, err_bf16, want, pcc):
        """Y in bfp8 (Y_BFP8): the result against fp32 Y; down's records against fp32 Y
        rounded on the host (each row's 16 consecutive values an exponent: round to nearest even, and toward zero),
        combine's sums against the host's of the records' values (fp32 in slot order, bf16)."""
        n8 = flag("Y_BFP8", True, lambda: moe_prefill.experts(mesh_device, d_x, w, nd1, nd2, e_col, LIMIT))
        g8 = first(n8).float()[0, 0]
        e8 = (g8 - gf).abs()
        line = (
            f"{n} tokens: Y in bfp8 against fp32: pcc {pcc(g8, gf):.7f}, max |d| {e8.max().item():.3e}, mean |d| "
            f"{e8.mean().item():.3e} (bf16 Y's {err_bf16.mean().item():.3e}), outputs differing "
            f"{(e8 > 0).float().mean().item() * 100:.2f} %"
        )
        if want is not None:
            line += f"; against torch: bfp8 Y {pcc(g8, want):.7f}, mean |d| {(g8 - want).abs().mean().item():.3e}"
        print(line, flush=True)
        ttnn.deallocate(n8)
        seg, ptok, pw, tpair = moe_prefill.plan(mesh_device, w)
        x_rm = ttnn.to_layout(d_x, ttnn.ROW_MAJOR_LAYOUT)
        hh = moe_prefill.gate_up(mesh_device, x_rm, nd1, seg, ptok, pw, LIMIT)
        y32 = flag("Y_BF16", False, lambda: moe_prefill.down(mesh_device, hh, nd2, seg, e_col))
        y8 = flag("Y_BFP8", True, lambda: moe_prefill.down(mesh_device, hh, nd2, seg, e_col))
        o8 = moe_prefill.combine(mesh_device, y8, tpair)
        hs = (first(seg).long() & 0xFFFFFFFF).reshape(-1)
        rows = torch.cat([torch.arange(int(hs[e]), int(hs[e]) + int(hs[128 + e])) for e in range(e_col)])
        v = first(y32).float()[0, 0][rows]  # [P, 4096]
        # a record a pair row's 1,024 columns: 64 exponents, then 1,024 mantissas
        rec = first(y8)[0, 0][rows].contiguous().view(torch.uint8).reshape(len(rows), 4, moe_prefill.Y8_REC)
        mant = rec[:, :, 64:].reshape(len(rows), 4096).long()
        ex = rec[:, :, :64].reshape(len(rows), 256).long()
        e_all = ex.repeat_interleave(16, dim=1)
        mag = (mant & 0x7F).double()
        dec = torch.where((mant >> 7) & 1 == 1, -mag, mag) * torch.pow(2.0, (e_all - 127 - 6).double())
        # the host's rounding: the group's exponent the largest of its values' (fp32 biased), 7 magnitude bits
        g = v.double().reshape(len(rows), 256, 16)
        _, ex_v = torch.frexp(g.abs())
        biased = torch.where(g == 0, torch.zeros_like(ex_v), ex_v - 1 + 127)
        e_h = biased.max(dim=-1).values  # [P, 256]
        scale = torch.pow(2.0, (6 - (e_h - 127)).double())[..., None]
        emu = {}
        for name, rnd in (("nearest even", torch.round), ("toward zero", torch.trunc)):
            m = rnd(g.abs() * scale).clamp(max=127)
            emu[name] = (torch.sign(g) * m / scale).reshape(len(rows), 4096)
        same_e = (ex == e_h).float().mean().item()
        line = (
            f"{n} tokens: Y8 records ({len(rows)} pair rows): exponents the same as the host's "
            f"{same_e * 100:.3f} %; values the same as the host's rounding: "
            + ", ".join(f"{k} {(dec == x).double().mean().item() * 100:.3f} %" for k, x in emu.items())
            + f"; relative error against fp32 Y {((dec - v.double()).abs().sum() / v.double().abs().sum()).item():.3e}"
        )
        print(line, flush=True)
        # combine against the host's sums of the records' values (slot order, fp32, then bf16)
        h_tp = (first(tpair).long() & 0xFFFFFFFF).reshape(n, 16)
        full = torch.zeros(int(hs[e_col]) + 32, 4096, dtype=torch.float64)
        full[rows] = dec
        want_o = torch.zeros(n, 4096, dtype=torch.float32)
        for t in range(n):
            for r in [int(r) for r in h_tp[t] if int(r) != 0xFFFFFFFF]:
                want_o[t] += full[r].float()
        o = first(o8).float()[0, 0]
        same_o = (o == want_o.to(torch.bfloat16).float()).float().mean().item()
        print(f"{n} tokens: combine of the Y8 records against the host's sums of their values: the same bits "
              f"{same_o * 100:.3f} %", flush=True)  # fmt: skip
        assert same_o > 0.99
        for tt in (seg, ptok, pw, tpair, x_rm, hh, y32, y8, o8):
            ttnn.deallocate(tt)

    def flag(name: str, value, fn):  # fn() with moe_prefill.<name> set to value
        keep = getattr(moe_prefill, name)
        setattr(moe_prefill, name, value)
        try:
            return fn()
        finally:
            setattr(moe_prefill, name, keep)

    for n in [int(v) for v in os.environ.get("GLM53_MOE_PREFILL_TOKENS", "512,4096").split(",") if v]:
        g = torch.Generator().manual_seed(n)
        logits = torch.randn(n, e_all, generator=g) * 2.0
        x = torch.randn(n, D, generator=g).to(torch.bfloat16)
        d_logits = put(mesh_device, logits[None, None], **rep)
        d_x = put(mesh_device, x[None, None], dims=(None, None), dtype=ttnn.bfloat16)
        w = ttnn.experimental.deepseek_prefill.moe_decode_route(d_logits, d_bias, d_base, n, K, e_col, 2.5)
        tile = first(w).float()[0, 0]
        ids, wts = tile[:, :16].contiguous().view(torch.int32), tile[:, 16:]

        # the plan against the rows read back
        seg, ptok, pw, tpair = moe_prefill.plan(mesh_device, w)
        u32 = lambda tt: first(tt).long() & 0xFFFFFFFF  # noqa: E731
        sent = 0xFFFFFFFF
        h_seg = u32(seg).reshape(-1)
        h_ptok = u32(ptok).reshape(-1)
        h_pw = first(pw).float().reshape(-1)
        h_tp = u32(tpair).reshape(n, 16)
        counts = torch.zeros(e_col, dtype=torch.int64)
        pairs = [[] for _ in range(e_col)]
        for t in range(n):
            for j in range(K):
                if wts[t, j] != 0:
                    e = int(ids[t, j])
                    pairs[e].append((t, wts[t, j].item()))
                    counts[e] += 1
        off, ok = 0, True
        for e in range(e_col):
            ok &= int(h_seg[e]) == off and int(h_seg[128 + e]) == int(counts[e])
            for i, (t, v) in enumerate(pairs[e]):
                ok &= int(h_ptok[off + i]) == t and h_pw[off + i].item() == v
            pad = off + len(pairs[e])
            end = off + -(-len(pairs[e]) // 32) * 32
            ok &= bool((h_ptok[pad:end] == sent).all()) and bool((h_pw[pad:end] == 0).all())
            off = end
        ok &= int(h_seg[e_col]) == off
        for t in range(n):
            rows = [int(r) for r in h_tp[t] if int(r) != sent]
            ok &= [int(h_ptok[r]) for r in rows] == [t] * len(rows) and len(rows) == int((wts[t, :K] != 0).sum())
        print(
            f"{n} tokens: plan matches the routing {ok} ({off} table rows, {int(counts.sum())} pairs, the most on an "
            f"expert {int(counts.max())})",
            flush=True,
        )
        assert ok

        # each stage against torch from the stage before it (the first 512 tokens' pairs)
        if n <= 512 and not prog:
            t_col_ = s_col // 32
            w1r = host_w1.reshape(e_col, t_col_, D, 64)
            gates = torch.cat([w1r[:, c, :, :32] for c in range(t_col_)], dim=-1)  # [E, D, S]
            ups = torch.cat([w1r[:, c, :, 32:] for c in range(t_col_)], dim=-1)
            downs = host_w2.reshape(Q, e_col, s_col, D // Q).permute(1, 2, 0, 3).reshape(e_col, s_col, D)
            x_rm = ttnn.to_layout(d_x, ttnn.ROW_MAJOR_LAYOUT)
            d_h = moe_prefill.gate_up(mesh_device, x_rm, nd1, seg, ptok, pw, LIMIT)
            hd = first(d_h).float()[0, 0]  # [M, 256]
            d_y = moe_prefill.down(mesh_device, d_h, nd2, seg, e_col)
            yd = first(d_y).float()[0, 0]  # [M, 4096]
            d_o = moe_prefill.combine(mesh_device, d_y, tpair)
            od = first(d_o).float()[0, 0]  # [N, 4096] (to_torch untilizes)
            h_err = y_err = 0.0
            bad_h = []
            for e in range(e_col):
                o0 = int(h_seg[e])
                for i, (t, v) in enumerate(pairs[e]):
                    xr = x[t].float()
                    gg = (xr @ gates[e]).to(torch.bfloat16).float()
                    uu = (xr @ ups[e]).to(torch.bfloat16).float()
                    hh = torch.nn.functional.silu(torch.clamp(gg, max=LIMIT)) * torch.clamp(uu, -LIMIT, LIMIT)
                    hh = (hh * v).to(torch.bfloat16).float()
                    d = (hd[o0 + i] - hh).abs().max().item() / max(hh.abs().max().item(), 1e-6)
                    if d > 0.02 and len(bad_h) < 6:
                        cols = torch.nonzero((hd[o0 + i] - hh).abs() > 0.02 * hh.abs().max())[:, 0].tolist()
                        bad_h.append((e, i, t, round(d, 3), cols[:8]))
                    h_err = max(h_err, d)
                    yy = hd[o0 + i] @ downs[e]
                    y_err = max(y_err, (yd[o0 + i] - yy).abs().max().item() / max(yy.abs().max().item(), 1e-6))
            want_o = torch.zeros(n, D)
            for t in range(n):
                for r in [int(r) for r in h_tp[t] if int(r) != sent]:
                    want_o[t] += yd[r]
            o_err = ((od - want_o.to(torch.bfloat16).float()).abs().max() / want_o.abs().max().clamp_min(1e-6)).item()
            print(
                f"{n} tokens, stages against torch (max |d| / max): gate_up {h_err:.3e}, down {y_err:.3e}, combine "
                f"{o_err:.3e}; bad h rows (e, i, token, err, columns) {bad_h}",
                flush=True,
            )
            for tt in (x_rm, d_h, d_y, d_o):
                ttnn.deallocate(tt)
        for tt in (seg, ptok, pw, tpair):
            ttnn.deallocate(tt)

        if os.environ.get("GLM53_MOE_PREFILL_STAGED") == "1" and not prog:  # each stage synchronized (a hang's place)
            seg, ptok, pw, tpair = moe_prefill.plan(mesh_device, w)
            x_rm = ttnn.to_layout(d_x, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.synchronize_device(mesh_device)
            print(f"{n} tokens: staged: plan done", flush=True)
            d_h = moe_prefill.gate_up(mesh_device, x_rm, nd1, seg, ptok, pw, LIMIT)
            ttnn.synchronize_device(mesh_device)
            print(f"{n} tokens: staged: gate_up done", flush=True)
            d_y = moe_prefill.down(mesh_device, d_h, nd2, seg, e_col)
            ttnn.synchronize_device(mesh_device)
            print(f"{n} tokens: staged: down done", flush=True)
            d_o = moe_prefill.combine(mesh_device, d_y, tpair)
            ttnn.synchronize_device(mesh_device)
            print(f"{n} tokens: staged: combine done", flush=True)
            for tt in (seg, ptok, pw, tpair, x_rm, d_h, d_y, d_o):
                ttnn.deallocate(tt)

        # the result: the new path, today's path, torch (up to 512 tokens)
        run_new = lambda: moe_prefill.experts(mesh_device, d_x, w, nd1, nd2, e_col, LIMIT)  # noqa: E731

        def run_old():
            parts = []
            for r0 in range(0, n, 512):
                xs = ttnn.slice(d_x, [0, 0, r0, 0], [1, 1, min(n, r0 + 512), D])
                ws = ttnn.slice(w, [0, 0, r0, 0], [1, 1, min(n, r0 + 512), 32])
                parts.append(
                    ttnn.experimental.deepseek_prefill.moe_decode_experts(xs, ws, nd1, nd2, e_col, s_col, LIMIT)
                )
            out = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)
            for p in parts if len(parts) > 1 else ():
                ttnn.deallocate(p)
            return out

        stress = int(os.environ.get("GLM53_MOE_PREFILL_STRESS", "0"))
        if stress:  # the new path over and over (it once hung, rarely), synchronized every 10
            for i in range(stress):
                ttnn.deallocate(run_new())
                if i % 10 == 9:
                    ttnn.synchronize_device(mesh_device)
            ttnn.synchronize_device(mesh_device)
            print(f"{n} tokens: {stress} runs of the new path without a hang", flush=True)
        if os.environ.get("GLM53_MOE_PREFILL_TRACED") == "1":  # device time: both paths captured in a trace
            for name, fn in (("new", run_new), ("today's", run_old)):
                ttnn.deallocate(fn())
                ttnn.synchronize_device(mesh_device)
                with trace_capture(mesh_device) as trace:
                    outs = [fn() for _ in range(5)]
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                sec = (time.perf_counter() - t0) / 5
                ttnn.release_trace(mesh_device, trace)
                for o in outs:
                    ttnn.deallocate(o)
                print(f"{n} tokens: {name} path traced {sec * 1e3:.2f} ms", flush=True)
        new, t_new = timed(mesh_device, run_new)
        old, t_old = timed(mesh_device, run_old)
        got, prev = first(new).float()[0, 0], first(old).float()[0, 0]
        pcc = lambda a, b: torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()  # noqa: E731
        line = (
            f"{n} tokens: new against today's pcc {pcc(got, prev):.7f}, max |d| {(got - prev).abs().max().item():.3e}"
            f" (max |y| {prev.abs().max().item():.3e}); {t_new * 1e3:.2f} ms against {t_old * 1e3:.2f} ms (eager)"
        )
        if n <= 512:
            want, _ = sliced_reference(x.float(), ids, wts, host_w1, host_w2, E=e_col, S=s_col)
            line += f"; against torch: new pcc {pcc(got, want):.7f}, today's {pcc(prev, want):.7f}"
            rel = lambda a, b: ((a - b).norm() / b.norm()).item()  # noqa: E731
            line += f"; relative error against torch: new {rel(got, want):.3e}, today's {rel(prev, want):.3e}"
        print(line, flush=True)
        checks = os.environ.get("GLM53_TEST_NO_ASSERT") != "1"  # (a study of lower fidelities: print only)
        if n <= 512 and checks:
            assert pcc(got, want) > 0.9999
        # (the weights are ttnn.rand's, another draw each run: at 4,096 tokens 0.99994 / 0.99989 in two runs;
        # GLM53_MOE_PREFILL_OLD_PCC lowers the bar for a run that checks something else)
        if checks:
            assert pcc(got, prev) > float(os.environ.get("GLM53_MOE_PREFILL_OLD_PCC", "0.9999"))
        # Y in fp32 (Y_BF16 off) against the default bf16 Y (each pair's sum rounded to bf16 before combine)
        nf = flag("Y_BF16", False, run_new)
        gf = first(nf).float()[0, 0]
        err = (got - gf).abs()
        line = (
            f"{n} tokens: Y in bf16 against fp32: pcc {pcc(got, gf):.7f}, max |d| {err.max().item():.3e}, mean |d| "
            f"{err.mean().item():.3e} (mean |y| {gf.abs().mean().item():.3e}), outputs differing "
            f"{(err > 0).float().mean().item() * 100:.2f} %"
        )
        if n <= 512:
            line += f"; against torch: bf16 Y {pcc(got, want):.7f}, fp32 Y {pcc(gf, want):.7f}"
            line += f", mean |d| {(got - want).abs().mean().item():.3e} / {(gf - want).abs().mean().item():.3e}"
        print(line, flush=True)
        if checks:
            assert pcc(got, gf) > 0.9999
        if os.environ.get("GLM53_MOE_PREFILL_Y8", "1") == "1" and checks:
            y8_check(n, w, gf, err, want if n <= 512 else None, pcc)
        ttnn.deallocate(nf)
        ttnn.deallocate(new)
        ttnn.deallocate(old)

        # the stages' times
        seg, ptok, pw, tpair = moe_prefill.plan(mesh_device, w)
        x_rm = ttnn.to_layout(d_x, ttnn.ROW_MAJOR_LAYOUT)
        ha = moe_prefill.gate_up(mesh_device, x_rm, nd1, seg, ptok, pw, LIMIT)
        hs = (first(seg).long() & 0xFFFFFFFF).reshape(-1)  # (the table's padding rows are left unwritten)
        pair_rows = torch.cat([torch.arange(int(hs[e]), int(hs[e]) + int(hs[128 + e])) for e in range(e_col)])
        # the readers on NoC 0 only against both NoCs (the defaults): the same values
        h1 = flag("GU5_NOCS", 1, lambda: moe_prefill.gate_up(mesh_device, x_rm, nd1, seg, ptok, pw, LIMIT))
        same_h = torch.equal(first(ha)[0, 0, pair_rows], first(h1)[0, 0, pair_rows])
        print(f"{n} tokens: gate_up, the reader on NoC 0 only against both NoCs: bit for bit {same_h}", flush=True)
        assert same_h
        ttnn.deallocate(h1)
        yb = moe_prefill.down(mesh_device, ha, nd2, seg, e_col)
        y1 = flag("DNCB_NOCS", 1, lambda: moe_prefill.down(mesh_device, ha, nd2, seg, e_col))
        ob = moe_prefill.combine(mesh_device, yb, tpair)
        o1 = flag("DNCB_NOCS", 1, lambda: moe_prefill.combine(mesh_device, yb, tpair))
        same_y = torch.equal(first(yb)[0, 0, pair_rows], first(y1)[0, 0, pair_rows])
        same_o = torch.equal(first(ob), first(o1))
        print(f"{n} tokens: down / combine on NoC 0 only against both NoCs: same bits {same_y} / {same_o}", flush=True)
        assert same_y and same_o
        for t in (yb, y1, ob, o1, ha):
            ttnn.deallocate(t)
        if os.environ.get("GLM53_MOE_PREFILL_TRACED") == "1":  # each stage alone in a trace (device time)
            h0 = moe_prefill.gate_up(mesh_device, x_rm, nd1, seg, ptok, pw, LIMIT)
            y0 = moe_prefill.down(mesh_device, h0, nd2, seg, e_col)
            y0f = flag("Y_BF16", False, lambda: moe_prefill.down(mesh_device, h0, nd2, seg, e_col))
            y08 = flag("Y_BFP8", True, lambda: moe_prefill.down(mesh_device, h0, nd2, seg, e_col))
            fns = {
                "plan": lambda: moe_prefill.plan(mesh_device, w),
                "to_layout": lambda: ttnn.to_layout(d_x, ttnn.ROW_MAJOR_LAYOUT),
                "gate_up": lambda: moe_prefill.gate_up(mesh_device, x_rm, nd1, seg, ptok, pw, LIMIT),
                **{
                    f"gate_up_probe{v}": (
                        lambda v=v: flag(
                            "GU5_PROBE", v, lambda: moe_prefill.gate_up(mesh_device, x_rm, nd1, seg, ptok, pw, LIMIT)
                        )
                    )
                    # compute 1 no tilize, 2 no finish, 4 no matmul; reader 8 W1 of the first expert only,
                    # 16 no x; writer 32 no exchange (wrong results)
                    for v in (7, 8, 16, 32, 24, 56, 63)
                },
                "gate_up_nocs1": lambda: flag(
                    "GU5_NOCS", 1, lambda: moe_prefill.gate_up(mesh_device, x_rm, nd1, seg, ptok, pw, LIMIT)
                ),
                **{
                    f"gate_up_q{q}": (
                        lambda q=q: flag(
                            "GU5_NOC1_QUARTERS",
                            q,
                            lambda: moe_prefill.gate_up(mesh_device, x_rm, nd1, seg, ptok, pw, LIMIT),
                        )
                    )
                    for q in (1, 3)
                },
                "down": lambda: moe_prefill.down(mesh_device, h0, nd2, seg, e_col),
                "combine": lambda: moe_prefill.combine(mesh_device, y0, tpair),
                "down_nocs1": lambda: flag("DNCB_NOCS", 1, lambda: moe_prefill.down(mesh_device, h0, nd2, seg, e_col)),
                "combine_nocs1": lambda: flag("DNCB_NOCS", 1, lambda: moe_prefill.combine(mesh_device, y0, tpair)),
                **{
                    f"{stage}_q{q}": (
                        lambda stage=stage, q=q: flag(
                            "DNCB_NOC1_QUARTERS",
                            q,
                            lambda: moe_prefill.down(mesh_device, h0, nd2, seg, e_col)
                            if stage == "down"
                            else moe_prefill.combine(mesh_device, y0, tpair),
                        )
                    )
                    for stage in ("down", "combine")
                    for q in (2, 3)
                },
                "down_yfp32": lambda: flag("Y_BF16", False, lambda: moe_prefill.down(mesh_device, h0, nd2, seg, e_col)),
                "combine_yfp32": lambda: moe_prefill.combine(mesh_device, y0f, tpair),
                "experts": run_new,
                "experts_yfp32": lambda: flag("Y_BF16", False, run_new),
                "down_ybfp8": lambda: flag("Y_BFP8", True, lambda: moe_prefill.down(mesh_device, h0, nd2, seg, e_col)),
                "combine_ybfp8": lambda: moe_prefill.combine(mesh_device, y08, tpair),
                "experts_ybfp8": lambda: flag("Y_BFP8", True, run_new),
                "tilize": lambda: ttnn.to_layout(ttnn.to_layout(d_x, ttnn.ROW_MAJOR_LAYOUT), ttnn.TILE_LAYOUT),
            }
            only = [k for k in os.environ.get("GLM53_MOE_PREFILL_TRACE_ONLY", "").split(",") if k]
            if only:  # (GLM53_MOE_PREFILL_TRACE_ONLY: these stages' times only)
                fns = {k: fns[k] for k in only}
            traced_ms = {}
            for name, fn in fns.items():

                def drop(o):
                    for t in o if isinstance(o, tuple) else (o,):
                        ttnn.deallocate(t)

                drop(fn())
                ttnn.synchronize_device(mesh_device)
                with trace_capture(mesh_device) as trace:
                    outs = [fn() for _ in range(5)]
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                traced_ms[name] = (time.perf_counter() - t0) / 5 * 1e3
                ttnn.release_trace(mesh_device, trace)
                for o in outs:
                    drop(o)
            print(
                f"{n} tokens: stages traced (ms) " + ", ".join(f"{k} {v:.2f}" for k, v in traced_ms.items()), flush=True
            )
            ttnn.deallocate(h0)
            ttnn.deallocate(y0)
            ttnn.deallocate(y0f)
            ttnn.deallocate(y08)
        stages = {}
        _, stages["plan"] = timed(mesh_device, lambda: moe_prefill.plan(mesh_device, w)[0])
        h, stages["gate_up"] = timed(
            mesh_device, lambda: moe_prefill.gate_up(mesh_device, x_rm, nd1, seg, ptok, pw, LIMIT)
        )
        y, stages["down"] = timed(mesh_device, lambda: moe_prefill.down(mesh_device, h, nd2, seg, e_col))
        out, stages["combine"] = timed(mesh_device, lambda: moe_prefill.combine(mesh_device, y, tpair))
        print(f"{n} tokens: stages (ms) " + ", ".join(f"{k} {v * 1e3:.2f}" for k, v in stages.items()), flush=True)
        for tt in (seg, ptok, pw, tpair, x_rm, h, y, out, w, d_x, d_logits):
            ttnn.deallocate(tt)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_moe_prefill_gate_up_zones(mesh_device):
    """gate | up alone at GLM53_MOE_PREFILL_TOKENS (4,096) tokens, run once (for a kernel zone profile)."""
    import torch

    n = int(os.environ.get("GLM53_MOE_PREFILL_TOKENS", "4096").split(",")[-1])
    e_all, e_col, s_col, base = 288, 72, 256, 144
    rand = lambda shape, dt: ttnn.rand(  # noqa: E731
        shape, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, low=-0.05, high=0.05
    )
    w1 = rand([1, 1, s_col // 32 * e_col * D, 64], ttnn.bfloat8_b)
    nd1 = ttnn.to_memory_config(w1, row_sharded(mesh_device, 2, 16 * 32))
    ttnn.deallocate(w1)
    rep = dict(dims=(None, None), dtype=ttnn.float32)
    d_bias = put(mesh_device, torch.zeros(1, 1, 1, e_all), **rep)
    bases = torch.zeros(1, 1, 1, 8, dtype=torch.int32)
    bases[..., 0] = base
    d_base = put(mesh_device, bases, dims=(None, None), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    g = torch.Generator().manual_seed(n)
    d_logits = put(mesh_device, (torch.randn(n, e_all, generator=g) * 2.0)[None, None], **rep)
    d_x = put(
        mesh_device,
        torch.randn(n, D, generator=g).to(torch.bfloat16)[None, None],
        dims=(None, None),
        dtype=ttnn.bfloat16,
    )
    w = ttnn.experimental.deepseek_prefill.moe_decode_route(d_logits, d_bias, d_base, n, K, e_col, 2.5)
    seg, ptok, pw, tpair = moe_prefill.plan(mesh_device, w)
    x_rm = ttnn.to_layout(d_x, ttnn.ROW_MAJOR_LAYOUT)
    for _ in range(int(os.environ.get("GLM53_MOE_PREFILL_ZONE_RUNS", "1"))):  # 3 overflowed the zones' buffer
        h = moe_prefill.gate_up(mesh_device, x_rm, nd1, seg, ptok, pw, LIMIT)
        ttnn.synchronize_device(mesh_device)
        stages = os.environ.get("GLM53_MOE_PREFILL_ZONE_STAGES", "")
        if "down" in stages or "combine" in stages:  # then down and combine once each (their kernels' spans)
            w2 = rand([1, 1, Q * e_col * s_col, D // Q], ttnn.bfloat8_b)
            nd2 = ttnn.to_memory_config(w2, row_sharded(mesh_device, D // Q // 32, e_col * s_col))
            ttnn.deallocate(w2)
            y = moe_prefill.down(mesh_device, h, nd2, seg, e_col)
            ttnn.synchronize_device(mesh_device)
            out = moe_prefill.combine(mesh_device, y, tpair)
            ttnn.synchronize_device(mesh_device)
            for t in (nd2, y, out):
                ttnn.deallocate(t)
        ttnn.deallocate(h)
    print(f"{n} tokens: gate_up run", flush=True)
