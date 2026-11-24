"""`ttnn.experimental.kda.decode_prepare` and `decode_finish` against torch on random inputs: q, k, v, the decay and
beta per (user, head) row and the shifted conv history; the gated RMS norm in o_proj's layout.  No checkpoint.  Each
case also prints the op's device time (a captured trace of 20 calls, replayed).

Run (a device test):
    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_kda_prepare.py -s
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
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE  # noqa: E402

H, DH, LOWER = 16, 128, -5.0
CH = 3 * H * DH


def pcc(a, b) -> float:
    return torch.corrcoef(torch.stack([a.double().flatten(), b.double().flatten()]))[0, 1].item()


def trace_seconds(mesh_device, call, reps: int = 20) -> float:
    """Device time of one `call()` (it returns a list of new tensors), from a captured trace of `reps` calls."""
    for t in call():
        ttnn.deallocate(t)
    ttnn.synchronize_device(mesh_device)
    with trace_capture(mesh_device) as trace:
        outs = [call() for _ in range(reps)]
    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
    seconds = (time.perf_counter() - t0) / reps
    ttnn.release_trace(mesh_device, trace)
    for out in outs:
        for t in out:
            ttnn.deallocate(t)
    return seconds


def reference(proj, fg, hist, taps, gate):
    """proj [B, >= CH + H] bf16, fg [B, >= H DH] bf16, hist 3 x [B, CH], taps [4, CH], gate [2, H DH] (fp32)."""
    b = proj.shape[0]
    cur = proj[:, :CH].float()
    acc = hist[0] * taps[0]
    acc = acc + hist[1] * taps[1]
    acc = acc + hist[2] * taps[2]
    acc = acc + cur * taps[3]
    qkv = torch.nn.functional.silu(acc).to(torch.bfloat16).float()
    q, k, v = (qkv[:, i * H * DH : (i + 1) * H * DH].reshape(b * H, DH) for i in range(3))
    q = q * (torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / DH**0.5)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    f = fg[:, : H * DH].float()
    e = torch.exp(LOWER * torch.sigmoid((f + gate[0]) * gate[1])).reshape(b * H, DH)
    beta = torch.sigmoid(proj[:, CH : CH + H].float()).to(torch.bfloat16).float().reshape(b * H, 1)
    return (q, k, v, e, beta), [hist[1], hist[2], cur]


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
@pytest.mark.parametrize(
    "users", [int(u) for u in os.environ.get("GLM53_KDA_PREPARE_USERS", "1,3,4,16,20").split(",")]
)  # (GLM53_KDA_ZONES=1: one eager call, untimed: a kernel zone profile)
def test_kda_prepare(mesh_device, users):
    g = torch.Generator().manual_seed(users)
    proj = (torch.randn(users, 6432, generator=g)).to(torch.bfloat16)
    fg = (torch.randn(users, 2 * H * DH, generator=g)).to(torch.bfloat16)
    hist = [torch.randn(users, CH, generator=g) for _ in range(3)]
    taps = torch.randn(4, CH, generator=g) * 0.5
    gate = torch.stack([torch.randn(H * DH, generator=g), torch.rand(H * DH, generator=g) + 0.5])
    want, want_hist = reference(proj, fg, hist, taps, gate)

    up = lambda t, dtype: ttnn.from_torch(
        t[None, None], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )  # noqa: E731
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()  # noqa: E731
    d_hist = [up(h, ttnn.float32) for h in hist]
    d_in = (up(proj, ttnn.bfloat16), up(fg, ttnn.bfloat16), up(taps, ttnn.float32), up(gate, ttnn.float32))
    call = lambda: list(  # noqa: E731
        ttnn.experimental.kda.decode_prepare(d_in[0], d_in[1], *d_hist, d_in[2], d_in[3], H, LOWER, 1.0 / DH**0.5)
    )
    outs = call()
    for name, got, ref in zip(("q", "k", "v", "decay", "beta"), outs, want):
        got = first(got).reshape(ref.shape)
        print(
            f"{users} users {name}: pcc {pcc(got, ref):.7f}, max |d| {(got - ref).abs().max().item():.3e}", flush=True
        )
    for i, (h, ref) in enumerate(zip(d_hist, want_hist)):
        got = first(h).reshape(ref.shape)
        print(f"{users} users history {i}: equal {torch.equal(got, ref)}, max |d| {(got - ref).abs().max().item():.3e}")
    for got, ref in zip(outs, want):
        assert pcc(first(got).reshape(ref.shape), ref) > 0.9999
    # GLM53_KDA_PREPARE_ROW_SPLIT=1: the plain call's units by grid row speed (`row_split`) on a fresh copy of the
    # history: its outputs and the shifted history bit for bit the default call's, and its time after the default's
    # (GLM53_KDA_ZONES=1 with it: the zone profile of the row_split call only)
    split = os.environ.get("GLM53_KDA_PREPARE_ROW_SPLIT") == "1"
    if split:
        s_hist = [up(h, ttnn.float32) for h in hist]
        s_call = lambda: list(  # noqa: E731
            ttnn.experimental.kda.decode_prepare(
                d_in[0], d_in[1], *s_hist, d_in[2], d_in[3], H, LOWER, 1.0 / DH**0.5, row_split=True
            )
        )
        s_outs = s_call()
        same = [torch.equal(first(a), first(b)) for a, b in zip(outs, s_outs)]
        same += [torch.equal(first(a), first(b)) for a, b in zip(d_hist, s_hist)]
        print(f"{users} users row_split: the same bits (q k v decay beta, histories 0-2): {same}", flush=True)
        assert all(same)
    if os.environ.get("GLM53_KDA_ZONES") != "1":
        print(f"{users} users prepare: {trace_seconds(mesh_device, call) * 1e6:.1f} us per call", flush=True)
        if split:
            us = trace_seconds(mesh_device, s_call) * 1e6
            print(f"{users} users prepare row_split: {us:.1f} us per call", flush=True)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
@pytest.mark.parametrize(
    "users", [int(u) for u in os.environ.get("GLM53_KDA_FINISH_USERS", "1,3,4,16,20,128").split(",")]
)  # 128: more units than cores (two on some); GLM53_KDA_ZONES=1: one eager call, untimed (a kernel zone profile)
def test_kda_finish(mesh_device, users):
    g = torch.Generator().manual_seed(100 + users)
    o = torch.randn(users * H, DH, generator=g) * 0.3
    fg = torch.randn(users, 2 * H * DH, generator=g).to(torch.bfloat16)
    o_norm = torch.rand(DH, generator=g) + 0.5
    eps = 1e-6
    ob = o.to(torch.bfloat16).float()
    gate = torch.sigmoid(fg[:, H * DH :].float().reshape(users * H, DH))
    want = (ob * torch.rsqrt((ob * ob).mean(-1, keepdim=True) + eps) * o_norm * gate).to(torch.bfloat16)
    want = want.reshape(users, H * DH)

    up = lambda t, dtype: ttnn.from_torch(
        t[None, None], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )  # noqa: E731
    d_in = (up(o, ttnn.float32), up(fg, ttnn.bfloat16), up(o_norm[None], ttnn.float32))
    out = ttnn.experimental.kda.decode_finish(*d_in, H, eps)
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(users, H * DH)
    diff = (got.float() - want.float()).abs()
    print(
        f"{users} users finish: pcc {pcc(got, want):.7f}, max |d| {diff.max().item():.3e}, equal {torch.equal(got, want)}"
    )
    assert pcc(got, want) > 0.9999
    if os.environ.get("GLM53_KDA_ZONES") != "1":
        seconds = trace_seconds(mesh_device, lambda: [ttnn.experimental.kda.decode_finish(*d_in, H, eps)])
        print(f"{users} users finish: {seconds * 1e6:.1f} us per call", flush=True)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
@pytest.mark.parametrize("users", [1, 3, 4, 8, 16, 32])
def test_kda_prepare_same(mesh_device, users):
    """decode_prepare's outputs (q, k, v, decay, beta and the history; with tokens=2, keep and saved also the checkpoint)
    on fixed random inputs, the plain step's form and the verify step's (tokens=2), saved to GLM53_KDA_SAME_DIR/<tag>
    (default out_dir("kda_same"); GLM53_KDA_SAME_TAG); with GLM53_KDA_SAME_AGAINST another tag's saved outputs compared bit for bit (a kernel
    change that must keep the values).  Each form's time (trace)."""
    import pathlib

    from models.demos.blackhole.glm53_flash.paths import out_dir

    root = pathlib.Path(os.environ.get("GLM53_KDA_SAME_DIR") or out_dir("kda_same"))
    tag, against = os.environ.get("GLM53_KDA_SAME_TAG", "now"), os.environ.get("GLM53_KDA_SAME_AGAINST")
    g = torch.Generator().manual_seed(700 + users)
    b = users
    proj = torch.randn(2 * b, 6432, generator=g).to(torch.bfloat16)
    fg = torch.randn(2 * b, 2 * H * DH, generator=g).to(torch.bfloat16)
    hist = [torch.randn(b, CH, generator=g) for _ in range(4)]
    taps = torch.randn(4, CH, generator=g) * 0.5
    gate = torch.stack([torch.randn(H * DH, generator=g), torch.rand(H * DH, generator=g) + 0.5])
    keep = torch.tensor([(u * 7 + 3) % 3 != 0 for u in range(b)], dtype=torch.int32)
    up = lambda t, dtype: ttnn.from_torch(  # noqa: E731
        t[None, None], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()  # noqa: E731
    tp, gc = up(taps, ttnn.float32), up(gate, ttnn.float32)
    flags = ttnn.from_torch(keep[None], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    scale = 1.0 / DH**0.5
    got, times = {}, {}
    # the plain step's form (one token a user)
    p1, f1 = up(proj[:b], ttnn.bfloat16), up(fg[:b], ttnn.bfloat16)
    hs = [up(x, ttnn.float32) for x in hist[:3]]
    outs = ttnn.experimental.kda.decode_prepare(p1, f1, *hs, tp, gc, H, LOWER, scale)
    for name, t in zip(("q", "k", "v", "decay", "beta"), outs):
        got[f"plain {name}"] = first(t)
    for i, t in enumerate(hs):
        got[f"plain history{i}"] = first(t)
    hs_t = [up(x, ttnn.float32) for x in hist[:3]]
    times["plain"] = trace_seconds(
        mesh_device, lambda: list(ttnn.experimental.kda.decode_prepare(p1, f1, *hs_t, tp, gc, H, LOWER, scale))
    )
    # the verify step's form (tokens=2, keep, saved)
    p2, f2 = up(proj, ttnn.bfloat16), up(fg, ttnn.bfloat16)
    hv = [up(x, ttnn.float32) for x in hist]
    outs = ttnn.experimental.kda.decode_prepare(
        p2, f2, *hv[:3], tp, gc, H, LOWER, scale, users=b, keep=flags, saved=hv[3], tokens=2
    )
    for name, t in zip(("q", "k", "v", "decay", "beta"), outs):
        got[f"two {name}"] = first(t)
    for i, t in enumerate(hv):
        got[f"two history{i}"] = first(t)
    hv_t = [up(x, ttnn.float32) for x in hist]
    times["two"] = trace_seconds(
        mesh_device,
        lambda: list(
            ttnn.experimental.kda.decode_prepare(
                p2, f2, *hv_t[:3], tp, gc, H, LOWER, scale, users=b, keep=flags, saved=hv_t[3], tokens=2
            )
        ),
    )
    root.mkdir(parents=True, exist_ok=True)
    torch.save(got, root / f"{tag}-{users}.pt")
    line = f"{users} users: prepare, plain {times['plain'] * 1e6:.1f} us, two tokens {times['two'] * 1e6:.1f} us"
    if against:
        old = torch.load(root / f"{against}-{users}.pt")
        diff = [k for k in got if not torch.equal(got[k], old[k])]
        line += f"; against {against}: {'bit for bit' if not diff else 'DIFFER: ' + ', '.join(diff)}"
        assert not diff
    print(line, flush=True)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_kda_spec_tokens(mesh_device):
    """A verify step's decode_prepare of T tokens a user (T - 1 drafts: `tokens=T`, keep codes 0 .. T - 1, T - 1
    checkpoint rows of the history) against T one-token calls from the history each user's keep code picks: q, k, v,
    the decay, beta (token k's rows k B H ..), the history and its checkpoint rows bit for bit; then its time (trace).
    (The delta rule of the verify step, the chunked form, is not bit for bit T plain steps: test_kda_decode_step's
    chunk test against float64.)  GLM53_KDA_SPEC_T (default 2,3,4,6,8), GLM53_KDA_SPEC_USERS (default 1,4,16)."""
    bad = []
    for users in (int(u) for u in os.environ.get("GLM53_KDA_SPEC_USERS", "1,4,16").split(",")):
        for n in (int(t) for t in os.environ.get("GLM53_KDA_SPEC_T", "2,3,4,6,8").split(",")):
            g = torch.Generator().manual_seed(900 + 10 * users + n)
            b, sr = users, n - 1
            proj = torch.randn(n * b, 6432, generator=g).to(torch.bfloat16)
            fg = torch.randn(n * b, 2 * H * DH, generator=g).to(torch.bfloat16)
            hist = [torch.randn(b, CH, generator=g) for _ in range(3)]
            saved = [torch.randn(b, CH, generator=g) for _ in range(sr)]
            taps = torch.randn(4, CH, generator=g) * 0.5
            gate = torch.stack([torch.randn(H * DH, generator=g), torch.rand(H * DH, generator=g) + 0.5])
            codes = torch.randint(0, n, (b,), generator=g, dtype=torch.int32)
            up = lambda t, dtype: ttnn.from_torch(  # noqa: E731
                t[None, None], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device
            )
            first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()  # noqa: E731
            tp, gc = up(taps, ttnn.float32), up(gate, ttnn.float32)
            scale = 1.0 / DH**0.5

            # one token at a time from the picked history (S_0 .. S_{sr-1}, h0, h1, h2)[c .. c + 2]
            seq = saved + hist
            chosen = [torch.stack([seq[int(c) + j][u] for u, c in enumerate(codes)]) for j in range(3)]
            r = [up(x, ttnn.float32) for x in chosen]
            outs, inputs_ = [], []
            for i in range(n):
                pi = up(proj[i * b : (i + 1) * b], ttnn.bfloat16)
                fi = up(fg[i * b : (i + 1) * b], ttnn.bfloat16)
                mid = ttnn.experimental.kda.decode_prepare(pi, fi, *r, tp, gc, H, LOWER, scale)
                outs.append([first(t).reshape(b * H, -1) for t in mid])
                inputs_.append(first(r[2]).reshape(b, CH))
            names = ("q", "k", "v", "decay", "beta")
            want = {nm: torch.cat([o[j] for o in outs]) for j, nm in enumerate(names)}
            want.update({f"history{i}": first(t).reshape(b, CH) for i, t in enumerate(r)})
            ext = chosen + inputs_  # S'_m <- ext[m + 1]
            want.update({f"saved{m}": ext[m + 1] for m in range(sr)})

            # the T-token op
            p, f = up(proj, ttnn.bfloat16), up(fg, ttnn.bfloat16)
            hn = [up(x, ttnn.float32) for x in hist]
            sv = ttnn.from_torch(
                torch.stack(saved)[None], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device
            )
            flags = ttnn.from_torch(codes[None], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)

            def ops():
                return list(
                    ttnn.experimental.kda.decode_prepare(
                        p, f, *hn, tp, gc, H, LOWER, scale, users=b, keep=flags, saved=sv, tokens=n
                    )
                )

            if os.environ.get("GLM53_KDA_ZONES") == "1":  # decode_prepare's kernel zones (the profiler on): eager runs
                for _ in range(int(os.environ.get("GLM53_KDA_ZONE_RUNS", "2"))):
                    mid = ops()
                    ttnn.synchronize_device(mesh_device)
                    for t in mid:
                        ttnn.deallocate(t)
                print(f"KSPEC users={users} T={n}: zone runs done", flush=True)
                continue
            mid = ops()
            got = {nm: first(t).reshape(n * b * H, -1) for nm, t in zip(names, mid)}
            got.update({f"history{i}": first(t).reshape(b, CH) for i, t in enumerate(hn)})
            svg = first(sv).reshape(sr, b, CH)
            got.update({f"saved{m}": svg[m] for m in range(sr)})
            same = {k: torch.equal(got[k].reshape(want[k].shape), want[k]) for k in want}
            if not all(same.values()):
                bad.append((users, n))
            # its time (the history advances call to call: values do not matter here)
            t_us = trace_seconds(mesh_device, ops) * 1e6
            print(
                f"KSPEC users={users} T={n}: {t_us:.1f} us (prepare), one call equals one token at a time: "
                f"{all(same.values())} {[k for k, v in same.items() if not v]}",
                flush=True,
            )
    assert not bad, f"differ: {bad}"


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_kda_prepare_pair_major(mesh_device):
    """decode_prepare's `pair_major` outputs (row (u H + h) T + k) against today's (row (u H + h) + k B H) on the same
    inputs and fresh copies of the history: q, k, v, decay and beta bit for bit after the rows' reorder, the history
    and its checkpoint rows the same; GLM53_KDA_SPEC_USERS (default 1,2,4,8,16: stacked up to 4 a row) x
    GLM53_KDA_SPEC_T (default 2,4,8), each user's keep code at random.  Prints a line a case (PAIRMAJOR), no assert."""
    for users in (int(u) for u in os.environ.get("GLM53_KDA_SPEC_USERS", "1,2,4,8,16").split(",")):
        for n in (int(t) for t in os.environ.get("GLM53_KDA_SPEC_T", "2,4,8").split(",")):
            g = torch.Generator().manual_seed(1300 + 10 * users + n)
            b, sr = users, max(n - 1, 1)
            proj = torch.randn(n * b, 6432, generator=g).to(torch.bfloat16)
            fg = torch.randn(n * b, 2 * H * DH, generator=g).to(torch.bfloat16)
            hist = [torch.randn(b, CH, generator=g) for _ in range(3)]
            saved = torch.randn(sr, b, CH, generator=g)
            taps = torch.randn(4, CH, generator=g) * 0.5
            gate = torch.stack([torch.randn(H * DH, generator=g), torch.rand(H * DH, generator=g) + 0.5])
            codes = torch.randint(0, n, (b,), generator=g, dtype=torch.int32)
            up = lambda t, dtype: ttnn.from_torch(  # noqa: E731
                t[None, None], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device
            )
            first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()  # noqa: E731
            p, f = up(proj, ttnn.bfloat16), up(fg, ttnn.bfloat16)
            tp, gc = up(taps, ttnn.float32), up(gate, ttnn.float32)
            flags = ttnn.from_torch(codes[None], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
            got = {}
            for name, pm in (("token", False), ("pair", True)):
                hn = [up(x, ttnn.float32) for x in hist]
                sv = ttnn.from_torch(saved[None], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device)
                outs = ttnn.experimental.kda.decode_prepare(
                    p, f, *hn, tp, gc, H, LOWER, 1.0 / DH**0.5, users=b, keep=flags, saved=sv, tokens=n, pair_major=pm
                )
                got[name] = [first(o).reshape(n * b * H, -1) for o in outs] + [first(x) for x in hn] + [first(sv)]
                for t in (*outs, *hn, sv):
                    ttnn.deallocate(t)
            pairs = b * H
            # today's row (pair + k pairs) -> pair_major's row (pair n + k)
            order = torch.tensor([(r % pairs) * n + r // pairs for r in range(n * pairs)])
            same = []
            for i, (a, c) in enumerate(zip(got["token"], got["pair"])):
                if i < 5:
                    re = torch.empty_like(a)
                    re[order] = a
                    same.append(torch.equal(re, c))
                else:
                    same.append(torch.equal(a, c))
            print(
                f"PAIRMAJOR users={users} T={n}: q k v decay beta / history 0-2 / saved the same after the reorder: "
                f"{same}",
                flush=True,
            )
