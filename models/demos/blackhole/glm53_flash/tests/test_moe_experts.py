"""`ttnn.experimental.deepseek_prefill.moe_decode_experts`: the decode MoE's routed experts in one op (64 gate | up
cores, 32 down cores), in the column layout (I = 256 of each of a mesh column's 72 experts on every chip).  Random
tensors, no checkpoint.

Run (a device test):
    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_moe_experts.py -s
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

from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, put  # noqa: E402

D, Q = 4096, 32
LIMIT = 10.0


def traced(mesh, fn, reps: int = 10) -> float:
    ttnn.deallocate(fn())  # compile
    ttnn.synchronize_device(mesh)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        outs = [fn() for _ in range(reps)]
    except BaseException:  # (never leave the capture open: the device's close then timed out and its fabric stuck)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.release_trace(mesh, trace)
        raise
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    seconds = (time.perf_counter() - t0) / reps
    ttnn.release_trace(mesh, trace)
    for o in outs:
        ttnn.deallocate(o)
    return seconds


def row_sharded(mesh, width_tiles: int, height: int = 32):
    """DRAM ND-sharded `height` rows tall (one tile row, or a core's whole slice) and `width_tiles` wide, the shards
    round-robin over the banks."""
    dram = mesh.dram_grid_size()
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=ttnn.Shape([height, width_tiles * 32]),
            grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))]),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


K = 8  # each token's experts


def fixed_weight(w: float) -> int:
    """route_build's fixed point of a pick's weight (moe_decode_experts_common.hpp fixed_weight: 2^20 a unit, from the
    fp32 bits of a bf16 value, truncated)."""
    import struct

    bits = struct.unpack("<I", struct.pack("<f", w))[0]
    e, m = (bits >> 23) & 0xFF, 128 | ((bits >> 16) & 0x7F)
    if e >= 114:
        return m << (e - 114) if e - 114 < 24 else 0xFFFFFFFF
    return m >> (114 - e) if 114 - e < 32 else 0


def prog_marked(ids, wts, local: int) -> list:
    """The progressive form's experts read with their correction (route_build.cpp's rule for GLM53_MOE_PROG_F): the
    picks' weight sums by expert in fixed point, the chosen experts D, m = floor(|D| (F 1088 - 576) / 576) of them
    (all from F = 1.06 on), the largest sums (equal: the lower expert); past half of D, all marked and the smallest
    (equal: the higher expert) unmarked."""
    milli = round(float(os.environ["GLM53_MOE_PROG_F"]) * 1000)
    # T1 (GLM53_MOE_PROG_ROWS, the program factory's rule): token r = mesh row r // trow's token j U + u counts with
    # q_j = round(256 P_j), (fixed_weight q_j) >> 8, when trow = N / 8 is T U with U <= 2
    q, n = [], ids.shape[0]
    if os.environ.get("GLM53_MOE_PROG_ROWS"):
        q = [round(float(v) * 256) for v in os.environ["GLM53_MOE_PROG_ROWS"].split(",")]
        if not (n % 8 == 0 and (n // 8) % len(q) == 0 and (n // 8) // len(q) <= 2):
            q = []
    trow = n // 8
    total, count = [0] * local, [0] * local
    for r in range(n):
        for c in range(16):
            w = float(wts[r, c])
            if not w > 0:
                continue
            e = int(ids[r, c])
            if q:
                total[e] += (fixed_weight(w) * q[(r % trow) // (trow // len(q))]) >> 8
            else:
                total[e] += fixed_weight(w)
            count[e] += 1
    chosen = [e for e in range(local) if count[e]]
    # the busiest mesh column's chosen experts (the routing's global picks in id columns 12 .. 15, + 1, two to a word);
    # none: this column's own
    cols = [set() for _ in range(4)]
    for r in range(ids.shape[0]):
        for c in range(12, 16):
            v = int(ids[r, c]) & 0xFFFFFFFF
            for g in (v & 0xFFFF, v >> 16):
                if 0 < g <= 4 * 72:
                    cols[(g - 1) // 72].add((g - 1) % 72)
    busiest = max(len(c) for c in cols)
    busiest = max(busiest, len(chosen))
    num, den = milli * 1088 * busiest, 576000 * len(chosen)
    m = 0 if num <= den else (num - den) // 576000
    m = min(m, len(chosen))
    if 2 * m <= len(chosen):
        marked = []
        for _ in range(m):
            best = None
            for e in chosen:
                if e not in marked and (best is None or total[e] > total[best]):
                    best = e
            marked.append(best)
        return marked
    marked = list(chosen)
    for _ in range(len(chosen) - m):
        best = None
        for e in marked:
            if best is None or total[e] <= total[best]:
                best = e
        marked.remove(best)
    return marked


def sliced_reference(x, ids, wts, w1, w2, E, S):
    """y [N, D] (bf16 values) from x [N, D], the chosen ids and weights [N, 16] (moe_decode_route's id tiles) and
    the layouts (T = S / 32 column tiles of each of the E experts) w1 [T E D, 64] ((gate, up) column tile c of
    expert e at tile rows (T e + c) D / 32 + k), w2 [Q E S, D / Q] (down core q's columns of expert e at rows
    (q E + e) S + i)."""
    import torch

    t = S // 32
    w1 = w1.reshape(E, t, D, 64)
    gates = torch.cat([w1[:, c, :, :32] for c in range(t)], dim=-1)  # [E, D, S]
    ups = torch.cat([w1[:, c, :, 32:] for c in range(t)], dim=-1)
    downs = w2.reshape(Q, E, S, D // Q).permute(1, 2, 0, 3).reshape(E, S, D)
    n = x.shape[0]
    dense = torch.zeros(n, E)
    for r in range(n):
        for c in range(16):
            if wts[r, c] > 0:
                dense[r, ids[r, c]] = wts[r, c]
    y = torch.zeros(n, D)
    for e in torch.nonzero(dense.sum(0) > 0)[:, 0].tolist():
        g = (x @ gates[e]).to(torch.bfloat16).float()
        u = (x @ ups[e]).to(torch.bfloat16).float()
        h = torch.nn.functional.silu(torch.clamp(g, max=LIMIT)) * torch.clamp(u, -LIMIT, LIMIT)
        h = (h * dense[:, e : e + 1]).to(torch.bfloat16).float()
        y = y + h @ downs[e]
    return y.to(torch.bfloat16).float(), int((dense.sum(0) > 0).sum())


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_moe_experts_column(mesh_device):
    """The column-sliced layout (I = 256: each chip holds 256 columns of each of its mesh column's 72 experts, the
    chosen ones in groups of up to 8): moe_decode_route with local 72 (the chosen ids of base .. base + 71, as e -
    base) against torch's routing, for 1 / 8 / 32 tokens (one tile row) and 64 / 128 / 256 (2 / 4 / 8 rows: packed
    units), the op against torch, and its time (trace)."""
    import torch

    e_all, e_col, s_col, base = 288, 72, 256, 144
    t_col = s_col // 32
    rand = lambda shape, dt: ttnn.rand(  # noqa: E731
        shape, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, low=-0.05, high=0.05
    )
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()  # noqa: E731
    # GLM53_MOE_PROG_F (the progressive form: each slot a BFP4 base and a BFP4 correction, here both random), the
    # corrections after the bases (w1 slots e_col + e, each down core's slice 2 e_col slots)
    prog = os.environ.get("GLM53_MOE_PROG_F") is not None
    slots = 2 * e_col if prog else e_col
    w1 = rand([1, 1, t_col * slots * D, 64], WDT)
    w2 = rand([1, 1, Q * slots * s_col, D // Q], WDT)
    host_w1, host_w2 = first(w1)[0, 0], first(w2)[0, 0]
    nd1 = ttnn.to_memory_config(w1, row_sharded(mesh_device, 2, 16 * 32))
    nd2 = ttnn.to_memory_config(w2, row_sharded(mesh_device, D // Q // 32, slots * s_col))
    ttnn.deallocate(w1)
    ttnn.deallocate(w2)
    rep = dict(dims=(None, None), dtype=ttnn.float32)
    d_bias = put(mesh_device, torch.zeros(1, 1, 1, e_all), **rep)
    bases = torch.zeros(1, 1, 1, 8, dtype=torch.int32)
    bases[..., 0] = base
    d_base = put(mesh_device, bases, dims=(None, None), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cases = []
    # the token counts, then 32 tokens whose choices in the column are held to `forced` experts (9 .. 20: the time
    # per chosen expert, ~9.5 us measured)
    # MOE_EXPERTS_COLUMN_CASES="tokens:forced,..": those cases only; MOE_EXPERTS_COLUMN_ZONES=1: no timing, the op run
    # eagerly 3 times a case instead (its last runs for a kernel zone profile)
    picked = os.environ.get("MOE_EXPERTS_COLUMN_CASES")
    zones = os.environ.get("MOE_EXPERTS_COLUMN_ZONES") == "1"
    sizes = (1, 8, 32, 64, 128, 256) + ((512,) if os.environ.get("MOE_EXPERTS_COLUMN_512") == "1" else ())
    cases_all = [(t, 0) for t in sizes] + [(32, f) for f in (9, 10, 11, 12, 13, 18, 20)]
    if picked:  # (any tokens:forced pair, e.g. 512:12: an expert with more than 32 tokens, several units)
        cases_all = [tuple(int(v) for v in c.split(":")) for c in picked.split(",")]
    for tokens, forced in cases_all:
        n = max(32, tokens)
        g = torch.Generator().manual_seed(tokens + 1000 * forced)
        logits = torch.randn(n, e_all, generator=g) * 2.0
        if forced:
            logits[:, base : base + e_col] = -100.0
            keep = base + torch.randperm(e_col, generator=g)[:forced]
            # (above 32 tokens a lower lift: at 10 the sigmoids of the held experts tie in fp32 and torch's top-k order
            # differs from the op's; still every token's choices in the column are among them)
            logits[:, keep] = (10.0 if tokens <= 32 else 3.0) + torch.randn(n, forced, generator=g)
        x = torch.randn(n, D, generator=g).to(torch.bfloat16)
        x[tokens:] = 0
        d_logits = put(mesh_device, logits[None, None], **rep)
        d_x = put(mesh_device, x[None, None], dims=(None, None), dtype=ttnn.bfloat16)
        w = ttnn.experimental.deepseek_prefill.moe_decode_route(d_logits, d_bias, d_base, tokens, K, e_col, 2.5)
        tile = first(w)[0, 0]
        ids, wts = tile[:, :16].contiguous().view(torch.int32), tile[:, 16:]
        # torch's routing (no bias) over all 288, then the column's: ids in increasing order, as e - base
        scores = torch.sigmoid(logits)
        top = torch.topk(scores, K, dim=-1).indices.sort(dim=-1).values
        total = torch.zeros(n)
        for i in range(K):
            total = total + scores.gather(1, top[:, i : i + 1])[:, 0]
        for r in range(tokens):
            mine = [e for e in top[r].tolist() if base <= e < base + e_col]
            want_w = [((scores[r, e] * (1.0 / (total[r] + 1e-20))) * 2.5).to(torch.bfloat16).float() for e in mine]
            assert ids[r, : len(mine)].tolist() == [e - base for e in mine], f"row {r}: the column's ids"
            for j, v in enumerate(want_w):
                assert abs(wts[r, j].item() - v.item()) <= 2**-8 * abs(v.item())
            assert (wts[r, len(mine) :] == 0).all()
        assert (wts[tokens:] == 0).all()
        ref_w1, ref_w2 = host_w1, host_w2
        if prog:  # the marked experts' weights base + correction, the others' the base
            marked = prog_marked(ids, wts, e_col)
            v1 = host_w1.reshape(2 * e_col, t_col, D, 64)
            v2 = host_w2.reshape(Q, 2 * e_col, s_col, D // Q)
            e1, e2 = v1[:e_col].clone(), v2[:, :e_col].clone()
            for e in marked:
                e1[e] += v1[e_col + e]
                e2[:, e] += v2[:, e_col + e]
            ref_w1, ref_w2 = e1.reshape(-1, 64), e2.reshape(-1, D // Q)
            n_chosen = len({int(ids[r, c]) for r in range(ids.shape[0]) for c in range(16) if wts[r, c] > 0})
            print(f"PROG F={os.environ['GLM53_MOE_PROG_F']}, {tokens} tokens: {len(marked)} of the {n_chosen} chosen "
                  f"experts read with their correction", flush=True)  # fmt: skip
        want, chosen = sliced_reference(x.float(), ids, wts, ref_w1, ref_w2, E=e_col, S=s_col)
        cases.append((tokens, d_x, w, want, chosen))

    # MOE_EXPERTS_COLUMN_AB="NAME=a/b,..": each case's y with the env switch NAME at a and at b (the program cache
    # cleared in between), compared bit for bit (then NAME back as it was); 64 / 128 tokens also y by mesh row
    # (row_tokens 8 / 16, the model's).  MOE_EXPERTS_COLUMN_AB_TIME=N: and each case's time (trace) at a and at b, N
    # rounds a / b in turn
    ab_rounds = int(os.environ.get("MOE_EXPERTS_COLUMN_AB_TIME", "0"))
    for item in (v for v in os.environ.get("MOE_EXPERTS_COLUMN_AB", "").split(",") if v):
        name, values = item.split("=")
        saved_ab = os.environ.get(name)
        outs, times = {}, {}
        for rnd in range(max(1, ab_rounds)):
            for v in values.split("/"):
                os.environ[name] = v
                mesh_device.clear_program_cache()
                for c, (tokens, d_x, w, want, chosen) in enumerate(cases):  # (by case: two may have the same tokens)
                    for rt in (0, tokens // 8) if tokens in (64, 128) else (0,):
                        run_ab = lambda: ttnn.experimental.deepseek_prefill.moe_decode_experts(  # noqa: E731
                            d_x, w, nd1, nd2, e_col, s_col, LIMIT, row_tokens=rt
                        )
                        if rnd == 0:
                            y = run_ab()
                            outs[(v, c, rt)] = first(y)[0, 0]
                            ttnn.deallocate(y)
                        if ab_rounds:
                            times.setdefault((v, c, rt), []).append(traced(mesh_device, run_ab) * 1e6)
        if saved_ab is None:
            os.environ.pop(name)
        else:
            os.environ[name] = saved_ab
        mesh_device.clear_program_cache()
        va, vb = values.split("/")
        for c, (tokens, d_x, w, want, chosen) in enumerate(cases):
            for rt in (0, tokens // 8) if tokens in (64, 128) else (0,):
                ya, yb = outs[(va, c, rt)], outs[(vb, c, rt)]
                same = torch.equal(ya, yb)
                print(
                    f"A/B {name} {va} vs {vb}, {tokens} tokens, row_tokens {rt}: "
                    f"{'bit for bit' if same else 'max |d| %.3e' % (ya - yb).abs().max().item()}"
                    + (
                        f", pcc vs torch {torch.corrcoef(torch.stack([yb.flatten(), want.flatten()]))[0, 1].item():.7f}"
                        f", relative error vs torch {va} {((ya - want).norm() / want.norm()).item():.3e} / {vb} "
                        f"{((yb - want).norm() / want.norm()).item():.3e}"
                        if rt == 0
                        else ""
                    ),
                    flush=True,
                )
                if ab_rounds:
                    ta, tb = times[(va, c, rt)], times[(vb, c, rt)]
                    med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
                    print(
                        f"  time {name}={va}: {med(ta):.1f} us, {name}={vb}: {med(tb):.1f} us "
                        f"({med(tb) - med(ta):+.1f}; "
                        f"rounds {', '.join(f'{x:.1f}' for x in ta)} / {', '.join(f'{x:.1f}' for x in tb)})",
                        flush=True,
                    )
    for tokens, d_x, w, want, chosen in cases:
        y = ttnn.experimental.deepseek_prefill.moe_decode_experts(d_x, w, nd1, nd2, e_col, s_col, LIMIT)
        got = first(y)[0, 0]
        ttnn.deallocate(y)
        diff = (got - want).abs()
        pcc = torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1].item() if chosen else 1.0
        print(
            f"column, {tokens} tokens, {chosen} of its experts chosen ({-(-chosen // (64 // t_col))}+ items, last "
            f"{chosen - (-(-chosen // 8) - 1) * 8 if tokens <= 32 and chosen else '-'}): pcc "
            f"{pcc:.7f}, max |d| {diff.max().item():.3e} (max |y| {want.abs().max().item():.3e}), rows past the "
            f"tokens zero {bool((got[tokens:] == 0).all())}",
            flush=True,
        )
        if not os.environ.get("MOE_EXPERTS_STREAM_SKIP"):  # (a stream reading nothing: the times only)
            assert pcc > 0.9999
            assert (got[tokens:] == 0).all()
        if tokens in (64, 128) and not zones:  # y by mesh row (row_tokens): the same rows, each row's at its tile's top
            rt = tokens // 8
            yp = ttnn.experimental.deepseek_prefill.moe_decode_experts(
                d_x, w, nd1, nd2, e_col, s_col, LIMIT, row_tokens=rt
            )
            gp = first(yp)[0, 0]
            ttnn.deallocate(yp)
            lay = torch.zeros(8 * 32, D)
            for r in range(8):
                lay[32 * r : 32 * r + rt] = got[r * rt : (r + 1) * rt].float()
            same = torch.equal(gp.float(), lay)
            print(f"  row_tokens {rt}: [{gp.shape[0]}, {gp.shape[1]}], equal to the rows moved {same}", flush=True)
            assert same
            sec = traced(
                mesh_device,
                lambda: ttnn.experimental.deepseek_prefill.moe_decode_experts(
                    d_x, w, nd1, nd2, e_col, s_col, LIMIT, row_tokens=rt
                ),
            )
            print(f"  row_tokens {rt}: {sec * 1e6:.1f} us", flush=True)
        if zones:
            for _ in range(3):
                ttnn.deallocate(
                    ttnn.experimental.deepseek_prefill.moe_decode_experts(d_x, w, nd1, nd2, e_col, s_col, LIMIT)
                )
            ttnn.synchronize_device(mesh_device)
            continue
        sec = traced(
            mesh_device,
            lambda: ttnn.experimental.deepseek_prefill.moe_decode_experts(d_x, w, nd1, nd2, e_col, s_col, LIMIT),
        )
        print(f"  the op: {sec * 1e6:.1f} us", flush=True)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_moe_experts_real_routing(mesh_device):
    """The column layout's op time (trace) on real routing: GLM53_TEST_ROUTING_FILE's cases (a torch.save'd list of
    dicts "tokens", "logits", "bias", "base", "label": the records' router logits and correction bias of some MoE
    layers, a verify step's tokens, each case the heaviest mesh column), random weights (GLM53_TEST_WDTYPE;
    GLM53_MOE_PROG_F: the progressive form's 2 slots each) and random x; 64 / 128 tokens a column by mesh row (row_tokens 8 / 16, the model's).  Times only (the
    routing and the op's values as test_moe_experts_column checks them).  MOE_EXPERTS_REAL_AB="NAME=a/b,..": first each
    case's y with the env switch NAME at a and at b (the program cache cleared in between), compared bit for bit.
    MOE_EXPERTS_REAL_PATHS (default "op"): the paths timed, each case: "op" the op in one call, "split" in calls of
    512 tokens (moe.py's `_routed_sliced` past EXPERTS_OP_CALL: the x and routing rows sliced, the parts concatenated),
    "prefill" moe_prefill.experts on the row-major x (`_routed_once_local`'s)."""
    import torch

    cases = torch.load(os.environ["GLM53_TEST_ROUTING_FILE"])
    only = [int(v) for v in os.environ.get("MOE_EXPERTS_REAL_TOKENS", "").split(",") if v]
    cases = [c for c in cases if not only or c["tokens"] in only]  # MOE_EXPERTS_REAL_TOKENS: these columns' sizes
    e_all, e_col, s_col = 288, 72, 256
    t_col = s_col // 32
    prog = os.environ.get("GLM53_MOE_PROG_F") is not None
    slots = 2 * e_col if prog else e_col
    rand = lambda shape, dt: ttnn.rand(  # noqa: E731
        shape, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, low=-0.05, high=0.05
    )
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()  # noqa: E731
    w1 = rand([1, 1, t_col * slots * D, 64], WDT)
    w2 = rand([1, 1, Q * slots * s_col, D // Q], WDT)
    nd1 = ttnn.to_memory_config(w1, row_sharded(mesh_device, 2, 16 * 32))
    nd2 = ttnn.to_memory_config(w2, row_sharded(mesh_device, D // Q // 32, slots * s_col))
    ttnn.deallocate(w1)
    ttnn.deallocate(w2)
    rep = dict(dims=(None, None), dtype=ttnn.float32)
    made = []  # per case: x, the routing, its chosen and marked counts
    for i, case in enumerate(cases):
        tokens = case["tokens"]
        g = torch.Generator().manual_seed(i)
        d_logits = put(mesh_device, case["logits"][None, None].float(), **rep)
        d_bias = put(mesh_device, case["bias"].view(1, 1, 1, e_all).float(), **rep)
        bases = torch.zeros(1, 1, 1, 8, dtype=torch.int32)
        bases[..., 0] = case["base"]
        d_base = put(mesh_device, bases, dims=(None, None), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        d_x = put(mesh_device, torch.randn(tokens, D, generator=g).to(torch.bfloat16)[None, None], dims=(None, None))
        w = ttnn.experimental.deepseek_prefill.moe_decode_route(d_logits, d_bias, d_base, tokens, K, e_col, 2.5)
        tile = first(w)[0, 0]
        ids, wts = tile[:, :16].contiguous().view(torch.int32), tile[:, 16:]
        chosen = len({int(ids[r, c]) for r in range(tokens) for c in range(16) if wts[r, c] > 0})
        marked = len(prog_marked(ids, wts, e_col)) if prog else 0
        made.append((case, d_x, w, chosen, marked))
        for t in (d_logits, d_bias, d_base):
            ttnn.deallocate(t)
    run_case = lambda d_x, w, tokens: ttnn.experimental.deepseek_prefill.moe_decode_experts(  # noqa: E731
        d_x, w, nd1, nd2, e_col, s_col, LIMIT, row_tokens=tokens // 8 if tokens in (64, 128) else 0
    )
    # MOE_EXPERTS_REAL_ZONES="c,..": those cases only, each run eagerly 3 times (for a kernel zone profile: its last
    # run), no A/B and no times
    zone_cases = [int(v) for v in os.environ.get("MOE_EXPERTS_REAL_ZONES", "").split(",") if v]
    for c in zone_cases:
        case, d_x, w, chosen, marked = made[c]
        for _ in range(3):
            y = run_case(d_x, w, case["tokens"])
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(y)
        print(f"real routing zones: case {c}: {case['label']}: {chosen} chosen, {marked} corrected", flush=True)
    if zone_cases:
        return
    for item in (v for v in os.environ.get("MOE_EXPERTS_REAL_AB", "").split(",") if v):
        name, values = item.split("=")
        saved_ab = os.environ.get(name)
        outs = {}
        for v in values.split("/"):
            os.environ[name] = v
            mesh_device.clear_program_cache()
            for c, (case, d_x, w, chosen, marked) in enumerate(made):
                y = run_case(d_x, w, case["tokens"])
                outs[(v, c)] = first(y)[0, 0]
                ttnn.deallocate(y)
        if saved_ab is None:
            os.environ.pop(name)
        else:
            os.environ[name] = saved_ab
        mesh_device.clear_program_cache()
        va, vb = values.split("/")
        same = [torch.equal(outs[(va, c)], outs[(vb, c)]) for c in range(len(made))]
        for c, (case, d_x, w, chosen, marked) in enumerate(made):
            if not same[c]:
                d = (outs[(va, c)] - outs[(vb, c)]).abs().max().item()
                print(f"real routing A/B {name} {va} vs {vb}: {case['label']}: max |d| {d:.3e}", flush=True)
        print(f"real routing A/B {name} {va} vs {vb}: {sum(same)} of {len(made)} cases bit for bit", flush=True)
    from models.demos.blackhole.glm53_flash.ttnn import moe_prefill

    def split(d_x, w, tokens, call=512):  # (moe.py's _routed_sliced past EXPERTS_OP_CALL)
        if tokens <= call:  # (one call: a whole-tensor slice is the tensor itself, not to be freed here)
            return run_case(d_x, w, tokens)
        parts = []
        for r0 in range(0, tokens, call):
            r1 = min(tokens, r0 + call)
            xs = ttnn.slice(d_x, [0, 0, r0, 0], [1, 1, r1, D])
            ws = ttnn.slice(w, [0, 0, r0, 0], [1, 1, r1, 32])
            parts.append(run_case(xs, ws, r1 - r0))
            ttnn.deallocate(xs)
            ttnn.deallocate(ws)
        out = ttnn.concat(parts, dim=2)
        for part in parts:
            ttnn.deallocate(part)
        return out

    paths = os.environ.get("MOE_EXPERTS_REAL_PATHS", "op").split(",")
    totals = {}
    for case, d_x, w, chosen, marked in made:
        tokens = case["tokens"]
        x_rm = ttnn.to_layout(d_x, ttnn.ROW_MAJOR_LAYOUT) if "prefill" in paths else None
        fns = {
            "op": lambda: run_case(d_x, w, tokens),
            "split": lambda: split(d_x, w, tokens),
            "prefill": lambda: moe_prefill.experts(mesh_device, x_rm, w, nd1, nd2, e_col, LIMIT, row_major=True),
        }
        for path in paths:
            sec = traced(mesh_device, fns[path])
            totals.setdefault((path, tokens), []).append(sec * 1e6)
            tag = "" if paths == ["op"] else f" [{path}]"
            print(
                f"real routing{tag}: {case['label']}: {chosen} chosen, {marked} read with their correction: "
                f"{sec * 1e6:.1f} us",
                flush=True,
            )
        if x_rm is not None:
            ttnn.deallocate(x_rm)
    for case, d_x, w, chosen, marked in made:
        ttnn.deallocate(d_x)
        ttnn.deallocate(w)
    for (path, tokens), v in totals.items():
        print(
            f"real routing [{path}], {tokens} tokens a column: mean {sum(v) / len(v):.1f} us over {len(v)} cases",
            flush=True,
        )
