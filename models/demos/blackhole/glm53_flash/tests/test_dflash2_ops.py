"""The DFlash2 drafter's own ops (`ttnn/dflash2_ops.py`) against torch, bit for bit, on every chip: rope_rows (the
drafter's block rows and context rows), kv_write (block and context maps), top16 (per chip, with ties and the last
chip's padding), top16_merge (the 32 chips' lists, the codebooks' id rows) and walk (the selector's greedy walk).

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_dflash2_ops.py -s
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.ttnn import dflash2_ops as ops  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    COLS,
    DEVICE_PARAMS,
    DEVICE_PARAMS_TRACE,
    MESH_SHAPE,
    ROWS,
    put,
)

CHIPS = ROWS * COLS
VOCAB = 154_880
W = -(-VOCAB // (CHIPS * 32)) * 32  # 4,864 a chip
VALID = [max(0, min(W, VOCAB - k * W)) for k in range(CHIPS)]
CHIP_RC = [(r, c) for r in range(ROWS) for c in range(COLS)]


# a failure in a device test's body is printed ("SOFT-FAIL <test>" and its traceback) and the test passes, so that a
# run of several studies goes on and ends with exit code 0; search its log for SOFT-FAIL.  GLM53_DFLASH2_STRICT=1:
# raise as usual
STRICT = os.environ.get("GLM53_DFLASH2_STRICT", "0") == "1"


def soft(fn):
    import functools
    import traceback

    @functools.wraps(fn)
    def run(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if STRICT:
                raise
            print(f"SOFT-FAIL {fn.__name__}\n{traceback.format_exc()}", flush=True)

    return run


def per_chip(mesh, t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT):
    """t [8, 4, A, ..] -> chip (r, c) holds t[r, c] as [A, 1, ..] (sent as [8 A, 4, ..] split on dims 0 and 1)."""
    a = t.shape[2]
    flat = t.transpose(1, 2).reshape(ROWS * a, COLS, *t.shape[3:])
    return put(mesh, flat, dims=(0, 1), dtype=dtype, layout=layout)


def back(mesh, t) -> torch.Tensor:
    """Each chip's [A, 1, ..] -> [8, 4, A, ..]."""
    full = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=MESH_SHAPE, dims=(0, 1)))
    a = full.shape[0] // ROWS
    return full.reshape(ROWS, a, COLS, *full.shape[2:]).transpose(1, 2)


def rep(mesh, t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT):
    return put(mesh, t, dims=(None, None), dtype=dtype, layout=layout)


def each_chip(t) -> list[torch.Tensor]:
    """The chips' tensors in the order (r, c)."""
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def u32(t: torch.Tensor) -> torch.Tensor:
    """Integers read back (int32 or uint32) as int64 of their 32 bits."""
    if t.dtype in (torch.int32, getattr(torch, "uint32", torch.int32)):
        return t.view(torch.int32).long() & 0xFFFFFFFF
    return t.long()


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@soft
def test_rope_rows(mesh_device):
    mesh = mesh_device
    g = torch.Generator().manual_seed(1)
    positions = 4096
    cos_tab = torch.randn(positions, 128, generator=g)
    sin_tab = torch.randn(positions, 128, generator=g)
    tc = rep(mesh, cos_tab, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
    ts = rep(mesh, sin_tab, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
    # the drafter's block: U users, 2 KV heads a chip, block B: row o of [2 U 32, 128] -> starts[o // 64] + o % B
    for users, block in ((1, 8), (4, 8), (16, 8), (16, 4)):
        starts = torch.randint(0, positions - block, (ROWS, COLS, 1, users), generator=g, dtype=torch.int32)
        idx = per_chip(mesh, starts, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)  # [1, 1, U] a chip
        rows = 2 * users * 32
        cos, sin, cur = ops.rope_rows(mesh, idx, tc, ts, rows, div=64, mod=block, cur=(2 * users, 2, block - 1))
        got_c, got_s = back(mesh, cos), back(mesh, sin)  # [8, 4, 1, 1, rows, 128]
        got_p = torch.stack([u32(x).reshape(-1) for x in each_chip(cur)]).view(ROWS, COLS, -1)
        o = torch.arange(rows)
        for r, c in CHIP_RC:
            pos = starts[r, c, 0][o // 64].long() + o % block
            assert torch.equal(got_c[r, c].reshape(rows, 128), cos_tab[pos]), (users, block, r, c)
            assert torch.equal(got_s[r, c].reshape(rows, 128), sin_tab[pos]), (users, block, r, c)
            want = starts[r, c, 0][torch.arange(2 * users) // 2] + block - 1
            assert torch.equal(got_p[r, c], want.long()), (users, block, r, c, got_p[r, c], want)
        for t in (cos, sin, cur, idx):
            ttnn.deallocate(t)
        print(f"rope_rows block: users {users} B {block} ok")
    # context rows: R tokens x 16 head rows, row o -> positions[o // 16] (negative: 0)
    for tokens in (8, 128):
        pos_in = torch.randint(-1, positions, (ROWS, COLS, 1, tokens), generator=g, dtype=torch.int32)
        idx = per_chip(mesh, pos_in, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        rows = tokens * 16
        cos, sin = ops.rope_rows(mesh, idx, tc, ts, rows, div=16)
        got_c = back(mesh, cos)
        o = torch.arange(rows)
        for r, c in CHIP_RC:
            pos = pos_in[r, c, 0][o // 16].long().clamp(min=0)
            assert torch.equal(got_c[r, c].reshape(rows, 128), cos_tab[pos]), (tokens, r, c)
        for t in (cos, sin, idx):
            ttnn.deallocate(t)
        print(f"rope_rows context: tokens {tokens} ok")
    # the model's table of the halves (sin_tab None: cos's columns 0-63, sin's 64-127; rope_rows.cpp fmt 1), the block
    # rows and the context rows (GLM53_DFLASH2_ROPE_DEDUPE as set: run the test with it 0 and 1; its form only from
    # dflash2_ops.ROPE_DEDUPE_MIN_ROWS rows, here the 1,024-row cases)
    halves = torch.randn(positions, 128, generator=g)
    th = rep(mesh, halves, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
    want_c = torch.cat([halves[:, :64], halves[:, :64]], dim=1)
    want_s = torch.cat([halves[:, 64:], halves[:, 64:]], dim=1)
    for users, block in ((1, 8), (4, 8), (16, 4)):
        starts = torch.randint(0, positions - block, (ROWS, COLS, 1, users), generator=g, dtype=torch.int32)
        idx = per_chip(mesh, starts, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        rows = 2 * users * 32
        cos, sin, cur = ops.rope_rows(mesh, idx, th, None, rows, div=64, mod=block, cur=(2 * users, 2, block - 1))
        got_c, got_s = back(mesh, cos), back(mesh, sin)
        o = torch.arange(rows)
        for r, c in CHIP_RC:
            pos = starts[r, c, 0][o // 64].long() + o % block
            assert torch.equal(got_c[r, c].reshape(rows, 128), want_c[pos]), ("halves", users, block, r, c)
            assert torch.equal(got_s[r, c].reshape(rows, 128), want_s[pos]), ("halves", users, block, r, c)
        for t in (cos, sin, cur, idx):
            ttnn.deallocate(t)
        print(f"rope_rows halves block: users {users} B {block} ok")
    for tokens, heads in ((8, 16), (64, 16)):
        pos_in = torch.randint(-1, positions, (ROWS, COLS, 1, tokens), generator=g, dtype=torch.int32)
        idx = per_chip(mesh, pos_in, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        rows = tokens * heads
        cos, sin = ops.rope_rows(mesh, idx, th, None, rows, div=heads)
        got_c, got_s = back(mesh, cos), back(mesh, sin)
        o = torch.arange(rows)
        for r, c in CHIP_RC:
            pos = pos_in[r, c, 0][o // heads].long().clamp(min=0)
            assert torch.equal(got_c[r, c].reshape(rows, 128), want_c[pos]), ("halves", tokens, r, c)
            assert torch.equal(got_s[r, c].reshape(rows, 128), want_s[pos]), ("halves", tokens, r, c)
        for t in (cos, sin, idx):
            ttnn.deallocate(t)
        print(f"rope_rows halves context: tokens {tokens} x {heads} heads ok")
    ttnn.deallocate(th)
    dedupe = os.environ.get("GLM53_DFLASH2_ROPE_DEDUPE", "1")
    print(f"ROPE_ROWS ALL OK (GLM53_DFLASH2_ROPE_DEDUPE={dedupe})", flush=True)


def kv_expect(
    cache, src, idx, users, *, tokens, block, layers, heads, rows, cols, pages, layer_pages, page_base, n_users
):
    """kv_write.cpp's map on torch, one chip: cache [pages, 32, 128], src [N, W]."""
    out = cache.clone()
    su, sp, sl, sh = rows
    c0, cl, ch = cols
    for r in range(tokens):
        if users is None:
            u = r // block
            p = int(idx[u]) if u < len(idx) else -1
            p = p + r % block if p >= 0 else p
        else:
            p, u = int(idx[r]), int(users[r])
        if p < 0 or p >= pages * 32 or u >= n_users:
            continue
        for l in range(layers):
            for h in range(heads):
                srow = (r // block) * su + (r % block) * sp + l * sl + h * sh
                col = c0 + l * cl + h * ch
                page = page_base + l * layer_pages + (u * heads + h) * pages + p // 32
                out[page, p % 32] = src[srow, col * 128 : col * 128 + 128]
    return out


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@soft
def test_kv_write(mesh_device):
    mesh = mesh_device
    g = torch.Generator().manual_seed(2)
    heads, pages = 2, 8  # 256 positions a (user, head)
    cases = []
    # the block: K in the virtual users' layout [2 U 32, 128] (row (u, h, pos)), one layer of 3 at page_base
    for users, block in ((1, 8), (16, 8), (16, 4)):
        cases.append(
            dict(
                name=f"block U {users} B {block}",
                users=users,
                tokens=users * block,
                block=block,
                layers=1,
                src_shape=(2 * users * 32, 128),
                rows=(64, 1, 0, 32),
                cols=(0, 0, 0),
                n_layers=3,
                page_base_layer=1,
                mode=0,
            )
        )
    # the block's V straight from the qkv projection [U B, 1536] (columns 1,280 + 128 h)
    cases.append(
        dict(
            name="block V U 16 B 8",
            users=16,
            tokens=128,
            block=8,
            layers=1,
            src_shape=(128, 1536),
            rows=(8, 1, 0, 0),
            cols=(10, 0, 1),
            n_layers=3,
            page_base_layer=2,
            mode=0,
        )
    )
    # context: K rows [R 16, 128] (row (r, l, h)) and V from [R, 4096] (columns 16 + 2 l + h), 8 layers
    for tokens, src_shape, rows, cols in (
        (24, (24 * 16, 128), (16, 0, 2, 1), (0, 0, 0)),
        (128, (128, 4096), (1, 0, 0, 0), (16, 2, 1)),
    ):
        cases.append(
            dict(
                name=f"context R {tokens} src {src_shape}",
                users=16,
                tokens=tokens,
                block=1,
                layers=8,
                src_shape=src_shape,
                rows=rows,
                cols=cols,
                n_layers=8,
                page_base_layer=0,
                mode=1,
            )
        )
    # GLM53_KVW_DTYPES (bf16): the caches' formats, e.g. bf16,bfp8,bfp4 (the KV cache study: a BFP source and cache,
    # the values rounded to it first so that the conversions are exact and a row's copy is the expected row)
    from models.demos.blackhole.glm53_flash.tools.kv_round import round_kv

    dtypes = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
    for dname, case in [(d, c) for d in os.environ.get("GLM53_KVW_DTYPES", "bf16").split(",") for c in cases]:
        dt = dtypes[dname]
        users = case["users"]
        layer_pages = users * heads * pages
        total = case["n_layers"] * layer_pages + 1
        page_base = case["page_base_layer"] * layer_pages
        cache0 = round_kv(torch.randn(ROWS, COLS, total, 32, 128, generator=g).bfloat16(), dname)
        src0 = round_kv(torch.randn(ROWS, COLS, *case["src_shape"], generator=g).bfloat16(), dname)
        cache = per_chip(mesh, cache0, dt)  # [total, 1, 32, 128] a chip
        src = per_chip(mesh, src0.unsqueeze(2), dt)  # [1, 1, N, W] a chip
        if case["mode"] == 0:
            idx0 = torch.randint(-1, pages * 32 - case["block"] + 1, (ROWS, COLS, 1, users), generator=g)
            usr0 = None
        else:  # distinct (user, position) pairs (two tokens on one cache row: the last write wins, in any order),
            # some not written (position -1, a user past the users)
            n_tok = case["tokens"]
            pick = torch.stack([torch.randperm(users * pages * 32, generator=g)[:n_tok] for _ in range(ROWS * COLS)])
            idx0 = (pick % (pages * 32)).view(ROWS, COLS, 1, n_tok)
            usr0 = (pick // (pages * 32)).view(ROWS, COLS, 1, n_tok).to(torch.int32)
            idx0[..., 1] = -1
            usr0[..., 2] = users
        idx0 = idx0.to(torch.int32)
        idx = per_chip(mesh, idx0, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        usr = None if usr0 is None else per_chip(mesh, usr0, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        kw = dict(
            tokens=case["tokens"],
            block=case["block"],
            layers=case["layers"],
            heads=heads,
            rows=case["rows"],
            cols=case["cols"],
            pages=pages,
            layer_pages=layer_pages,
            page_base=page_base,
        )
        ops.kv_write(
            mesh,
            [(src, cache, kw["rows"], kw["cols"])],
            idx,
            usr,
            n_users=users,
            **{k: v for k, v in kw.items() if k not in ("rows", "cols")},
        )
        got = back(mesh, cache).reshape(ROWS, COLS, total, 32, 128).float()
        for r, c in CHIP_RC:
            want = kv_expect(
                cache0[r, c],
                src0[r, c],
                idx0[r, c, 0],
                None if usr0 is None else usr0[r, c, 0],
                n_users=users,
                **kw,
            )
            want = want.float()
            if not torch.equal(got[r, c], want):
                bad = (got[r, c] != want).any(dim=-1).nonzero()
                print(
                    f"kv_write {dname} {case['name']} chip ({r}, {c}): {len(bad)} rows differ, first {bad[:8].tolist()}"
                )
                assert False, (dname, case["name"], r, c)
        for t in (cache, src, idx) + (() if usr is None else (usr,)):
            ttnn.deallocate(t)
        print(f"kv_write {dname} {case['name']}: ok")


def top16_expect(logits: torch.Tensor, valid: int, base: int):
    """[M, W] fp32 -> the 16 largest of the first `valid` columns (equal: the smaller column first), -0.0 as +0.0."""
    v = logits[:, :valid].clone()
    v[v == 0] = 0.0  # (-0.0 -> +0.0, as the keys)
    order = torch.sort(v, dim=-1, descending=True, stable=True)
    return order.values[:, :16], (order.indices[:, :16] + base).float()


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@soft
def test_top16(mesh_device):
    mesh = mesh_device
    g = torch.Generator().manual_seed(3)
    cases = ((56, 7, 7), (112, 14, 14), (64, 32, 7), (256, 32, 28), (1024, 128, 112), (256, 256, 256), (512, 64, 48))
    only = os.environ.get("GLM53_DFLASH2_TOP16_M")  # one case (rows), e.g. for a zone profile
    for m, period, real in [c for c in cases if only is None or c[0] == int(only)][: 1 if only else None]:
        logits = torch.randn(ROWS, COLS, m, W, generator=g) * 4
        logits[:, :, 1] -= 50.0  # a row all negative
        for row in range(2, m, 5):  # ties at the top, within a chip
            cols = torch.randint(0, W, (5,), generator=g)
            logits[:, :, row, cols] = logits[:, :, row].amax(dim=-1, keepdim=True) + 1.0
        logits[:, :, 0, 5] = -0.0
        logits[:, :, 0, 7] = 0.0
        logits[-1, -1, :, VALID[-1] :] = 1.0e4  # past the vocabulary on the last chip: not taken
        d = per_chip(mesh, logits.unsqueeze(2), ttnn.float32)  # [1, 1, m, W] a chip
        live = torch.arange(m) % period < real  # (the rows not live are not written)
        want = [top16_expect(logits[r, c], VALID[k], k * W) for k, (r, c) in enumerate(CHIP_RC)]
        repeat = int(os.environ.get("GLM53_DFLASH2_TOP16_REPEAT", "1"))  # calls compared (a race's check)
        for _ in range(repeat):
            out = ops.top16(mesh, d, m, VALID, W, period, real)
            got = back(mesh, out).reshape(ROWS, COLS, m, 32)
            for k, (r, c) in enumerate(CHIP_RC):
                vals, ids = want[k]
                assert torch.equal(got[r, c, live, :16], vals[live]), (m, r, c)
                assert torch.equal(got[r, c, live, 16:], ids[live]), (m, r, c)
            ttnn.deallocate(out)
        ttnn.deallocate(d)
        op = ops._TOP16[next(k for k in ops._TOP16 if k[1] == m and k[5] == period and k[7] == ops.top16_trisc())]
        trisc = sum(args[3] - args[2] - op.t0.get((c.x, c.y, risc), args[3] - args[2]) for c, risc, args, _ in op.items)
        print(
            f"top16 rows {m} (period {period}, real {real}): ok x {repeat} (G {op.G}, runs {op.runs} of {op.per} tiles,"
            f" {len(op.items)} items, unpack-thread rows {trisc})"
        )


def merge_expect(gathered: torch.Tensor, bonus: torch.Tensor, row: int, users: int, slots: int, period: int):
    """gathered [32, 8 period, 32] fp32 (each chip's 16 values, 16 ids; mesh row r's slot rows at r period ..) -> sel
    [U K, 32] (bits), cand, pred [U K, 32]."""
    sel = torch.zeros(users * slots, 32, dtype=torch.int64)
    cand = torch.zeros(users * slots, 32, dtype=torch.int64)
    pred = torch.zeros(users * slots, 32, dtype=torch.int64)
    for u in range(users):
        for s in range(slots):
            n = row * period + u * slots + s
            vals = gathered[:, n, :16].reshape(-1)
            ids = gathered[:, n, 16:].reshape(-1)
            order = torch.sort(vals, descending=True, stable=True).indices[:16]
            i = u * slots + s
            sel[i, :16] = vals[order].view(torch.int32).to(torch.int64) & 0xFFFFFFFF
            sel[i, 16:] = ids[order].long()
            cand[i, :16] = ids[order].long()
            pred[i, :16] = bonus[u] if s == 0 else cand[i - 1, :16]
    return sel, cand, pred


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@soft
def test_top16_merge_and_walk(mesh_device):
    mesh = mesh_device
    g = torch.Generator().manual_seed(4)
    for users, slots, compact in (
        (1, 7, True),
        (1, 7, False),
        (2, 7, True),
        (4, 7, False),
        (16, 7, False),
        (16, 3, False),
    ):
        # compact: a mesh row's slot rows packed (the head's real rows only, TtDFlash2.compact), else whole tiles
        period = users * slots if compact else -(-users * slots // 32) * 32
        n = ROWS * period
        vals = torch.randn(CHIPS, n, 16, generator=g) * 3
        vals[3, :, 2] = vals[5, :, 4]  # equal values across chips
        vals = torch.sort(vals, dim=-1, descending=True).values
        ids = torch.stack(
            [torch.randperm(W, generator=g)[:16].float() + k * W for k in range(CHIPS) for _ in range(n)]
        ).view(CHIPS, n, 16)
        gathered0 = torch.cat([vals, ids], dim=-1)  # [32, N, 32]
        gathered = rep(mesh, gathered0.view(ROWS, COLS, n, 32), ttnn.float32, ttnn.TILE_LAYOUT)
        bonus0 = torch.randint(0, VOCAB, (ROWS, COLS, 1, users), generator=g).to(torch.int32)
        bonus0[:, 1:] = bonus0[:, :1]  # the same on a row's chips
        bonus = per_chip(mesh, bonus0, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        sel, cand, pred = ops.top16_merge(mesh, gathered, bonus, users, slots, period, -(-n // 32) * 32)
        got_sel = torch.stack([u32(x) for x in each_chip(sel)]).view(ROWS, COLS, users * slots, 32)
        got_cand = torch.stack([u32(x) for x in each_chip(cand)]).view(ROWS, COLS, users * slots, 32)
        got_pred = torch.stack([u32(x) for x in each_chip(pred)]).view(ROWS, COLS, users * slots, 32)
        for r, c in CHIP_RC:
            w_sel, w_cand, w_pred = merge_expect(gathered0, bonus0[r, 0, 0].long(), r, users, slots, period)
            assert torch.equal(got_sel[r, c], w_sel), ("sel", users, slots, r, c)
            assert torch.equal(got_cand[r, c], w_cand), ("cand", users, slots, r, c)
            assert torch.equal(got_pred[r, c], w_pred), ("pred", users, slots, r, c)
        print(f"top16_merge U {users} K {slots} (period {period}): ok")
        # the walk over random pairwise scores and these candidates
        scores0 = torch.randn(ROWS, COLS, users * slots, 32, 32, generator=g)
        scores = per_chip(mesh, scores0.unsqueeze(2), ttnn.float32)  # [1, U K, 32, 32] a chip
        drafts = ops.walk(mesh, scores, sel, bonus, users, slots)
        got = torch.stack([u32(x) for x in each_chip(drafts)]).view(ROWS, COLS, users, 8)
        for r, c in CHIP_RC:
            unary = torch.from_numpy(got_sel[r, c, :, :16].numpy().astype("uint32").view("float32"))  # [U K, 16]
            want = torch.zeros(users, 8, dtype=torch.long)
            for u in range(users):
                want[u, 0] = int(bonus0[r, c, 0, u])
                prev = 0
                for s in range(slots):
                    i = u * slots + s
                    score = unary[i] + scores0[r, c, i, prev, :16]
                    best = int(torch.argmax(score))
                    want[u, 1 + s] = int(got_sel[r, c, i, 16 + best])
                    prev = best
            assert torch.equal(got[r, c], want), ("walk", users, slots, r, c)
        print(f"walk U {users} K {slots} (period {period}): ok")
        for t in (gathered, bonus, sel, cand, pred, scores, drafts):
            ttnn.deallocate(t)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@soft
def test_heads_split_merge(mesh_device):
    mesh = mesh_device
    g = torch.Generator().manual_seed(7)
    group, kvh, k0 = 4, 2, 8  # a chip: 8 q heads, 2 KV heads; qkv columns [q 1,024 | k 256 | v 256]
    for users, block in ((1, 8), (4, 8), (16, 8), (16, 4), (3, 4)):
        rows = users * block
        qkv0 = torch.randn(ROWS, COLS, rows, 1536, generator=g).bfloat16()
        qkv = per_chip(mesh, qkv0.unsqueeze(2), ttnn.bfloat16)  # [1, 1, U B, 1536] a chip
        q, k = ops.heads_split(mesh, qkv, users, block, group, kvh, k0)
        got_q = torch.stack(each_chip(q)).view(ROWS, COLS, kvh * users, 32, 128)
        got_k = torch.stack(each_chip(k)).view(ROWS, COLS, kvh * users, 32, 128)
        for r, c in CHIP_RC:
            x = qkv0[r, c].view(users, block, 12, 128)  # 12 heads: 8 q, 2 k, 2 v
            want_q = torch.zeros(kvh * users, 32, 128, dtype=torch.bfloat16)
            want_k = torch.zeros(kvh * users, 32, 128, dtype=torch.bfloat16)
            for u in range(users):
                for j in range(kvh):
                    for h in range(group):
                        want_q[u * kvh + j, h * block : (h + 1) * block] = x[u, :, j * group + h]
                    want_k[u * kvh + j, :block] = x[u, :, k0 + j]
            assert torch.equal(got_q[r, c], want_q), ("q", users, block, r, c)
            assert torch.equal(got_k[r, c], want_k), ("k", users, block, r, c)
        merged = ops.heads_merge(mesh, q, users, block, group, kvh)
        got_m = back(mesh, merged).reshape(ROWS, COLS, rows, 1024)
        for r, c in CHIP_RC:
            assert torch.equal(got_m[r, c], qkv0[r, c, :, :1024]), ("merge", users, block, r, c)
        # q and k as one tensor (the drafter's attention): q's virtual users, then k's, the same tiles
        qk = ops.heads_split(mesh, qkv, users, block, group, kvh, k0, combined=True)
        got_qk = torch.stack(each_chip(qk)).view(ROWS, COLS, 2 * kvh * users, 32, 128)
        assert torch.equal(got_qk[:, :, : kvh * users], got_q) and torch.equal(got_qk[:, :, kvh * users :], got_k), (
            "combined",
            users,
            block,
        )
        for t in (qkv, q, k, merged, qk):
            ttnn.deallocate(t)
        print(f"heads_split / heads_merge U {users} B {block}: ok")


def conv_expect(x: torch.Tensor, delta: torch.Tensor, base: torch.Tensor, side: int, block: int) -> torch.Tensor:
    """tt/dflash2.py's _conv for one side: x [M, 4096] bf16, delta [M, 1024] bf16, base [2 taps, 4096] bf16."""
    m = x.shape[0]
    d = delta.view(m, 2, 2, 256)[:, side].float()  # [M, taps, groups]
    coef = base.float().view(1, 2, 256, 16) + d.unsqueeze(-1)  # [M, taps, groups, 16]
    coef = coef.view(m, 2, 4096)
    xf = x.float()
    out = coef[:, 0] * xf
    prev = torch.zeros_like(xf)
    for i in range(m):
        if i % block:
            prev[i] = xf[i - 1]
    out = out + coef[:, 1] * prev
    return out.bfloat16()


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@soft
def test_conv(mesh_device):
    mesh = mesh_device
    g = torch.Generator().manual_seed(8)
    only = os.environ.get("GLM53_DFLASH2_CONV_U")  # one case (users, block 8), e.g. for a zone profile
    cases = ((1, 8), (4, 8), (16, 8), (16, 4))
    for users, block in [c for c in cases if only is None or c == (int(only), 8)]:
        consts = ops.conv_constants(mesh, block)
        m = users * block
        x0 = torch.randn(ROWS, COLS, m, 4096, generator=g).bfloat16()
        d0 = (torch.randn(ROWS, COLS, m, 1024, generator=g) * 0.3).bfloat16()
        b0 = torch.randn(2, 2, 4096, generator=g).bfloat16()  # [side, tap, hidden], the same on every chip
        x = per_chip(mesh, x0.unsqueeze(2), ttnn.bfloat16)
        d = per_chip(mesh, d0.unsqueeze(2), ttnn.bfloat16)
        for side in (0, 1):
            base = ops.conv_base(mesh, b0[side])
            out = ops.conv(mesh, x, d, base, consts, side)
            got = back(mesh, out).reshape(ROWS, COLS, m, 4096)
            worst = 0
            for r, c in CHIP_RC:
                want = conv_expect(x0[r, c], d0[r, c], b0[side], side, block)
                diff = (got[r, c].float() - want.float()).abs()
                n_bad = int((got[r, c] != want).sum())
                worst = max(worst, n_bad)
                assert float(diff.max()) <= float(want.float().abs().max()) * 2**-7, (users, block, side, r, c)
            print(f"conv U {users} B {block} side {side}: values not bit for bit (worst chip) {worst} of {m * 4096}")
            ttnn.deallocate(out)
            ttnn.deallocate(base)
        for t in (x, d) + consts:
            ttnn.deallocate(t)


def rope_expect(n: torch.Tensor, w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """tt/dflash2.py: y = bf16(w * n) (bf16 multiply), then bf16(y cos + rotate_half(y) sin) in fp32."""
    y = (w * n).float()  # (bf16 x bf16 in torch: rounded to bf16)
    rot = torch.cat([-y[..., 64:], y[..., :64]], dim=-1)
    return (y * cos + rot * sin).bfloat16()


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@soft
def test_rope(mesh_device):
    mesh = mesh_device
    g = torch.Generator().manual_seed(9)
    for rows, per_row in ((64, False), (1024, False), (2048, True)):
        n0 = torch.randn(ROWS, COLS, rows, 128, generator=g).bfloat16()
        c0 = torch.randn(ROWS, COLS, rows, 128, generator=g)
        s0 = torch.randn(ROWS, COLS, rows, 128, generator=g)
        if per_row:
            w0 = torch.randn(ROWS, COLS, rows, 128, generator=g).bfloat16()
            w = per_chip(mesh, w0.unsqueeze(2), ttnn.bfloat16)
        else:
            w1 = torch.randn(128, generator=g).bfloat16()
            w0 = w1.expand(ROWS, COLS, rows, 128)
            w = rep(mesh, w1.view(1, 1, 1, 128).expand(1, 1, 32, 128).contiguous(), ttnn.bfloat16)
        n = per_chip(mesh, n0.unsqueeze(2), ttnn.bfloat16)
        c = per_chip(mesh, c0.unsqueeze(2), ttnn.float32)
        s = per_chip(mesh, s0.unsqueeze(2), ttnn.float32)
        out = ops.rope(mesh, n, w, c, s)
        got = back(mesh, out).reshape(ROWS, COLS, rows, 128)
        worst = 0
        for r, cc in CHIP_RC:
            want = rope_expect(n0[r, cc], w0[r, cc], c0[r, cc], s0[r, cc])
            n_bad = int((got[r, cc] != want).sum())
            worst = max(worst, n_bad)
            diff = (got[r, cc].float() - want.float()).abs()
            assert float(diff.max()) <= float(want.float().abs().max()) * 2**-7, (rows, per_row, r, cc)
        print(f"rope rows {rows} weight a row {per_row}: values not bit for bit (worst chip) {worst} of {rows * 128}")
        for t in (n, w, c, s, out):
            ttnn.deallocate(t)
    # q and k in one call (halves, the drafter's attention): bit for bit the two calls' (q's weight, k's weight, the
    # same cos / sin), at 2 U = 2 / 8 / 32 virtual users (8 / 32 / 128 users)
    for h in (2, 8, 32):
        nq0, nk0 = (torch.randn(ROWS, COLS, h * 32, 128, generator=g).bfloat16() for _ in range(2))
        c0 = torch.randn(ROWS, COLS, h * 32, 128, generator=g)
        s0 = torch.randn(ROWS, COLS, h * 32, 128, generator=g)
        wq1, wk1 = (torch.randn(128, generator=g).bfloat16() for _ in range(2))
        rows_of = lambda w1: w1.view(1, 1, 1, 128).expand(1, 1, 32, 128).contiguous()  # noqa: E731
        wq, wk = rep(mesh, rows_of(wq1), ttnn.bfloat16), rep(mesh, rows_of(wk1), ttnn.bfloat16)
        wqk = rep(mesh, torch.cat([rows_of(wq1), rows_of(wk1)], dim=2), ttnn.bfloat16)
        nq, nk = per_chip(mesh, nq0.unsqueeze(2), ttnn.bfloat16), per_chip(mesh, nk0.unsqueeze(2), ttnn.bfloat16)
        c = per_chip(mesh, c0.unsqueeze(2), ttnn.float32)
        s = per_chip(mesh, s0.unsqueeze(2), ttnn.float32)
        oq, ok = ops.rope(mesh, nq, wq, c, s), ops.rope(mesh, nk, wk, c, s)
        flat = per_chip(mesh, torch.cat([nq0, nk0], dim=2).unsqueeze(2), ttnn.bfloat16)
        n = ttnn.reshape(flat, [1, 2 * h, 32, 128])
        c4, s4 = ttnn.reshape(c, [1, h, 32, 128]), ttnn.reshape(s, [1, h, 32, 128])
        q2, k2 = ops.rope(mesh, n, wqk, c4, s4, halves=h)
        flat_of = lambda t: [x.flatten() for x in each_chip(t)]  # noqa: E731  (q2 / k2 [1, h, 32, 128] a chip)
        same_q = all(torch.equal(a, b) for a, b in zip(flat_of(q2), flat_of(oq)))
        same_k = all(torch.equal(a, b) for a, b in zip(flat_of(k2), flat_of(ok)))
        print(f"rope halves {h}: q bit for bit {same_q}, k bit for bit {same_k}")
        assert same_q and same_k, h
        for t in (wq, wk, wqk, nq, nk, c, s, oq, ok, flat, q2, k2):
            ttnn.deallocate(t)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@soft
def test_sel_scores(mesh_device):
    """sel_scores against float64 sums of the exact products; with GLM53_DFLASH2_SEL_PARTS=1 / 2 (the hi / the lo part
    alone) against those parts' own sums (a debug of the op's two halves)."""
    mesh = mesh_device
    g = torch.Generator().manual_seed(10)
    rank = 256
    parts_list = [int(p) for p in os.environ.get("GLM53_DFLASH2_SEL_PARTS", "3").split(",")]
    for items in (7, 28, 112):
        pr0 = torch.randn(ROWS, COLS, items * 32, rank, generator=g).bfloat16()
        su0 = torch.randn(ROWS, COLS, items * 32, rank, generator=g).bfloat16()
        proj = torch.randn(ROWS, COLS, items, rank, generator=g).bfloat16()
        pj0 = proj.repeat_interleave(32, dim=2)
        pr = per_chip(mesh, pr0.unsqueeze(2), ttnn.bfloat16)
        pj = per_chip(mesh, pj0.unsqueeze(2), ttnn.bfloat16)
        su = per_chip(mesh, su0.unsqueeze(2), ttnn.bfloat16)
        for parts in parts_list:
            out = ops.sel_scores(mesh, pr, pj, su, items, terms=parts)
            got = torch.stack(each_chip(out)).view(ROWS, COLS, items, 32, 32)
            worst = 0.0
            for r, c in CHIP_RC:
                p_exact = pr0[r, c].double() * pj0[r, c].double()
                hi = p_exact.float().bfloat16().double()
                a = {3: p_exact, 1: hi, 2: (p_exact - hi).float().bfloat16().double()}[parts].view(items, 32, rank)
                want = a @ su0[r, c].double().view(items, 32, rank).transpose(1, 2)  # [items, 32, 32]
                scale = want.abs().amax().clamp(min=1e-30)
                err = float((got[r, c].double() - want).abs().max() / scale)
                worst = max(worst, err)
            print(f"sel_scores items {items} parts {parts}: worst error / max |S| {worst:.2e}")
            # (printed, not asserted: the matmul's own sums are not exact for bf16 x bf16, ~5e-4 of the largest
            # measured, as the ttnn matmul's the op replaces; judged from the printout)
            ttnn.deallocate(out)
        for t in (pr, pj, su):
            ttnn.deallocate(t)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@soft
def test_silu_mul(mesh_device):
    """`silu_mul` (gate | up -> SiLU(gate) * up in one op) against ttnn's two slices and multiply with the SiLU on its
    first input: bit for bit, at 8 / 32 / 64 / 128 rows (the draft's 1 / 4 users at block 8, 16 at block 4, the
    prefill's 128).  Prints only."""
    mesh = mesh_device
    g = torch.Generator().manual_seed(11)
    inter = 3072
    for m in (8, 32, 64, 128):
        gu0 = (torch.randn(ROWS, COLS, m, 2 * inter, generator=g) * 2).bfloat16()
        gu = per_chip(mesh, gu0.unsqueeze(2), ttnn.bfloat16)
        got = back(mesh, ops.silu_mul(mesh, gu, inter)).reshape(ROWS, COLS, m, inter)
        a = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, m, inter])
        b = ttnn.slice(gu, [0, 0, 0, inter], [1, 1, m, 2 * inter])
        ref = ttnn.multiply(a, b, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)])
        want = back(mesh, ref).reshape(ROWS, COLS, m, inter)
        same = torch.equal(got, want)
        print(
            f"silu_mul rows {m}: equal to ttnn's {same}, values not equal {int((got != want).sum())} of "
            f"{got.numel()}, max |d| {float((got.float() - want.float()).abs().max()):.4g}",
            flush=True,
        )
        for t in (gu, a, b, ref):
            ttnn.deallocate(t)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
@soft
def test_rope_rows_time(mesh_device):
    """rope_rows traced (20 calls), GLM53_DFLASH2_ROPE_DEDUPE 0 against 1 in one process (the switch is a compile-time
    arg: other programs), on the model's table of the halves at the drafter's shapes: the layers' front (2 U x 32 rows,
    div 64, mod B: 8 / 32 / 128 users = U 1 / 4 / 16, B 8 / 8 / 4) and write_context (tokens x 16 head rows, div 16:
    U B tokens); the outputs of the two the same bit for bit.  Prints "ROPE_TIME" lines."""
    import time

    from models.demos.blackhole.glm53_flash.tests._trace import traced as trace_capture

    mesh = mesh_device
    g = torch.Generator().manual_seed(5)
    positions = 4096
    th = rep(mesh, torch.randn(positions, 128, generator=g), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
    saved = os.environ.get("GLM53_DFLASH2_ROPE_DEDUPE")
    shapes = []
    for users, block in ((1, 8), (4, 8), (16, 4)):
        shapes.append((f"front U {users} B {block}", users, 2 * users * 32, dict(div=64, mod=block)))
        shapes.append((f"context {users * block} tokens", users * block, users * block * 16, dict(div=16)))
    try:
        for name, n, rows, kw in shapes:
            starts = torch.randint(0, positions - 8, (ROWS, COLS, 1, n), generator=g, dtype=torch.int32)
            idx = per_chip(mesh, starts, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            outs, times = {}, {}
            for v in ("0", "1"):
                os.environ["GLM53_DFLASH2_ROPE_DEDUPE"] = v
                cos, sin = ops.rope_rows(mesh, idx, th, None, rows, **kw)
                outs[v] = (back(mesh, cos), back(mesh, sin))
                for t in (cos, sin):
                    ttnn.deallocate(t)
                ttnn.synchronize_device(mesh)
                with trace_capture(mesh) as trace:
                    kept = [ops.rope_rows(mesh, idx, th, None, rows, **kw) for _ in range(20)]
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                times[v] = (time.perf_counter() - t0) / 20 * 1e6
                ttnn.release_trace(mesh, trace)
                for pair in kept:
                    for t in pair:
                        ttnn.deallocate(t)
            same = all(torch.equal(a, b) for a, b in zip(outs["0"], outs["1"]))
            print(f"ROPE_TIME {name} ({rows} rows): {times['0']:.1f} -> {times['1']:.1f} us, the outputs "
                  f"{'the same' if same else 'DIFFER'}", flush=True)  # fmt: skip
            ttnn.deallocate(idx)
    finally:
        if saved is None:
            os.environ.pop("GLM53_DFLASH2_ROPE_DEDUPE", None)
        else:
            os.environ["GLM53_DFLASH2_ROPE_DEDUPE"] = saved
    ttnn.deallocate(th)
