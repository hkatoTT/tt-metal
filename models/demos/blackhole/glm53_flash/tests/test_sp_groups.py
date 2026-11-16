"""The 8-row prefill (sequence parallel) with more than one request in a round: each request's logits against the same
prompt alone (one request, all 8 rows), and against the one-row prefill (a round of slots 0 / 6 / 11 at once, rows
0 / 1 / 2: blocks of 6, 1 and 1 rows, once gave other logits than alone, the same wrong ones in every slot).  Synthetic
weights (the values only need to agree), eager, each round through `Glm53Generator._prefill_sp_round` directly (so the
8-row form whatever `sp_plan` would choose).

Each case is a round of requests on mesh rows (user 0 of each row), e.g. "0+4" (two requests, blocks of 4 and 4 rows),
"0+1" (blocks of 7 and 1): request i takes prompt i (distinct random prompts), compared with prompt i alone.
    GLM53_SPG_CASES  the cases (default "0+4,0+1,1+2")
    GLM53_SPG_LEN    the prompts' length (default 1024)
    GLM53_SPG_CACHE  the cache rows (default 2048: the dense cache; 9728: the long one)
    GLM53_SPG_T      the segment length (default 512)
    GLM53_SPG_LAYERS the layers: all, kda (layers 0-2: KDA, dense MLPs), dsa (layer 3 alone: DSA, MoE), or a count
    GLM53_SPG_MTP    1 / 0: with / without the MTP layer (default GLM53_MTP)
    GLM53_SPG_SPEC   the generator's drafts (1: MTP's verify step as serving with MTP; default 0)
    GLM53_SPG_REAL   1: the real weights
Each case also through the one-row prefill (`prefill_many` without the 8-row form: the requests on their own rows in
one wave), against each prompt alone on one row.
Run (a device test): GLM53_DEVICE_TEST=1 GLM53_SYNTHETIC_WEIGHTS=1 pytest ... -s
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

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, ShapeOnlyCheckpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.config import DSA, KDA  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

USERS = 4


def build(mesh, cache: int):
    layers = os.environ.get("GLM53_SPG_LAYERS", "all")
    mtp = {"1": True, "0": False}.get(os.environ.get("GLM53_SPG_MTP", ""))  # (default: GLM53_MTP)
    n = {"all": None, "kda": 3, "dsa": 4}[layers] if layers in ("all", "kda", "dsa") else int(layers)
    real = os.environ.get("GLM53_SPG_REAL") == "1"  # the real weights (a few layers load quickly)
    model = TtGlm53Model(
        mesh, Glm53Checkpoint() if real else ShapeOnlyCheckpoint(), max_seq=cache, num_layers=n, mtp=mtp
    )
    if layers == "dsa":  # layer 3 alone (the first DSA layer, with its MoE)
        model.layers = model.layers[3:]
        model.dsa = model.layers[0].attention
        model.has_kda = False
    kinds = [layer.kind for layer in model.layers]
    print(
        f"layers {layers}: {len(kinds)} ({kinds.count(KDA)} KDA, {kinds.count(DSA)} DSA), MTP "
        f"{model.mtp is not None}, cache {cache}, long {model.dsa.long if model.dsa is not None else None}",
        flush=True,
    )
    return model


@pytest.mark.timeout(1400)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_sp_groups(mesh_device):
    n = int(os.environ.get("GLM53_SPG_LEN", "1024"))
    cache = int(os.environ.get("GLM53_SPG_CACHE", "2048"))
    t = int(os.environ.get("GLM53_SPG_T", "512"))
    cases = [[int(r) for r in c.split("+")] for c in os.environ.get("GLM53_SPG_CASES", "0+4,0+1,1+2").split(",")]
    t0 = time.time()
    model = build(mesh_device, cache)
    assert model.sp_ready, "dsa.PREFILL_SP is needed"
    gen = Glm53Generator(model, USERS, chunk=(128, 512))
    print(f"loaded in {time.time() - t0:.0f} s; prompts of {n}, segments of {t}", flush=True)
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(7)
    most = max(len(c) for c in cases)
    prompts = [torch.randint(0, vocab, (n,), generator=g).tolist() for _ in range(most)]

    def sp_round(rows: list, which: list) -> list:
        """One round of requests on `rows` (user 0 of each), prompt which[i] for the i-th -> their logits."""
        requests = [(row * USERS, prompts[w]) for row, w in zip(rows, which)]
        members = {row: i for i, row in enumerate(rows)}
        blocks = gen.sp_blocks(members)
        out = [None] * len(requests)
        t1 = time.time()
        gen._prefill_sp_round(requests, members, t, blocks, False, out)
        segments = -(-n // t)
        waves = max(-(-segments // len(blocks[r])) for r in rows)
        print(
            f"  round rows {rows}: blocks {[blocks[r] for r in rows]}, {waves} waves, {time.time() - t1:.1f} s",
            flush=True,
        )
        return [o.float() for o in out]

    # the references: each prompt alone (all 8 rows, from row 0's block), and on its own row (one-row chunks)
    alone = [sp_round([0], [i])[0] for i in range(most)]
    gen.sp = False
    rows_ref = [gen.prefill_many([(0, prompts[i])], chunk=t, traced=False)[0].float() for i in range(most)]
    gen.sp = True
    worst = 0.0
    for i in range(most):
        scale = alone[i].abs().max().item()
        d = (alone[i] - rows_ref[i]).abs().max().item()
        worst = max(worst, d / scale)
        print(
            f"prompt {i} alone over the rows vs one row: max |d| {d:.3e} ({d / scale:.2e} of the max), argmax "
            f"{int(alone[i].argmax())} / {int(rows_ref[i].argmax())}",
            flush=True,
        )
    results = {}
    for rows in cases:
        got = sp_round(rows, list(range(len(rows))))
        for i, (row, o) in enumerate(zip(rows, got)):
            scale = alone[i].abs().max().item()
            d = (o - alone[i]).abs().max().item()
            results[(tuple(rows), row)] = d / scale
            print(
                f"  case {'+'.join(map(str, rows))}, row {row} (prompt {i}): max |d| {d:.3e} ({d / scale:.2e} of the "
                f"max), argmax {int(o.argmax())} (alone {int(alone[i].argmax())}), bitwise {torch.equal(o, alone[i])}",
                flush=True,
            )
        gen.sp = False  # the same requests on their own rows, one wave of one-row chunks
        got = gen.prefill_many([(row * USERS, prompts[i]) for i, row in enumerate(rows)], chunk=t, traced=False)
        gen.sp = True
        for i, (row, o) in enumerate(zip(rows, got)):
            o = o.float()
            scale = rows_ref[i].abs().max().item()
            d = (o - rows_ref[i]).abs().max().item()
            print(
                f"  case {'+'.join(map(str, rows))} on one row each, row {row} (prompt {i}): max |d| {d:.3e} "
                f"({d / scale:.2e} of the max) against prompt {i} alone on row 0, bitwise {torch.equal(o, rows_ref[i])}",
                flush=True,
            )
    bad = {k: v for k, v in results.items() if v > max(10 * worst, 1e-2)}
    print(f"over the tolerance (10 x the alone / one-row difference {worst:.2e}, at least 1e-2): {bad}", flush=True)


@pytest.mark.timeout(1400)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_one_row_users(mesh_device):
    """The one-row prefill (no 8-row form) of the same prompt alone in slots GLM53_SPG_SLOTS (default 0, 1, 4, 5, 6: row
    0's users 0 / 1, row 1's users 0 / 1 / 2), each against slot 0's: the same computation on another user's pages and
    state, so only the rounding of other rows may differ (users 2 / 3 of the long cache once gave other first tokens).
    Then the 8-row form alone in the same slots against slot 0's (GLM53_SPG_SP_TOO=1, default)."""
    n = int(os.environ.get("GLM53_SPG_LEN", "1024"))
    cache = int(os.environ.get("GLM53_SPG_CACHE", "2048"))
    t = int(os.environ.get("GLM53_SPG_T", "512"))
    slots = [int(s) for s in os.environ.get("GLM53_SPG_SLOTS", "0,1,4,5,6").split(",")]
    model = build(mesh_device, cache)
    gen = Glm53Generator(model, USERS, chunk=(128, 512))
    g = torch.Generator().manual_seed(11)
    prompt = torch.randint(0, model.config.vocab_size, (n,), generator=g).tolist()
    for sp in (False, True) if os.environ.get("GLM53_SPG_SP_TOO", "1") == "1" else (False,):
        gen.sp = sp
        if sp and not model.sp_ready:
            continue
        outs = {}
        for s in slots:
            t1 = time.time()
            if sp:
                row = s // USERS
                out = [None]
                gen._prefill_sp_round([(s, prompt)], {row: 0}, t, gen.sp_blocks([row]), False, out)
                outs[s] = out[0].float()
            else:
                outs[s] = gen.prefill_many([(s, prompt)], chunk=t, traced=False)[0].float()
            ref = outs[slots[0]]
            scale = ref.abs().max().item()
            d = (outs[s] - ref).abs().max().item()
            print(
                f"{'8-row' if sp else 'one-row'} prefill, slot {s} (row {s // USERS}, user {s % USERS}), "
                f"{time.time() - t1:.1f} s: max |d| {d:.3e} ({d / scale:.2e} of the max) against slot {slots[0]}, "
                f"argmax {int(outs[s].argmax())} / {int(ref.argmax())}, bitwise {torch.equal(outs[s], ref)}",
                flush=True,
            )


@pytest.mark.timeout(1400)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_one_row_traced(mesh_device):
    """The one-row prefill through the captured forms (as serving: `warmup(traced=(128, 512))`, every chunk replayed)
    against the eager one-row prefill in slot 0, bit for bit (eager is the same in every slot): the prompt alone
    in slots 0 / 1 / 4 / 5 / 6 / 11, then in slots 0 / 6 / 11 at once (rows 0 / 1 / 2, users 0 / 2 / 3: a serving
    run's first round), eager and traced; then the 8-row form alone in slot 0, traced against eager."""
    n = int(os.environ.get("GLM53_SPG_LEN", "1024"))
    cache = int(os.environ.get("GLM53_SPG_CACHE", "2048"))
    t = int(os.environ.get("GLM53_SPG_T", "512"))
    model = build(mesh_device, cache)
    gen = Glm53Generator(model, USERS, chunk=(128, 512), spec=int(os.environ.get("GLM53_SPG_SPEC", "0")))
    t0 = time.time()
    print(f"warm-up {gen.warmup(greedy=True, logits=False, traced=(128, 512))} in {time.time() - t0:.0f} s", flush=True)
    g = torch.Generator().manual_seed(11)
    prompt = torch.randint(0, model.config.vocab_size, (n,), generator=g).tolist()
    gen.sp = False
    ref = gen.prefill_many([(0, prompt)], chunk=t, traced=False)[0].float()
    scale = ref.abs().max().item()

    def show(label, outs, slots):
        for s, o in zip(slots, outs):
            o = o.float()
            d = (o - ref).abs().max().item()
            print(
                f"{label}, slot {s} (row {s // USERS}, user {s % USERS}): max |d| {d:.3e} ({d / scale:.2e} of the "
                f"max), argmax {int(o.argmax())} / {int(ref.argmax())}, bitwise {torch.equal(o, ref)}",
                flush=True,
            )

    for s in (0, 1, 4, 5, 6, 11):
        before = dict(gen.counts)
        out = gen.prefill_many([(s, prompt)], chunk=t, traced=True)
        replayed = gen.counts["traced_chunks"] - before["traced_chunks"]
        show(f"one-row traced alone ({replayed} chunks replayed)", out, [s])
    together = [0, 6, 11]
    show("one-row eager together", gen.prefill_many([(s, prompt) for s in together], chunk=t, traced=False), together)
    before = dict(gen.counts)
    out = gen.prefill_many([(s, prompt) for s in together], chunk=t, traced=True)
    replayed = gen.counts["traced_chunks"] - before["traced_chunks"]
    show(f"one-row traced together ({replayed} chunks replayed)", out, together)
    # a serving run's round as serving does it: slots 0 / 1 / 6 / 11, the generator's own plan (8-row where it chooses)
    gen.sp = model.sp_ready
    served = [0, 1, 6, 11]
    before = dict(gen.counts)
    out = gen.prefill_many([(s, prompt) for s in served])
    show(f"serving's prefill_many ({gen.counts['waves'] - before['waves']} waves)", out, served)
    if gen.spec:
        print(f"  first drafts {[gen.first.get(s) for s in served]}", flush=True)
    gen.sp = False
    if model.sp_ready:
        gen.sp = True
        eager = [None]
        gen._prefill_sp_round([(0, prompt)], {0: 0}, t, gen.sp_blocks([0]), False, eager)
        traced = [None]
        gen._prefill_sp_round([(0, prompt)], {0: 0}, t, gen.sp_blocks([0]), True, traced)
        d = (traced[0].float() - eager[0].float()).abs().max().item()
        print(
            f"8-row alone in slot 0, traced against eager: max |d| {d:.3e}, bitwise {torch.equal(traced[0], eager[0])}",
            flush=True,
        )
    gen.release()


@pytest.mark.timeout(1400)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_first_drafts(mesh_device):
    """MTP's first draft after the prefill (`Glm53Generator._first_drafts`, spec 1) of the same prompt, which must be the
    same in every slot (the logits are; slots 0 / 6 / 11 together once gave drafts 2970 / 2970 / 2746):
    the one-row prefill (traced, as serving) alone in each slot 0 .. 15, then at once in GLM53_SPG_TOGETHER (default
    0+6+11, 0+1+2+3, 3+7+11+15); each slot's logits' argmax and first draft."""
    n = int(os.environ.get("GLM53_SPG_LEN", "8192"))
    cache = int(os.environ.get("GLM53_SPG_CACHE", "9216"))
    model = build(mesh_device, cache)
    assert model.mtp is not None, "the MTP layer is needed (GLM53_SPG_MTP=1)"
    gen = Glm53Generator(model, USERS, chunk=(128, 512), spec=1)
    t0 = time.time()
    print(f"warm-up {gen.warmup(greedy=True, logits=False, traced=(128, 512))} in {time.time() - t0:.0f} s", flush=True)
    g = torch.Generator().manual_seed(11)
    prompt = torch.randint(0, model.config.vocab_size, (n,), generator=g).tolist()
    gen.sp = False
    alone = {}
    for s in range(gen.slots // 2):  # rows 0 .. 3 (the users 0 .. 3 of each)
        gen.first.clear()
        out = gen.prefill_many([(s, prompt)])[0]
        alone[s] = (int(out.argmax()), gen.first.get(s))
    ref = alone[0]
    print(f"alone (argmax, first draft) by slot: {alone}", flush=True)
    print(f"  slots whose draft differs from slot 0's ({ref}): {[s for s, v in alone.items() if v != ref]}", flush=True)
    for spec in os.environ.get("GLM53_SPG_TOGETHER", "0+6+11,0+1+2+3,3+7+11+15").split(","):
        slots = [int(x) for x in spec.split("+")]
        gen.first.clear()
        outs = gen.prefill_many([(s, prompt) for s in slots])
        got = {s: (int(o.argmax()), gen.first.get(s)) for s, o in zip(slots, outs)}
        print(
            f"together {spec}: {got}; differs from slot 0 alone: {[s for s, v in got.items() if v != ref]}", flush=True
        )
    gen.release()
