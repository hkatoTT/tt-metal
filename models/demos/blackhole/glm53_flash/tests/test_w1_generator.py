"""The L1 landing of the next MoE layer's chosen experts' weights (GLM53_W1, ttnn/w1_landing.py) in the serving
generator: the landing held from the model's load (by default for the DFlash2 step's shape, `w1_landing.landing_c`: the
model made with its dflash_shape as the vLLM adapter makes it; GLM53_W1=0 off, GLM53_W1=C that many slots), the MoE
layers armed by the generator for its DFlash2 step, the DRISCs started at its warm-up's end, and prefills between the
verify steps as a server's: a long run's tokens against the same run without the landing, bit for bit (two processes:
GLM53_WG_DUMP, then GLM53_WG_COMPARE).

Synthetic weights (target and drafter: seeded per tensor, the same in every run) and GLM53_TEST_FORCE_ACCEPT=1 (each
seat's accepted drafts pseudo-random in 0 .. K from its position and seat, `verify_accept.forced_accept`: 0 .. K drafts
taken although the synthetic drafter's are never right), so that the flow (which request ends when, which seats a
prefill fills) is the same whenever the tokens are.  Requests with seeded random prompts of GLM53_WG_PROMPTS tokens
(lo-hi) and outputs of GLM53_WG_OUT (lo-hi) take the free seats as they come (`prefill_many`, its first token and the
drafter's first drafts), the others go on with the traced step (`dflash_verify`), until GLM53_WG_STEPS verify steps.
Per run: its steps, prefill calls, requests done, ms a step (median), and with the landing the DRISCs' windows and the
first mesh row's 4 chips' blocks from the landing / read again (one chip a mesh column).

    GLM53_WG_USERS (4: 32 seats; 16: 128), GLM53_WG_K (7; 3 at 16 users), GLM53_WG_STEPS (2000), GLM53_WG_MAX_SEQ
    (2048), GLM53_WG_PROMPTS (100-500), GLM53_WG_OUT (600-1200), GLM53_WG_REFILL_MIN (1: the free seats are filled
    once at least this many are free; seats wait empty meanwhile), GLM53_WG_DUMP (a path: this run's tokens),
    GLM53_WG_COMPARE (a path: each request's tokens against it; "WG COMPARE ... same" / "DIFFER")
    GLM53_WG_CHUNKS (128,512: the prefill's chunk lengths, each captured; 128,512,1024 the adapter's, the SP
    waves of 1,024 tokens a row)
"VERDICT WG PASS" when the run finished (and with GLM53_WG_COMPARE every request the same), else "VERDICT WG FAIL"
(the test ends normally either way).  Run (a device test): GLM53_DEVICE_TEST=1 pytest .../tests/test_w1_generator.py -s
"""

from __future__ import annotations

import json
import os
import statistics
import time
import traceback

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.checkpoint import ShapeOnlyCheckpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.tests._trace import is_device_timeout  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.dflash2 import TtDFlash2, load_weights  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

CHUNKS = tuple(int(x) for x in os.environ.get("GLM53_WG_CHUNKS", "128,512").split(","))  # (the prefill's chunks)


def span(name: str, default: str) -> tuple:
    lo, _, hi = os.environ.get(name, default).partition("-")
    return int(lo), int(hi or lo)


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_w1_generator(mesh_device):
    try:
        ok = _run(mesh_device)
    except Exception as e:
        if is_device_timeout(e):  # (a device timeout: the run ends non-zero)
            raise
        traceback.print_exc()
        print("STUDY FAILED (printed, not raised)", flush=True)
        ok = False
    print(f"VERDICT WG {'PASS' if ok else 'FAIL'}", flush=True)


def _run(mesh) -> bool:
    os.environ.setdefault("GLM53_TEST_FORCE_ACCEPT", "1")
    users = int(os.environ.get("GLM53_WG_USERS", "4"))
    k = int(os.environ.get("GLM53_WG_K", "7" if users <= 4 else "3"))
    steps_want = int(os.environ.get("GLM53_WG_STEPS", "2000"))
    max_seq = int(os.environ.get("GLM53_WG_MAX_SEQ", "2048"))
    p_lo, p_hi = span("GLM53_WG_PROMPTS", "100-500")
    o_lo, o_hi = span("GLM53_WG_OUT", "600-1200")
    refill_min = int(os.environ.get("GLM53_WG_REFILL_MIN", "1"))
    # GLM53_WG_FIRST_END=m: every m-th request (by its number) ends at its prefill, its first token its only output: its
    # seat sits out at the idle position the prefill gave it until the next request
    first_end = int(os.environ.get("GLM53_WG_FIRST_END", "0"))
    assert p_hi + o_hi + k + 1 < max_seq, (p_hi, o_hi, max_seq)
    t0 = time.time()
    model = TtGlm53Model(mesh, ShapeOnlyCheckpoint(), max_seq=max_seq, mtp=False, dflash_shape=(users, k + 1))
    cfg, w, mask = load_weights(None, synthetic=True)
    drafter = TtDFlash2(
        mesh, cfg, w, mask, users=users, max_ctx=max_seq + 8, block=4 if k <= 3 else 8,
        head_weight=model.lm_head.mesh_weight, embedding=model.embed,
    )  # fmt: skip
    gen = Glm53Generator(model, users_per_row=users, chunk=CHUNKS, dflash=(drafter, k))
    seconds = gen.warmup(greedy=False, logits=False, traced=CHUNKS)
    w1 = model.w1
    print(f"WG: {gen.slots} seats ({users} a mesh row), K {k}, cache {max_seq}; GLM53_W1 landing "
          f"{'C %d at %#x, running %s' % (w1.c, w1.addr, w1.running) if w1 else 'off'}; loaded and warmed up in "
          f"{time.time() - t0:.0f} s: {seconds}", flush=True)  # fmt: skip
    before = w1.counts(range(4)) if w1 else None
    if os.environ.get("GLM53_WG_L1_BLOCKS") == "1":  # (a study: the L1 buffers held after the warm-up, high to low)
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
        held = sorted(
            ((int(b["address"]), int(b["size"])) for b in view.block_table if b.get("allocated") == "yes"), reverse=True
        )
        print(f"WG L1 blocks after the warm-up ({len(held)}; address, bytes; high to low): {held[:48]}", flush=True)

    g = torch.Generator().manual_seed(1180 + 10 * users + k)
    vocab = model.config.vocab_size

    def new_request():
        n = int(torch.randint(p_lo, p_hi + 1, (1,), generator=g))
        out = int(torch.randint(o_lo, o_hi + 1, (1,), generator=g))
        return torch.randint(0, vocab, (n,), generator=g).tolist(), out

    slots = gen.slots
    owner = [None] * slots  # seat -> request id
    reqs = []  # (prompt length, wanted, tokens)
    bonus = torch.zeros(slots, dtype=torch.int64)
    drafts = torch.zeros(slots, k, dtype=torch.int64)
    pos = torch.full((slots,), -1, dtype=torch.int64)
    keep = torch.full((slots,), k, dtype=torch.int64)
    step_s, prefill_calls, done, steps = [], 0, 0, 0
    prefill_s: dict = {}  # requests in the call -> its seconds (to its return: the logits read back)
    interleaved: set = set()  # the requests prefilled in a round of interleaved blocks
    forms: dict = {}  # request -> its prefill form
    waves_seen: list = []  # (the current call's waves: (own row -> length, the rounds chosen))
    own_rounds = gen.sp_rounds

    def spy(lengths, chunk=None):
        got = own_rounds(lengths, chunk)
        waves_seen.append((dict(lengths), got))
        return got

    gen.sp_rounds = spy
    flow_t0 = time.perf_counter()
    while steps < steps_want:
        arriving = []
        free = sum(o is None for o in owner)
        for s in range(slots) if free >= min(refill_min, slots) else ():
            if owner[s] is None:
                prompt, want = new_request()
                owner[s] = len(reqs)
                reqs.append([len(prompt), want, []])
                arriving.append((s, prompt))
        if arriving:  # the new requests' prefill, their first token and the drafter's first drafts
            prefill_calls += 1
            waves_seen.clear()
            a = time.perf_counter()
            logits = gen.prefill_many(arriving, traced=True)
            # each request's prefill form (as prefill_many waves them: a row's longest first), from the rounds it chose:
            # ("sp", segment length, its rows) or ("row", the chunks); the same form gives the same values whichever rows
            per_row: dict = {}
            for i, (s, pr) in enumerate(arriving):
                per_row.setdefault(s // users, []).append(i)
            for v in per_row.values():
                v.sort(key=lambda i: -len(arriving[i][1]))
            for wi, (lens, rounds) in enumerate(waves_seen):  # (not k: the drafts a step)
                for r in lens:
                    i = per_row[r][wi]
                    rid = owner[arriving[i][0]]
                    if rounds is None:
                        forms[rid] = ("row", tuple(t for _, t in gen.schedule(lens[r])))
                    else:
                        t, b = next((t, b) for o, t, b in rounds if r in o)
                        forms[rid] = ("sp", t, len(b[r]))
                        if b != gen.sp_blocks(list(b)):
                            interleaved.add(rid)
            prefill_s.setdefault(len(arriving), []).append(time.perf_counter() - a)
            for (s, prompt), lg in zip(arriving, logits):
                first = int(lg.reshape(-1)[:vocab].argmax())
                reqs[owner[s]][2] = [first]
                bonus[s], pos[s], keep[s] = first, len(prompt), k
                drafts[s] = torch.tensor(gen.first.pop(s))
                if first_end and owner[s] % first_end == 0:  # (GLM53_WG_FIRST_END: done at its prefill)
                    reqs[owner[s]][1] = 1
                    owner[s], pos[s] = None, -1
                    done += 1
        toks = torch.cat([bonus[:, None], drafts], dim=1)
        poss = torch.where(pos[:, None] >= 0, pos[:, None] + torch.arange(k + 1)[None, :], -1)
        a = time.perf_counter()
        n, b, d = gen.dflash_verify(toks, poss, keep)
        step_s.append(time.perf_counter() - a)
        steps += 1
        for s in range(slots):
            i = owner[s]
            if i is None:
                continue
            ns = int(n[s])
            reqs[i][2] += drafts[s, :ns].tolist() + [int(b[s])]
            bonus[s], drafts[s], keep[s], pos[s] = b[s], d[s], ns, pos[s] + ns + 1
            if len(reqs[i][2]) >= reqs[i][1]:  # done: the seat takes the next request
                owner[s], pos[s] = None, -1
                done += 1
    flow_s = time.perf_counter() - flow_t0
    print(f"WG prefill calls (requests: count, median ms): "
          f"{ {m: (len(v), round(1e3 * statistics.median(v), 1)) for m, v in sorted(prefill_s.items())} }; "
          f"the flow {flow_s:.1f} s ({sum(map(sum, prefill_s.values())):.1f} s in prefill calls)", flush=True)  # fmt: skip
    stats = {}
    if w1 is not None:
        ttnn.synchronize_device(mesh)
        stats = w1.stop() if w1.running else {}
        after = w1.counts(range(4))
        stats["blocks"] = [a - b for a, b in zip(after, before)]
    gen.release()
    tokens = sum(len(r[2]) for r in reqs)
    print(f"WG run: {steps} verify steps, {prefill_calls} prefill calls, {len(reqs)} requests ({done} done), {tokens} "
          f"tokens, {1000 * statistics.median(step_s):.2f} ms a step (median, with the host); landing {stats or 'off'} "
          f"(blocks: the first mesh row's 4 chips' gate | up from the landing, read again, down from it, read again)",
          flush=True)  # fmt: skip
    # (GLM53_DFLASH2_NEXT_INPUTS: the host writes the device's next inputs replaced, GLM53_DFLASH2_NEXT_CHECK's checks)
    print(f"WG next inputs: {getattr(gen, 'dflash_writes', None)}", flush=True)
    out = {str(i): r[2] for i, r in enumerate(reqs)}
    if os.environ.get("GLM53_WG_DUMP"):
        with open(os.environ["GLM53_WG_DUMP"], "w") as f:
            json.dump({"steps": steps, "requests": out, "forms": {str(i): list(v) for i, v in forms.items()}}, f)
        print(f"WG dump: {os.environ['GLM53_WG_DUMP']}", flush=True)
    ok = steps >= steps_want
    if os.environ.get("GLM53_WG_COMPARE"):
        theirs = json.load(open(os.environ["GLM53_WG_COMPARE"]))["requests"]
        differ = [i for i in out if theirs.get(i) != out[i]]
        first = next((i for i in out if theirs.get(i) != out[i]), None)
        print(f"WG COMPARE against {os.environ['GLM53_WG_COMPARE']}: {len(out) - len(differ)} / {len(out)} requests "
              f"{'same' if not differ else 'DIFFER (first request %s)' % first}", flush=True)  # fmt: skip
        mine = json.loads(json.dumps({str(i): list(v) for i, v in forms.items()}))  # (as the dump holds them)
        changed = {i for i in mine if theirs_forms.get(i) != mine[i]} if (theirs_forms := json.load(
            open(os.environ["GLM53_WG_COMPARE"])).get("forms")) is not None else set()  # fmt: skip
        inter = {str(i) for i in interleaved}
        print(f"WG FORMS: {len(differ)} requests differ; {len(changed)} took another prefill form than the dump's "
              f"({len(inter)} of them in interleaved rounds); differ and not changed {len(set(differ) - changed)}, "
              f"changed and not differ {len(changed - set(differ))}", flush=True)  # fmt: skip
        ok = ok and not differ and len(theirs) == len(out)
    return ok
