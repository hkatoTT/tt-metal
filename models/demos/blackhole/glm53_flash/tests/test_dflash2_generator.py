"""DFlash2 in the serving generator: `Glm53Generator(..., dflash=(drafter, K))` with
requests arriving into freed slots as a server takes them: the 11 greedy records over the 8 slots (one a row), each
generating its own number of tokens, a new request prefilled (eager, its taps into the drafter's context, its first
drafts) as soon as a slot is free while the others go on with the traced step (`dflash_verify`).  Each record's tokens
against the plain greedy decode's (the same generator form without DFlash2, one user a row: a verify step of K + 1 <= 4
tokens a user rounds as the plain decode), the tokens a verify step, the slots reused.  The real weights.

    GLM53_DFLASH2_K (3; a comma list: each K on the one load), GLM53_DFLASH2_TOKENS (24: the longest record's tokens;
    the others fewer, staggered),
    GLM53_DFLASH2_PREFILL_TRACED (0; 1: the DFlash2 generator's prefill from its traces of 128 / 512 tokens, the taps
    their outputs)
    GLM53_DFLASH2_KV_ROUNDS (a comma list of dflash2.KV_ROUND codes, each K run with each: 0 the drafter's K / V as
    they are, 8 / 4 rounded as a BFP8 / BFP4 cache holds them: a study of the drafter's cache precision)
Run (a device test): GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_dflash2_generator.py -s
"""

from __future__ import annotations

import gc
import json
import os
import time
import traceback
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash import checkpoint as checkpoint_module  # noqa: E402
from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import dflash2 as dflash2_mod  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import dsa as dsa_mod  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.dflash2 import TtDFlash2, load_weights  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

RECORDS = Path(checkpoint_module.__file__).parent / "tools" / "acceptance" / "greedy-v1"
MAX_SEQ = 1024
CHUNKS = (128, 512)


def guarded(run) -> None:
    """A study's run: an exception is printed, not raised (the run ends cleanly, with exit code 0), raised with
    GLM53_DFLASH2_STRICT=1."""
    try:
        run()
    except Exception:
        if os.environ.get("GLM53_DFLASH2_STRICT") == "1":
            raise
        traceback.print_exc()
        print("STUDY FAILED (printed, not raised)", flush=True)


def plain_greedy(model, prompts: list, wants: list) -> list:
    """The plain greedy decode (traced), one user a row, the prompts in waves of 8 -> each prompt's tokens."""
    gen = Glm53Generator(model, users_per_row=1, chunk=CHUNKS)
    gen.warmup(greedy=True, logits=False)
    out = []
    for w0 in range(0, len(prompts), ROWS):
        wave = list(range(w0, min(w0 + ROWS, len(prompts))))
        logits = gen.prefill_many([(r, prompts[i]) for r, i in enumerate(wave)], traced=False)
        got = [[int(x.reshape(-1)[: model.config.vocab_size].argmax())] for x in logits]
        tokens = torch.zeros(ROWS, dtype=torch.int64)
        positions = torch.full((ROWS,), -1, dtype=torch.int64)
        for r, i in enumerate(wave):
            tokens[r], positions[r] = got[r][0], len(prompts[i])
        for _ in range(max(wants[i] for i in wave) - 1):
            nxt = gen.decode(tokens, positions, greedy=True)
            for r in range(len(wave)):
                got[r].append(int(nxt[r]))
            tokens = nxt
            positions = torch.where(positions >= 0, positions + 1, positions)
        out += [g[: wants[i]] for g, i in zip(got, wave)]
    gen.release()
    return out


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dflash2_generator(mesh_device):
    guarded(lambda: _dflash2_generator(mesh_device))


def _dflash2_generator(mesh_device):
    # GLM53_DFLASH2_K: one K or a comma list of them, each on the same load
    ks = [int(v) for v in os.environ.get("GLM53_DFLASH2_K", "3").split(",")]
    longest = int(os.environ.get("GLM53_DFLASH2_TOKENS", "24"))
    t0 = time.time()
    dsa_mod.PREFILL_SP = False  # (one request a row, the same in the reference)
    model = TtGlm53Model(mesh_device, Glm53Checkpoint(), max_seq=MAX_SEQ, mtp=False)
    cfg, w, mask = load_weights()
    records = [json.loads(f.read_text()) for f in sorted(RECORDS.glob("*.json"))]
    prompts = [r["prompt_token_ids"] for r in records]
    wants = [longest - 3 * (i % 4) for i in range(len(records))]  # (staggered: the slots free at different steps)
    print(f"loaded in {time.time() - t0:.0f} s; {len(records)} records, tokens {wants}", flush=True)

    plain = plain_greedy(model, prompts, wants)
    print(f"plain greedy: {time.time() - t0:.0f} s", flush=True)
    rounds = [int(v) for v in os.environ.get("GLM53_DFLASH2_KV_ROUNDS", "").split(",") if v] or [None]
    for k in ks:
        for r in rounds:
            if r is not None:
                dflash2_mod.KV_ROUND = r
                print(f"== the drafter's K / V rounding {r} (dflash2.KV_ROUND)", flush=True)
            guarded(lambda k=k: _dflash2_run(mesh_device, model, cfg, w, mask, records, prompts, wants, plain, k, t0))


def _dflash2_run(mesh_device, model, cfg, w, mask, records, prompts, wants, plain, k, t0):
    """DFlash2 with K = k drafts in the generator on the loaded model, against `plain`."""
    drafter = TtDFlash2(
        mesh_device, cfg, w, mask, users=1, max_ctx=MAX_SEQ + 8, block=k + 1, head_weight=model.lm_head.mesh_weight,
        embedding=model.embed,
    )  # fmt: skip
    gen = Glm53Generator(model, users_per_row=1, chunk=CHUNKS, dflash=(drafter, k))
    traced = os.environ.get("GLM53_DFLASH2_PREFILL_TRACED", "0") == "1"
    seconds = gen.warmup(greedy=False, logits=False, traced=CHUNKS if traced else ())
    print(f"DFlash2 generator warmed up: {seconds}; {time.time() - t0:.0f} s", flush=True)

    slots = gen.slots
    queue = list(range(len(records)))
    owner = [None] * slots  # slot -> record
    out = [[] for _ in records]
    bonus = torch.zeros(slots, dtype=torch.int64)
    drafts = torch.zeros(slots, k, dtype=torch.int64)
    pos = torch.full((slots,), -1, dtype=torch.int64)  # each slot's bonus position, -1: free
    keep = torch.full((slots,), k, dtype=torch.int64)
    accepted = [[] for _ in records]
    used = [0] * slots
    step_s = []
    while queue or any(o is not None for o in owner):
        free = [s for s in range(slots) if owner[s] is None]
        arriving = []
        for s in free:
            if not queue:
                break
            i = queue.pop(0)
            owner[s], used[s] = i, used[s] + 1
            arriving.append((s, i))
        if arriving:  # the new requests' prefill (one a row), their first token and drafts
            logits = gen.prefill_many([(s, prompts[i]) for s, i in arriving], traced=traced)
            for (s, i), lg in zip(arriving, logits):
                first = int(lg.reshape(-1)[: model.config.vocab_size].argmax())
                out[i] = [first]
                bonus[s], pos[s], keep[s] = first, len(prompts[i]), k
                drafts[s] = torch.tensor(gen.first.pop(s))
        toks = torch.cat([bonus[:, None], drafts], dim=1)
        poss = torch.where(pos[:, None] >= 0, pos[:, None] + torch.arange(k + 1)[None, :], -1)
        a = time.perf_counter()
        n, b, d = gen.dflash_verify(toks, poss, keep)
        step_s.append(time.perf_counter() - a)
        for s in range(slots):
            i = owner[s]
            if i is None:
                continue
            ns = int(n[s])
            out[i] += drafts[s, :ns].tolist() + [int(b[s])]
            accepted[i].append(ns)
            bonus[s], drafts[s], keep[s], pos[s] = b[s], d[s], ns, pos[s] + ns + 1
            if len(out[i]) >= wants[i]:  # done: the slot is free for the next request
                owner[s], pos[s] = None, -1
    gen.release()

    same_all = True
    tokens = steps = 0
    for i, rec in enumerate(records):
        got = out[i][: wants[i]]
        diff = next((j for j in range(wants[i]) if got[j] != plain[i][j]), None)
        same_all = same_all and diff is None
        tokens += sum(x + 1 for x in accepted[i])
        steps += len(accepted[i])
        print(
            f"generator {rec['id']}: {len(accepted[i])} steps, accepted {accepted[i]}; "
            + ("the same tokens as the plain greedy decode" if diff is None else f"first difference at token {diff}"),
            flush=True,
        )
    rest = sorted(step_s[2:])
    print(
        f"DFlash2 generator K = {k} (KV round {dflash2_mod.KV_ROUND}): {tokens / steps:.3f} tokens a verify step, "
        f"slots used {used}, step median "
        f"{rest[len(rest) // 2] * 1e3:.2f} ms (host included); prefill {gen.counts}; every record the same as the "
        f"plain greedy decode: {same_all}; {time.time() - t0:.0f} s",
        flush=True,
    )
    del gen, drafter
    gc.collect()
