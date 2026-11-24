"""The interleaved sequence-parallel blocks (generator.SP_INTERLEAVE, GLM53_SP_INTERLEAVE) against the same requests
in consecutive blocks of the same sizes, bit for bit: synthetic weights (target and drafter, seeded per tensor), the
generator as the vLLM adapter makes it for DFlash2 (4 users a mesh row, K = 7, chunks 128 / 512 / 1,024 all captured,
the experts' L1 landing GLM53_W1, the adapter's two warm-up calls).  Synthetic weights give a row's segment the same
values whichever row computes it (test_sp_groups: a request in a 4-row block = the request alone), so a request's
segments on interleaved rows must give the bits of its segments on consecutive rows.

Each case is a round of requests on mesh rows (user 0 of each), e.g. "0+1+4+5:1024":
  A  interleave on: `prefill_many` (traced), its rounds printed (one round, interleaved blocks where the consecutive
     ones take more waves), each request's logits; then GLM53_SPI_STEPS DFlash2 verify steps of those seats (the
     others sitting out; GLM53_TEST_FORCE_ACCEPT=1), each step's bonus and drafts.
  B  each request alone (`_prefill_sp_round`, traced) on the consecutive block of as many rows as A gave it, ending at
     its own row (the same waves and segments a row), then the same verify steps.
  "SPI SAME" when every logit and every step's ids match, else "SPI DIFFER"; also the round with the interleave off (one
  row each where A interleaved: other values, printed only).

    GLM53_SPI_CASES  ","-list (default "0+1+4+5:1024,0+1+2+3:1024,0+1+2:1000,0+1:1024,3+4+6:1024,0+2+4+6:1024")
    GLM53_SPI_CACHE  the cache rows (2048; 9728: the long cache, its indexer)
    GLM53_SPI_STEPS  verify steps after each prefill (3)
"VERDICT SPI PASS" when every case is the same.  Run (a device test): GLM53_DEVICE_TEST=1 pytest ... -s
"""

from __future__ import annotations

import os
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
from models.demos.blackhole.glm53_flash.ttnn import generator as generator_module  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.dflash2 import TtDFlash2, load_weights  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

USERS, K, CHUNKS = 4, 7, (128, 512, 1024)


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_sp_interleave(mesh_device):
    try:
        ok = _run(mesh_device)
    except Exception as e:
        if is_device_timeout(e):  # (a device timeout: the run ends non-zero)
            raise
        traceback.print_exc()
        print("STUDY FAILED (printed, not raised)", flush=True)
        ok = False
    print(f"VERDICT SPI {'PASS' if ok else 'FAIL'}", flush=True)


def _run(mesh) -> bool:
    os.environ.setdefault("GLM53_TEST_FORCE_ACCEPT", "1")
    cache = int(os.environ.get("GLM53_SPI_CACHE", "2048"))
    steps = int(os.environ.get("GLM53_SPI_STEPS", "3"))
    spec = os.environ.get("GLM53_SPI_CASES", "0+1+4+5:1024,0+1+2+3:1024,0+1+2:1000,0+1:1024,3+4+6:1024,0+2+4+6:1024")
    cases = []
    for c in spec.split(","):
        rows, _, n = c.partition(":")
        cases.append(([int(r) for r in rows.split("+")], int(n)))
    t0 = time.time()
    model = TtGlm53Model(mesh, ShapeOnlyCheckpoint(), max_seq=cache, mtp=False, dflash_shape=(USERS, K + 1))
    cfg, w, mask = load_weights(None, synthetic=True)
    drafter = TtDFlash2(
        mesh, cfg, w, mask, users=USERS, max_ctx=cache + 8, block=8,
        head_weight=model.lm_head.mesh_weight, embedding=model.embed,
    )  # fmt: skip
    gen = Glm53Generator(model, USERS, chunk=CHUNKS, dflash=(drafter, K))
    s1 = gen.warmup(greedy=False, logits=False)
    s2 = gen.warmup(greedy=True, logits=True, tokens=gen.chunk, traced=CHUNKS)
    print(f"SPI: {gen.slots} seats, cache {cache}, long {model.dsa.long}; "
          f"GLM53_W1 landing {model.w1.c if model.w1 else 'off'}; "
          f"loaded and warmed up in {time.time() - t0:.0f} s ({s1}, {s2})", flush=True)  # fmt: skip
    vocab = model.config.vocab_size
    slots = gen.slots

    def verify_run(seats: dict, logits: dict) -> list:
        """`steps` verify steps of `seats` (slot -> prompt length; the others sitting out) from their prefill's
        greedy first token and the drafter's first drafts -> each step's (bonus, drafts) of those seats."""
        bonus = torch.zeros(slots, dtype=torch.int64)
        drafts = torch.zeros(slots, K, dtype=torch.int64)
        pos = torch.full((slots,), -1, dtype=torch.int64)
        keep = torch.full((slots,), K, dtype=torch.int64)
        for s, n in seats.items():
            bonus[s], pos[s] = int(logits[s].reshape(-1)[:vocab].argmax()), n
            first = gen.first.pop(s, None)
            drafts[s] = torch.tensor(first) if first is not None else 0
        out = []
        for _ in range(steps):
            toks = torch.cat([bonus[:, None], drafts], dim=1)
            poss = torch.where(pos[:, None] >= 0, pos[:, None] + torch.arange(K + 1)[None, :], -1)
            n, b, d = gen.dflash_verify(toks, poss, keep)
            got = []
            for s in seats:
                ns = int(n[s])
                got.append((int(b[s]), d[s].tolist(), ns))
                bonus[s], drafts[s], keep[s], pos[s] = b[s], d[s], ns, pos[s] + ns + 1
            out.append(got)
        return out

    all_same = True
    for ci, (rows, n) in enumerate(cases):
        g = torch.Generator().manual_seed(2051 + ci)
        prompts = {r: torch.randint(0, vocab, (n,), generator=g).tolist() for r in rows}
        user = ci % USERS
        seat = {r: r * USERS + user for r in rows}
        lengths = {r: n for r in rows}
        # A: interleave on, through prefill_many as serving calls it
        os.environ[generator_module.SP_INTERLEAVE] = "1"
        rounds = gen.sp_rounds(lengths)
        out = gen.prefill_many([(seat[r], prompts[r]) for r in rows], traced=True)
        a_logits = {seat[r]: o.float() for r, o in zip(rows, out)}
        a_steps = verify_run({seat[r]: n for r in rows}, a_logits)
        if rounds is None or len(rounds) != 1:
            print(f"SPI case {ci} rows {rows} x {n}: rounds {rounds} (not one round over the rows: no B)", flush=True)
            continue
        _, t, blocks = rounds[0]
        # B: each request alone on a consecutive block of its A block's size (the same waves and segments a row)
        b_logits = {}
        for r in rows:
            size = len(blocks[r])
            ref = {r: tuple((r - size + 1 + j) % ROWS for j in range(size))}
            o = [None]
            gen.idle_pos[seat[r]] = min(n, gen.max_seq - 2)  # (as prefill_many sets it)
            gen._prefill_sp_round([(seat[r], prompts[r])], {r: 0}, t, ref, True, o)
            b_logits[seat[r]] = o[0].float()
        b_steps = verify_run({seat[r]: n for r in rows}, b_logits)
        same_logits = all(torch.equal(a_logits[s], b_logits[s]) for s in a_logits)
        worst = max((a_logits[s] - b_logits[s]).abs().max().item() for s in a_logits)
        same_steps = a_steps == b_steps
        same = same_logits and same_steps
        all_same = all_same and same
        # off: the consecutive blocks only (one row each where A interleaved): other values, printed only
        os.environ[generator_module.SP_INTERLEAVE] = "0"
        off_rounds = gen.sp_rounds(lengths)
        out = gen.prefill_many([(seat[r], prompts[r]) for r in rows], traced=True)
        off = {seat[r]: o.float() for r, o in zip(rows, out)}
        for s in off:
            gen.first.pop(s, None)
        d_off = max((off[s] - a_logits[s]).abs().max().item() for s in off)
        print(f"SPI case {ci} rows {rows} x {n}: t {t}, blocks {blocks}, logits {'same' if same_logits else 'DIFFER'} "
              f"(max |d| {worst:.3e}), {steps} verify steps {'same' if same_steps else 'DIFFER'} "
              f"({a_steps[-1][:2]} / {b_steps[-1][:2]}) -> {'SPI SAME' if same else 'SPI DIFFER'}; off: "
              f"{'one row each' if off_rounds is None else off_rounds}, max |d| {d_off:.3e} from A", flush=True)  # fmt: skip
    os.environ.pop(generator_module.SP_INTERLEAVE, None)
    gen.release()
    return all_same
