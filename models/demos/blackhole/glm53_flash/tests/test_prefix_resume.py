"""The prefix store's snapshot and restore (`Glm53Generator.prefix_snapshot` / `prefix_restore`) through the vLLM
adapter's chunked prefill (tools/glm53_vllm.py CHUNKED, as tests/test_vllm_chunked_prefill.py): a prompt's first
16,384 tokens prefilled on one seat in two calls of 8,192 and snapshotted, the mesh rows' state then overwritten by
another prompt's prefill and decode steps, and the snapshot restored into another seat of the same mesh row, which goes
on from 16,384 (`prefill_forward` from start_pos 16,384): its logits after the prompt and its greedy continuation
(DFlash2 verify steps with that seat alone) against the same prompt prefilled whole on that seat.  They must be the
same bit for bit.

Arms, in one process (the adapter warmed up as vLLM does), vLLM's state slots on the adapter's physical seats (slot k on
mesh row k mod 8, user k // 8; U = 4 a row):
  (A) the whole prompt in one call on slot 13 (row 5, user 1), then GLM53_PR_STEPS verify steps of it alone;
  (A2) the same again (the whole prefill's run-to-run sameness);
  (B) the prompt's first two chunks on slot 5 (row 5, user 0), the snapshot at 16,384 (timed);
  (C) another prompt (GLM53_PR_OTHER tokens, over the rows: every row's wave state and extra user rewritten) on slot 5
      itself, and 3 verify steps of it;
  (D) the restore into slot 13 (timed), the rest of the prompt from 16,384, the verify steps alone: against (A);
  (E) (GLM53_PR_CROSS=1, default) the entry's row bypassed: restored into slot 6 (row 6, user 0), the rest from 16,384:
      its logits against the whole prompt on slot 6 (A6), to see whether the rows' roles in the KDA wave states matter
      (the drafter's K / V are not on row 6: only the logits are compared).
GLM53_PR_LEN (default 16,684: two full chunks and 300 tokens).  Synthetic weights by default (GLM53_SYNTHETIC_WEIGHTS=0:
real).  Prints the verdict ("PREFIX RESUME BITS: PASS") and exits 0 (a study):

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_prefix_resume.py -s
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
spec_decode = pytest.importorskip("vllm_tt_plugin.spec_decode")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, ShapeOnlyCheckpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.prefix_store import GRID, prefix_keys  # noqa: E402

USERS = 4  # a row: 32 seats
SEATS = ROWS * USERS


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_prefix_resume(mesh_device):
    os.environ.setdefault("GLM53_SYNTHETIC_WEIGHTS", "1")
    os.environ["GLM53_MTP"] = "0"
    from models.demos.blackhole.glm53_flash.tools import glm53_vllm

    glm53_vllm.MTP, glm53_vllm.DFLASH2, glm53_vllm.SPEC, glm53_vllm.CHUNKED = False, True, True, True
    # (greedy requests only: the generator is made here without GLM53_DFLASH2_SAMPLING, which the adapter's start-up
    # sets when SAMPLING is on, its default where the plugin certifies sampled ids)
    glm53_vllm.SAMPLING = False
    synthetic = os.environ["GLM53_SYNTHETIC_WEIGHTS"] == "1"
    n = int(os.environ.get("GLM53_PR_LEN", "16684"))
    steps = int(os.environ.get("GLM53_PR_STEPS", "16"))
    other_len = int(os.environ.get("GLM53_PR_OTHER", "9000"))
    cross = os.environ.get("GLM53_PR_CROSS", "1") == "1"
    src, dst, far = 5, 13, 6  # vLLM's state slots: row 5 user 0, row 5 user 1, row 6 user 0
    length = (n - 1) // GRID * GRID
    assert length >= GRID, (n, GRID)
    t0 = time.time()
    chunks = glm53_vllm.prefill_chunks()
    max_len = -(-(n + 8 * (steps + 2)) // max(chunks)) * max(chunks)
    model = TtGlm53Model(
        mesh_device, ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint(), max_seq=max_len, mtp=False,
        dflash_shape=glm53_vllm.w1_shape(SEATS),  # (the experts' L1 landing GLM53_W1 by default, as `load_model`)
    )  # fmt: skip
    from models.demos.blackhole.glm53_flash.ttnn.dflash2 import TtDFlash2, load_weights

    k = glm53_vllm.dflash2_k(SEATS)
    cfg, w, dmask = load_weights(os.environ.get("GLM53_DFLASH2_DRAFTER"), synthetic=synthetic)
    drafter = TtDFlash2(
        mesh_device, cfg, w, dmask, users=USERS, max_ctx=max_len + 8, block=4 if k <= 3 else 8,
        head_weight=model.lm_head.mesh_weight, embedding=model.embed,
    )  # fmt: skip
    gen = Glm53Generator(model, USERS, chunk=chunks, spec=0, dflash=(drafter, k), prefix_entries=2)
    adapter = glm53_vllm.Glm53ForCausalLM(gen, max_len)
    phys = adapter.phys
    print(
        f"loaded in {time.time() - t0:.0f} s; cache {gen.max_seq}, K {k}, prompt {n}, snapshot at {length}; slots "
        f"{src} / {dst} / {far} on physical seats {phys[src]} / {phys[dst]} / {phys[far]} (rows "
        f"{phys[src] // USERS} / {phys[dst] // USERS} / {phys[far] // USERS}); store "
        f"{gen.prefix.bytes_a_chip() / 2**20:.0f} MiB a chip",
        flush=True,
    )
    assert phys[src] // USERS == phys[dst] // USERS != phys[far] // USERS
    t0 = time.time()
    adapter.warmup_model_prefill()
    adapter.warmup_model_decode()
    print(f"warmed up in {time.time() - t0:.0f} s", flush=True)

    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(29)
    prompt = torch.randint(0, vocab, (n,), generator=g)
    other = torch.randint(0, vocab, (other_len,), generator=g)
    mode = spec_decode.ACCEPT_MODE_ARGMAX_IDS
    counts = [1] * SEATS

    def prefill(slot: int, p: torch.Tensor, begin: int, end: int) -> torch.Tensor:
        """One prefill call as the plugin makes it -> the host logits [V] fp32 (host sampling)."""
        out = adapter.prefill_forward(
            p[:end].view(1, end),
            torch.tensor([begin]),
            torch.tensor([end]),
            empty_slots=torch.tensor([slot]),
            sampling_params=None,
        )
        return out[0, 0, :]

    def propose() -> list:
        out = adapter.propose_draft_tokens(
            k, torch.zeros(SEATS, 1, dtype=torch.int32), torch.zeros(SEATS, dtype=torch.int64), torch.tensor(counts)
        )
        return out.draft_token_ids.tolist()

    def verify(active: dict, count: int) -> None:
        for _ in range(count):
            tokens = torch.zeros(SEATS, k + 1, dtype=torch.int64)
            pos = torch.full((SEATS, k + 1), -1, dtype=torch.int64)
            valid = torch.zeros(SEATS, dtype=torch.int32)
            for s, r in active.items():
                tokens[s] = torch.tensor([r["tok"]] + r["draft"])
                pos[s] = r["pos"] + torch.arange(k + 1)
                valid[s] = k
            out = adapter.decode_forward(
                tokens,
                pos,
                spec_mode=mode,
                num_valid_drafts=valid,
                accepted_counts=torch.tensor(counts, dtype=torch.int32),
                sampling_params=None,
            )
            ids = out.argmax_ids.tolist()
            for s, r in active.items():
                a = 0
                while a < k and r["draft"][a] == ids[s][a]:
                    a += 1
                r["pos"], r["tok"] = r["pos"] + a + 1, ids[s][a]
                r["toks"].extend(ids[s][: a + 1])
                counts[s] = a + 1
            drafts = propose()
            for s, r in active.items():
                r["draft"] = drafts[s]

    def decode_alone(slot: int, logits: torch.Tensor, end: int, count: int) -> list:
        """The slot alone after its prefill: its first token and `count` verify steps -> its tokens."""
        for s in range(SEATS):
            counts[s] = 1
        tok = int(logits.argmax())
        run = {slot: dict(tok=tok, pos=end, toks=[tok])}
        run[slot]["draft"] = propose()[slot]
        verify(run, count)
        return run[slot]["toks"]

    def timed(fn):
        ttnn.synchronize_device(mesh_device)
        t1 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh_device)
        return out, time.perf_counter() - t1

    # (A) / (A2) the whole prompt on slot 13
    logits_a = prefill(dst, prompt, 0, n)
    toks_a = decode_alone(dst, logits_a, n, steps)
    logits_a2 = prefill(dst, prompt, 0, n)
    toks_a2 = decode_alone(dst, logits_a2, n, steps)
    # (B) the first two chunks on slot 5, the snapshot at `length`
    for begin in range(0, length, GRID):
        prefill(src, prompt, begin, begin + GRID)
    keys = prefix_keys(prompt[:length])
    entry, t_snap = timed(lambda: gen.prefix_snapshot(phys[src], length, keys))
    # (C) another prompt over the rows on slot 5 itself, and verify steps of it
    t1 = time.time()
    logits_c = prefill(src, other, 0, other_len)
    decode_alone(src, logits_c, other_len, 3)
    t_other = time.time() - t1
    # (D) the restore into slot 13, the rest from `length`
    found = gen.prefix.find(prefix_keys(prompt), n - 1)
    restored, t_restore = timed(lambda: gen.prefix_restore(entry, phys[dst]))
    t1 = time.time()
    logits_d = prefill(dst, prompt, length, n)
    t_rest = time.time() - t1
    toks_d = decode_alone(dst, logits_d, n, steps)
    print(
        f"snapshot {t_snap * 1e3:.1f} ms, restore {t_restore * 1e3:.1f} ms (entry {entry} of {restored} tokens; the "
        f"prompt's lookup {found}); the other prompt and its steps {t_other:.2f} s; the rest of the prompt "
        f"({n - length} tokens) {t_rest:.3f} s",
        flush=True,
    )
    verdict = []
    for name, lg, tk in (("whole again", logits_a2, toks_a2), ("restored and resumed", logits_d, toks_d)):
        same_l, same_t = torch.equal(logits_a, lg), toks_a == tk
        diff = (logits_a - lg).abs().max().item()
        first = next((i for i, (x, y) in enumerate(zip(toks_a, tk)) if x != y), None)
        print(
            f"{name} against the whole: logits {'the same' if same_l else 'DIFFERENT'} (max |diff| {diff:.3g}); the "
            f"seat's {steps} verify steps {'the same' if same_t else f'DIFFERENT from token {first}'} ({len(tk)} tokens)",
            flush=True,
        )
        verdict.append(same_l and same_t)
    verdict.append(found == (entry, length))
    print(f"PREFIX RESUME BITS: {'PASS' if all(verdict) else 'FAIL'} {verdict}", flush=True)
    if model.w1 is not None:  # (the landing by default: its DRISCs run from the warm-up's end over every call above)
        ttnn.synchronize_device(mesh_device)
        print(f"PR GLM53_W1 landing: C {model.w1.c}, DRISCs running {model.w1.running}; the first mesh row's 4 chips' "
              f"(gate | up blocks from the landing, read again, down halves from it, read again) "
              f"{model.w1.counts(range(4))}", flush=True)  # fmt: skip

    if not cross:
        return
    # (E) the entry into another row's seat (the rows' roles in the KDA wave states moved), logits only
    logits_a6 = prefill(far, prompt, 0, n)
    e = gen.prefix.entries[entry]
    e.row = phys[far] // USERS  # (the same-row rule bypassed: a study)
    gen.prefix_restore(entry, phys[far])
    logits_e = prefill(far, prompt, length, n)
    same = torch.equal(logits_a6, logits_e)
    diff = (logits_a6 - logits_e).abs().max().item()
    agree = int(logits_a6.argmax()) == int(logits_e.argmax())
    print(
        f"PREFIX RESUME CROSS-ROW (row {phys[src] // USERS} -> {phys[far] // USERS}): logits "
        f"{'the same' if same else 'DIFFERENT'} (max |diff| {diff:.3g}, argmax {'the same' if agree else 'different'})",
        flush=True,
    )
