"""Serving's flow through the vLLM adapter without vLLM: `Glm53ForCausalLM` (MTP, one draft) called in the plugin's order,
requests arriving and ending at different times on 32 seats that are used again: a served request must accept the
drafts as the generator called directly does (~0.9), whatever its seat held before.

The adapter warmed up as vLLM does (`warmup_model_prefill`, then `warmup_model_decode`).  Each flow (GLM53_AF_FLOWS,
"length:per_prefill:requests:outlo-outhi" comma separated; default "8192:8:40:48-96,1024:8:40:48-96,8192:1:16:48-96,
8192:8:8:900-1000"): `requests` random prompts of `length` tokens, each with an output length drawn from outlo-outhi
(an 8K prompt with up to 1,024 tokens out: positions past 8,192 + 2,048 pools); in a loop, the lowest free
seats (vLLM's state slots) take up to `per_prefill` waiting requests in one `prefill_forward` (the first token its
greedy id), `propose_draft_tokens` gives the new rows their first drafts, then one `decode_forward(spec_mode=..)` over
all 32 seats (rows without a request at -1), the commit (the draft accepted when it equals the first greedy id: two
tokens, else one), `propose_draft_tokens` for the next drafts; a request that has its output frees its seat.  Per
request: its acceptance (accepted / proposed drafts), whether its seat (the physical slot) held an earlier request,
and its prefill's size; then by first use and reuse, by prefill size, and by the output position (bands of 128).
A flow with a fifth field "plugin" (or "plugin-gen") follows the plugin's own bookkeeping instead (`plugin_flow`: state
slot names, the input batch's holes and condense, `slot_remap`, no first draft after a prefill) and counts the
requests that broke (the second half of their drafts accepted under 0.3; a broken request makes about one token a
verify; a warm-up that left a later capture's buffers in an earlier one's scratch once broke most reused seats).
GLM53_AF_SPEC=0: no MTP (the plain greedy decode); GLM53_AF_POOL (default 8) distinct prompts, each request's first
64 tokens against its prompt's first run; GLM53_AF_PROMPT=repeat: each prompt a random 64-token block over and over, and
`pattern` the share of a request's first 64 tokens that continue it (a check of the text on its own).
A short regression run (synthetic weights, about 3 minutes; the guard stops at a buffer allocated after the warm-up and
alive at a replay): TT_METAL_TRACE_ALLOC_TRACKING=1 GLM53_TRACE_ALLOC_GUARD=1 GLM53_SYNTHETIC_WEIGHTS=1
GLM53_AF_FLOWS=1024:8:40:20-40:plugin,8192:3:9:20-40:plugin
    GLM53_AF_MAX_LEN (default 9408: max_model_len; the cache rounded up to 9728), real weights by default
    (GLM53_SYNTHETIC_WEIGHTS=1: random, the plumbing only).  Needs vllm_tt_plugin installed:
    GLM53_DEVICE_TEST=1 GLM53_MTP=1 pytest .../tests/test_vllm_adapter_flow.py -s
GLM53_AF_SAMPLE_RIDS (with GLM53_AF_VLLM_SAMPLING=1 and GLM53_AF_SAMPLE): only these requests sampled, with seeds of
their own (two runs compare), so that the steps mix sampled and greedy rows and run greedy-only between them.
GLM53_AF_DFLASH2=1: the simple flows with DFlash2 (K drafts a step for the 32 seats, the drafter on the device; its
weights random with GLM53_SYNTHETIC_WEIGHTS=1, else GLM53_DFLASH2_DRAFTER), the plugin's acceptance up to the first
draft that differs; the acceptance per draft and the tokens a verify step.  GLM53_TEST_FORCE_ACCEPT=1 (`DFlash2Step`,
`verify_accept.forced_accept`): each seat's accepted drafts n pseudo-random in 0 .. K from its position and seat, the
same rule in every arm, so that the synthetic weights' runs (no draft accepted) keep 0 .. K drafts too.
The prefill chunks are the adapter's (GLM53_PREFILL_CHUNK, `glm53_vllm.prefill_chunks`), the traced ones its
GLM53_PREFILL_TRACE; after the warm-up an "AF region" line: the trace region and DRAM taken.  The simple flows run the
plain greedy decode too (GLM53_AF_SPEC=0) and dump their requests' tokens (GLM53_AF_DUMP, the first of each request's
output length); GLM53_AF_COMPARE=<a dump>: each request's tokens against another run's (GLM53_AF_COMPARE_STRICT=1:
every request the same, else the test fails).
GLM53_VLLM_SPREAD_ROWS=1 (the adapter's `_place`): a prefill call's new requests on free
seats of distinct mesh rows; the summary's "AF waves" line counts the prefill waves against the calls (one a call when
the rows allow), and GLM53_AF_COMPARE against a run without it shows whether the requests' tokens stay the same.  The
simple flows' GLM53_AF_SEAT_ORDER=samerow takes a mesh row's free seats first (a prefill's requests share rows).
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")
spec_decode = pytest.importorskip("vllm_tt_plugin.spec_decode")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, ShapeOnlyCheckpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import common as common_module  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

USERS = 4  # a row: 32 seats
SEATS = ROWS * USERS


class Greedy:  # (the adapter only checks that sampling params are given: device greedy ids)
    temperature = [0.0] * SEATS
    top_k = [1] * SEATS


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_vllm_adapter_flow(mesh_device):
    dumped = {}  # GLM53_AF_DUMP: flow -> request -> its tokens
    spec = int(os.environ.get("GLM53_AF_SPEC", "1"))  # 0: no MTP (the plain greedy decode; plugin flows only)
    pool = int(os.environ.get("GLM53_AF_POOL", "8"))  # plugin flows: prompts from this many distinct ones (0: all new)
    # plugin flows: "repeat": each prompt a random block of BLOCK tokens over and over, whose continuation (the block
    # again) a healthy model writes with confidence, so a request's text is checked on its own (`pattern`)
    prompt_kind = os.environ.get("GLM53_AF_PROMPT", "random")
    block_len = 64
    # GLM53_AF_DFLASH2=1: DFlash2's drafter (K drafts a step, the adapter's K for the seats) in place of MTP; the basic
    # flows only (the plugin's bookkeeping, `plugin_flow`, keeps MTP's one draft)
    dflash = os.environ.get("GLM53_AF_DFLASH2") == "1"
    os.environ["GLM53_MTP"] = str(0 if dflash else spec)
    # GLM53_AF_VLLM_SAMPLING=1 (with GLM53_AF_DFLASH2): the adapter's sampled requests (GLM53_VLLM_SAMPLING: the
    # verify step's draws, and its logits step); GLM53_AF_SAMPLE="T:P:K" every request sampled
    # so (else greedy); GLM53_AF_LOGITS_EVERY=n every n-th verify step in the plugin's "logits" mode, accepted here as the
    # plugin's host walk does (a greedy row by the argmax, a sampled row by a draw), and the greedy rows' acceptance from
    # the logits checked against the device's own in that step (the flow's "logits" entry)
    vllm_sampling = dflash and os.environ.get("GLM53_AF_VLLM_SAMPLING") == "1"
    if vllm_sampling:
        os.environ["GLM53_DFLASH2_SAMPLING"] = os.environ["GLM53_DFLASH2_LOGITS"] = "1"
    sample = (
        tuple(float(x) for x in os.environ["GLM53_AF_SAMPLE"].split(":")) if os.environ.get("GLM53_AF_SAMPLE") else None
    )
    logits_every = int(os.environ.get("GLM53_AF_LOGITS_EVERY", "0"))
    # GLM53_AF_SAMPLE_RIDS="lo-hi[,lo-hi]" (with GLM53_AF_SAMPLE): only these requests of each flow (by their number,
    # hi excluded) sampled, each with its own seed 7000 + its number (the same draws in every run: two runs' tokens
    # compare, GLM53_AF_COMPARE), the rest greedy; the steps then mix both and run greedy-only while no sampled request
    # is in (GLM53_DFLASH2_SAMPLING_SPLIT's two forms).  Unset: every request sampled, its seed the adapter's
    sample_rids = None
    if os.environ.get("GLM53_AF_SAMPLE_RIDS"):
        sample_rids = set()
        for part in os.environ["GLM53_AF_SAMPLE_RIDS"].split(","):
            lo, hi = (int(x) for x in part.split("-"))
            sample_rids.update(range(lo, hi))
    assert vllm_sampling or not (sample or logits_every), "GLM53_AF_SAMPLE / _LOGITS_EVERY: GLM53_AF_VLLM_SAMPLING=1"
    from models.demos.blackhole.glm53_flash.tools import glm53_vllm

    glm53_vllm.MTP = bool(spec) and not dflash
    glm53_vllm.DFLASH2 = dflash
    glm53_vllm.SPEC = glm53_vllm.MTP or dflash
    glm53_vllm.SAMPLING = vllm_sampling
    max_len = int(os.environ.get("GLM53_AF_MAX_LEN", "9408"))
    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1"
    t0 = time.time()
    chunks = glm53_vllm.prefill_chunks()  # (the adapter's: GLM53_PREFILL_CHUNK, default 128,512,1024)
    chunk = max(chunks)
    max_seq = -(-max_len // chunk) * chunk
    model = TtGlm53Model(
        mesh_device,
        ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint(),
        max_seq=max_seq,
        mtp=glm53_vllm.MTP,
        dflash_shape=glm53_vllm.w1_shape(SEATS),  # (the expert-weight landing by default, as `load_model`)
    )
    dk, dfl = 0, None
    if dflash:  # as the adapter's `load_model` makes it
        from models.demos.blackhole.glm53_flash.ttnn.dflash2 import TtDFlash2, load_weights

        dk = glm53_vllm.dflash2_k(SEATS)
        # GLM53_AF_DRAFTER_REAL=1: the drafter's checkpoint with a synthetic target (synthetic drafter weights differ by
        # shape: its context rows' norm constants, `_ctx_norm_rows(R)`, are another random tensor for each R)
        real_drafter = os.environ.get("GLM53_AF_DRAFTER_REAL") == "1"
        cfg, w, dmask = load_weights(os.environ.get("GLM53_DFLASH2_DRAFTER"), synthetic=synthetic and not real_drafter)
        drafter = TtDFlash2(
            mesh_device, cfg, w, dmask, users=USERS, max_ctx=max_seq + 8, block=4 if dk <= 3 else 8,
            head_weight=model.lm_head.mesh_weight, embedding=model.embed,
        )  # fmt: skip
        dfl = (drafter, dk)
    gen = Glm53Generator(model, USERS, chunk=chunks, spec=0 if dflash else spec, dflash=dfl)
    adapter = glm53_vllm.Glm53ForCausalLM(gen, max_len)
    phys0 = list(adapter.phys)  # (each flow starts from it: `flow`)
    prefill_calls = [0]  # (the "AF waves" line: the generator's waves against the adapter's prefill calls)
    adapter_prefill = adapter.prefill_forward

    def counted_prefill(*args, **kwargs):
        prefill_calls[0] += 1
        return adapter_prefill(*args, **kwargs)

    adapter.prefill_forward = counted_prefill
    device_accept = {}  # (GLM53_AF_LOGITS_EVERY: the device's own acceptance in the last DFlash2 step, by slot)
    if dflash:
        inner_verify = gen.dflash_verify

        def recorded_verify(*args, **kwargs):
            got = inner_verify(*args, **kwargs)
            device_accept["n"], device_accept["bonus"] = got[0].clone(), got[1].clone()
            return got

        gen.dflash_verify = recorded_verify
    print(
        f"loaded in {time.time() - t0:.0f} s; cache {gen.max_seq}, max_model_len {max_len}, prefill traced "
        f"{adapter.prefill_traced}",
        flush=True,
    )
    t0 = time.time()
    adapter.warmup_model_prefill()
    adapter.warmup_model_decode()
    print(f"warmed up (the adapter's two calls) in {time.time() - t0:.0f} s", flush=True)
    waves0 = gen.counts["waves"]
    from models.demos.blackhole.glm53_flash.ttnn import model as model_module

    dram = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)  # (one chip's banks; every chip holds the same)
    banks = int(dram.num_banks)
    print(
        f"AF region: chunks {chunks}, traced {adapter.prefill_traced}, cache {gen.max_seq}: the trace region holds "
        f"{model_module.trace_bytes(mesh_device) / 2**20:.1f} MiB ({len(gen.prefill_traces)} prefill forms, "
        f"{sum(c.bytes for c in gen.prefill_traces.values()) / 2**20:.1f} MiB of them) of "
        f"{common_module.TRACE_REGION_MB} MiB; DRAM a chip {int(dram.total_bytes_allocated_per_bank) * banks / 2**30:.2f} "
        f"GiB allocated, {int(dram.total_bytes_free_per_bank) * banks / 2**30:.2f} GiB free",
        flush=True,
    )
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(13)
    mode = spec_decode.ACCEPT_MODE_ARGMAX_IDS
    used: set = set()  # physical seats that have held a request (over all flows)
    uses: dict = {}  # physical seat -> the requests it has held (over the plugin flows)

    nd = dk or 1  # drafts a step
    # the simple flows' free seats in this order: lowest (default), highest, or shift:k (seat s at (s + k) mod 32)
    seat_order = os.environ.get("GLM53_AF_SEAT_ORDER", "lowest")
    # GLM53_AF_VLLM_CUT=1 (simple flows; read at each flow, an arm may set it): vLLM's own end of a request at
    # max_model_len (0.31 scheduler: a step's new tokens <= max_model_len - computed - 1): a seat's valid drafts min(K,
    # max_len - pos - 2), the request ends when its tokens reach max_len (pos + 1 >= max_len), its budget to that end;
    # else (the default, the earlier form) a request ends before its drafts would be cut (pos + K + 1 >= max_len) (read
    # at each flow: an arm may set it)
    # GLM53_AF_GATHER="m:wait" (simple flows; read at each flow, an arm may set it): the scheduler's gathering
    # (`glm53_vllm_scheduler`, GLM53_PREFILL_BATCH m, its wait counted in this flow's loop turns): while requests run,
    # a prefill waits until m waiting requests could be admitted (at most the free seats, and one a free mesh row when
    # the adapter spreads them: GLM53_VLLM_SPREAD_ROWS, on by default here with it), or `wait` turns; with the run-ahead
    # the admission told while it holds is none (GLM53_PREFILL_BATCH_AHEAD's), unless it ends at the next turn.
    # GLM53_VLLM_AHEAD_SEATS=1 (the adapter's): before a prefill the scheduler's admission after a prefill step (a decode
    # comes next: none), so that the adapter may launch the next verify at the prefill's end

    walk_gen = torch.Generator().manual_seed(29)  # (the logits steps' draws for sampled rows)

    def dist(row: torch.Tensor) -> torch.Tensor:
        """A sampled row's distribution from its logits: temperature, top-k, top-p (GLM53_AF_SAMPLE), as vLLM's."""
        temp, top_p, top_k = sample
        x = row.double() / temp
        if top_k > 0:
            x = torch.where(x < torch.topk(x, int(top_k)).values[-1], float("-inf"), x)
        pr = torch.softmax(x, -1)
        srt, idx = pr.sort(descending=True)
        keep = torch.zeros_like(pr)
        keep[idx[(srt.cumsum(0) - srt) < top_p]] = 1.0
        pr = pr * keep
        return pr / pr.sum()

    def drawn(r: dict) -> bool:
        """Whether a request is sampled (GLM53_AF_SAMPLE, GLM53_AF_SAMPLE_RIDS)."""
        return sample is not None and (sample_rids is None or r["rid"] in sample_rids)

    def logits_ids(lg: torch.Tensor, ds: list, sampled: bool = True) -> list:
        """A logits step's row -> ids [K + 1] as the plugin's host walk commits them (the drafts it accepts, then its
        correction or bonus, repeated): a greedy row the argmax at each column; a sampled row draft j accepted with
        p(d_j), else a draw from p without it, a bonus drawn after them all."""
        if sample is None or not sampled:
            return lg.argmax(-1).tolist()
        for j, d in enumerate(ds):
            pr = dist(lg[j])
            if float(torch.rand(1, generator=walk_gen)) < float(pr[d]):
                continue
            pr[d] = 0.0
            x = int(torch.multinomial(pr / pr.sum(), 1, generator=walk_gen))
            return ds[:j] + [x] * (len(ds) - j + 1)
        return ds + [int(torch.multinomial(dist(lg[len(ds)]), 1, generator=walk_gen))]

    def propose(counts: list) -> list:
        """Each seat's next drafts: a token (MTP), or a list of K (DFlash2)."""
        out = adapter.propose_draft_tokens(
            nd,
            torch.zeros(SEATS, 1, dtype=torch.int32),
            torch.zeros(SEATS, dtype=torch.int64),
            torch.tensor(counts, dtype=torch.int32),
        )
        return out.draft_token_ids.tolist() if dk else out.draft_token_ids[:, 0].tolist()

    def flow(length: int, per_prefill: int, total: int, lo: int, hi: int) -> dict:
        # each flow from the adapter's first slot permutation: the spreading (`_place`) moves it, and an arm compared
        # with another must place the same requests on the same seats (the gathering's arms, from the second on, had
        # started where the one before left it: 32 / 48 requests the same from the first prefill)
        adapter.phys = list(phys0)
        # and from no vLLM slot in use: the adapter's `active` (the last call's rows) is what `_place` keeps off, and a
        # flow's last verify left its rows there (once: the 1024:4:24 flow's "1" arm started with the "dry" arm's last
        # slot 8 busy, so its first request went to seat 16 instead of 0: 18 / 24 requests the same as "dry")
        adapter.active = set()
        gather = os.environ.get("GLM53_AF_GATHER")
        gather = tuple(int(x) for x in gather.split(":")) if gather else None
        spread_saved = glm53_vllm.SPREAD_ROWS
        if gather:
            glm53_vllm.SPREAD_ROWS = os.environ.get("GLM53_VLLM_SPREAD_ROWS", "1") == "1"
        try:
            return flow_inner(length, per_prefill, total, lo, hi, gather)
        finally:
            glm53_vllm.SPREAD_ROWS = spread_saved

    def flow_inner(length: int, per_prefill: int, total: int, lo: int, hi: int, gather) -> dict:
        held = None  # (GLM53_AF_GATHER: the turns the current hold has lasted)
        vllm_cut = os.environ.get("GLM53_AF_VLLM_CUT") == "1"
        seats_on = os.environ.get("GLM53_VLLM_AHEAD_SEATS", "0") == "1"
        # GLM53_AF_PLUGIN_POST_PREFILL=1 (read at each flow): a request's first verify after its prefill with no drafts,
        # as the vLLM TT plugin sends it (its draft gathering's "post-prefill default": the runner proposes after a
        # verify only); else (the default, the earlier form) the drafter's first drafts proposed after the prefill
        no_first = os.environ.get("GLM53_AF_PLUGIN_POST_PREFILL") == "1"
        # GLM53_AF_ACCEPT_CAP=1 (read at each flow; with GLM53_TEST_FORCE_ACCEPT): the plugin's acceptance at most the
        # device's n, as with a real acceptance (the first draft past n differs from the bonus there).  Forced, the bonus
        # may equal the next draft and the plugin then accepts more than the device: a step launched ahead on the
        # device's next inputs is thrown away and the rerun moves that seat a position on, so the row the thrown step
        # wrote there (the DSA latent, the drafter's context) keeps another computation of the same token (once:
        # 47 / 48 requests the same, one bonus differing 12 steps later)
        accept_cap = os.environ.get("GLM53_AF_ACCEPT_CAP") == "1"
        # GLM53_AF_FALSE_ENDS=m (read at each flow): every m-th request (by its number), once at half its output, has
        # a budget saying it ends at the next step (1 token left) while it goes on: an end foreseen wrongly (as a stop
        # id the vLLM side ignores under min_tokens).  With GLM53_VLLM_AHEAD_SEATS the step launched with its seat
        # sitting out is then thrown away: the seat must sit out past its tokens (its next position), never at its idle
        # position 0, whose rows are its prompt's (`Glm53ForCausalLM._given_inputs`); "1" against "dry" checks it
        false_ends = int(os.environ.get("GLM53_AF_FALSE_ENDS", "0"))
        # GLM53_AF_PREFILL_HOST=1 (read at each flow): each prefill sampled on the host, as served with
        # sample_on_device_mode "decode_only" (SAMPLING): the call without sampling params returns the logits and the
        # test (the plugin) takes their argmax; the scheduler's publication of the requests whose first token is that
        # argmax (`glm53_vllm_rows.publish_first_argmax`, by vLLM slot here: no page tables) before the call returns
        host_prefill = os.environ.get("GLM53_AF_PREFILL_HOST") == "1"
        # GLM53_AF_FIRST_END=m (read at each flow): every m-th request (by its number) ends at its prefill, its first
        # token its only output (max_tokens 1, a first stop token, an abort): its seat never takes a verify as a
        # request, the seat's idle position the prefill gave it
        first_end = int(os.environ.get("GLM53_AF_FIRST_END", "0"))

        def gather_choice(free_n: int, turns):
            """The scheduler's gathering choice now ("plain" / "hold" / "release"; None: no gathering)."""
            if gather is None:
                return None
            from models.demos.blackhole.glm53_flash.tools import glm53_vllm_rows as rows_g

            return rows_g.gather(
                waiting=len(pending), free_seats=free_n, rows=rows_g.free_rows() if glm53_vllm.SPREAD_ROWS else None,
                decodes=len(active), partial=False, held_ms=turns, batch=gather[0], wait_ms=gather[1],
            )  # fmt: skip

        pending = [
            (
                torch.randint(0, vocab, (length,), generator=g).tolist(),
                int(torch.randint(lo, hi + 1, (1,), generator=g)),
            )
            for _ in range(total)
        ]
        active: dict = {}  # state slot -> the request's progress
        done, counts = [], [1] * SEATS
        rid = 0  # (the requests in their order: the same in any run of the flow)
        bands: dict = {}  # (position - length) // 128 -> [accepted, proposed]
        steps = prefills = 0
        events = []  # the flow's prefills and verify steps in order (GLM53_AF_COMPARE: the first that differs)
        # GLM53_AF_LOGITS_EVERY: the logits steps, the greedy rows checked against the device's acceptance and those
        # that agreed; [accepted, proposed] drafts in the logits steps and the device-accepted ones
        lstat = dict(steps=0, rows=0, agree=0, logits=[0, 0], device=[0, 0])
        t1 = time.time()
        while pending or active:
            free = [s for s in range(SEATS) if s not in active]
            if seat_order == "highest":  # (GLM53_AF_SEAT_ORDER: the same requests on other seats, other mesh rows)
                free = free[::-1]
            elif seat_order.startswith("shift:"):
                k = int(seat_order.split(":")[1])
                free = sorted(free, key=lambda s: (s + k) % SEATS)
            elif seat_order == "samerow":  # the free seats of one mesh row (as named at the start) first: a prefill's
                free = sorted(
                    free, key=lambda s: (s % ROWS, s)
                )  # requests share rows (GLM53_VLLM_SPREAD_ROWS spreads them)
            # GLM53_VLLM_AHEAD ("1" or "dry"): the scheduler's part (glm53_vllm_scheduler): no prefill while the adapter
            # has a verify step launched ahead and requests run (a decode step instead)
            from models.demos.blackhole.glm53_flash.tools import glm53_vllm_rows as rows_mod

            ahead_mode = os.environ.get("GLM53_VLLM_AHEAD", "0") in ("1", "dry")
            # (GLM53_VLLM_AHEAD_HOLD=0: no hold, the scheduler's switch: a prefill may follow a launched step, which it
            # throws away, KDA's state undone first)
            hold = (
                ahead_mode and rows_mod.ahead() and bool(active) and os.environ.get("GLM53_VLLM_AHEAD_HOLD", "1") == "1"
            )
            choice = gather_choice(len(free), held) if (pending and free and not hold) else None
            if choice == "hold":  # (GLM53_AF_GATHER: a decode turn instead, the hold's clock on)
                held = (held or 0) + 1
            elif choice is not None:
                held = None
            if (
                pending and free and not hold and choice != "hold"
            ):  # the lowest free seats take the waiting requests, up to per_prefill at once
                cap = len(free)
                if choice is not None and glm53_vllm.SPREAD_ROWS and rows_mod.free_rows() is not None:
                    cap = min(cap, rows_mod.free_rows())  # (the scheduler's: one a free mesh row)
                take = free[: min(per_prefill, len(pending), cap)]
                if ahead_mode and seats_on:  # (the scheduler after a prefill step: a decode comes next)
                    rows_mod.publish_admission(False)
                reqs, pending = pending[: len(take)], pending[len(take) :]
                tokens = torch.zeros(len(take), length, dtype=torch.int64)
                for i, (prompt, _) in enumerate(reqs):
                    tokens[i] = torch.tensor(prompt)
                if host_prefill and not take:  # (no seat to take: vLLM sends no prefill)
                    ids = torch.zeros(0, 1, dtype=torch.int64)
                elif host_prefill:  # (the scheduler's publication after the prefill step, then the host's argmax)
                    rows_mod.publish_first_argmax(take)
                    logits = adapter.prefill_forward(
                        tokens,
                        torch.zeros(len(take), dtype=torch.int64),
                        torch.tensor([length] * len(take)),
                        empty_slots=torch.tensor(take),
                    )
                    ids = logits[:, 0, :].argmax(-1, keepdim=True)
                else:
                    ids = adapter.prefill_forward(
                        tokens,
                        torch.zeros(len(take), dtype=torch.int64),
                        torch.tensor([length] * len(take)),
                        empty_slots=torch.tensor(take),
                        sampling_params=Greedy(),
                    )
                prefills += 1
                for i, s in enumerate(take):
                    p = adapter.phys[s]
                    active[s] = dict(
                        pos=length,
                        tok=int(ids[i, 0]),
                        out=1,
                        want=reqs[i][1],
                        proposed=0,
                        accepted=0,
                        reused=p in used,
                        batch=len(take),
                        phys=p,
                        rid=rid,
                        toks=[int(ids[i, 0])],
                    )
                    rid += 1
                    used.add(p)
                    counts[s] = 1
                if (spec or dk) and no_first:  # (the plugin's post-prefill default: no drafts at the first verify)
                    for s in take:
                        active[s]["draft"] = [0] * nd if dk else 0
                        active[s]["nodraft"] = True
                elif spec or dk:
                    drafts = propose(counts)
                    for s in take:
                        active[s]["draft"] = drafts[s]
                events.append(
                    (
                        "prefill",
                        prefills,
                        steps,
                        tuple(take),
                        tuple(int(ids[i, 0]) for i in range(len(take))),
                        tuple(tuple(active[s]["draft"]) if dk else (active[s].get("draft"),) for s in take),
                    )
                )
                for s in take:  # (GLM53_AF_FIRST_END: done at the prefill, its seat free again)
                    if first_end and active[s]["rid"] % first_end == 0:
                        active[s]["want"] = 1
                        done.append(active.pop(s))
                        counts[s] = 1
            if not (spec or dk):  # the plain greedy decode (GLM53_AF_SPEC=0): a token a step, device sampling
                tokens = torch.zeros(SEATS, 1, dtype=torch.int64)
                start = torch.full((SEATS,), -1, dtype=torch.int64)
                for s, r in active.items():
                    tokens[s, 0], start[s] = r["tok"], r["pos"]
                got = adapter.decode_forward(tokens, start, slot_remap=None, sampling_params=Greedy()).view(-1).tolist()
                steps += 1
                for s in list(active):
                    r = active[s]
                    r["pos"], r["tok"], r["out"] = r["pos"] + 1, got[s], r["out"] + 1
                    r["toks"].append(got[s])
                    if r["out"] >= r["want"] or r["pos"] + 2 >= max_len:
                        done.append(r)
                        del active[s]
                continue
            tokens = torch.zeros(SEATS, nd + 1, dtype=torch.int64)
            start = torch.full((SEATS, nd + 1), -1, dtype=torch.int64)
            valid = torch.zeros(SEATS, dtype=torch.int32)
            for s, r in active.items():
                tokens[s] = torch.tensor([r["tok"]] + (r["draft"] if dk else [r["draft"]]))
                start[s] = r["pos"] + torch.arange(nd + 1)
                valid[s] = max(0, min(nd, max_len - r["pos"] - 2)) if vllm_cut else nd
                if r.pop("nodraft", False):  # (GLM53_AF_PLUGIN_POST_PREFILL)
                    valid[s] = 0
            if ahead_mode:  # (the scheduler's: whether a waiting request could take a seat now, each request's end)
                free_now = sum(s not in active for s in range(SEATS))
                admit = bool(pending) and free_now > 0
                if admit and gather is not None and os.environ.get("GLM53_PREFILL_BATCH_AHEAD", "1") == "1":
                    nxt = gather_choice(free_now, held)  # (GLM53_PREFILL_BATCH_AHEAD: no admission while it holds)
                    admit = nxt != "hold" or (held or 0) + 1 >= gather[1]
                rows_mod.publish_admission(admit)
                budgets = {
                    s: (
                        min(r["want"] - r["out"], max_len - 1 - r["pos"] if vllm_cut else max_len - nd - 1 - r["pos"]),
                        frozenset(),
                    )
                    for s, r in active.items()
                }
                for s, r in active.items():  # (GLM53_AF_FALSE_ENDS: an end foreseen wrongly, once)
                    if false_ends and r["rid"] % false_ends == 0 and r["out"] >= r["want"] // 2 and not r.get("fe"):
                        budgets[s], r["fe"] = (1, frozenset()), True
                rows_mod.publish_budgets(budgets)  # (keyed by the batch row: no page tables here; the ends below)
            in_logits = bool(logits_every) and (steps + 1) % logits_every == 0
            params = None  # (a device-sampled step's parameters: GLM53_AF_SAMPLE; a logits step gets none, as served)
            if sample is not None and not in_logits:
                temp, top_p, top_k = sample
                on = [s in active and drawn(active[s]) for s in range(SEATS)]
                params = SimpleNamespace(
                    temperature=[temp if on[s] else 0.0 for s in range(SEATS)],
                    top_k=[int(top_k)] * SEATS,
                    top_p=[top_p] * SEATS,
                    seed=[7000 + active[s]["rid"] if on[s] and sample_rids is not None else None for s in range(SEATS)],
                )
            out = adapter.decode_forward(
                tokens,
                start,
                spec_mode=spec_decode.ACCEPT_MODE_LOGITS if in_logits else mode,
                num_valid_drafts=valid,
                accepted_counts=torch.tensor(counts, dtype=torch.int32),
                sampling_params=params,
            )
            if in_logits:
                assert tuple(out.logits.shape) == (SEATS, nd + 1, vocab), tuple(out.logits.shape)
                ids = [
                    logits_ids(out.logits[s], active[s]["draft"], drawn(active[s])) if s in active else [0] * (nd + 1)
                    for s in range(SEATS)
                ]
                lstat["steps"] += 1
                if sample is None:  # the argmax walk on the logits against the device's own acceptance in the step
                    for s, r in active.items():
                        a = 0
                        while a < nd and r["draft"][a] == ids[s][a]:
                            a += 1
                        p = adapter.phys[s]
                        lstat["rows"] += 1
                        lstat["agree"] += int(
                            a == int(device_accept["n"][p]) and ids[s][a] == int(device_accept["bonus"][p])
                        )
            else:
                ids = out.argmax_ids.tolist()
            steps += 1
            step_ev = []
            for s, r in active.items():
                ds = (r["draft"] if dk else [r["draft"]])[: int(valid[s])]  # (GLM53_AF_VLLM_CUT: the valid ones)
                a = 0  # the plugin's acceptance: the drafts up to the first that differs from the greedy id there
                while a < len(ds) and ds[a] == ids[s][a]:
                    a += 1
                if accept_cap and "n" in device_accept:  # (GLM53_AF_ACCEPT_CAP)
                    a = min(a, int(device_accept["n"][adapter.phys[s]]))
                step_ev.append((s, tuple(ds), a, tuple(ids[s][: a + 1])))
                r["proposed"] += len(ds)
                r["accepted"] += a
                kind = lstat["logits" if in_logits else "device"]
                kind[0] += a
                kind[1] += len(ds)
                band = bands.setdefault((r["pos"] - length) // 128, [0, 0])
                band[0] += a
                band[1] += len(ds)
                r["pos"], r["tok"], r["out"] = r["pos"] + a + 1, ids[s][a], r["out"] + a + 1
                r["toks"].extend(ids[s][: a + 1])
                counts[s] = a + 1
            events.append(("verify", prefills, steps, tuple(sorted(step_ev))))
            host_ms = float(os.environ.get("GLM53_AF_HOST_MS", "0"))  # (vLLM's own host work between two calls)
            if host_ms > 0:
                time.sleep(host_ms / 1e3)
            drafts = propose(counts)
            for s in list(active):
                r = active[s]
                r["draft"] = drafts[s]
                # GLM53_AF_SURPRISE_ENDS=m: every m-th request (by its number) also ends at half its output, an end the
                # budgets do not foresee (as a stop string's): a step launched ahead is then thrown away and undone
                surprise = surprise_ends and r["rid"] % surprise_ends == 0 and r["out"] >= r["want"] // 2
                if (
                    r["out"] >= r["want"]
                    or surprise
                    or (r["pos"] + 1 >= max_len if vllm_cut else r["pos"] + nd + 1 >= max_len)
                ):
                    done.append(r)
                    del active[s]
                    counts[s] = 1
        rate = lambda rs: sum(r["accepted"] for r in rs) / max(1, sum(r["proposed"] for r in rs))  # noqa: E731
        first = [r for r in done if not r["reused"]]
        again = [r for r in done if r["reused"]]
        by_batch = {b: rate([r for r in done if r["batch"] == b]) for b in sorted({r["batch"] for r in done})}
        summary = dict(
            requests=len(done),
            prefills=prefills,
            steps=steps,
            seconds=round(time.time() - t1, 1),
            acceptance=round(rate(done), 3),
            tokens_a_step=round(1 + nd * rate(done), 3),
            first_use=(len(first), round(rate(first), 3)),
            reused=(len(again), round(rate(again), 3)),
            by_prefill_size={b: round(v, 3) for b, v in by_batch.items()},
            by_position={f"+{128 * b}": round(a / max(1, n), 3) for b, (a, n) in sorted(bands.items())},
        )
        if logits_every:
            summary["logits"] = dict(
                steps=lstat["steps"],
                greedy_rows_agree=(lstat["agree"], lstat["rows"]),
                acceptance_logits=round(lstat["logits"][0] / max(1, lstat["logits"][1]), 3),
                acceptance_device=round(lstat["device"][0] / max(1, lstat["device"][1]), 3),
            )
        print(f"flow {length}:{per_prefill}:{total}: {summary}", flush=True)
        dumped[f"{length}:{per_prefill}:{total}"] = {r["rid"]: list(r["toks"][: r["want"]]) for r in done}
        dumped[f"{length}:{per_prefill}:{total}:events"] = events
        dumped[f"{length}:{per_prefill}:{total}:seats"] = {
            r["rid"]: (r["phys"], r["reused"], r["batch"]) for r in done
        }  # (rid -> its seat's facts)
        print(
            "  per request (phys seat, reused, prefill size, accepted / proposed): "
            + ", ".join(
                f"{r['phys']}{'R' if r['reused'] else ''}/{r['batch']}:{r['accepted']}/{r['proposed']}" for r in done
            ),
            flush=True,
        )
        return summary

    def plugin_flow(length: int, per_prefill: int, total: int, lo: int, hi: int, first: str) -> dict:
        """The plugin's own bookkeeping (vllm_tt_plugin/model_runner.py): its state slot names
        (`_alloc_prefill_state_slots`: a prefill row k takes name k unless held, else the lowest free name), the input
        batch's rows (finished requests leave holes, new requests fill the lowest holes, `condense` moves the last
        rows into the rest), the decode's `slot_remap` (row i reads name remap[i]: each row's name, then the free
        names; None for the identity) and the names settling to the rows after it, decode rows padded to 32 with
        position -1; no proposal after a prefill (`propose_after_plain_step` runs after decode steps only), so a new
        request's first verify carries no draft (num_valid 0) and its count is 1.  first == "gen": the first draft
        given anyway (the generator's, as `flow`).  The free seats are filled (up to per_prefill a prefill, several
        prefills in a row) before each verify."""
        if prompt_kind == "repeat":
            blocks = [torch.randint(0, vocab, (block_len,), generator=g).tolist() for _ in range(pool or total)]
            distinct = [(b * (-(-length // block_len)))[:length] for b in blocks]
        else:
            distinct = [torch.randint(0, vocab, (length,), generator=g).tolist() for _ in range(pool or total)]

        def pattern(r) -> float:  # "repeat": the share of its first 64 tokens that continue the block
            if prompt_kind != "repeat":
                return float("nan")
            b, n = blocks[r["which"]], min(64, len(r["toks"]))
            return sum(r["toks"][i] == b[(length + i) % block_len] for i in range(n)) / max(1, n)

        pending = [(k % len(distinct), int(torch.randint(lo, hi + 1, (1,), generator=g))) for k in range(total)]
        reqs: dict = {}  # request id -> its progress
        batch: list = []  # the input batch's rows: request ids, None a hole
        names: dict = {}  # running request id -> its state slot name (the plugin's _req_state_slot)
        done, bands = [], {}
        steps = prefills = remaps = 0
        t1 = time.time()

        def update(new: list) -> None:  # vLLM's _update_states: new requests into the lowest holes, then condense
            holes = sorted(i for i, r in enumerate(batch) if r is None)
            for rid in new:
                if holes:
                    batch[holes.pop(0)] = rid
                else:
                    batch.append(rid)
            while True:
                while batch and batch[-1] is None:
                    batch.pop()
                if None not in batch:
                    break
                batch[batch.index(None)] = batch.pop()

        while pending or any(r is not None for r in batch):
            while pending and sum(r is not None for r in batch) < SEATS:  # the prefill steps
                held = {names[r] for r in batch if r is not None}
                n = min(per_prefill, len(pending), SEATS - len(held))
                new, slots = [], []
                for row in range(n):
                    rid = len(reqs)
                    which, want = pending.pop(0)
                    prompt = distinct[which]
                    slot = row if row not in held else min(s for s in range(SEATS) if s not in held)
                    held.add(slot)
                    names[rid] = slot
                    reqs[rid] = dict(prompt=prompt, want=want, which=which, rid=rid)
                    new.append(rid)
                    slots.append(slot)
                tokens = torch.tensor([reqs[rid].pop("prompt") for rid in new], dtype=torch.int64)
                ids = adapter.prefill_forward(
                    tokens,
                    torch.zeros(n, dtype=torch.int64),
                    torch.tensor([length] * n),
                    empty_slots=torch.tensor(slots),
                    sampling_params=Greedy(),
                )
                prefills += 1
                phys = [adapter.phys[s] for s in slots]
                mesh_rows = [p // USERS for p in phys]
                for i, rid in enumerate(new):
                    p = phys[i]
                    draft = adapter.gen.first.get(p) if first == "gen" else None
                    reqs[rid].update(
                        pos=length,
                        tok=int(ids[i, 0]),
                        out=1,
                        count=1,
                        draft=draft,
                        proposed=0,
                        accepted=0,
                        bits=[],
                        reused=p in used,
                        batch=n,
                        shared_row=mesh_rows.count(mesh_rows[i]) > 1,
                        phys=p,
                        verifies=0,
                        mark=None,
                        between=None,
                        nth=uses.get(p, 0) + 1,
                        toks=[int(ids[i, 0])],
                    )
                    used.add(p)
                    uses[p] = uses.get(p, 0) + 1
                update(new)
            update([])
            rows = list(batch)
            want = [names[rid] for rid in rows]
            remap = want + [s for s in range(SEATS) if s not in set(want)]
            identity = remap == list(range(SEATS))
            owner = {s: rid for rid, s in names.items()}
            for row, s in enumerate(remap):  # the names settle to the rows
                if s in owner:
                    names[owner[s]] = row
            remaps += not identity
            if not spec:  # the plain greedy decode (the plugin's ordinary step, device sampling)
                tokens = torch.zeros(SEATS, 1, dtype=torch.int64)
                start = torch.full((SEATS,), -1, dtype=torch.int64)
                for row, rid in enumerate(rows):
                    tokens[row, 0], start[row] = reqs[rid]["tok"], reqs[rid]["pos"]
                got = (
                    adapter.decode_forward(
                        tokens,
                        start,
                        slot_remap=None if identity else torch.tensor(remap, dtype=torch.int32),
                        sampling_params=Greedy(),
                    )
                    .view(-1)
                    .tolist()
                )
                steps += 1
                for row, rid in enumerate(rows):
                    r = reqs[rid]
                    r["pos"], r["tok"], r["out"] = r["pos"] + 1, got[row], r["out"] + 1
                    r["toks"].append(got[row])
                    if r["out"] >= r["want"] or r["pos"] + 2 >= max_len:
                        done.append(r)
                        batch[row] = None
                        del names[rid]
                continue
            tokens = torch.zeros(SEATS, 2, dtype=torch.int64)
            start = torch.full((SEATS, 2), -1, dtype=torch.int64)
            valid = torch.zeros(SEATS, dtype=torch.int32)
            counts = torch.ones(SEATS, dtype=torch.int32)
            for row, rid in enumerate(rows):
                r = reqs[rid]
                tokens[row] = torch.tensor([r["tok"], r["draft"] if r["draft"] is not None else 0])
                start[row] = torch.tensor([r["pos"], r["pos"] + 1])
                valid[row] = int(r["draft"] is not None)
                counts[row] = r["count"]
            out = adapter.decode_forward(
                tokens,
                start,
                slot_remap=None if identity else torch.tensor(remap, dtype=torch.int32),
                spec_mode=mode,
                num_valid_drafts=valid,
                accepted_counts=counts,
                sampling_params=None,
            )
            ids = out.argmax_ids.tolist()
            steps += 1
            committed = [1] * SEATS
            for row, rid in enumerate(rows):
                r = reqs[rid]
                g0, g1 = ids[row]
                r["verifies"] += 1
                if r["verifies"] == 1:
                    r["mark"] = prefills
                elif r["verifies"] == 2:  # prefills between its first and second verify
                    r["between"] = prefills - r["mark"]
                a = r["draft"] is not None and r["draft"] == g0
                if r["draft"] is not None:
                    r["proposed"] += 1
                    r["accepted"] += int(a)
                    r["bits"].append(int(a))
                    band = bands.setdefault((r["pos"] - length) // 128, [0, 0])
                    band[0] += int(a)
                    band[1] += 1
                r["pos"], r["tok"], r["out"] = (
                    (r["pos"] + 2, g1, r["out"] + 2) if a else (r["pos"] + 1, g0, r["out"] + 1)
                )
                r["toks"] += [g0, g1] if a else [g0]
                r["count"] = committed[row] = 2 if a else 1
            drafts = propose(committed)
            for row, rid in enumerate(rows):
                r = reqs[rid]
                r["draft"] = drafts[row]
                if r["out"] >= r["want"] or r["pos"] + 2 >= max_len:
                    done.append(r)
                    batch[row] = None
                    del names[rid]
        rate = lambda rs: sum(r["accepted"] for r in rs) / max(1, sum(r["proposed"] for r in rs))  # noqa: E731
        late = lambda r: sum(r["bits"][len(r["bits"]) // 2 :]) / max(
            1, len(r["bits"]) - len(r["bits"]) // 2
        )  # noqa: E731
        broken = [r for r in done if len(r["bits"]) >= 40 and late(r) < 0.3]

        def split(key) -> dict:
            out = {}
            for r in done:
                k = key(r)
                n, b = out.get(k, (0, 0))
                out[k] = (n + 1, b + (r in broken))
            return {k: f"{b} / {n}" for k, (n, b) in sorted(out.items(), key=lambda kv: str(kv[0]))}

        def same(a: list, b: list) -> int:
            m = 0
            while m < min(len(a), len(b), 64) and a[m] == b[m]:
                m += 1
            return m

        refs = {}  # prompt -> its first occurrence on a seat's first use, else its first occurrence in the flow
        for r in sorted(done, key=lambda r: (r["nth"] != 1, r["rid"])):
            refs.setdefault(r["which"], r)
        text = {}  # seat use -> the tokens equal to the reference's (of 64) per request
        for r in done:
            ref = refs.get(r["which"])
            if ref is not None and ref is not r:
                text.setdefault(r["nth"], []).append(same(r["toks"], ref["toks"]))
        for r in done:
            ref = refs.get(r["which"])
            r["same"] = same(r["toks"], ref["toks"]) if ref is not None and ref is not r else None
        first_use = [r for r in done if not r["reused"]]
        again = [r for r in done if r["reused"]]
        summary = dict(
            requests=len(done),
            prefills=prefills,
            steps=steps,
            remaps=remaps,
            seconds=round(time.time() - t1, 1),
            acceptance=round(rate(done), 3),
            tokens_a_step=round(1 + nd * rate(done), 3),
            first_use=(len(first_use), round(rate(first_use), 3)),
            reused=(len(again), round(rate(again), 3)),
            broken=len(broken),
            broken_by_reuse=split(lambda r: "reused" if r["reused"] else "first"),
            broken_by_prefill_size=split(lambda r: r["batch"]),
            broken_by_shared_row=split(lambda r: r["shared_row"]),
            broken_by_prefills_between_1st_2nd_verify=split(lambda r: r["between"]),
            broken_by_seat_use=split(lambda r: r["nth"]),
            acceptance_by_seat_use={
                k: round(rate([r for r in done if r["nth"] == k]), 3) for k in sorted({r["nth"] for r in done})
            },
            by_position={f"+{128 * b}": round(a / max(1, n), 3) for b, (a, n) in sorted(bands.items())},
            same_text_as_first_by_seat_use={k: sorted(v) for k, v in sorted(text.items())},
        )
        if prompt_kind == "repeat":
            by_use = {}
            for r in done:
                by_use.setdefault(r["nth"], []).append(pattern(r))
            summary["pattern_by_seat_use"] = {
                k: dict(mean=round(sum(v) / len(v), 3), low=sorted(round(x, 2) for x in v)[:4])
                for k, v in sorted(by_use.items())
            }
        print(
            f"flow {length}:{per_prefill}:{total} ({'plugin, first draft given' if first == 'gen' else 'plugin'}"
            f"{'' if spec else ', no MTP'}): "
            f"{summary}",
            flush=True,
        )
        print(
            "  per request (phys seat, its use of the seat, prefill size, shared row, prefills between its 1st and 2nd "
            "verify: accepted / proposed, the second half's rate, tokens equal to its prompt's first run (on a fresh seat if any), "
            "the block's continuation): "
            + ", ".join(
                f"{r['phys']}#{r['nth']}/{r['batch']}{'s' if r['shared_row'] else ''}"
                f"/{r['between']}:{r['accepted']}/{r['proposed']}:{late(r):.2f}:{r['same']}:{pattern(r):.2f}"
                for r in done
            ),
            flush=True,
        )
        seats: dict = {}
        for r in done:
            seats.setdefault(r["phys"], []).append(r)
        print(
            "  per seat (its requests in turn: the second half's rate): "
            + ", ".join(
                f"{p}: " + " ".join(f"{late(r):.2f}" for r in sorted(rs, key=lambda r: r["nth"]))
                for p, rs in sorted(seats.items())
            ),
            flush=True,
        )
        dumped[f"{length}:{per_prefill}:{total}"] = {r["rid"]: list(r["toks"]) for r in done}
        return summary

    results = {}
    flows = os.environ.get("GLM53_AF_FLOWS", "8192:8:40:48-96,1024:8:40:48-96,8192:1:16:48-96,8192:8:8:900-1000")
    both = os.environ.get("GLM53_AF_SPREAD_BOTH") == "1"  # each simple flow without and then with the spreading
    # GLM53_AF_AHEAD_ARMS (e.g. "dry,1,0" or "dry/nohold,1/nohold"): each simple flow under each GLM53_VLLM_AHEAD in
    # turn (the same requests; "/nohold": GLM53_VLLM_AHEAD_HOLD=0), each launched arm's ("1") tokens and events against
    # its "dry" one's (the same holds; GLM53_AF_COMPARE_STRICT=1: all the same, else the test fails), each arm's time;
    # with GLM53_VLLM_TIMELINE an "arm" line before each.  An arm may set more for itself after "@": "1@VAR=value+VAR=value"
    # (e.g. "dry@GLM53_VLLM_AHEAD_EDGE=vllm,1@GLM53_VLLM_AHEAD_EDGE=vllm"); "1<rest>" is compared with "dry<rest>"
    ahead_arms = [a for a in os.environ.get("GLM53_AF_AHEAD_ARMS", "").split(",") if a]
    # GLM53_AF_PAIRS="A|B,..": more arms compared strictly as the "1" / "dry" pairs (e.g. a switch that keeps the values:
    # "1@X|1@X+GLM53_DFLASH2_FIRST_DRAFTS=0")
    pairs_more = [tuple(x.split("|", 1)) for x in os.environ.get("GLM53_AF_PAIRS", "").split(",") if "|" in x]
    # GLM53_AF_AHEAD_LAX_ARMS: the launched arms ("1<rest>") whose differences from their "dry" are reported only
    lax_arms = {a for a in os.environ.get("GLM53_AF_AHEAD_LAX_ARMS", "").split(",") if a}
    surprise_ends = int(os.environ.get("GLM53_AF_SURPRISE_ENDS", "0"))
    ahead_differ: list = []
    for idx, name in enumerate(flows.split(",")):
        length, per, total, out, *kind = name.split(":")
        lo, hi = (int(x) for x in out.split("-"))
        kind = kind[0] if kind else "simple"
        if kind == "simple" and both:  # the same requests (the prompts drawn again from one seed) both ways
            key = f"{length}:{per}:{total}"
            for spread in (False, True):
                glm53_vllm.SPREAD_ROWS = spread
                g.manual_seed(1000 + idx)
                w0 = gen.counts["waves"]
                run = name + ("+spread" if spread else "")
                results[run] = flow(int(length), int(per), int(total), lo, hi)
                results[run]["waves"] = gen.counts["waves"] - w0
                dumped[run] = dumped.pop(key)
            a, b = dumped[name], dumped[name + "+spread"]
            same = sum(a[r] == b.get(r) for r in a)
            first = [next((i for i, (x, y) in enumerate(zip(a[r], b.get(r, []))) if x != y), None) for r in a]
            print(
                f"AF spread {name}: {same} / {len(a)} requests the same tokens without and with the spreading (the "
                f"first differing token of each: {first}); acceptance {results[name]['acceptance']} -> "
                f"{results[name + '+spread']['acceptance']}, waves {results[name]['waves']} -> "
                f"{results[name + '+spread']['waves']}",
                flush=True,
            )
        elif kind == "simple" and ahead_arms:  # the same requests under each GLM53_VLLM_AHEAD, against the first's
            key = f"{length}:{per}:{total}"
            saved = os.environ.get("GLM53_VLLM_AHEAD"), os.environ.get("GLM53_VLLM_AHEAD_HOLD")
            for arm in ahead_arms:
                head, _, extra = arm.partition("@")
                os.environ["GLM53_VLLM_AHEAD"] = head.split("/")[0]
                os.environ["GLM53_VLLM_AHEAD_HOLD"] = "0" if head.endswith("/nohold") else "1"
                arm_env = dict(kv.split("=", 1) for kv in extra.split("+") if kv)
                arm_saved = {var: os.environ.get(var) for var in arm_env}
                os.environ.update(arm_env)
                g.manual_seed(1000 + idx)
                w0 = dict(gen.dflash_writes)
                run = f"{name}+ahead={arm}"
                glm53_vllm._record("arm", time.time(), arm=run)
                results[run] = flow(int(length), int(per), int(total), lo, hi)
                writes = {w: v - w0.get(w, 0) for w, v in gen.dflash_writes.items() if v != w0.get(w, 0)}
                res = results[run]
                print(f"AF ahead {run}: {res['seconds']} s, {res['steps']} verify steps, {res['prefills']} prefills; "
                      f"{1e3 * res['seconds'] / max(1, res['steps']):.2f} ms a step; next inputs {writes}",
                      flush=True)  # fmt: skip
                for part in ("", ":events", ":seats"):
                    dumped[run + part] = dumped.pop(key + part)
                for var, value in arm_saved.items():
                    if value is None:
                        os.environ.pop(var, None)
                    else:
                        os.environ[var] = value
            for var, value in zip(("GLM53_VLLM_AHEAD", "GLM53_VLLM_AHEAD_HOLD"), saved):
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value
            # (each launched arm "1[/nohold]" against its "dry[/nohold]": the same choices and holds, none launched)
            checks = [
                (a, "dry" + a[1:])
                for a in ahead_arms
                if a.partition("@")[0].split("/")[0] == "1" and "dry" + a[1:] in ahead_arms
            ]
            checks += [(b, a) for a, b in pairs_more if a in ahead_arms and b in ahead_arms]
            for arm, ref_arm in checks:
                ref = f"{name}+ahead={ref_arm}"
                run = f"{name}+ahead={arm}"
                a, b = dumped[run], dumped[ref]
                same = sum(a[r] == b.get(r) for r in a)
                ea, eb = dumped[run + ":events"], dumped[ref + ":events"]
                at = next((i for i, (x, y) in enumerate(zip(ea, eb)) if x != y), None)
                if at is None and len(ea) != len(eb):
                    at = min(len(ea), len(eb))
                print(f"AF ahead {run} against {ref}: {same} / {len(a)} requests the same tokens; events "
                      f"{'all the same' if at is None else f'first differing #{at}: {ea[at] if at < len(ea) else None} / {eb[at] if at < len(eb) else None}'}"
                      f" ({len(ea)} / {len(eb)})", flush=True)  # fmt: skip
                if same != len(a) or at is not None:
                    if arm in lax_arms:  # (GLM53_AF_AHEAD_LAX_ARMS: reported, not asserted)
                        print(f"AF ahead {run}: differs, not asserted (GLM53_AF_AHEAD_LAX_ARMS)", flush=True)
                    else:
                        ahead_differ.append(run)
        elif kind == "simple":
            results[name] = flow(int(length), int(per), int(total), lo, hi)
        elif dflash:  # (the plugin's bookkeeping here keeps MTP's one draft)
            print(f"flow {name}: skipped with GLM53_AF_DFLASH2", flush=True)
        else:
            results[name] = plugin_flow(
                int(length), int(per), int(total), lo, hi, "gen" if kind == "plugin-gen" else ""
            )
    print(f"summary {results}", flush=True)
    if dflash:  # (GLM53_DFLASH2_NEXT_INPUTS: the verify steps whose host write the device's next inputs replaced)
        print(f"AF next inputs: {gen.dflash_writes}", flush=True)
    print(f"AF ids check (GLM53_PREFILL_IDS_CHECK: [rows checked, differing]): {model.ids_check}", flush=True)
    if model.w1 is not None:  # the expert-weight landing (GLM53_W1): the DRISCs over the flows
        ttnn.synchronize_device(mesh_device)
        stats = model.w1.stop() if model.w1.running else {}
        print(f"AF W1: C {model.w1.c}, the landing at {model.w1.addr:#x}; DRISCs {stats}; the first mesh row's 4 chips' "
              f"(gate | up blocks from the landing, read again, down halves from it, read again) "
              f"{model.w1.counts(range(4))}", flush=True)  # fmt: skip
    print(
        f"AF waves: {gen.counts['waves'] - waves0} prefill waves (a round over the rows one) in {prefill_calls[0]} "
        f"prefill calls; GLM53_VLLM_SPREAD_ROWS {glm53_vllm.SPREAD_ROWS}",
        flush=True,
    )
    if os.environ.get("GLM53_AF_DUMP"):  # every finished request's tokens (to compare two runs)
        torch.save(dumped, os.environ["GLM53_AF_DUMP"])
        print(f"wrote {os.environ['GLM53_AF_DUMP']}", flush=True)
    adapter.release_persistent_capture()
    if os.environ.get("GLM53_AF_COMPARE"):  # another run's dump: each request's tokens against it
        other = torch.load(os.environ["GLM53_AF_COMPARE"])
        differ = []
        skip_new = os.environ.get("GLM53_AF_COMPARE_SKIP_NEW") == "1"  # (names the other run lacks: not compared)
        for name, reqs in dumped.items():
            if name.endswith(":seats"):
                continue
            if skip_new and name not in other:
                print(f"AF compare {name}: not in {os.environ['GLM53_AF_COMPARE']} (skipped)", flush=True)
                continue
            if name.endswith(":events"):  # the first prefill or verify step that differs, and what it held
                theirs = other.get(name)
                if theirs is None:
                    continue
                at = next((i for i, (a, b) in enumerate(zip(reqs, theirs)) if a != b), None)
                if at is None:
                    print(f"AF compare {name}: all {len(reqs)} events the same (the other run {len(theirs)})",
                          flush=True)  # fmt: skip
                    continue
                a, b = reqs[at], theirs[at]
                last_pf = [e for e in reqs[:at] if e[0] == "prefill"][-3:]
                if a[0] == "verify" and b[0] == "verify":
                    mine, ref = {x[0]: x for x in a[3]}, {x[0]: x for x in b[3]}
                    seats = [s for s in sorted(set(mine) | set(ref)) if mine.get(s) != ref.get(s)]
                    detail = {s: (mine.get(s), ref.get(s)) for s in seats[:4]}
                else:
                    seats, detail = [], (a, b)
                print(f"AF compare {name}: the first differing event is #{at} of {len(reqs)} ({a[0]}, prefill "
                      f"#{a[1]}, step {a[2]}); seats {seats}; here / there {detail}; the prefills before it "
                      f"{[(e[1], e[2], e[3]) for e in last_pf]}", flush=True)  # fmt: skip
                differ.append(f"{name}@{at}")
                continue
            theirs = other.get(name, {})
            same, firsts = 0, []
            for rid, toks in sorted(reqs.items()):
                ref = theirs.get(rid)
                n = min(len(toks), len(ref)) if ref is not None else 0
                at = next((j for j in range(n) if toks[j] != ref[j]), None)
                if ref is not None and at is None:
                    same += 1
                else:
                    firsts.append(at if ref is not None else "missing")
                    differ.append(f"{name}#{rid}")
            print(f"AF compare {name}: {same} / {len(reqs)} requests the same tokens as {os.environ['GLM53_AF_COMPARE']}"
                  f"; first differing tokens of the rest {firsts}", flush=True)  # fmt: skip
        if os.environ.get("GLM53_AF_COMPARE_STRICT") == "1":
            assert not differ, f"requests whose tokens differ: {differ}"
    if ahead_arms and os.environ.get("GLM53_AF_COMPARE_STRICT") == "1":
        assert (
            not ahead_differ
        ), f"flows whose tokens or events differ between the GLM53_VLLM_AHEAD arms: {ahead_differ}"
    if common_module.TRACE_ALLOC_GUARD != "0":
        print(f"trace allocation guard: {common_module._GUARD['flagged']} buffer(s) flagged", flush=True)
        assert common_module._GUARD["flagged"] == 0, "buffers allocated after the warm-up were alive at a replay"
