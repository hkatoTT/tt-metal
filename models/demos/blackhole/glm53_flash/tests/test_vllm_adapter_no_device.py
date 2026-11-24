"""The vLLM adapter's slot bookkeeping (tools/glm53_vllm.py) on a stand-in generator: no device, no vLLM.

vLLM's state slots name physical slots through a host-side permutation; prefill writes the slot `empty_slots` names,
decode row i runs on the physical slot vLLM's slot i names after the plugin's `slot_remap` (row i reads old slot
remap[i]), and the outputs come back in row order.  The permutation starts interleaved over the 8 mesh rows, so one
prefill call's requests reach the generator together on different rows.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.tools.glm53_vllm import Glm53ForCausalLM  # noqa: E402

V = 7


class FakeGenerator:
    """Physical slots hold (prompt, generated) histories; the 'model' predicts the last token + 1 (mod V) or, from
    logits, a one-hot at that token."""

    def __init__(self, slots: int):
        self.slots = slots
        self.history = {}
        self.traces = {}
        self.chunk = 32
        self.chunks = (32,)
        self.decoded = []
        self.prefill_calls = []
        self.dflash_ahead = None  # (no DFlash2 step launched ahead: the adapter checks it before each call)

    def prefill_many(self, requests, chunk=None, greedy_ids=False):
        self.prefill_calls.append([slot for slot, _ in requests])
        out = []
        for slot, tokens in requests:
            self.history[slot] = list(tokens)
            nxt = (tokens[-1] + 1) % V
            out.append(nxt if greedy_ids else torch.nn.functional.one_hot(torch.tensor(nxt), V).float())
        return out

    def decode(self, tokens, positions, greedy):
        self.decoded.append((tokens.clone(), positions.clone()))
        nxt = torch.zeros(self.slots, dtype=torch.int64)
        for s in range(self.slots):
            if positions[s] >= 0:
                assert positions[s] == len(self.history[s]), f"slot {s}: position {positions[s]}"
                self.history[s].append(int(tokens[s]))
                nxt[s] = (int(tokens[s]) + 1) % V
        if greedy:
            return nxt
        return torch.nn.functional.one_hot(nxt, V).float()


class Params:
    def __init__(self, n):
        self.temperature = [0.0] * n
        self.top_k = [V] * n


def test_spread_rows(monkeypatch):
    """GLM53_VLLM_SPREAD_ROWS: a prefill's new requests on free physical slots of distinct mesh rows (one wave), the
    slots in use left where they are, and the rows with a free slot published for the scheduler."""
    from models.demos.blackhole.glm53_flash.tools import glm53_vllm as adapter_module
    from models.demos.blackhole.glm53_flash.tools import glm53_vllm_rows

    glm53_vllm_rows.reset()
    gen = FakeGenerator(8 * 4)  # 8 mesh rows x 4 users: physical slot p on row p // 4
    adapter = Glm53ForCausalLM(gen, max_seq_len=64)
    row = lambda slot: adapter.phys[slot] // 4  # noqa: E731

    def prefill(slots):
        adapter.prefill_forward(
            tokens=torch.ones(len(slots), 2, dtype=torch.int64),
            start_pos=[0] * len(slots),
            prompt_lens=[2] * len(slots),
            empty_slots=slots,
        )

    # without spreading: 22 requests on rows 0 .. 5 (2 of their 24 seats left free), rows 6 and 7 empty
    first = [s for s in range(32) if row(s) < 6][:22]
    prefill(first)
    monkeypatch.setattr(adapter_module, "SPREAD_ROWS", True)
    before = {s: adapter.phys[s] for s in first}
    # a batch of 4 whose vLLM slots name two seats of one row and two of another: 4 distinct rows with a free seat
    free = [s for s in range(32) if s not in first]
    r0 = row(free[0])
    batch = [s for s in free if row(s) == r0][:2] + [s for s in free if row(s) != r0][:2]
    prefill(batch)
    assert len({row(s) for s in batch}) == 4
    assert all(adapter.phys[s] == p for s, p in before.items())  # the requests in use stayed
    assert sorted(adapter.phys) == list(range(32))  # still one physical slot a vLLM slot
    occupied = [0] * 8
    for s in first + batch:
        occupied[row(s)] += 1
    assert glm53_vllm_rows.free_rows() == sum(c < 4 for c in occupied)
    # a decode in which only the batch runs: the slots in use and the free rows follow it
    start = torch.full((32,), -1, dtype=torch.int64)
    start[batch] = 2
    adapter.decode_forward(tokens=torch.ones(32, 1, dtype=torch.int64), start_pos=start)
    assert adapter.active == set(batch)
    assert glm53_vllm_rows.free_rows() == 8
    glm53_vllm_rows.reset()


def test_prefill_device_sampling_ids():
    """Device sampling (greedy): the prefill's first tokens as ids [B, 1] (the generator's greedy_ids), the argmax of
    the logits host sampling gets."""
    tokens, lens = torch.tensor([[1, 2, 3], [4, 5, 0]]), [3, 2]
    logits = Glm53ForCausalLM(FakeGenerator(8), max_seq_len=64).prefill_forward(
        tokens=tokens, start_pos=[0, 0], prompt_lens=lens, empty_slots=[0, 1]
    )
    ids = Glm53ForCausalLM(FakeGenerator(8), max_seq_len=64).prefill_forward(
        tokens=tokens, start_pos=[0, 0], prompt_lens=lens, empty_slots=[0, 1], sampling_params=Params(2)
    )
    assert ids.shape == (2, 1) and ids.dtype == torch.int64
    assert ids[:, 0].tolist() == logits[:, 0].argmax(-1).tolist()


def test_slot_bookkeeping():
    gen = FakeGenerator(8)
    adapter = Glm53ForCausalLM(gen, max_seq_len=64)
    # two requests prefilled into vLLM slots 0 and 1 (front-packed rows), then a third into slot 2
    out = adapter.prefill_forward(
        tokens=torch.tensor([[1, 2, 3], [4, 5, 0]]), start_pos=[0, 0], prompt_lens=[3, 2], empty_slots=[0, 1]
    )
    assert out.shape == (2, 1, V) and out[0, 0].argmax() == 4 and out[1, 0].argmax() == 6
    assert gen.history == {0: [1, 2, 3], 1: [4, 5]}
    # decode both (device sampling): row i is slot i
    ids = adapter.decode_forward(
        tokens=torch.tensor([[4], [6]]), start_pos=torch.tensor([3, 2]), sampling_params=Params(2)
    )
    assert ids.tolist() == [[5], [0]]
    # request in slot 0 finishes; the survivor (slot 1) is gathered to row 0: remap row 0 <- slot 1
    remap = [1, 0, 2, 3, 4, 5, 6, 7]
    out = adapter.decode_forward(
        tokens=torch.tensor([[0]]), start_pos=torch.tensor([3]), slot_remap=torch.tensor(remap)
    )
    assert out.shape == (1, 1, V) and out[0, 0].argmax() == 1
    assert gen.history[1] == [4, 5, 6, 0], "the survivor must keep its physical slot"
    assert adapter.phys[:2] == [1, 0]
    # a new request into vLLM slot 1 lands on physical slot 0 (the finished request's)
    adapter.prefill_forward(tokens=torch.tensor([[2, 2]]), start_pos=[0], prompt_lens=[2], empty_slots=[1])
    assert gen.history[0] == [2, 2]
    ids = adapter.decode_forward(
        tokens=torch.tensor([[1], [3], [0]]), start_pos=torch.tensor([4, 2, -1]), sampling_params=Params(3)
    )
    assert ids.tolist() == [[2], [4], [0]]
    assert gen.history[1] == [4, 5, 6, 0, 1] and gen.history[0] == [2, 2, 3]
    _, positions = gen.decoded[-1]
    assert positions.tolist() == [2, 4, -1, -1, -1, -1, -1, -1], "inactive rows and slots off the batch idle"


def test_slots_interleave_over_rows():
    gen = FakeGenerator(16)  # 2 users per mesh row: physical slot s is user s % 2 of row s // 2
    adapter = Glm53ForCausalLM(gen, max_seq_len=64)
    assert adapter.phys == [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]
    n = 10
    adapter.prefill_forward(
        tokens=torch.arange(n).view(n, 1), start_pos=[0] * n, prompt_lens=[1] * n, empty_slots=list(range(n))
    )
    assert gen.prefill_calls == [[0, 2, 4, 6, 8, 10, 12, 14, 1, 3]], "one call: user 0 of every row, then user 1"
    assert all(gen.history[adapter.phys[k]] == [k] for k in range(n))


def test_refusals(expect_error):
    adapter = Glm53ForCausalLM(FakeGenerator(8), max_seq_len=8)
    with expect_error(ValueError, "only whole prompts"):
        adapter.prefill_forward(tokens=torch.tensor([[1, 2]]), start_pos=[1], prompt_lens=[2], empty_slots=[0])
    with expect_error(ValueError, "exceeds max_model_len"):
        adapter.prefill_forward(tokens=torch.zeros(1, 9, dtype=torch.int64), start_pos=[0], prompt_lens=[9])
    params = Params(1)
    params.temperature, params.top_k = [0.7], [50]
    adapter.prefill_forward(tokens=torch.tensor([[1]]), start_pos=[0], prompt_lens=[1], empty_slots=[0])
    with expect_error(ValueError, "greedy only"):
        adapter.decode_forward(tokens=torch.tensor([[1]]), start_pos=torch.tensor([1]), sampling_params=params)
    with expect_error(ValueError, "greedy only"):  # (a prefill's device sampling is the argmax too)
        adapter.prefill_forward(
            tokens=torch.tensor([[1]]), start_pos=[0], prompt_lens=[1], empty_slots=[1], sampling_params=params
        )


def test_slot_sampling():
    """SAMPLING's parameters by physical slot: each row's on its slot; a slot without a row and a greedy row greedy (k
    1); a row without a seed its slot's base."""
    from types import SimpleNamespace

    from models.demos.blackhole.glm53_flash.tools.glm53_vllm import slot_sampling

    params = SimpleNamespace(
        temperature=[1.0, 0.0, 0.7], top_k=[32, 1, 0], top_p=[0.95, 1.0, 0.9], seed=[None, None, 5]
    )
    k, p, t, seed = slot_sampling(params, [3, 0, 5], 8, list(range(100, 108)))
    assert k.tolist() == [1, 1, 1, 32, 1, 0, 1, 1]
    assert t.tolist() == pytest.approx([0, 0, 0, 1.0, 0, 0.7, 0, 0])
    assert p.tolist() == pytest.approx([1, 1, 1, 0.95, 1, 0.9, 1, 1])
    assert seed.tolist() == [100, 101, 102, 103, 104, 5, 106, 107]
    k, p, t, seed = slot_sampling(None, [3], 8, [7] * 8)  # (a step the device does not sample: every slot greedy)
    assert k.tolist() == [1] * 8 and t.tolist() == [0.0] * 8 and seed.tolist() == [7] * 8


def test_verify_logits_by_slot():
    """The DFlash2 logits step's rows (row r's virtual user j U + u) by physical slot r U + u and position j."""
    from models.demos.blackhole.glm53_flash.ttnn.dflash2_step import verify_logits_by_slot

    rows, k, users, vocab = 2, 2, 3, 5
    x = torch.arange(rows * (k + 1) * users * vocab, dtype=torch.float32).view(rows, (k + 1) * users, vocab)
    out = verify_logits_by_slot(x, k, users)
    assert out.shape == (rows * users, k + 1, vocab)
    for r in range(rows):
        for u in range(users):
            for j in range(k + 1):
                assert torch.equal(out[r * users + u, j], x[r, j * users + u])


def test_check_chunked(monkeypatch, expect_error):
    """CHUNKED's start-up check (`check_chunked`): chunks on the grid, a step's prompt tokens equal to the alignment,
    Glm53Scheduler; else a clear error.  Nothing to check with vLLM's chunked prefill off or CHUNKED off."""
    import sys
    import types

    from models.demos.blackhole.glm53_flash.tools import glm53_vllm as adapter_module

    config = types.ModuleType("vllm_tt_plugin.config")
    config.get_tt_prefill_chunk_alignment = lambda c: c.align  # (the plugin's: additional_config tt, default 128)
    monkeypatch.setitem(sys.modules, "vllm_tt_plugin", types.ModuleType("vllm_tt_plugin"))
    monkeypatch.setitem(sys.modules, "vllm_tt_plugin.config", config)
    monkeypatch.setattr(adapter_module, "CHUNKED", True)

    def cfg(align=8192, tokens=8192, sched=adapter_module.SCHEDULER_PATH, on=True, threshold=0):
        s = types.SimpleNamespace(
            enable_chunked_prefill=on,
            max_num_batched_tokens=tokens,
            long_prefill_token_threshold=threshold,
            scheduler_cls=sched,
        )
        return types.SimpleNamespace(scheduler_config=s, align=align)

    class Glm53Scheduler:  # (a stand-in: the check goes by the class names)
        pass

    class Derived(Glm53Scheduler):
        pass

    adapter_module.check_chunked(cfg())
    adapter_module.check_chunked(cfg(align=16384, tokens=16384))
    adapter_module.check_chunked(cfg(sched=Derived))  # (the class itself, as the plugin resolves it)
    adapter_module.check_chunked(cfg(align=128, tokens=2048, sched=None, on=False))  # (vLLM's chunked prefill off)
    with expect_error(ValueError, "prefill_chunk_alignment 128"):  # (the plugin's default)
        adapter_module.check_chunked(cfg(align=128, tokens=128))
    with expect_error(ValueError, "prompt tokens 16384"):  # (a cut-short prompt could share its step)
        adapter_module.check_chunked(cfg(tokens=16384))
    with expect_error(ValueError, "prompt tokens 4096"):
        adapter_module.check_chunked(cfg(threshold=4096))
    with expect_error(ValueError, "--scheduler-cls"):
        adapter_module.check_chunked(cfg(sched="vllm_tt_plugin.scheduler.TTScheduler"))
    monkeypatch.setattr(adapter_module, "CHUNKED", False)
    adapter_module.check_chunked(cfg(align=128))  # (whole prompts: nothing to check)
