"""Device KDA decode (v0) against `tt/` stepping one token at a time from an empty state (the recurrent form).

8 rows x 4 users, 5 steps, the first and the last KDA layer.  Run (a device test):
    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_kda_device.py
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.config import Glm53Config  # noqa: E402
from models.demos.blackhole.glm53_flash.tt.kda import Kda, KdaWeights  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    DEVICE_PARAMS,
    MESH_SHAPE,
    ROWS,
    SpGroup,
    eager_place,
    gather_rows,
    put,
)
from models.demos.blackhole.glm53_flash.ttnn.kda import TtKda  # noqa: E402

USERS, STEPS = int(os.environ.get("GLM53_KDA_TEST_USERS", "4")), 5  # (users a row: 64 for the 512-user batch)


def pcc(a, b) -> float:
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer", [0, 44])
def test_kda_device(mesh_device, layer, pcc_thresholds):
    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = KdaWeights.from_checkpoint(checkpoint, layer)
    reference = Kda(weights, config)
    module = TtKda(mesh_device, weights, config)

    g = torch.Generator().manual_seed(layer)
    xs = (torch.randn(STEPS, ROWS, 1, USERS, config.hidden_size, generator=g) * 0.5).to(torch.bfloat16)
    state = module.empty_state(USERS)
    ref_state = reference.empty_state(ROWS * USERS)
    for step in range(STEPS):
        out, state = module.step(put(mesh_device, xs[step], dims=(0, None)), state)
        got = gather_rows(mesh_device, out).reshape(ROWS * USERS, -1)
        with torch.no_grad():
            want, ref_state = reference(xs[step].reshape(ROWS * USERS, 1, -1), ref_state)
        score = pcc(got, want[:, 0])
        print(f"layer {layer} step {step}: pcc {score:.6f}")
        assert score >= pcc_thresholds["test_kda_device"], f"step {step}: pcc {score:.6f}"
    got_state = ttnn.to_torch(
        state.recurrent, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=MESH_SHAPE, dims=(0, 1))
    )
    score = pcc(got_state, ref_state.recurrent)
    print(f"layer {layer} state after {STEPS} steps: pcc {score:.6f}")
    assert score >= pcc_thresholds["test_kda_device"]


def chunk_inputs(mesh, module, t, valid, users, slots, rows=None, traced=False):
    """A wave's chunk inputs as the model makes them: keep [1, 1, 1, 1] per row and `TtKda.prefill_picks` (with
    `slots`, the wave's last chunk; `traced`: the form a captured prefill uses, run eagerly here)."""
    rows = [True] * ROWS if rows is None else rows
    keep = put(mesh, torch.tensor([1.0 if r else 0.0 for r in rows]).view(ROWS, 1, 1, 1), dims=(0, None))
    valid = [v if r else t for v, r in zip(valid, rows)]
    return keep, TtKda.prefill_picks(eager_place(mesh), t, valid, module.kernel, users, slots, rows, traced)


def read_states(mesh_device, module, state, users):
    """-> recurrent [ROWS, B, 64, 128, 128], conv history 3 x [ROWS, B, 4 x 6144] (fp32, the chips' heads in order)."""
    recurrent = ttnn.to_torch(
        state.recurrent, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=MESH_SHAPE, dims=(0, 1))
    ).reshape(ROWS, users, -1, module.dh, module.dh)
    history = [
        ttnn.to_torch(
            h, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, 3))
        ).reshape(ROWS, users, -1)
        for h in state.history
    ]
    return recurrent, history


def conv_order(config, module) -> torch.Tensor:
    """The device's conv history channels (chip c's q | k | v, its heads, d) as indices into the reference's."""
    width, per_chip = config.kda_num_heads * module.dh, module.heads * module.dh
    return torch.cat(
        [
            torch.arange(per_chip) + part * width + c * per_chip
            for c in range(config.kda_num_heads // module.heads)
            for part in range(3)
        ]
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer, length", [(0, 66), (44, 70)])
@pytest.mark.parametrize("traced", [False, True])
def test_kda_device_prefill(mesh_device, layer, length, traced, pcc_thresholds):
    """A prompt of `length` tokens per mesh row in chunks of 64 and 32 (the second padded: 2 or 6 real tokens), row
    r's into user (r + 1) % 4 of 4 (ttnn.experimental.kda, the wave state carried between the chunks; the last
    chunk, every row still in its prompt, writes straight into the users), then 2 decode steps; the outputs, each row's user's recurrent state and conv history against
    `tt/` (which runs the prompt in one call), the other users untouched; then a wave in which rows 4..7 have no
    request must leave their states as they were.  `traced`: the users written as a captured prefill writes them (by
    one-hot over all the users, not paged_fill_cache on the rows with a request)."""
    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = KdaWeights.from_checkpoint(checkpoint, layer)
    reference = Kda(weights, config)
    module = TtKda(mesh_device, weights, config)
    users, hidden = 4, config.hidden_size
    slots = [(r + 1) % users for r in range(ROWS)]
    g = torch.Generator().manual_seed(layer + 11)
    prompt = (torch.randn(ROWS, length, hidden, generator=g) * 0.5).to(torch.bfloat16)
    state = module.empty_state(users)
    got = []
    for start, t in ((0, 64), (64, 32)):
        valid = min(t, length - start)
        chunk = torch.zeros(ROWS, 1, t, hidden, dtype=torch.bfloat16)
        chunk[:, 0, :valid] = prompt[:, start : start + valid]
        keep, picks = chunk_inputs(
            mesh_device, module, t, [valid] * ROWS, users, slots if start else None, traced=traced
        )
        out, state = module.prefill(put(mesh_device, chunk, dims=(0, None)), state, start, keep, picks)
        got.append(gather_rows(mesh_device, out).reshape(ROWS, t, -1)[:, :valid])
    got = torch.cat(got, dim=1)
    with torch.no_grad():
        want, ref_state = reference(prompt, None)
    score = pcc(got, want)
    print(f"layer {layer} prefill of {length} tokens (64 + {length - 64}): pcc {score:.6f}", flush=True)
    assert score >= pcc_thresholds["test_kda_device"]

    recurrent, history = read_states(mesh_device, module, state, users)
    mine = lambda t: torch.stack([t[r, s] for r, s in enumerate(slots)])  # noqa: E731
    score = pcc(mine(recurrent), ref_state.recurrent)
    print(f"layer {layer} each row's user's state after the prompt: pcc {score:.6f}", flush=True)
    assert score >= pcc_thresholds["test_kda_device"]
    for r, s in enumerate(slots):
        others = [u for u in range(users) if u != s]
        assert not recurrent[r, others].any() and not any(h[r, others].any() for h in history), "another user changed"
    # (the device's projection rounds differently from torch's, so not bit-equal; a row from the wrong position of
    # these random inputs would score near 0)
    order = conv_order(config, module)
    for i, h in enumerate(history):
        score = pcc(mine(h), ref_state.conv[:, order, i + 1])
        print(f"layer {layer} conv history row {i}: pcc {score:.6f}", flush=True)
        assert score >= 0.9999, f"conv history row {i} differs"
    print(f"layer {layer} conv history matches the reference, other users untouched", flush=True)

    # decode continues from each row's prefilled user (the other users are fresh)
    for step in range(2):
        x = (torch.randn(ROWS, 1, users, hidden, generator=g) * 0.5).to(torch.bfloat16)
        out, state = module.step(put(mesh_device, x, dims=(0, None)), state)
        got = mine(gather_rows(mesh_device, out).reshape(ROWS, users, -1))
        with torch.no_grad():
            want, ref_state = reference(torch.stack([x[r, 0, s : s + 1] for r, s in enumerate(slots)]), ref_state)
        score = pcc(got, want[:, 0])
        print(f"layer {layer} decode after the prompt, step {step}: pcc {score:.6f}", flush=True)
        assert score >= pcc_thresholds["test_kda_device"]

    # a one-chunk wave with requests on rows 0..3 (into user 3) only: rows 4..7 keep their states
    before = read_states(mesh_device, module, state, users)
    half = [3] * 4 + [None] * 4
    rows = [s is not None for s in half]
    keep, picks = chunk_inputs(mesh_device, module, 32, [32] * ROWS, users, half, rows=rows, traced=traced)
    other = (torch.randn(ROWS, 1, 32, hidden, generator=g) * 0.5).to(torch.bfloat16)
    module.prefill(put(mesh_device, other, dims=(0, None)), state, 0, keep, picks)
    after = read_states(mesh_device, module, state, users)
    for b, a in zip([before[0], *before[1]], [after[0], *after[1]]):
        assert torch.equal(a[4:], b[4:]), "a row without a request changed its state"
        assert not torch.equal(a[:4, 3], b[:4, 3]), "a row with a request did not write its user"
        assert torch.equal(a[:4, :3], b[:4, :3]), "a row with a request wrote another user"
    print("rows without a request unchanged, rows with one rewrote their user only", flush=True)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("traced", [False, True])
def test_kda_device_prefill_rows(mesh_device, traced, pcc_thresholds):
    """Every mesh row its own prompt in the same chunks (the generator's `prefill_many`): lengths 66, 65, 64, 63, 34,
    33, 2, 1 in chunks of 64 and 32, row r's into user r % 4 of 4, a row whose prompt has ended keeping its wave state
    until the last chunk writes it (through the wave state, as some rows have ended).  Each row's outputs, user state and conv history against `tt/` run on that row's
    prompt alone; this covers a history taken from the previous chunk (1 and 2 tokens in the second chunk) and from
    the empty start (prompts of 1 and 2).  `traced`: the users written as a captured prefill writes them."""
    layer, lengths = 0, [66, 65, 64, 63, 34, 33, 2, 1]
    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = KdaWeights.from_checkpoint(checkpoint, layer)
    reference = Kda(weights, config)
    module = TtKda(mesh_device, weights, config)
    users, hidden = 4, config.hidden_size
    slots = [r % users for r in range(ROWS)]
    g = torch.Generator().manual_seed(23)
    prompt = (torch.randn(ROWS, max(lengths), hidden, generator=g) * 0.5).to(torch.bfloat16)
    state = module.empty_state(users)
    got = [[] for _ in range(ROWS)]
    for start, t in ((0, 64), (64, 32)):
        valid = [min(t, n - start) for n in lengths]
        rows = [v > 0 for v in valid]
        keep, picks = chunk_inputs(mesh_device, module, t, valid, users, slots if start else None, rows, traced)
        chunk = torch.zeros(ROWS, 1, t, hidden, dtype=torch.bfloat16)
        for r in range(ROWS):
            if rows[r]:
                chunk[r, 0, : valid[r]] = prompt[r, start : start + valid[r]]
        out, state = module.prefill(put(mesh_device, chunk, dims=(0, None)), state, start, keep, picks)
        full = gather_rows(mesh_device, out).reshape(ROWS, t, -1)
        for r in range(ROWS):
            if rows[r]:
                got[r].append(full[r, : valid[r]])

    recurrent, history = read_states(mesh_device, module, state, users)
    order = conv_order(config, module)
    for r, (n, s) in enumerate(zip(lengths, slots)):
        others = [u for u in range(users) if u != s]
        assert not recurrent[r, others].any() and not any(h[r, others].any() for h in history), "another user changed"
        with torch.no_grad():
            want, ref = reference(prompt[r : r + 1, :n], None)
        scores = [pcc(torch.cat(got[r]), want[0]), pcc(recurrent[r, s], ref.recurrent[0])]
        for i, h in enumerate(history):
            ref_row = ref.conv[0, order, i + 1]
            if ref_row.any():
                scores.append(pcc(h[r, s], ref_row))
            else:  # before the prompt's first token: zeros
                assert not h[r, s].any(), f"row {r} ({n} tokens): conv history row {i} should be zeros"
        print(f"row {r}, {n} tokens: output, state, conv history pcc {[round(x, 6) for x in scores]}", flush=True)
        assert min(scores[:2]) >= pcc_thresholds["test_kda_device"], f"row {r} ({n} tokens)"
        assert min(scores[2:]) >= 0.9999, f"row {r} ({n} tokens): conv history differs"


def sp_schedule(requests: dict, t: int):
    """Requests' sequence-parallel waves as the generator makes them (`Glm53Generator.sp_blocks`, `_prefill_sp_round`):
    {own row: (prompt length, user)} -> per wave the groups and each group's (row, start, real tokens)."""
    from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator

    blocks = Glm53Generator.sp_blocks(requests)
    segs = {o: [(s0, min(t, n - s0)) for s0 in range(0, n, t)] for o, (n, _) in requests.items()}
    waves = []
    for k in range(max(-(-len(segs[o]) // len(blocks[o])) for o in segs)):
        groups = []
        for o in sorted(segs):
            block = blocks[o]
            part = segs[o][k * len(block) : (k + 1) * len(block)]
            if part:
                rows = block[len(block) - len(part) :]
                last = (k + 1) * len(block) >= len(segs[o])
                groups.append(SpGroup(rows, tuple(v for _, v in part), requests[o][1], part[0][0], last))
        waves.append(groups)
    return waves


def run_sp(mesh_device, module, users, t, prompts: dict, waves, state):
    """Run `sp_schedule`'s waves through `TtKda.prefill` ({own row: prompt [L, hidden]}) -> {own row: outputs}."""
    hidden = next(iter(prompts.values())).shape[-1]
    keep = put(mesh_device, torch.ones(ROWS, 1, 1, 1), dims=(0, None))
    got = {o: [] for o in prompts}
    for groups in waves:
        chunk = torch.zeros(ROWS, 1, t, hidden, dtype=torch.bfloat16)
        for grp in groups:
            for i, (r, v) in enumerate(zip(grp.rows, grp.valid)):
                chunk[r, 0, :v] = prompts[grp.owner][grp.start + i * t : grp.start + i * t + v]
        picks = TtKda.sp_picks(eager_place(mesh_device), t, groups, module.kernel, users)
        out, state = module.prefill(put(mesh_device, chunk, dims=(0, None)), state, 0, keep, picks)
        full = gather_rows(mesh_device, out).reshape(ROWS, t, -1)
        for grp in groups:
            got[grp.owner] += [full[r, :v] for r, v in zip(grp.rows, grp.valid)]
        picks.deallocate()
    return {o: torch.cat(parts) for o, parts in got.items()}, state


def run_alone(mesh_device, module, users, t, prompt, owner, slot):
    """The prompt on its own row, chunk by chunk (`prefill`'s per-row form) -> (outputs, state)."""
    hidden = prompt.shape[-1]
    state, mine = module.empty_state(users), []
    rows = [r == owner for r in range(ROWS)]
    slots = [slot if r else None for r in rows]
    for start in range(0, prompt.shape[0], t):
        v = min(t, prompt.shape[0] - start)
        chunk = torch.zeros(ROWS, 1, t, hidden, dtype=torch.bfloat16)
        chunk[owner, 0, :v] = prompt[start : start + v]
        last = start + t >= prompt.shape[0]
        keep, picks = chunk_inputs(mesh_device, module, t, [v] * ROWS, users, slots if last else None, rows)
        out, state = module.prefill(put(mesh_device, chunk, dims=(0, None)), state, start, keep, picks)
        mine.append(gather_rows(mesh_device, out).reshape(ROWS, t, -1)[owner, :v])
    return torch.cat(mine), state


def rel(a, b) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "t, requests",
    [
        (64, {5: (232, 2)}),  # one wave of 4 segments, the last partial
        (32, {5: (322, 2)}),  # two waves, a 2-token last segment (shorter than the conv history)
        (32, {5: (257, 2)}),  # a 1-token second wave
        (64, {5: (232, 2), 1: (322, 3)}),  # two requests on 4 rows each; row 1's in two waves, its last 2 tokens
        (512, {5: (1500, 2)}),  # segments of 512 (the prefill's chunk): 3 segments
        (512, {5: (1100, 2), 1: (900, 3)}),  # two requests, segments of 512
    ],
)
def test_kda_device_prefill_sp(mesh_device, t, requests, pcc_thresholds):
    """Requests' prompts over the mesh rows (`TtKda.sp_picks`, `_prefill_sp`, the generator's schedule): segments of
    t tokens, each request on the block of rows that ends at its own row ({own row: (length, user of 4)}).  Each
    request's outputs, user state and conv history against `tt/` and against the same prompt prefilled on its row
    alone (`prefill`'s per-row form; the chain's bf16 maps make the difference); every other user untouched."""
    layer, users = 0, 4
    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = KdaWeights.from_checkpoint(checkpoint, layer)
    reference = Kda(weights, config)
    module = TtKda(mesh_device, weights, config)
    g = torch.Generator().manual_seed(31)
    prompts = {
        o: (torch.randn(n, config.hidden_size, generator=g) * 0.5).to(torch.bfloat16) for o, (n, _) in requests.items()
    }
    waves = sp_schedule(requests, t)
    got, state = run_sp(mesh_device, module, users, t, prompts, waves, module.empty_state(users))
    recurrent, history = read_states(mesh_device, module, state, users)
    order = conv_order(config, module)
    for owner, (length, slot) in requests.items():
        mine, alone = run_alone(mesh_device, module, users, t, prompts[owner], owner, slot)
        recurrent1, history1 = read_states(mesh_device, module, alone, users)
        with torch.no_grad():
            want, ref = reference(prompts[owner][None], None)
        st, st1 = recurrent[owner, slot], recurrent1[owner, slot]
        print(
            f"row {owner}: {length} tokens in segments of {t} ({len(waves)} waves, {len(requests)} requests): output "
            f"pcc {pcc(got[owner], want[0]):.6f} (row alone {pcc(mine, want[0]):.6f}), vs the row alone rel "
            f"{rel(got[owner], mine):.2e}; state pcc {pcc(st, ref.recurrent[0]):.6f} (alone "
            f"{pcc(st1, ref.recurrent[0]):.6f}), vs alone rel {rel(st, st1):.2e}",
            flush=True,
        )
        assert pcc(got[owner], want[0]) >= pcc_thresholds["test_kda_device"]
        assert pcc(st, ref.recurrent[0]) >= pcc_thresholds["test_kda_device"]
        for i, (h, h1) in enumerate(zip(history, history1)):
            assert torch.equal(h[owner, slot], h1[owner, slot]), f"row {owner}: conv history row {i} differs"
            assert pcc(h[owner, slot], ref.conv[0, order, i + 1]) >= 0.9999, f"row {owner}: conv history row {i}"
    for r in range(ROWS):
        for u in range(users):
            if (r, u) not in {(o, s) for o, (_, s) in requests.items()}:
                assert not recurrent[r, u].any() and not any(
                    h[r, u].any() for h in history
                ), f"row {r} user {u} changed"
    print("conv histories as the rows alone's, every other user untouched", flush=True)
