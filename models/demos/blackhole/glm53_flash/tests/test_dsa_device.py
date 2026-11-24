"""Device DSA decode (v0, dense MLA) against `tt/` stepping one token at a time (the indexer keeps every token here).

8 rows x 4 to 128 users, 6 steps, the first and the last DSA layer.  Run (a device test):
    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_dsa_device.py
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
from models.demos.blackhole.glm53_flash.tt.dsa import Dsa, DsaWeights  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import dsa as dsa_module  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    COLS,
    DEVICE_PARAMS,
    MESH_SHAPE,
    ROWS,
    eager_place,
    gather_rows,
    put,
)
from models.demos.blackhole.glm53_flash.ttnn.dsa import TtDsa  # noqa: E402

STEPS = int(os.environ.get("GLM53_DSA_TEST_STEPS", "6"))


def pcc(a, b) -> float:
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer", [int(v) for v in os.environ.get("GLM53_DSA_TEST_LAYERS", "3,43").split(",")])
@pytest.mark.parametrize("sdpa_hifi4", [False, True])
def test_dsa_device(mesh_device, layer, sdpa_hifi4, pcc_thresholds):
    """Users per row 4 / 16 / 32 / 128 (128: two sdpa decode calls, one user per core) in one case per layer and
    config: the fixture opens a new mesh per case, so the layer's weights are uploaded once for all of them."""
    from models.demos.blackhole.glm53_flash.ttnn.common import HIFI4_FP32

    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = DsaWeights.from_checkpoint(checkpoint, layer)
    reference = Dsa(weights, config)
    max_seq = int(os.environ.get("GLM53_DSA_TEST_MAX_SEQ", "128"))  # (the model's decode: 512)
    module = TtDsa(
        mesh_device, weights, config, max_seq=max_seq, sdpa_compute_config=HIFI4_FP32 if sdpa_hifi4 else None
    )
    for users in (int(v) for v in os.environ.get("GLM53_DSA_TEST_USERS", "4,16,32,128").split(",")):
        check_dsa(mesh_device, module, reference, config, layer, users, sdpa_hifi4, pcc_thresholds)


def check_dsa(mesh_device, module, reference, config, layer, users, sdpa_hifi4, pcc_thresholds):
    USERS = users

    g = torch.Generator().manual_seed(layer)
    xs = (torch.randn(STEPS, ROWS, 1, USERS, config.hidden_size, generator=g) * 0.5).to(torch.bfloat16)
    state, ref_state = module.empty_state(USERS), None
    for step in range(STEPS):
        out, state = module.step(
            put(mesh_device, xs[step], dims=(0, None)),
            state,
            module.positions_on_device(torch.full((ROWS, USERS), step)),
        )
        got = gather_rows(mesh_device, out).reshape(ROWS * USERS, -1)
        with torch.no_grad():
            want, ref_state, _ = reference(xs[step].reshape(ROWS * USERS, 1, -1), ref_state)
        score = pcc(got, want[:, 0])
        print(f"layer {layer} users {users} sdpa_hifi4 {sdpa_hifi4} step {step}: pcc {score:.6f}")
        assert score >= pcc_thresholds["test_dsa_device"], f"step {step}: pcc {score:.6f}"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer", [3])
@pytest.mark.parametrize("users", [4])
def test_dsa_device_long(mesh_device, layer, users, pcc_thresholds):
    """Through the cache's 128-row boundaries up to its last row (cache 256 rows, positions 0..255)."""
    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = DsaWeights.from_checkpoint(checkpoint, layer)
    reference = Dsa(weights, config)
    max_seq = 256
    module = TtDsa(mesh_device, weights, config, max_seq=max_seq)
    g = torch.Generator().manual_seed(layer)
    state, ref_state, worst = module.empty_state(users), None, 1.0
    for step in range(max_seq):
        x = (torch.randn(ROWS, 1, users, config.hidden_size, generator=g) * 0.5).to(torch.bfloat16)
        out, state = module.step(
            put(mesh_device, x, dims=(0, None)), state, module.positions_on_device(torch.full((ROWS, users), step))
        )
        got = gather_rows(mesh_device, out).reshape(ROWS * users, -1)
        with torch.no_grad():
            want, ref_state, _ = reference(x.reshape(ROWS * users, 1, -1), ref_state)
        score = pcc(got, want[:, 0])
        worst = min(worst, score)
        if step in (0, 31, 32, 127, 128, 129, 254, 255):
            print(f"layer {layer} users {users} position {step}: pcc {score:.6f}", flush=True)
        assert score >= pcc_thresholds["test_dsa_device"], f"position {step}: pcc {score:.6f}"
    print(f"layer {layer} users {users}: worst pcc over positions 0..{max_seq - 1}: {worst:.6f}", flush=True)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer", [3])
def test_dsa_device_prefill(mesh_device, layer, pcc_thresholds):
    """A 96-token prompt per mesh row in two chunks (64 + 32), row r's into user r % 4 of 4 (the page tables per
    row), then 2 decode steps for those users; a row without a request (rows 4..7 in a second pass) must leave every
    user's cache as it was, and a row with one must change only its own user's."""
    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = DsaWeights.from_checkpoint(checkpoint, layer)
    reference = Dsa(weights, config)
    users, length = 4, 96
    slots = [r % users for r in range(ROWS)]
    module = TtDsa(mesh_device, weights, config, max_seq=256)
    g = torch.Generator().manual_seed(layer + 7)
    prompt = (torch.randn(ROWS, length, config.hidden_size, generator=g) * 0.5).to(torch.bfloat16)
    state = module.empty_state(users)
    got = []
    for start, end in ((0, 64), (64, 96)):
        tables = module.prefill_tables(eager_place(mesh_device), users, slots, start, end - start)
        out, state = module.prefill(put(mesh_device, prompt[:, None, start:end], dims=(0, None)), state, start, tables)
        got.append(gather_rows(mesh_device, out).reshape(ROWS, end - start, -1))
    got = torch.cat(got, dim=1)
    with torch.no_grad():
        want, ref_state, _ = reference(prompt, None)
    score = pcc(got, want)
    print(f"layer {layer} prefill of {length} tokens in 2 chunks: pcc {score:.6f}", flush=True)
    assert score >= pcc_thresholds["test_dsa_device"]

    # decode continues in each row's user (the other users are fresh at position 0)
    for step in range(2):
        x = (torch.randn(ROWS, 1, users, config.hidden_size, generator=g) * 0.5).to(torch.bfloat16)
        positions = torch.zeros(ROWS, users, dtype=torch.int64)
        for r, s in enumerate(slots):
            positions[r, s] = length + step
        out, state = module.step(put(mesh_device, x, dims=(0, None)), state, module.positions_on_device(positions))
        full = gather_rows(mesh_device, out).reshape(ROWS, users, -1)
        got = torch.stack([full[r, s] for r, s in enumerate(slots)])
        with torch.no_grad():
            want, ref_state, _ = reference(torch.stack([x[r, 0, s : s + 1] for r, s in enumerate(slots)]), ref_state)
        score = pcc(got, want[:, 0])
        print(f"layer {layer} decode after the prefill, position {length + step}: pcc {score:.6f}", flush=True)
        assert score >= pcc_thresholds["test_dsa_device"]

    # rows 4..7 without a request: a new prefill (rows 0..3 into user 2) leaves their caches unchanged
    def caches():
        full = ttnn.to_torch(
            state.keys, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=MESH_SHAPE, dims=(0, 1))
        )
        pages = full.reshape(ROWS, -1, *full.shape[1:])[:, : users * module.pages, :, :, :64]
        return pages.reshape(ROWS, users, module.pages, *pages.shape[2:])

    before = caches()
    half = [2] * 4 + [None] * 4
    other = (torch.randn(ROWS, 64, config.hidden_size, generator=g) * 0.5).to(torch.bfloat16)
    tables = module.prefill_tables(eager_place(mesh_device), users, half, 0, 64)
    module.prefill(put(mesh_device, other[:, None], dims=(0, None)), state, 0, tables)
    after = caches()
    assert torch.equal(after[4:], before[4:]), "a row without a request changed its cache"
    assert not torch.equal(after[:4, 2, :2], before[:4, 2, :2]), "a row with a request did not write its user"
    assert torch.equal(after[:4, 2, 2:], before[:4, 2, 2:]), "a 64-token chunk wrote past its 2 pages"
    others = [0, 1, 3]
    assert torch.equal(after[:4, others], before[:4, others]), "a row with a request wrote another user"
    print("rows without a request unchanged, rows with one wrote their user's first 2 pages only", flush=True)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_dsa_device_indexer(mesh_device, pcc_thresholds):
    """Decode past the dense limit (the indexer, `TtDsa._step_indexer`): the reference's state after a long context
    (the per-token projections of random inputs only, no attention needed) loaded into a cache of 8,192 tokens, then a
    few steps against tt/ stepping the same tokens (every mesh row the same users).  GLM53_DSA_INDEXER_START (4,097),
    _STEPS (4), _USERS (2 a row), _LAYER (3)."""
    import torch.nn.functional as F

    from models.demos.blackhole.glm53_flash.tt.common import rms_norm
    from models.demos.blackhole.glm53_flash.tt.dsa import INDEXER_NORM_EPS, DsaState

    layer = int(os.environ.get("GLM53_DSA_INDEXER_LAYER", "3"))
    start = int(os.environ.get("GLM53_DSA_INDEXER_START", "4097"))
    steps = int(os.environ.get("GLM53_DSA_INDEXER_STEPS", "4"))
    users = int(os.environ.get("GLM53_DSA_INDEXER_USERS", "2"))
    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    w = DsaWeights.from_checkpoint(checkpoint, layer)
    ref = Dsa(w, config)
    g = torch.Generator().manual_seed(layer)
    x0 = torch.randn(users, start, config.hidden_size, generator=g).to(torch.bfloat16)
    latent = rms_norm(F.linear(x0, w.kv_a), w.kv_a_norm, ref.eps)
    index_key = F.layer_norm(
        F.linear(x0, w.index_wk), (ref.index_dim,), w.index_k_norm_weight, w.index_k_norm_bias, INDEXER_NORM_EPS
    )
    index_gate = F.linear(x0, w.index_pool_gate)
    state_ref = DsaState(latent=latent, index_key=index_key, index_gate=index_gate)

    module = TtDsa(mesh_device, w, config, max_seq=8192)
    assert module.long
    state = module.load_reference_state(latent, index_key, index_gate, w)
    worst = 1.0
    for i in range(steps):
        x = torch.randn(users, 1, config.hidden_size, generator=g).to(torch.bfloat16)
        want, state_ref, sel = ref(x, state_ref)
        positions = torch.full((ROWS, users), start + i, dtype=torch.int64)
        out, state = module.step(
            put(mesh_device, x.reshape(1, 1, users, -1), dims=(None, None)),
            state,
            module.positions_on_device(positions),
        )
        got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(users, -1)
        ttnn.deallocate(out)
        for u in range(users):
            p = torch.corrcoef(torch.stack([got[u], want[u, 0].float()]))[0, 1].item()
            worst = min(worst, p)
            print(
                f"layer {layer} position {start + i} user {u}: pcc {p:.6f} (selected {int((sel.tokens[u, 0] >= 0).sum())} "
                f"tokens)",
                flush=True,
            )
    print(f"layer {layer}, indexer decode from position {start}: worst pcc {worst:.6f}", flush=True)
    if os.environ.get("GLM53_DSA_INDEXER_PRINT_ONLY") != "1":  # (a probe's run: printed, not failed)
        assert worst > 0.99


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_dsa_device_indexer_prefill(mesh_device, pcc_thresholds):
    """A long cache's prefill (`TtDsa._prefill_indexer`) and the decode after it: two prompts of
    GLM53_DSA_IP_LENGTH (3,001) tokens, row r's prompt r % 2 into user r % 4 of 4, in chunks of GLM53_DSA_IP_CHUNK
    (512) into a GLM53_DSA_IP_CACHE (4,096) token cache, then GLM53_DSA_IP_STEPS (4) decode steps, against `tt/`
    (layer GLM53_DSA_IP_LAYER, 3).  The chunks past 2,048 tokens pick their queries' tokens; the last one ends inside
    a pool, which decode completes."""
    import time

    layer = int(os.environ.get("GLM53_DSA_IP_LAYER", "3"))
    length = int(os.environ.get("GLM53_DSA_IP_LENGTH", "3001"))
    chunk = int(os.environ.get("GLM53_DSA_IP_CHUNK", "512"))
    cache = int(os.environ.get("GLM53_DSA_IP_CACHE", "4096"))
    steps = int(os.environ.get("GLM53_DSA_IP_STEPS", "4"))
    users = 4
    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = DsaWeights.from_checkpoint(checkpoint, layer)
    reference = Dsa(weights, config)
    module = TtDsa(mesh_device, weights, config, max_seq=cache)
    assert module.long
    g = torch.Generator().manual_seed(layer + 11)
    prompts = (torch.randn(2, length, config.hidden_size, generator=g) * 0.5).to(torch.bfloat16)
    t0 = time.time()
    with torch.no_grad():
        want, ref_state, _ = reference(prompts, None)
    print(f"reference prefill of 2 x {length} tokens: {time.time() - t0:.1f} s", flush=True)
    slots = [r % users for r in range(ROWS)]
    rows_prompt = torch.stack([prompts[r % 2] for r in range(ROWS)])  # [ROWS, L, H]
    state = module.empty_state(users)
    worst = 1.0
    for start in range(0, length, chunk):
        k = min(chunk, length - start)
        x = torch.zeros(ROWS, 1, chunk, config.hidden_size, dtype=torch.bfloat16)
        x[:, 0, :k] = rows_prompt[:, start : start + k]
        tables = module.prefill_tables(eager_place(mesh_device), users, slots, start, chunk, [k] * ROWS)
        t0 = time.time()
        out, state = module.prefill(put(mesh_device, x, dims=(0, None)), state, start, tables)
        got = gather_rows(mesh_device, out).reshape(ROWS, chunk, -1)[:, :k]
        seconds = time.time() - t0
        ttnn.deallocate(out)
        scores = [pcc(got[r], want[r % 2, start : start + k]) for r in range(ROWS)]
        worst = min(worst, *scores)
        print(
            f"layer {layer} prefill [{start}, {start + k}): pcc min {min(scores):.6f} max {max(scores):.6f} "
            f"({seconds:.2f} s with the read back)",
            flush=True,
        )
    for step in range(steps):
        x = (torch.randn(2, 1, config.hidden_size, generator=g) * 0.5).to(torch.bfloat16)
        xs = torch.zeros(ROWS, 1, users, config.hidden_size, dtype=torch.bfloat16)
        positions = torch.zeros(ROWS, users, dtype=torch.int64)
        for r, s in enumerate(slots):
            xs[r, 0, s] = x[r % 2, 0]
            positions[r, s] = length + step
        out, state = module.step(put(mesh_device, xs, dims=(0, None)), state, module.positions_on_device(positions))
        full = gather_rows(mesh_device, out).reshape(ROWS, users, -1)
        ttnn.deallocate(out)
        with torch.no_grad():
            want_step, ref_state, _ = reference(x, ref_state)
        scores = [pcc(full[r, s], want_step[r % 2, 0]) for r, s in enumerate(slots)]
        worst = min(worst, *scores)
        print(f"layer {layer} decode at {length + step}: pcc min {min(scores):.6f} max {max(scores):.6f}", flush=True)
    print(
        f"layer {layer}, long-cache prefill of {length} tokens and {steps} decode steps: worst pcc {worst:.6f}",
        flush=True,
    )
    if os.environ.get("GLM53_DSA_INDEXER_PRINT_ONLY") != "1":  # (a probe's run: printed, not failed)
        assert worst > 0.99


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "requests, max_seq, t",
    [
        ({5: (424, 2)}, 512, 128),  # one wave of 4 segments, the last partial
        ({5: (1157, 2)}, 2048, 128),  # two waves
        ({5: (424, 2), 1: (300, 3)}, 512, 128),  # two requests on 4 rows each
        ({5: (3001, 2)}, 4096, 512),  # a long cache: one wave, the queries past 2,048 picking their tokens
        ({5: (3001, 2)}, 4096, 128),  # three waves
        ({5: (3001, 2), 1: (2500, 3)}, 4096, 512),  # two requests on 4 rows each, two waves
        ({5: (1500, 2)}, 2048, 512),  # a dense cache in segments of 512 (the prefill's chunk)
        ({5: (1100, 2), 1: (900, 3)}, 2048, 512),
    ],
)
def test_dsa_device_prefill_sp(mesh_device, requests, max_seq, t, monkeypatch, pcc_thresholds):
    """Requests' prompts over the mesh rows (`TtDsa.sp_tables`, `dsa.PREFILL_SP`; the generator's schedule): segments
    of T tokens, each request on the block of rows that ends at its own row ({own row: (length, user of 4)}), then 2
    decode steps for those users; against `tt/`, every other user's cache untouched.  A dense cache goes through
    `_prefill_sp`; a long one (the indexer) through `_prefill_indexer`, gathered."""
    from models.demos.blackhole.glm53_flash.tests.test_kda_device import sp_schedule

    monkeypatch.setattr(dsa_module, "PREFILL_SP", True)
    layer, users = 3, 4
    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = DsaWeights.from_checkpoint(checkpoint, layer)
    reference = Dsa(weights, config)
    module = TtDsa(mesh_device, weights, config, max_seq=max_seq)
    hidden = config.hidden_size
    g = torch.Generator().manual_seed(layer + 17)
    prompts = {o: (torch.randn(n, hidden, generator=g) * 0.5).to(torch.bfloat16) for o, (n, _) in requests.items()}
    state = module.empty_state(users)
    assert module.users(state) == users + 1  # (the extra user, PREFILL_SP)
    waves = sp_schedule(requests, t)
    got = {o: [] for o in requests}
    for groups in waves:
        chunk = torch.zeros(ROWS, 1, t, hidden, dtype=torch.bfloat16)
        for grp in groups:
            for i, (r, v) in enumerate(zip(grp.rows, grp.valid)):
                chunk[r, 0, :v] = prompts[grp.owner][grp.start + i * t : grp.start + i * t + v]
        tables = module.sp_tables(eager_place(mesh_device), users + 1, groups, t)
        out, state = module.prefill(put(mesh_device, chunk, dims=(0, None)), state, 0, tables)
        full = gather_rows(mesh_device, out).reshape(ROWS, t, -1)
        for grp in groups:
            got[grp.owner] += [full[r, :v] for r, v in zip(grp.rows, grp.valid)]
        for z in tables:
            if isinstance(z, ttnn.Tensor):
                ttnn.deallocate(z)
    # a long cache's selection can pick a near-tied pool differently: the indexer prefill test's bound
    least = 0.99 if module.long else pcc_thresholds["test_dsa_device"]
    refs = {}
    for o, (n, u) in requests.items():
        with torch.no_grad():
            want, refs[o], _ = reference(prompts[o][None], None)
        # the same prompt on its row alone in chunks of T (the one-row form), in a fresh cache
        alone_state, alone = module.empty_state(users), []
        slots = [u if r == o else None for r in range(ROWS)]
        for s0 in range(0, n, t):
            v = min(t, n - s0)
            x = torch.zeros(ROWS, 1, t, hidden, dtype=torch.bfloat16)
            x[o, 0, :v] = prompts[o][s0 : s0 + v]
            tb = module.prefill_tables(eager_place(mesh_device), users + 1, slots, s0, t, [v] * ROWS)
            out, alone_state = module.prefill(put(mesh_device, x, dims=(0, None)), alone_state, s0, tb)
            alone.append(gather_rows(mesh_device, out).reshape(ROWS, t, -1)[o, :v])
        alone_state.deallocate()
        alone = torch.cat(alone)
        mine = torch.cat(got[o])
        score = pcc(mine, want[0])
        diff = ((mine.float() - alone.float()).norm() / alone.float().norm()).item()
        print(
            f"layer {layer}, row {o}: prefill of {n} tokens over the rows in {t}s ({len(waves)} waves, {len(requests)} "
            f"requests): pcc {score:.6f} (row alone {pcc(alone, want[0]):.6f}), vs the row alone rel {diff:.2e}",
            flush=True,
        )
        if os.environ.get("GLM53_DSA_INDEXER_PRINT_ONLY") != "1":  # (a probe's run: printed, not failed)
            assert score >= least
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=MESH_SHAPE, dims=(0, 1))
    if (
        module.long and module.shard_kv()
    ):  # (the DSA cache sharded over the row's 4 chips: a chip's quarter of the latents; the pools' keys only)
        pools = ttnn.to_torch(state.pools, mesh_composer=composer)  # [8 N, 4, 32, 128]
        pools = pools.reshape(ROWS, -1, *pools.shape[1:])[:, : users * module.pool_pages].reshape(ROWS, users, -1)
        caches = [pools]
    elif module.long:  # the latents row-major, each user's max_seq rows; the pools' keys paged
        rm = ttnn.to_torch(state.rm, mesh_composer=composer).reshape(ROWS, COLS, -1, module.rank)[:, 0]
        mine = rm[:, : users * max_seq].reshape(ROWS, users, max_seq, module.rank)
        pools = ttnn.to_torch(state.pools, mesh_composer=composer)  # [8 N, 4, 32, 128]
        pools = pools.reshape(ROWS, -1, *pools.shape[1:])[:, : users * module.pool_pages].reshape(ROWS, users, -1)
        caches = [mine, pools]
    else:
        full = ttnn.to_torch(state.keys, mesh_composer=composer)
        pages = full.reshape(ROWS, -1, *full.shape[1:])[:, : users * module.pages]
        caches = [pages.reshape(ROWS, users, module.pages, *pages.shape[2:])]
    mine_users = {(o, u) for o, (_, u) in requests.items()}
    for cache in caches:
        for r in range(ROWS):
            for u in range(users):
                if (r, u) not in mine_users:
                    assert not cache[r, u].any(), f"row {r} user {u}'s cache changed"
    print("every other user's cache untouched by the prefill", flush=True)

    # decode continues in those users (the others at position 0, writing their first row)
    for step in range(2):
        x = (torch.randn(ROWS, 1, users, hidden, generator=g) * 0.5).to(torch.bfloat16)
        positions = torch.zeros(ROWS, users, dtype=torch.int64)
        for o, (n, u) in requests.items():
            positions[o, u] = n + step
        out, state = module.step(put(mesh_device, x, dims=(0, None)), state, module.positions_on_device(positions))
        full = gather_rows(mesh_device, out).reshape(ROWS, users, -1)
        for o, (n, u) in requests.items():
            with torch.no_grad():
                want, refs[o], _ = reference(x[o, 0, u : u + 1][None], refs[o])
            score = pcc(full[o, u], want[0, 0])
            print(f"layer {layer}, row {o}: decode after the prefill, position {n + step}: pcc {score:.6f}", flush=True)
            if os.environ.get("GLM53_DSA_INDEXER_PRINT_ONLY") != "1":
                assert score >= least
