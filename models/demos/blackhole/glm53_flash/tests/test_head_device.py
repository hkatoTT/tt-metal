"""Embedding -> final norm -> LM head on the device against `tt/` (each row its own tokens).

Run (a device test): GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_head_device.py
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.config import Glm53Config  # noqa: E402
from models.demos.blackhole.glm53_flash.tt.common import rms_norm  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    DEVICE_PARAMS,
    DEVICE_PARAMS_TRACE,
    MESH_SHAPE,
    ROWS,
    gather_rows,
    put,
)
from models.demos.blackhole.glm53_flash.ttnn.head import TtEmbedding, TtLmHead, TtRmsNorm  # noqa: E402

P = "model.language_model."


def pcc(a, b) -> float:
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_head_device(mesh_device, pcc_thresholds):
    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    seq = 32
    ids = torch.randint(0, config.vocab_size, (ROWS, seq), generator=torch.Generator().manual_seed(0))
    table, norm, head = (
        checkpoint.tensor(P + "embed_tokens.weight"),
        checkpoint.tensor(P + "norm.weight"),
        checkpoint.tensor("lm_head.weight"),
    )

    embed = TtEmbedding(mesh_device, table)
    final_norm = TtRmsNorm(mesh_device, norm, config.rms_norm_eps)
    lm_head = TtLmHead(mesh_device, head)
    dev_ids = put(mesh_device, ids.to(torch.int32), dims=(0, None), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    emb = embed(dev_ids)
    logits = lm_head(final_norm(emb))

    want_emb = F.embedding(ids, table)
    want_logits = F.linear(rms_norm(want_emb, norm, config.rms_norm_eps).float(), head.float())
    got_emb = gather_rows(mesh_device, emb).reshape(ROWS, seq, -1)
    got_logits = gather_rows(mesh_device, logits).reshape(ROWS, seq, -1)
    assert torch.equal(got_emb, want_emb), "the embedding lookup is a copy and must be exact"
    for r in range(ROWS):
        score = pcc(got_logits[r], want_logits[r])
        top1 = (got_logits[r].argmax(-1) == want_logits[r].argmax(-1)).float().mean().item()
        print(f"row {r}: logits pcc {score:.6f}, top-1 agreement {top1:.3f}")
        assert score >= pcc_thresholds["test_head_device"]


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_head_greedy_mesh(mesh_device):
    """The greedy head over the whole mesh (`greedy_mesh`: each chip a 1/32 of the vocabulary for every row's tokens)
    against torch's argmax of the fp32 logits, 1 / 4 / 16 / 64 tokens a row (BF16 head)."""
    checkpoint = Glm53Checkpoint()
    head = checkpoint.tensor("lm_head.weight")
    lm_head = TtLmHead(mesh_device, head)
    for seq in (1, 4, 16, 64):
        g = torch.Generator().manual_seed(seq)
        x = (torch.randn(ROWS, seq, head.shape[1], generator=g) * 2.0).to(torch.bfloat16)
        dev_x = put(mesh_device, x[:, None], dims=(0, None))
        mesh_ids = ttnn.to_torch(ttnn.get_device_tensors(lm_head.greedy_mesh(dev_x))[0]).reshape(ROWS, seq)
        logits = F.linear(x.float(), head.float())
        top2 = logits.topk(2, dim=-1).values
        clear = (top2[..., 0] - top2[..., 1]) > 1e-2 * top2[..., 0].abs().clamp(min=1.0)  # not a near tie
        want = logits.argmax(-1).float()
        same_torch = (mesh_ids == want)[clear].float().mean().item()
        print(
            f"{seq} tokens a row: the mesh head's ids equal torch's argmax for {same_torch:.4f} of the "
            f"{int(clear.sum())} clear ones",
            flush=True,
        )
        assert same_torch == 1.0


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("tokens", [8, 32, 64])
def test_rms_norm_sharded(mesh_device, tokens):
    """A decode step's norm ([1, 1, T, 4096], T <= 64: one or two tile rows) interleaved (one core) against width-sharded over
    16 / 32 / 64 cores: the time per call (synced, host included) and the output against the interleaved one and a
    float64 reference.  Random input and weight (the final norm's scale)."""
    import time

    g = torch.Generator().manual_seed(tokens)
    x = torch.randn(ROWS, 1, tokens, 4096, generator=g).to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(4096, generator=g)).to(torch.bfloat16)
    want = rms_norm(x.double(), w.double(), 1e-5)
    dev_x = put(mesh_device, x, dims=(0, None))
    base = None
    for cores in (0, 16, 32, 64):
        norm = TtRmsNorm(mesh_device, w, 1e-5, shard_cores=cores)
        out = gather_rows(mesh_device, norm(dev_x)).reshape(ROWS, 1, tokens, -1)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(20):
            ttnn.deallocate(norm(dev_x))
        ttnn.synchronize_device(mesh_device)
        us = (time.perf_counter() - t0) / 20 * 1e6
        base = out if base is None else base
        diff = (out.float() - base.float()).abs().max().item()
        print(
            f"tokens {tokens}, {cores or 'interleaved'} cores: {us:.1f} us; vs interleaved max |d| {diff:.3e}, "
            f"pcc vs float64 {pcc(out, want):.7f}",
            flush=True,
        )
        assert pcc(out, want) > 0.9999


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_head_pick(mesh_device):
    """`greedy_mesh` with the pick over the mesh in one op (GLM53_HEAD_PICK: `head_pick.head_pick`) against the
    all-gathers and the max / compare / min, at 1 / 2 / 4 / 8 / 16 / 32 tokens a row (the BF16 LM head, random hidden
    states): the same ids on every chip, torch's argmax where the top two are not a near tie, and each form's time
    (trace of 20 calls).  GLM53_HEAD_PICK_REPEAT: calls of the one-op form compared with the first (a stress)."""
    import time

    from models.demos.blackhole.glm53_flash.ttnn import head as head_mod

    checkpoint = Glm53Checkpoint()
    head = checkpoint.tensor("lm_head.weight")
    lm_head = TtLmHead(mesh_device, head)
    saved = head_mod.HEAD_PICK
    head_mod.HEAD_PICK_MAX_ROWS = 256  # (every case here)
    repeat = int(os.environ.get("GLM53_HEAD_PICK_REPEAT", "0"))
    ok = True

    def all_chips(t):
        return [ttnn.to_torch(d).reshape(-1) for d in ttnn.get_device_tensors(t)]

    def traced(fn, reps=20):
        ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh_device)
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        outs = [fn() for _ in range(reps)]
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        sec = (time.perf_counter() - t0) / reps
        ttnn.release_trace(mesh_device, trace)
        for o in outs:
            ttnn.deallocate(o)
        return sec

    try:
        for seq in (1, 2, 4, 8, 16, 32):
            g = torch.Generator().manual_seed(200 + seq)
            x = (torch.randn(ROWS, seq, head.shape[1], generator=g) * 2.0).to(torch.bfloat16)
            dev_x = put(mesh_device, x[:, None], dims=(0, None))
            ids, times = {}, {}
            for name, flag in (("ops", False), ("pick", True)):
                head_mod.HEAD_PICK = flag
                out = lm_head.greedy_mesh(dev_x)
                ids[name] = all_chips(out)
                ttnn.deallocate(out)
                times[name] = traced(lambda: lm_head.greedy_mesh(dev_x)) * 1e6
            logits = F.linear(x.float(), head.float())
            top2 = logits.topk(2, dim=-1).values
            clear = ((top2[..., 0] - top2[..., 1]) > 1e-2 * top2[..., 0].abs().clamp(min=1.0)).reshape(-1)
            want = logits.argmax(-1).float().reshape(-1)
            same_ops = all(torch.equal(a[: ROWS * seq], b[: ROWS * seq]) for a, b in zip(ids["ops"], ids["pick"]))
            same_chips = all(torch.equal(a[: ROWS * seq], ids["pick"][0][: ROWS * seq]) for a in ids["pick"])
            torch_ok = (ids["pick"][0][: ROWS * seq] == want)[clear].float().mean().item()
            stress = 0
            head_mod.HEAD_PICK = True
            for _ in range(repeat):
                out = lm_head.greedy_mesh(dev_x)
                stress += sum(int((a[: ROWS * seq] != ids["pick"][0][: ROWS * seq]).sum()) for a in all_chips(out))
                ttnn.deallocate(out)
            print(
                f"{seq} tokens a row: the pick op's ids {'equal' if same_ops else 'DIFFER from'} the ops' on every chip, "
                f"{'the same' if same_chips else 'NOT the same'} on all 32; torch's argmax for {torch_ok:.4f} of the "
                f"{int(clear.sum())} clear ones; greedy_mesh {times['ops']:.1f} -> {times['pick']:.1f} us"
                + (f"; {repeat} more calls: {stress} ids differ" if repeat else ""),
                flush=True,
            )
            ok &= same_ops and same_chips and torch_ok == 1.0 and stress == 0
    finally:
        head_mod.HEAD_PICK = saved
    assert ok
