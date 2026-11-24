"""Device mHC against `tt/`: the mixing weights, the block input and the written streams, on two layers' weights.

Each of the 8 rows gets its own tokens.  Run (a device test):
    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_mhc_device.py
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
from models.demos.blackhole.glm53_flash.tests._trace import traced as trace_capture  # noqa: E402
from models.demos.blackhole.glm53_flash.tt.mhc import HyperConnection, HyperConnectionWeights  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    DEVICE_PARAMS,
    DEVICE_PARAMS_TRACE,
    MESH_SHAPE,
    ROWS,
    gather_rows,
    put,
)
from models.demos.blackhole.glm53_flash.ttnn.mhc import (  # noqa: E402
    TtHyperConnection,
    streams_from_device,
    streams_to_device,
)


def pcc(a, b) -> float:
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer, block", [(0, "attn"), (20, "ffn")])
@pytest.mark.parametrize(  # 1 and 5 users per row use padded tokens; GLM53_MHC_TEST_SEQS (e.g. 64: 512 users)
    "seq", [int(v) for v in os.environ.get("GLM53_MHC_TEST_SEQS", "1,5,32").split(",")]
)
def test_mhc_device(mesh_device, layer, block, seq, pcc_thresholds):
    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = HyperConnectionWeights.from_checkpoint(checkpoint, layer, block)
    kw = dict(
        hc_mult=config.hc_mult,
        sinkhorn_iters=config.hc_sinkhorn_iters,
        hc_eps=config.hc_eps,
        norm_eps=config.rms_norm_eps,
    )
    reference = HyperConnection(weights, **kw)
    module = TtHyperConnection(mesh_device, weights, **kw)

    g = torch.Generator().manual_seed(layer)
    n, d = config.hc_mult, config.hidden_size
    emb = checkpoint.tensor("model.language_model.embed_tokens.weight")[
        torch.randint(0, config.vocab_size, (ROWS, seq), generator=g)
    ]
    streams = (emb.float().unsqueeze(2) + torch.randn(ROWS, seq, n, d, generator=g) * emb.float().std()).to(
        torch.bfloat16
    )
    block_out = (torch.randn(ROWS, seq, d, generator=g) * 0.5).to(torch.bfloat16)

    dev_streams = streams_to_device(mesh_device, streams)
    # GLM53_MHC_TEST_CHUNKED=1: the decode's read (above 32 tokens by the ops, `read(chunked=True)`)
    post, comb, block_input = module.read(dev_streams, seq, chunked=os.environ.get("GLM53_MHC_TEST_CHUNKED") == "1")
    new = module.write(put(mesh_device, block_out.reshape(ROWS, 1, seq, d), dims=(0, None)), dev_streams, post, comb)

    with torch.no_grad():
        r_post, r_comb, r_input = reference.read(streams)
        r_new = HyperConnection.write(block_out, streams, r_post, r_comb)
    got = {
        "post": gather_rows(mesh_device, post)[..., :n].reshape(ROWS, seq, n),
        "comb": gather_rows(mesh_device, comb)[..., : n * n].reshape(ROWS, seq, n, n),
        "block_input": gather_rows(mesh_device, block_input).reshape(ROWS, seq, d),
        "new_streams": streams_from_device(mesh_device, new, n, seq),
    }
    want = {"post": r_post, "comb": r_comb, "block_input": r_input, "new_streams": r_new}
    for name in got:
        score = pcc(got[name], want[name])
        diff = (got[name].float() - want[name].float()).abs().max().item()
        print(f"layer {layer} {block} {name}: pcc {score:.7f} max|diff| {diff:.3e}")
        assert score >= pcc_thresholds["test_mhc_device"], f"{name}: pcc {score:.7f}"


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("seq", [128, 512])
def test_mhc_grouped(mesh_device, seq):
    """The grouped form (prefill chunks) against the full form: bit for bit, and the time of each (synced)."""
    import time

    from models.demos.blackhole.glm53_flash.ttnn import mhc

    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = HyperConnectionWeights.from_checkpoint(checkpoint, 20, "attn")
    kw = dict(
        hc_mult=config.hc_mult,
        sinkhorn_iters=config.hc_sinkhorn_iters,
        hc_eps=config.hc_eps,
        norm_eps=config.rms_norm_eps,
    )
    module = TtHyperConnection(mesh_device, weights, **kw)
    g = torch.Generator().manual_seed(seq)
    n, d = config.hc_mult, config.hidden_size
    streams = (torch.randn(ROWS, seq, n, d, generator=g) * 0.05).to(torch.bfloat16)
    block_out = (torch.randn(ROWS, seq, d, generator=g) * 0.5).to(torch.bfloat16)
    dev_streams = streams_to_device(mesh_device, streams)
    dev_out = put(mesh_device, block_out.reshape(ROWS, 1, seq, d), dims=(0, None))

    def run():
        post, comb, block_input = module.read(dev_streams, seq)
        new = module.write(dev_out, dev_streams, post, comb)
        return post, comb, block_input, new

    results, times = {}, {}
    saved = mhc.GROUP_MIN
    try:
        for name, group_min in (("full", 0), ("grouped", 32)):
            mhc.GROUP_MIN = group_min
            outs = run()
            results[name] = [gather_rows(mesh_device, t) for t in outs]
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(5):
                for t in run():
                    ttnn.deallocate(t)
            ttnn.synchronize_device(mesh_device)
            times[name] = (time.perf_counter() - t0) / 5
    finally:
        mhc.GROUP_MIN = saved
    print(
        f"seq {seq}: read + write full {times['full'] * 1e3:.2f} ms, grouped {times['grouped'] * 1e3:.2f} ms (synced, host included)"
    )
    for label, a, b in zip(("post", "comb", "block_input", "new_streams"), results["full"], results["grouped"]):
        a, b = a.reshape(-1)[: b.numel()], b.reshape(-1)
        same = torch.equal(a, b)
        print(f"  {label}: {'bit for bit' if same else 'max |d| %.3e' % (a.float() - b.float()).abs().max().item()}")
        assert same, label


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
@pytest.mark.parametrize("seq", [1, 4, 16])  # the decode steps of 8 / 32 / 128 users
def test_mhc_stream_write(mesh_device, seq):
    """The write (one op, `mhc_stream_write`) against `tt/`'s write on the device's own post and comb (the write
    alone), and its time (trace)."""
    import time

    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = HyperConnectionWeights.from_checkpoint(checkpoint, 20, "ffn")
    kw = dict(
        hc_mult=config.hc_mult,
        sinkhorn_iters=config.hc_sinkhorn_iters,
        hc_eps=config.hc_eps,
        norm_eps=config.rms_norm_eps,
    )
    module = TtHyperConnection(mesh_device, weights, **kw)
    g = torch.Generator().manual_seed(seq)
    n, d = config.hc_mult, config.hidden_size
    streams = (torch.randn(ROWS, seq, n, d, generator=g) * 0.05).to(torch.bfloat16)
    block_out = (torch.randn(ROWS, seq, d, generator=g) * 0.5).to(torch.bfloat16)
    dev_streams = streams_to_device(mesh_device, streams)
    dev_out = put(mesh_device, block_out.reshape(ROWS, 1, seq, d), dims=(0, None))
    post, comb, block_input = module.read(dev_streams, seq)
    ttnn.deallocate(block_input)

    new = module.write(dev_out, dev_streams, post, comb)
    got = streams_from_device(mesh_device, new, n, seq).float()
    ttnn.deallocate(new)
    h_post = gather_rows(mesh_device, post)[..., :n].reshape(ROWS, seq, n)
    h_comb = gather_rows(mesh_device, comb)[..., : n * n].reshape(ROWS, seq, n, n)
    want = HyperConnection.write(block_out, streams, h_post, h_comb).float()
    ttnn.synchronize_device(mesh_device)
    reps = 20
    with trace_capture(mesh_device) as trace:
        outs = [module.write(dev_out, dev_streams, post, comb) for _ in range(reps)]
    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
    sec = (time.perf_counter() - t0) / reps
    ttnn.release_trace(mesh_device, trace)
    for t in outs:
        ttnn.deallocate(t)
    diff = (got - want).abs().max().item()
    print(f"seq {seq}: write {sec * 1e6:.1f} us; against tt/'s write max |d| {diff:.3e}")
    assert diff <= 2**-6 * want.abs().max().item()


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
@pytest.mark.parametrize(
    "seq", [int(s) for s in os.environ.get("GLM53_MHC_WR_SEQ", "1,2,4,8,16,32,64").split(",")]
)  # decode: 8 / 32 / 128 users' plain and verify steps, 64: 128 users' T = 4 and 512 users' plain step
def test_mhc_stream_write_read(mesh_device, seq):
    """A sub-layer's write and the next sub-layer's read as one op (`mhc_stream_write_read`, mhc.WRITE_READ: the new
    streams handed to the read's slice cores in L1) against `write` then `read` (decode): the new streams, post, comb
    and the normed input bit for bit, and both forms' times (trace).  GLM53_MHC_WR_PREFILL=1: the prefill's groups
    form of 32 tokens (e.g. GLM53_MHC_WR_SEQ=128,256,512); GLM53_TEST_STRICT=0: a difference printed only;
    GLM53_MHC_WR_ZONE=1: no traces, the one op run once more last (for a kernel zone profile)."""
    import time

    from models.demos.blackhole.glm53_flash.ttnn import mhc
    from models.demos.blackhole.glm53_flash.ttnn.head import TtRmsNorm

    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    kw = dict(
        hc_mult=config.hc_mult,
        sinkhorn_iters=config.hc_sinkhorn_iters,
        hc_eps=config.hc_eps,
        norm_eps=config.rms_norm_eps,
    )
    attn = TtHyperConnection(mesh_device, HyperConnectionWeights.from_checkpoint(checkpoint, 20, "attn"), **kw)
    ffn = TtHyperConnection(mesh_device, HyperConnectionWeights.from_checkpoint(checkpoint, 20, "ffn"), **kw)
    gamma = checkpoint.tensor("model.language_model.layers.20.post_attention_layernorm.weight")
    norm = TtRmsNorm(mesh_device, gamma, config.rms_norm_eps)
    in_norm = TtRmsNorm(
        mesh_device, checkpoint.tensor("model.language_model.layers.20.input_layernorm.weight"), config.rms_norm_eps
    )
    g = torch.Generator().manual_seed(seq)
    n, d = config.hc_mult, config.hidden_size
    emb = checkpoint.tensor("model.language_model.embed_tokens.weight")[
        torch.randint(0, config.vocab_size, (ROWS, seq), generator=g)
    ]
    streams = (emb.float().unsqueeze(2) + torch.randn(ROWS, seq, n, d, generator=g) * emb.float().std()).to(
        torch.bfloat16
    )
    block_out = (torch.randn(ROWS, seq, d, generator=g) * 0.5).to(torch.bfloat16)
    dev_streams = streams_to_device(mesh_device, streams)
    dev_out = put(mesh_device, block_out.reshape(ROWS, 1, seq, d), dims=(0, None))
    prefill = os.environ.get("GLM53_MHC_WR_PREFILL") == "1"
    post, comb, x = attn.read(dev_streams, seq, in_norm, chunked=not prefill)
    ttnn.deallocate(x)
    saved = mhc.WRITE_READ
    saved64, mhc.WRITE_READ_64 = mhc.WRITE_READ_64, True
    got, times = {}, {}
    try:
        for name, flag in (("two ops", False), ("one op", True)):
            mhc.WRITE_READ = flag
            run = lambda: attn.write_read(dev_out, dev_streams, post, comb, ffn, seq, norm, prefill)  # noqa: E731
            new, (p2, c2, h) = run()
            got[name] = [
                streams_from_device(mesh_device, new, n, seq).float(),
                gather_rows(mesh_device, p2)[..., :n].reshape(ROWS, seq, n),
                gather_rows(mesh_device, c2)[..., : n * n].reshape(ROWS, seq, n, n),
                gather_rows(mesh_device, h).reshape(ROWS, seq, d).float(),
            ]
            for t in (new, p2, c2, h):
                ttnn.deallocate(t)
            ttnn.synchronize_device(mesh_device)
            if os.environ.get("GLM53_MHC_WR_ZONE") == "1":  # (a kernel zone profile: no traces, the one op run last)
                times[name] = 0.0
                if flag:
                    for t in (lambda r: (r[0], *r[1]))(run()):
                        ttnn.deallocate(t)
                    ttnn.synchronize_device(mesh_device)
                continue
            reps = 20
            with trace_capture(mesh_device) as trace:
                outs = [run() for _ in range(reps)]
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            times[name] = (time.perf_counter() - t0) / reps
            ttnn.release_trace(mesh_device, trace)
            for new, rest in outs:
                for t in (new, *rest):
                    ttnn.deallocate(t)
    finally:
        mhc.WRITE_READ = saved
        mhc.WRITE_READ_64 = saved64
    same = [torch.equal(a, b) for a, b in zip(got["two ops"], got["one op"])]
    print(
        f"seq {seq}: write + read {times['two ops'] * 1e6:.1f} us -> one op {times['one op'] * 1e6:.1f} us; bit for "
        f"bit (new streams, post, comb, normed input): {same}",
        flush=True,
    )
    ref = os.environ.get("GLM53_MHC_WR_REF")  # (A/B across runs: the results saved there, or compared with them)
    if ref:
        path = f"{ref}-{seq}{'-prefill' if prefill else ''}.pt"
        if os.environ.get("GLM53_MHC_WR_REF_SAVE") == "1":
            torch.save(got, path)
            print(f"seq {seq}: saved to {path}", flush=True)
        else:
            old = torch.load(path)
            for name in got:
                eq = [torch.equal(a, b) for a, b in zip(old[name], got[name])]
                print(f"seq {seq}: {name} against {path}: {eq}", flush=True)
                same += eq
    stress = int(os.environ.get("GLM53_MHC_WR_STRESS", "0"))
    if stress:  # the one op over and over in a trace (its hand-offs, semaphores, multicasts), the last result unchanged
        mhc.WRITE_READ = True
        try:
            trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            try:
                new, (p2, c2, h) = attn.write_read(dev_out, dev_streams, post, comb, ffn, seq, norm, prefill)
            finally:
                ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
            for _ in range(stress):
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            last = [
                streams_from_device(mesh_device, new, n, seq).float(),
                gather_rows(mesh_device, h).reshape(ROWS, seq, d).float(),
            ]
            ttnn.release_trace(mesh_device, trace)
            for t in (new, p2, c2, h):
                ttnn.deallocate(t)
        finally:
            mhc.WRITE_READ = saved
        kept = torch.equal(last[0], got["one op"][0]) and torch.equal(last[1], got["one op"][3])
        print(
            f"seq {seq}: {stress} replays of the one op without a hang, the last equal to the first {kept}", flush=True
        )
        same.append(kept)
    if os.environ.get("GLM53_TEST_STRICT", "1") == "1":
        assert all(same)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
@pytest.mark.parametrize("seq", [1, 4, 16, 32, 64, 128])  # the decode steps of 8 / 32 / 128 / 256 / 512 / 1,024 users
def test_mhc_stream_read(mesh_device, seq):
    """The read in three ops (`mhc_stream_project`, `mhc_stream_mix`, P @ streams; above 32 tokens 32 at a time,
    STREAM_READ_CHUNKED) against the composite: the outputs' largest differences, and each one's time (trace)."""
    import time

    from models.demos.blackhole.glm53_flash.ttnn import mhc

    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = HyperConnectionWeights.from_checkpoint(checkpoint, 20, "attn")
    kw = dict(
        hc_mult=config.hc_mult,
        sinkhorn_iters=config.hc_sinkhorn_iters,
        hc_eps=config.hc_eps,
        norm_eps=config.rms_norm_eps,
    )
    module = TtHyperConnection(mesh_device, weights, **kw)
    g = torch.Generator().manual_seed(seq)
    n, d = config.hc_mult, config.hidden_size
    emb = checkpoint.tensor("model.language_model.embed_tokens.weight")[
        torch.randint(0, config.vocab_size, (ROWS, seq), generator=g)
    ]
    streams = (emb[:, :, None, :] + torch.randn(ROWS, seq, n, d, generator=g) * 0.01).to(torch.bfloat16)
    dev_streams = streams_to_device(mesh_device, streams)

    results, times = {}, {}
    saved = mhc.STREAM_READ
    try:
        for name, flag in (("composite", False), ("ops", True)):
            mhc.STREAM_READ = flag
            outs = module.read(dev_streams, seq, chunked=True)
            results[name] = [gather_rows(mesh_device, t) for t in outs]
            for t in outs:
                ttnn.deallocate(t)
            ttnn.synchronize_device(mesh_device)
            reps = 20
            with trace_capture(mesh_device) as trace:
                captured = [module.read(dev_streams, seq, chunked=True) for _ in range(reps)]
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            times[name] = (time.perf_counter() - t0) / reps
            ttnn.release_trace(mesh_device, trace)
            for o in captured:
                for t in o:
                    ttnn.deallocate(t)
    finally:
        mhc.STREAM_READ = saved
    print(f"seq {seq}: read composite {times['composite'] * 1e6:.1f} us, ops {times['ops'] * 1e6:.1f} us")
    for label, a, b in zip(("post", "comb", "block_input"), results["composite"], results["ops"]):
        a, b = a.float().reshape(-1), b.float().reshape(-1)
        rel = ((a - b).abs().max() / a.abs().max()).item()
        print(f"  {label}: max |d| / max |a| {rel:.2e}, pcc {pcc(a, b):.7f}")
        assert pcc(a, b) > 0.99999, label


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
@pytest.mark.parametrize("seq", [1, 4, 5, 16, 32, 64])  # the decode steps of 8 / 32 / 128 / 256 / 512 users; 5 pads
def test_mhc_stream_read_fused(mesh_device, seq):
    """The read and the block's RMS norm in one op (`mhc_stream_read`, 8 / 16 slice cores) against the three ops and
    `TtRmsNorm`, and the normed block input against torch (`tt/`'s read, then the RMS norm in fp32); each form's time
    (trace).  32 / 64 tokens a row: the decode's read (`chunked`), the groups form of 8 tokens a group
    (GLM53_MHC_READ_DECODE_G8, GLM53_MHC_TEST_G8_SLICES slice cores a group: 8) against the chunked ops."""
    import time

    from models.demos.blackhole.glm53_flash.ttnn import mhc
    from models.demos.blackhole.glm53_flash.ttnn.head import TtRmsNorm

    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = HyperConnectionWeights.from_checkpoint(checkpoint, 20, "attn")
    kw = dict(
        hc_mult=config.hc_mult,
        sinkhorn_iters=config.hc_sinkhorn_iters,
        hc_eps=config.hc_eps,
        norm_eps=config.rms_norm_eps,
    )
    reference = HyperConnection(weights, **kw)
    module = TtHyperConnection(mesh_device, weights, **kw)
    gamma = checkpoint.tensor("model.language_model.layers.20.input_layernorm.weight")
    norm = TtRmsNorm(mesh_device, gamma, config.rms_norm_eps)
    g = torch.Generator().manual_seed(seq)
    n, d = config.hc_mult, config.hidden_size
    emb = checkpoint.tensor("model.language_model.embed_tokens.weight")[
        torch.randint(0, config.vocab_size, (ROWS, seq), generator=g)
    ]
    streams = (emb.float().unsqueeze(2) + torch.randn(ROWS, seq, n, d, generator=g) * emb.float().std()).to(
        torch.bfloat16
    )
    dev_streams = streams_to_device(mesh_device, streams)
    with torch.no_grad():
        r_post, r_comb, r_input = reference.read(streams)
        x = r_input.float()
        r_normed = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + config.rms_norm_eps) * gamma.float()).to(
            torch.bfloat16
        )
    want = {"post": r_post, "comb": r_comb, "normed": r_normed}

    results, times = {}, {}
    saved = mhc.READ_FUSED, mhc.READ_DECODE_G8
    chunked = seq in (32, 64)
    fused32 = int(os.environ.get("GLM53_MHC_TEST_G8_SLICES", "8"))
    try:
        for name, flag in (("ops", False), ("fused", True)):
            if chunked:
                mhc.READ_FUSED, mhc.READ_DECODE_G8 = True, fused32 if flag else 0
            else:
                mhc.READ_FUSED = flag
            outs = module.read(dev_streams, seq, norm, chunked=chunked)
            results[name] = {
                "post": gather_rows(mesh_device, outs[0])[..., :n].reshape(ROWS, seq, n),
                "comb": gather_rows(mesh_device, outs[1])[..., : n * n].reshape(ROWS, seq, n, n),
                "normed": gather_rows(mesh_device, outs[2]).reshape(ROWS, seq, d),
            }
            for t in outs:
                ttnn.deallocate(t)
            ttnn.synchronize_device(mesh_device)
            reps = 20
            with trace_capture(mesh_device) as trace:
                captured = [module.read(dev_streams, seq, norm, chunked=chunked) for _ in range(reps)]
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            times[name] = (time.perf_counter() - t0) / reps
            ttnn.release_trace(mesh_device, trace)
            for o in captured:
                for t in o:
                    ttnn.deallocate(t)
    finally:
        mhc.READ_FUSED, mhc.READ_DECODE_G8 = saved
    print(f"seq {seq}: read + norm: ops {times['ops'] * 1e6:.1f} us, fused {times['fused'] * 1e6:.1f} us", flush=True)
    for label in ("post", "comb", "normed"):
        a, b, w = (results["ops"][label].float(), results["fused"][label].float(), want[label].float())
        rel = ((a - b).abs().max() / a.abs().max()).item()
        print(
            f"  {label}: fused vs ops max |d| / max |a| {rel:.2e}, pcc {pcc(a, b):.7f}; vs torch: ops pcc "
            f"{pcc(a, w):.7f} max |d| {(a - w).abs().max().item():.3e}, fused pcc {pcc(b, w):.7f} max |d| "
            f"{(b - w).abs().max().item():.3e}",
            flush=True,
        )
        if label == "normed" and pcc(a, b) <= 0.99999:  # where the fused output differs
            col = torch.arange(a.shape[-1])
            for name, m in (("cols % 32 < 16", col % 32 < 16), ("cols % 32 >= 16", col % 32 >= 16)):
                print(f"    {name}: pcc {pcc(a[..., m], b[..., m]):.5f}", flush=True)
            ratio = (b / a.where(a.abs() > 1e-3, torch.tensor(float("nan")))).flatten()
            print(f"    fused / ops: median {ratio.nanmedian().item():.4f}", flush=True)
            print(f"    ops row 0 cols 0..7 {a[0, 0, :8].tolist()}", flush=True)
            print(f"    fused row 0 cols 0..7 {b[0, 0, :8].tolist()}", flush=True)
            print(f"    ops row 0 cols 16..19 {a[0, 0, 16:20].tolist()}, fused {b[0, 0, 16:20].tolist()}", flush=True)
            per_tile = [pcc(a[..., 32 * c : 32 * c + 32], b[..., 32 * c : 32 * c + 32]) for c in range(0, 128, 8)]
            print(f"    pcc per column tile (every 8th): {[round(v, 3) for v in per_tile]}", flush=True)
        assert pcc(a, b) > 0.99999, label


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
@pytest.mark.parametrize("seq", [128, 256, 512])  # prefill chunks a row (2 x 128: 256)
def test_mhc_stream_read_groups(mesh_device, seq):
    """The prefill's read and the block's RMS norm by `mhc_stream_read`'s groups form (GLM53_MHC_READ_GROUPS: 32
    tokens a group) against the composite and `TtRmsNorm`, both against torch; each form's time (trace)."""
    import time

    from models.demos.blackhole.glm53_flash.ttnn import mhc
    from models.demos.blackhole.glm53_flash.ttnn.head import TtRmsNorm

    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = HyperConnectionWeights.from_checkpoint(checkpoint, 20, "attn")
    kw = dict(
        hc_mult=config.hc_mult,
        sinkhorn_iters=config.hc_sinkhorn_iters,
        hc_eps=config.hc_eps,
        norm_eps=config.rms_norm_eps,
    )
    reference = HyperConnection(weights, **kw)
    module = TtHyperConnection(mesh_device, weights, **kw)
    gamma = checkpoint.tensor("model.language_model.layers.20.input_layernorm.weight")
    norm = TtRmsNorm(mesh_device, gamma, config.rms_norm_eps)
    g = torch.Generator().manual_seed(seq)
    n, d = config.hc_mult, config.hidden_size
    emb = checkpoint.tensor("model.language_model.embed_tokens.weight")[
        torch.randint(0, config.vocab_size, (ROWS, seq), generator=g)
    ]
    streams = (emb.float().unsqueeze(2) + torch.randn(ROWS, seq, n, d, generator=g) * emb.float().std()).to(
        torch.bfloat16
    )
    dev_streams = streams_to_device(mesh_device, streams)
    with torch.no_grad():
        r_post, r_comb, r_input = reference.read(streams)
        x = r_input.float()
        r_normed = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + config.rms_norm_eps) * gamma.float()).to(
            torch.bfloat16
        )
    want = {"post": r_post, "comb": r_comb, "normed": r_normed}

    results, times = {}, {}
    saved = mhc.READ_GROUPS
    try:
        for name, flag in (("composite", False), ("groups", True)):
            mhc.READ_GROUPS = flag
            outs = module.read(dev_streams, seq, norm)
            results[name] = {
                "post": gather_rows(mesh_device, outs[0])[..., :n].reshape(ROWS, seq, n),
                "comb": gather_rows(mesh_device, outs[1])[..., : n * n].reshape(ROWS, seq, n, n),
                "normed": gather_rows(mesh_device, outs[2]).reshape(ROWS, seq, d),
            }
            for t in outs:
                ttnn.deallocate(t)
            ttnn.synchronize_device(mesh_device)
            reps = 10
            with trace_capture(mesh_device) as trace:
                captured = [module.read(dev_streams, seq, norm) for _ in range(reps)]
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            times[name] = (time.perf_counter() - t0) / reps
            ttnn.release_trace(mesh_device, trace)
            for o in captured:
                for t in o:
                    ttnn.deallocate(t)
    finally:
        mhc.READ_GROUPS = saved
    print(
        f"seq {seq}: read + norm: composite {times['composite'] * 1e6:.1f} us, groups {times['groups'] * 1e6:.1f} us",
        flush=True,
    )
    stress = int(os.environ.get("GLM53_MHC_GROUPS_STRESS", "0"))
    if stress:  # the groups form over and over in a trace (its multicasts and semaphores), the last result unchanged
        mhc.READ_GROUPS = True
        try:
            with trace_capture(mesh_device) as trace:
                captured = module.read(dev_streams, seq, norm)
            for _ in range(stress):
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            last = gather_rows(mesh_device, captured[2]).reshape(ROWS, seq, d)
            ttnn.release_trace(mesh_device, trace)
            for t in captured:
                ttnn.deallocate(t)
        finally:
            mhc.READ_GROUPS = saved
        same = torch.equal(last, results["groups"]["normed"])
        print(
            f"seq {seq}: {stress} replays of the groups form without a hang, the last equal to the first {same}",
            flush=True,
        )
        assert same
    for label in ("post", "comb", "normed"):
        a, b, w = (results["composite"][label].float(), results["groups"][label].float(), want[label].float())
        print(
            f"  {label}: groups vs composite pcc {pcc(a, b):.7f}; vs torch: composite pcc {pcc(a, w):.7f} max |d| "
            f"{(a - w).abs().max().item():.3e}, groups pcc {pcc(b, w):.7f} max |d| {(b - w).abs().max().item():.3e}",
            flush=True,
        )
        assert pcc(b, w) > 0.9999, label


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
@pytest.mark.parametrize("seq", [1, 5, 16, 32])  # 1 / 4 SFPU slots; 5 users per row use padded tokens
def test_mhc_sinkhorn_sfpu(mesh_device, seq):
    """`mhc_stream_mix` with comb's Sinkhorn on the SFPU (the default) against the tile passes (MHC_SINKHORN_SFPU=0):
    comb against a float64 reference from the same streams, how far its column sums are from 1, the other outputs bit
    for bit, and the read's time (trace)."""
    import time

    from models.demos.blackhole.glm53_flash.ttnn import mhc

    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = HyperConnectionWeights.from_checkpoint(checkpoint, 20, "attn")
    kw = dict(
        hc_mult=config.hc_mult,
        sinkhorn_iters=config.hc_sinkhorn_iters,
        hc_eps=config.hc_eps,
        norm_eps=config.rms_norm_eps,
    )
    module = TtHyperConnection(mesh_device, weights, **kw)
    g = torch.Generator().manual_seed(seq)
    n, d = config.hc_mult, config.hidden_size
    emb = checkpoint.tensor("model.language_model.embed_tokens.weight")[
        torch.randint(0, config.vocab_size, (ROWS, seq), generator=g)
    ]
    streams = (emb.float().unsqueeze(2) + torch.randn(ROWS, seq, n, d, generator=g) * emb.float().std()).to(
        torch.bfloat16
    )
    dev_streams = streams_to_device(mesh_device, streams)

    # comb in float64 from the same streams (HyperConnection.read's steps)
    x = streams.double().reshape(ROWS, seq, n * d)
    flat = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + config.rms_norm_eps)
    mixed = flat @ weights.fn.double().T
    scale, base, eps = weights.scale.double(), weights.base.double(), config.hc_eps
    want = torch.softmax(mixed[..., 2 * n :].reshape(ROWS, seq, n, n) * scale[2] + base[2 * n :].reshape(n, n), -1)
    want = want + eps
    want = want / (want.sum(-2, keepdim=True) + eps)
    for _ in range(config.hc_sinkhorn_iters - 1):
        want = want / (want.sum(-1, keepdim=True) + eps)
        want = want / (want.sum(-2, keepdim=True) + eps)

    results, times = {}, {}
    saved_read, saved_env = mhc.STREAM_READ, os.environ.get("MHC_SINKHORN_SFPU")
    try:
        mhc.STREAM_READ = True
        for name, flag in (("tiles", "0"), ("sfpu", "1")):
            os.environ["MHC_SINKHORN_SFPU"] = flag  # read when the program is made
            mesh_device.clear_program_cache()
            outs = module.read(dev_streams, seq)
            results[name] = [gather_rows(mesh_device, t) for t in outs]
            for t in outs:
                ttnn.deallocate(t)
            ttnn.synchronize_device(mesh_device)
            reps = 20
            with trace_capture(mesh_device) as trace:
                captured = [module.read(dev_streams, seq) for _ in range(reps)]
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            times[name] = (time.perf_counter() - t0) / reps
            ttnn.release_trace(mesh_device, trace)
            for o in captured:
                for t in o:
                    ttnn.deallocate(t)
    finally:
        mhc.STREAM_READ = saved_read
        if saved_env is None:
            os.environ.pop("MHC_SINKHORN_SFPU", None)
        else:
            os.environ["MHC_SINKHORN_SFPU"] = saved_env
        mesh_device.clear_program_cache()

    print(f"seq {seq}: read with the tile passes {times['tiles'] * 1e6:.1f} us, SFPU {times['sfpu'] * 1e6:.1f} us")
    errors = {}
    for name in ("tiles", "sfpu"):
        comb = results[name][1][..., : n * n].reshape(ROWS, seq, n, n).double()
        errors[name] = (comb - want).abs().max().item()
        col = (comb.sum(-2) - (want.sum(-2))).abs().max().item()
        print(f"  {name}: comb max |d| vs float64 {errors[name]:.3e}, column sums' max |d| {col:.3e}")
    for label, i in (("post", 0), ("block_input", 2)):
        same = torch.equal(results["tiles"][i], results["sfpu"][i])
        print(f"  {label}: {'bit for bit' if same else 'differs'}")
        assert same, label
    assert errors["sfpu"] <= max(1.5 * errors["tiles"], 1e-5), errors
