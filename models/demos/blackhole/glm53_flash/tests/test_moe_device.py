"""The device MoE (the experts op on the experts sliced by mesh column) against `tt/`, including the routing it chose
(`TtMoe.route`, the same choice by ttnn ops).

    GLM53_MOE_SEQS = comma list of tokens per row for test_moe_device (default 1,4,16,32,64,128)
    GLM53_MOE_COL_AB = 1: each of them twice, with the MoE's column ops off and on (moe.MOE_COL_CUSTOM, switched in
        the process: the same weights), each against tt/ (bf16 sums by expert id, as HF) and against the same MoE in
        float64 (from the same bf16 weights and input): PCC, max |d|, RMS and mean signed error; and off against on
Run (a device test): GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_moe_device.py
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
from models.demos.blackhole.glm53_flash.tt.common import clamped_swiglu  # noqa: E402
from models.demos.blackhole.glm53_flash.tt.moe import Moe, MoeWeights  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import moe as moe_module  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    DEVICE_PARAMS,
    MESH_SHAPE,
    ROWS,
    gather_rows,
    put,
)
from models.demos.blackhole.glm53_flash.ttnn.moe import TtMoe  # noqa: E402


def pcc(a, b) -> float:
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("layer", [3, 44])
def test_moe_device(mesh_device, layer, pcc_thresholds):
    """Tokens per row 1 / 4 / 16 / 32 / 64 / 128 in one case per layer: the fixture opens a new mesh per case, so the
    layer's weights are uploaded once for all of them (1 and 16 are the decode steps of 8 and 128 users, below 32 the
    row collectives pad)."""
    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    import time

    t0 = time.time()
    weights = MoeWeights(checkpoint, layer)
    # GLM53_TEST_EXPERT_DTYPE (bfp8 / bfp4): the routed experts in that format (the model's GLM53_EXPERT_DTYPE), for a
    # comparison in the same test (the progressive form, GLM53_MOE_PROG_F, takes its own)
    dt = os.environ.get("GLM53_TEST_EXPERT_DTYPE")
    kw = {"expert_dtype": {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[dt]} if dt else {}
    module, reference = TtMoe(mesh_device, weights, config, **kw), Moe(weights, config)
    print(f"layer {layer}: MoE weights uploaded in {time.time() - t0:.1f}s")
    seqs = [int(v) for v in os.environ.get("GLM53_MOE_SEQS", "1,4,16,32,64,128").split(",")]
    ab = os.environ.get("GLM53_MOE_COL_AB") == "1"
    for seq in seqs:
        if not ab:
            check_moe(mesh_device, module, reference, config, layer, seq, pcc_thresholds)
            continue
        outs = {}
        saved = moe_module.MOE_COL_CUSTOM
        try:
            for on in (False, True):
                moe_module.MOE_COL_CUSTOM = on
                outs[on] = check_moe(mesh_device, module, reference, config, layer, seq, pcc_thresholds)
        finally:
            moe_module.MOE_COL_CUSTOM = saved
        compare_col_ops(reference, config, layer, seq, outs)


def moe_float64(reference, config, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """The MoE of x [N, D] (bf16) for the given routing in float64 throughout, from the same bf16 weights."""
    x64 = x.double()
    out = torch.zeros_like(x64)
    limit = config.swiglu_limit

    def mlp(w, rows):
        h = clamped_swiglu(rows @ w.gate.double().T, rows @ w.up.double().T, limit)
        return h @ w.down.double().T

    for e in torch.unique(indices).tolist():
        slot, token = torch.where(indices.T == e)
        out.index_add_(0, token, mlp(reference.w.expert(e), x64[token]) * weights[token, slot, None].double())
    return out + mlp(reference.w.shared, x64)


def compare_col_ops(reference, config, layer, seq, outs):
    """The column ops off and on (`check_moe`'s outputs) against tt/ and float64, over the tokens whose expert sets
    both runs share with the reference."""
    (off, same_off, expected, x, routing), (on, same_on, _, _, _) = outs[False], outs[True]
    same = (same_off & same_on).flatten()
    exact = moe_float64(reference, config, x.reshape(-1, x.shape[-1]), routing.indices, routing.weights)
    ref = expected.reshape(-1, expected.shape[-1]).double()
    rows = {"off": off.reshape(-1, off.shape[-1]).double(), "on": on.reshape(-1, on.shape[-1]).double()}
    for name, target in (("tt/ (bf16 sums)", ref), ("float64", exact)):
        for key, got in rows.items():
            d = (got - target)[same]
            print(
                f"layer {layer} seq {seq} col ops {key} vs {name}: pcc {pcc(got[same], target[same]):.7f}, max |d| "
                f"{d.abs().max().item():.3e}, rms {d.pow(2).mean().sqrt().item():.3e}, mean {d.mean().item():+.2e}",
                flush=True,
            )
    d = (ref - exact)[same]
    print(
        f"layer {layer} seq {seq} tt/ vs float64: max |d| {d.abs().max().item():.3e}, rms "
        f"{d.pow(2).mean().sqrt().item():.3e}, mean {d.mean().item():+.2e}",
        flush=True,
    )
    d = rows["on"] - rows["off"]
    print(
        f"layer {layer} seq {seq} on vs off (all tokens): max |d| {d.abs().max().item():.3e}, rms "
        f"{d.pow(2).mean().sqrt().item():.3e}, identical {(d == 0).double().mean().item() * 100:.2f} %",
        flush=True,
    )


def check_moe(mesh_device, module, reference, config, layer, seq, pcc_thresholds):
    g = torch.Generator().manual_seed(layer * 100 + seq)
    x = (torch.randn(ROWS, 1, seq, config.hidden_size, generator=g) * 0.3).to(torch.bfloat16)
    dev_x = put(mesh_device, x, dims=(0, None))
    out = gather_rows(mesh_device, module(dev_x)).reshape(ROWS, seq, -1)
    x_all = ttnn.all_gather(dev_x, dim=2, cluster_axis=0)
    dense = gather_rows(mesh_device, module.route(x_all))
    with torch.no_grad():
        expected, routing = reference(x.reshape(ROWS, seq, -1))
    logits = gather_rows(mesh_device, module.router_logits(x_all))[0, 0].reshape(ROWS * seq, -1)
    err = (logits[:, : config.num_experts] - routing.logits).abs().max().item()
    print(f"layer {layer} seq {seq}: router logits max |d| vs tt/'s fp32 {err:.2e}")

    # routing: the same 8 experts per token, except where the reference's 8th and 9th choices nearly tie (the device
    # adds the router's products in another order)
    dense = dense[0, 0].reshape(ROWS * seq, -1)[:, : config.num_experts]
    got_sets = [set((row > 0).nonzero().flatten().tolist()) for row in dense]
    want_sets = [set(r.tolist()) for r in routing.indices]
    choice = routing.logits.sigmoid() + reference.w.router_bias
    ranked = choice.sort(dim=-1, descending=True).values
    gaps = ranked[:, config.num_experts_per_token - 1] - ranked[:, config.num_experts_per_token]
    differ = [i for i, (a, b) in enumerate(zip(got_sets, want_sets)) if a != b]
    print(
        f"layer {layer} seq {seq}: same expert set for {len(want_sets) - len(differ)}/{len(want_sets)} tokens; "
        f"8th-9th gaps where they differ: {[f'{gaps[i]:.2e}' for i in differ]} (median gap {gaps.median():.2e})"
    )
    assert all(gaps[i] < 1e-4 for i in differ), "an expert set differs where the reference had a clear 8th choice"
    same = torch.tensor([i not in differ for i in range(len(want_sets))]).view(ROWS, seq)
    if os.environ.get("GLM53_TEST_MOE_EXACT") == "1":  # against the float64 MoE too (the same bf16 weights and input)
        flat = x.reshape(ROWS * seq, -1)
        exact = moe_float64(reference, config, flat, routing.indices, routing.weights).view(ROWS, seq, -1)
        got = out.double()
        keep = same.flatten()
        dg = (got.reshape(ROWS * seq, -1) - exact.reshape(ROWS * seq, -1))[keep]
        dr = (expected.double().reshape(ROWS * seq, -1) - exact.reshape(ROWS * seq, -1))[keep]
        ex = exact.reshape(ROWS * seq, -1)[keep]
        print(
            f"layer {layer} seq {seq}: against float64: device relative error {(dg.norm() / ex.norm()).item():.3e}, "
            f"pcc {pcc(got.reshape(ROWS * seq, -1)[keep], ex):.6f}; tt/ (bf16) {(dr.norm() / ex.norm()).item():.3e}",
            flush=True,
        )
    for r in range(ROWS):
        score = pcc(out[r][same[r]], expected[r][same[r]])
        print(
            f"layer {layer} seq {seq} row {r}: pcc {score:.6f} over {int(same[r].sum())} tokens with the same experts"
        )
        if os.environ.get("GLM53_TEST_NO_ASSERT") != "1":  # (a study of a lower precision: print only)
            assert score >= pcc_thresholds["test_moe_device"], f"row {r}: pcc {score:.6f}"
    return out, same, expected, x, routing


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_prog_split(mesh_device):
    """The progressive form's device split (moe.py `_prog_split`: base = BFP4(W), correction = BFP4(W - base) by
    the packer) on a random bf16 matrix and on layer 3's expert 0 (gate, down): the base against the host's BFP4
    packing of W, base + correction against W (relative error, against BFP8(W)'s and BFP4(W)'s), the correction's
    size.  Prints only."""
    from models.demos.blackhole.glm53_flash.ttnn.moe import _prog_split

    checkpoint = Glm53Checkpoint()
    weights = MoeWeights(checkpoint, 3)
    g = torch.Generator().manual_seed(0)
    mats = {
        "random": (torch.randn(512, 1024, generator=g) * 0.02).to(torch.bfloat16),
        "layer 3 expert 0 gate": weights.expert(0).gate[:512, :1024].contiguous().to(torch.bfloat16),
        "layer 3 expert 0 down": weights.expert(0).down[:512, :1024].contiguous().to(torch.bfloat16),
    }
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()  # noqa: E731
    for name, w in mats.items():
        d = ttnn.from_torch(
            w[None, None],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        base, corr = _prog_split(d)
        b, c = first(base)[0, 0], first(corr)[0, 0]
        wf = w.float()
        host4 = ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
        host8 = ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
        from models.demos.blackhole.glm53_flash.tools.bfp_study import bfp_round_exact

        want_b = bfp_round_exact(wf, "bfp4")
        want_c = bfp_round_exact(wf - want_b, "bfp4")
        rel = lambda a: ((a - wf).norm() / wf.norm()).item()  # noqa: E731
        print(
            f"prog split, {name}: dtypes {base.dtype} / {corr.dtype}; base == host BFP4 {torch.equal(b, host4)} "
            f"(max |d| {(b - host4).abs().max().item():.3e}), == bfp_round {torch.equal(b, want_b)}; correction == "
            f"bfp_round(w - base) {torch.equal(c, want_c)} (max |d| {(c - want_c).abs().max().item():.3e}); relative "
            f"error: base {rel(b):.3e} (host BFP4 {rel(host4):.3e}), base + correction {rel(b + c):.3e} (host BFP8 "
            f"{rel(host8):.3e}); |corr| / |W| {(c.norm() / wf.norm()).item():.3e}",
            flush=True,
        )
        for t in (d, base, corr):
            ttnn.deallocate(t)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_prog_layout(mesh_device):
    """The progressive form's loaded weights in the experts op's layout (TtMoe, layer 3, GLM53_MOE_PROG_F set): on a few
    chips, a few experts' (and the shared expert's) base and correction slots of e_w1 (gate | up column tiles) and e_w2
    (down cores' slices) against the CPU's split of the checkpoint (tools/bfp_study.bfp_round_exact of the dense copy's
    16-value groups along the output features).  Prints only."""
    from models.demos.blackhole.glm53_flash.tools.bfp_study import bfp_round_exact

    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = MoeWeights(checkpoint, 3)
    module = TtMoe(mesh_device, weights, config)
    # (without GLM53_MOE_PROG_F: the one set in the experts' format, against the host packer's values)
    fmt = "bfp4" if module.prog or module.e_w1.dtype == ttnn.bfloat4_b else "bfp8"
    sets = 2 if module.prog else 1
    n, s_, q = module.op_slots, 256, 32
    d = config.hidden_size
    t = s_ // 32
    chips = ttnn.get_device_tensors(module.e_w1)
    chips2 = ttnn.get_device_tensors(module.e_w2)
    cols = 4
    for r, c in ((0, 0), (3, 2), (7, 3)):
        idx = r * cols + c
        w1 = ttnn.to_torch(chips[idx]).float()[0, 0]  # [T 2n D, 64]
        w2 = ttnn.to_torch(chips2[idx]).float()[0, 0]  # [Q 2n S, D / Q]
        for e in (0, 5, 71, n - 1):
            if e < module.op_experts:
                ew = weights.expert(c * module.op_experts + e)
                gate, up, down = ew.gate, ew.up, ew.down
            else:  # (the shared expert's slot)
                gate, up, down = weights.shared.gate, weights.shared.up, weights.shared.down
            g = gate[r * s_ : (r + 1) * s_].T.float()  # [D, S]
            u = up[r * s_ : (r + 1) * s_].T.float()
            dn = down[:, r * s_ : (r + 1) * s_].T.float()  # [S, D]
            bad = []
            for name, wt in (("gate", g), ("up", u)):
                b = bfp_round_exact(wt, fmt)
                cr = bfp_round_exact(wt - b, "bfp4")
                for ci in (0, t - 1):
                    half = slice(0, 32) if name == "gate" else slice(32, 64)
                    for which, ref, slot in (("base", b, e), ("corr", cr, n + e))[:sets]:
                        got = w1[(slot * t + ci) * d : (slot * t + ci + 1) * d, half]
                        if not torch.equal(got, ref[:, 32 * ci : 32 * ci + 32]):
                            bad.append(
                                f"{name} {which} col {ci}: max |d| {(got - ref[:, 32 * ci : 32 * ci + 32]).abs().max().item():.3e}"
                            )
            b = bfp_round_exact(dn, fmt if module.down_dtype != ttnn.bfloat4_b else "bfp4")
            cr = bfp_round_exact(dn - b, "bfp4")
            for core in (0, q - 1):
                cw = d // q
                for which, ref, slot in (("base", b, e), ("corr", cr, n + e))[:sets]:
                    got = w2[(core * sets * n + slot) * s_ : (core * sets * n + slot + 1) * s_]
                    if not torch.equal(got, ref[:, core * cw : (core + 1) * cw]):
                        bad.append(
                            f"down {which} core {core}: max |d| {(got - ref[:, core * cw : (core + 1) * cw]).abs().max().item():.3e}"
                        )
            print(f"prog layout chip ({r}, {c}) slot {e}: {'all equal' if not bad else '; '.join(bad[:6])}", flush=True)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_prog_layout_steps(mesh_device):
    """Where the loaded progressive weights part from the CPU's split (test_prog_layout found base and correction off
    by a few BFP4 steps): chip (0, 0)'s dense gate copy of layer 3 ([D, 72 x 256], as TtMoe uploads it) split on the
    device from the host's tiling and from the upload path's (row-major, tiled on the device), each against the CPU's
    split of the same copy (bfp_round_exact) and the host packer's BFP4 of it: how many values differ, where (rows,
    columns mod 32 / 16), the largest; then one expert's base pair after the experts op's slice and concat.  Prints."""
    from models.demos.blackhole.glm53_flash.tools.bfp_study import bfp_round_exact
    from models.demos.blackhole.glm53_flash.ttnn import prog_split
    from models.demos.blackhole.glm53_flash.ttnn.common import upload_shards

    checkpoint = Glm53Checkpoint()
    weights = MoeWeights(checkpoint, 3)
    ws = [weights.expert(e) for e in range(72)]
    dense = torch.cat([w.gate[0:256].T for w in ws], dim=1).contiguous()  # chip (0, 0): rows 0 .. 255 of each
    wf = dense.float()
    want_b = bfp_round_exact(wf, "bfp4")
    want_c = bfp_round_exact(wf - want_b, "bfp4")
    host4 = ttnn.to_torch(ttnn.from_torch(dense, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
    print(
        f"cpu split vs host BFP4 packer: base equal {torch.equal(want_b, host4)}, differing {(want_b != host4).sum().item()}",
        flush=True,
    )
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()[0, 0]  # noqa: E731

    def report(name, b, c):
        db, dc = b != want_b, c != want_c
        rows, cols = torch.nonzero(db, as_tuple=True)
        print(
            f"{name}: base differing {int(db.sum())} of {db.numel()} (max |d| {(b - want_b).abs().max().item():.3e}), "
            f"correction differing {int(dc.sum())} (max |d| {(c - want_c).abs().max().item():.3e}); first base "
            f"differences at {[(int(r), int(q)) for r, q in list(zip(rows, cols))[:6]]}, columns mod 32 "
            f"{sorted(set((cols % 32).tolist()))[:16]}, rows mod 32 {sorted(set((rows % 32).tolist()))[:16]}",
            flush=True,
        )
        if int(db.sum()):
            r, q = int(rows[0]), int(cols[0])
            g0 = (q // 16) * 16
            print(f"  row {r} group {g0}: w {wf[r, g0:g0 + 16].tolist()}", flush=True)
            print(f"  device base {b[r, g0:g0 + 16].tolist()}", flush=True)
            print(f"  cpu base    {want_b[r, g0:g0 + 16].tolist()}", flush=True)

    d_host = ttnn.from_torch(
        dense[None, None],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    b, c = prog_split.split(mesh_device, d_host)
    report("split of the host-tiled copy", first(b), first(c))
    d_up = upload_shards(mesh_device, lambda r, q: dense[None, None], dtype=ttnn.bfloat16)
    b2, c2 = prog_split.split(mesh_device, d_up)
    report("split of the upload path's copy", first(b2), first(c2))
    gb = ttnn.slice(b2, [0, 0, 0, 5 * 256], [1, 1, 4096, 5 * 256 + 32], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    got = first(gb)
    print(
        f"slice of expert 5's column tile 0: equal to the split's {torch.equal(got, first(b2)[:, 1280:1312])}, to the "
        f"cpu's {torch.equal(got, want_b[:, 1280:1312])}",
        flush=True,
    )


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_prog_layout_build(mesh_device):
    """The experts op's layout steps after the split (test_prog_layout_steps: the split and a slice exact): chip
    (0, 0)'s dense gate and up copies of layer 3 split, then as TtMoe builds e_w1: (gate, up) column tile pairs by
    concat, 9 pairs stacked by concat along the rows, the stack placed ND-sharded; each against the CPU's split of the
    same values.  Prints."""
    from models.demos.blackhole.glm53_flash.tools.bfp_study import bfp_round_exact
    from models.demos.blackhole.glm53_flash.ttnn import prog_split
    from models.demos.blackhole.glm53_flash.ttnn.common import upload_shards

    checkpoint = Glm53Checkpoint()
    weights = MoeWeights(checkpoint, 3)
    ws = [weights.expert(e) for e in range(72)]
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()[0, 0]  # noqa: E731
    mc = ttnn.DRAM_MEMORY_CONFIG
    dev, want = {}, {}
    for name in ("gate", "up"):
        dense = torch.cat([getattr(w, name)[0:256].T for w in ws], dim=1).contiguous()
        wb = bfp_round_exact(dense.float(), "bfp4")
        want[name] = wb
        d_up = upload_shards(mesh_device, lambda r, q, dense=dense: dense[None, None], dtype=ttnn.bfloat16)
        dev[name], _ = prog_split.split(mesh_device, d_up)
    pairs, refs = [], []
    for e, c in [(0, 0), (0, 7), (5, 0), (5, 3), (17, 2), (40, 6), (63, 1), (70, 5), (71, 7)]:
        s0 = e * 256 + c * 32
        g = ttnn.slice(dev["gate"], [0, 0, 0, s0], [1, 1, 4096, s0 + 32], memory_config=mc)
        u = ttnn.slice(dev["up"], [0, 0, 0, s0], [1, 1, 4096, s0 + 32], memory_config=mc)
        p = ttnn.concat([g, u], dim=-1, memory_config=mc)
        ref = torch.cat([want["gate"][:, s0 : s0 + 32], want["up"][:, s0 : s0 + 32]], dim=1)
        got = first(p)
        print(
            f"pair ({e}, {c}): dtype {p.dtype}, equal {torch.equal(got, ref)}, differing {int((got != ref).sum())}, "
            f"max |d| {(got - ref).abs().max().item():.3e}",
            flush=True,
        )
        pairs.append(p)
        refs.append(ref)
    st = ttnn.concat(pairs, dim=-2, memory_config=mc)
    ref = torch.cat(refs, dim=0)
    got = first(st)
    print(
        f"stack of 9: dtype {st.dtype}, equal {torch.equal(got, ref)}, differing {int((got != ref).sum())}, max |d| "
        f"{(got - ref).abs().max().item():.3e}",
        flush=True,
    )
    dram = mesh_device.dram_grid_size()
    nd = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=ttnn.Shape([16 * 32, 64]),
            grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))]),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    pl = ttnn.to_memory_config(st, nd)
    got = first(pl)
    print(
        f"placed: equal {torch.equal(got, ref)}, differing {int((got != ref).sum())}, max |d| "
        f"{(got - ref).abs().max().item():.3e}",
        flush=True,
    )


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_prog_layout_keep(mesh_device):
    """TtMoe's own split copies (GLM53_TEST_PROG_KEEP=1 keeps them: the six splits in the load's order, gate, up,
    down, then the shared expert's three slices) on chips (0, 0) and (7, 3), each against the CPU's split of the same
    values; then the loaded e_w1's first pairs against those kept copies (the layout steps alone).  Prints only."""
    from models.demos.blackhole.glm53_flash.tools.bfp_study import bfp_round_exact

    os.environ["GLM53_TEST_PROG_KEEP"] = "1"
    checkpoint = Glm53Checkpoint()
    config = Glm53Config.from_dict(checkpoint.config)
    weights = MoeWeights(checkpoint, 3)
    module = TtMoe(mesh_device, weights, config)
    assert module.prog, "set GLM53_MOE_PROG_F"
    s_ = 256
    ne = module.op_experts
    (gb, ub, db, sb), (gc, uc, dc, sc) = module._prog_dense
    cols = 4

    def chip(t, idx):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[idx]).float()[0, 0]

    for r, c in ((0, 0), (7, 3)):
        idx = r * cols + c
        ws = [weights.expert(e) for e in range(c * ne, (c + 1) * ne)]
        dense = {
            "gate": torch.cat([w.gate[r * s_ : (r + 1) * s_].T for w in ws], dim=1),
            "up": torch.cat([w.up[r * s_ : (r + 1) * s_].T for w in ws], dim=1),
            "down": torch.cat([w.down[:, r * s_ : (r + 1) * s_].T for w in ws], dim=0),
            "shared gate": weights.shared.gate[r * s_ : (r + 1) * s_].T,
            "shared up": weights.shared.up[r * s_ : (r + 1) * s_].T,
            "shared down": weights.shared.down[:, r * s_ : (r + 1) * s_].T,
        }
        got = dict(zip(dense, ((gb, gc), (ub, uc), (db, dc), (sb[0], sc[0]), (sb[1], sc[1]), (sb[2], sc[2]))))
        for name, w in dense.items():
            wf = w.float()
            want_b = bfp_round_exact(wf, "bfp4")
            want_c = bfp_round_exact(wf - want_b, "bfp4")
            b, cr = chip(got[name][0], idx), chip(got[name][1], idx)
            nb, nc = int((b != want_b).sum()), int((cr != want_c).sum())
            where = ""
            if nb:
                rows, qs = torch.nonzero(b != want_b, as_tuple=True)
                where = (
                    f"; first at {[(int(x), int(y)) for x, y in list(zip(rows, qs))[:4]]}, rows mod 32 "
                    f"{sorted(set((rows % 32).tolist()))[:12]}, columns mod 32 {sorted(set((qs % 32).tolist()))[:12]}"
                )
            print(
                f"chip ({r}, {c}) {name} {tuple(w.shape)}: base differing {nb} of {b.numel()} (max |d| "
                f"{(b - want_b).abs().max().item():.3e}), correction differing {nc} (max |d| "
                f"{(cr - want_c).abs().max().item():.3e}){where}",
                flush=True,
            )
        # e_w1's pairs against the kept copies: slot e, column tile ci at rows (e T + ci) D
        w1 = chip(module.e_w1, idx)
        d, t = config.hidden_size, s_ // 32
        bg, bu = chip(gb, idx), chip(ub, idx)
        for e, ci in ((0, 0), (5, 7), (71, 3)):
            pair = w1[(e * t + ci) * d : (e * t + ci + 1) * d]
            ref = torch.cat(
                [bg[:, e * s_ + 32 * ci : e * s_ + 32 * ci + 32], bu[:, e * s_ + 32 * ci : e * s_ + 32 * ci + 32]], 1
            )
            print(
                f"chip ({r}, {c}) e_w1 slot {e} tile {ci} against the kept split: differing {int((pair != ref).sum())}, "
                f"max |d| {(pair - ref).abs().max().item():.3e}",
                flush=True,
            )


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_concat_block_float(mesh_device):
    """ttnn.concat of tiled block-float tensors along the width and the rows (measured: (gate, up) pairs of BFP4 column
    tiles by concat(dim=-1) differed from the CPU in ~10 % of the values, the stacks by concat(dim=-2) did not add any):
    host-packed bf16 / BFP8 / BFP4 [4096, 32] pieces of layer 3's expert 0 gate and up, concatenated on the device,
    against the host packing of the same pieces put side by side.  Prints only."""
    checkpoint = Glm53Checkpoint()
    weights = MoeWeights(checkpoint, 3)
    g = weights.expert(0).gate[0:32].T.contiguous()  # [4096, 32]
    u = weights.expert(0).up[0:32].T.contiguous()
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()[0, 0]  # noqa: E731
    mc = ttnn.DRAM_MEMORY_CONFIG
    for dtype in (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b):
        put = lambda w: ttnn.from_torch(  # noqa: E731
            w[None, None],
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=mc,
        )
        dg, du = put(g), put(u)
        hg, hu = first(dg), first(du)
        for dim, ref in ((-1, torch.cat([hg, hu], 1)), (-2, torch.cat([hg, hu], 0))):
            got = first(ttnn.concat([dg, du], dim=dim, memory_config=mc))
            bad = got != ref
            where = ""
            if int(bad.sum()):
                rows, qs = torch.nonzero(bad, as_tuple=True)
                where = (
                    f"; in the first piece {int((qs < 32).sum() if dim == -1 else (rows < 4096).sum())}, columns "
                    f"{sorted(set(qs.tolist()))[:20]}, rows mod 32 {sorted(set((rows % 32).tolist()))[:20]}; "
                    f"e.g. ({int(rows[0])}, {int(qs[0])}): got {got[rows[0], qs[0]].item():.6g}, want "
                    f"{ref[rows[0], qs[0]].item():.6g}"
                )
            print(
                f"concat {dtype} dim {dim}: differing {int(bad.sum())} of {bad.numel()} (max |d| "
                f"{(got - ref).abs().max().item():.3e}){where}",
                flush=True,
            )
