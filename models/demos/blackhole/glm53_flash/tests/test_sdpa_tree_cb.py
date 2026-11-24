"""The drafter's paged SDPA decode alone (ttnn.transformer.paged_scaled_dot_product_attention_decode with
TtDFlash2's program config: q chunk 32, K chunk 128, the 12 x 10 grid; its reduction tree's writer, with its c_16
wait): at the drafter's shapes (2 U virtual users a chip of 32 query rows and one KV head, U = 1 / 4 / 16: 8 / 32 /
128 users), BFP8 K / V caches of random values, positions per virtual user swept over 0 .. 2,047 (1 .. 16 K chunks:
the reduction tree's cores with 0, 1, 2, 3 or more children, the forms from position 256 / 384 on), with
TT_SDPA_DECODE_TREE_CB off and on; and the drafter's ring (a cache past 4,096 positions: the cache of 4,096 positions,
cache_position_modulo and sliding_window_size 4,096) at positions 4,095 .. 4,223 every third and a few beyond (the
sliding window's mask with its start in each tile of a chunk: dataflow_common.hpp generate_sliding_window_mask).  Each
call's output saved under a tag (chip 0's tensor and every chip's hash) and compared bit for bit with another tag's
(e.g. another build's kernels); against a torch reference (PCC, a sanity check of the test itself); the same call
repeated GLM53_STC_REPEAT times, every repeat's bits the first's.

    GLM53_STC_USERS   U,.. (default 1,4,16)
    GLM53_STC_FORMS   tree CB forms, 0,1 (default both)
    GLM53_STC_RINGS   0 (the plain cache of 2,048) and / or 4096 (the ring), default 0,4096
    GLM53_STC_REPEAT  repeats of each call (default 3)
    GLM53_STC_DIR     (default /tmp/glm53_sdpa_tree), GLM53_STC_TAG (this run's), GLM53_STC_AGAINST

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_sdpa_tree_cb.py -s

Prints "STC .." lines and "STC ALL SAME" / "STC DIFFER"."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE  # noqa: E402

CTX, HD, ROWS_Q, PAGE = 2048, 128, 32, 32
K_CHUNK = 128  # (dflash2.SDPA_K_CHUNK)
SWEEP = [0, 31, 100, 127, 128, 200, 255, 256, 300, 383, 384, 450, 511, 512, 600, 700, 767, 900, 1023, 1024, 1200,
         1400, 1535, 1600, 1800, 1900, 2000, 2047]  # fmt: skip
RING = 4096  # (dflash2.RING: the drafter's K / V ring past it)
SWEEP_RING = list(range(4095, 4224, 3)) + [4223, 4224, 4300, 5000, 6143, 8183, 8191]


def rep(mesh):
    return ttnn.ReplicateTensorToMesh(mesh)


def hashes(t) -> list:
    return [
        hashlib.sha256(ttnn.to_torch(x).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        for x in ttnn.get_device_tensors(t)
    ]


def reference(q, k, v, table, pos, scale, ring=0):
    """torch: per virtual user b, its 32 query rows over its positions 0 .. pos[b] (fp32); in the ring (positions past
    it, the window of `ring` positions): every slot of the ring (the window's positions are all of them)."""
    out = torch.zeros_like(q, dtype=torch.float32)
    for b in range(q.shape[1]):
        n = ring if ring else int(pos[b]) + 1
        pages = table[b, : -(-n // PAGE)]
        kb = k[pages].reshape(-1, HD)[:n].float()
        vb = v[pages].reshape(-1, HD)[:n].float()
        s = q[0, b].float() @ kb.T * scale
        out[0, b] = torch.softmax(s, dim=-1) @ vb
    return out


def pcc(a, b) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_sdpa_tree_cb(mesh_device):
    mesh = mesh_device
    users = [int(x) for x in os.environ.get("GLM53_STC_USERS", "1,4,16").split(",")]
    forms = [int(x) for x in os.environ.get("GLM53_STC_FORMS", "0,1").split(",")]
    rings = [int(x) for x in os.environ.get("GLM53_STC_RINGS", "0,4096").split(",")]
    repeat = int(os.environ.get("GLM53_STC_REPEAT", "3"))
    root = Path(os.environ.get("GLM53_STC_DIR", "/tmp/glm53_sdpa_tree"))
    tag, against = os.environ.get("GLM53_STC_TAG", "now"), os.environ.get("GLM53_STC_AGAINST")
    root.mkdir(parents=True, exist_ok=True)
    scale = HD**-0.5
    program = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),
        q_chunk_size=32,
        k_chunk_size=K_CHUNK,
        exp_approx_mode=False,
    )
    saved, worst, bad_repeat = {}, 1.0, 0
    undo = os.environ.get("TT_SDPA_DECODE_TREE_CB")
    try:
        for form in forms:
            os.environ["TT_SDPA_DECODE_TREE_CB"] = str(form)
            mesh.clear_program_cache()
            for ring, u in [(r, u) for r in rings for u in users]:
                vu = 2 * u
                pages = (ring or CTX) // PAGE
                sweep = SWEEP_RING if ring else SWEEP
                extra = {"sliding_window_size": ring, "cache_position_modulo": ring} if ring else {}
                g = torch.Generator().manual_seed(2024 + u + ring)
                k = torch.randn(vu * pages, 1, PAGE, HD, generator=g).bfloat16()
                v = torch.randn(vu * pages, 1, PAGE, HD, generator=g).bfloat16()
                table = (torch.arange(vu)[:, None] * pages + torch.randperm(pages, generator=g)[None]).to(torch.int32)
                kd = ttnn.from_torch(
                    k, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep(mesh)
                )
                vd = ttnn.from_torch(
                    v, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep(mesh)
                )
                # (the reference on the values the device holds: BFP8 back to the host)
                kq = ttnn.to_torch(ttnn.get_device_tensors(kd)[0]).bfloat16()
                vq = ttnn.to_torch(ttnn.get_device_tensors(vd)[0]).bfloat16()
                td = ttnn.from_torch(table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                     mesh_mapper=rep(mesh))  # fmt: skip
                calls = -(-len(sweep) // vu) + 1
                for j in range(calls):
                    pos = torch.tensor([sweep[(b + j * vu + j) % len(sweep)] for b in range(vu)], dtype=torch.int32)
                    q = torch.randn(1, vu, ROWS_Q, HD, generator=g).bfloat16()
                    qd = ttnn.from_torch(
                        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep(mesh)
                    )
                    cd = ttnn.from_torch(pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                         mesh_mapper=rep(mesh))  # fmt: skip
                    first = None
                    for r in range(repeat):
                        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                            qd,
                            kd,
                            vd,
                            td,
                            is_causal=True,
                            cur_pos_tensor=cd,
                            scale=scale,
                            program_config=program,
                            **extra,
                        )
                        h = hashes(o)
                        if r == 0:
                            first = h
                            key = f"form{form} ring{ring} u{u} call{j}"
                            out0 = ttnn.to_torch(ttnn.get_device_tensors(o)[0])
                            saved[key] = {"hashes": h, "pos": pos.tolist(), "out0": out0}
                            p = pcc(out0, reference(q, kq, vq, table, pos, scale, ring))
                            worst = min(worst, p)
                            chips_same = len(set(h)) == 1
                            print(f"STC {key}: positions {pos.tolist()[:8]}{'..' if vu > 8 else ''} pcc {p:.5f}, "
                                  f"{'all chips the same' if chips_same else 'CHIPS DIFFER'}", flush=True)  # fmt: skip
                        elif h != first:
                            bad_repeat += 1
                            print(f"STC REPEAT DIFFER form{form} ring{ring} u{u} call{j} repeat {r}", flush=True)
                        ttnn.deallocate(o)
                    ttnn.deallocate(qd)
                    ttnn.deallocate(cd)
                for t in (kd, vd, td):
                    ttnn.deallocate(t)
    finally:
        if undo is None:
            os.environ.pop("TT_SDPA_DECODE_TREE_CB", None)
        else:
            os.environ["TT_SDPA_DECODE_TREE_CB"] = undo
        mesh.clear_program_cache()
    torch.save(saved, root / f"{tag}.pt")
    print(f"STC saved {len(saved)} calls under {root / (tag + '.pt')}; worst pcc {worst:.5f}; repeats that differed "
          f"{bad_repeat}", flush=True)  # fmt: skip
    if against:
        ref = torch.load(root / f"{against}.pt")
        diff = [key for key in saved if key not in ref or saved[key]["hashes"] != ref[key]["hashes"]]
        same = len(saved) - len(diff)
        print(f"STC {tag} against {against}: {same} / {len(saved)} calls the same bits on every chip", flush=True)
        for key in diff[:10]:
            print(f"STC DIFFER {key}", flush=True)
        print("STC ALL SAME" if not diff and not bad_repeat else "STC DIFFER", flush=True)
