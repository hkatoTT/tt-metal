"""DSA's attention in a long wave's first chunk (`TtDsa._first_wave`: the serving one-row form's 1,024 tokens a row) by
Q chunk and pairing, for value-same speed: (A) its first half's SDPA (`_first_attend`, q [1, 16, 512, 256] over the
expanded keys / values [1, 16, 512, 256], causal; the model passes no program config: an explicit one of any Q / K
sums otherwise) and (B) its later half's (`_absorbed_attend` at start 512: q~ [1, 16, 512, 512] over the latent pages
by chunked_flash_mla_prefill, K chunk 128; the first arm's Q chunk 128, the largest of 128 / 64 / 32 dividing 512;
GLM53_DSA_FIRST_QCHUNK sets the model's, 64 by default).  A query row's online softmax runs over the same K chunks
whatever its Q chunk, so the values should not change; each arm bit for bit against the first arm ("the model's" in
the prints) and its time (traced).  TT_SDPA_PAIR_FIT 1 and TT_SDPA_KV_SHARED 1 unless set, and the pairing arms
(PAIR_FIT 1 / 2 / 0) with the program cache cleared around them.  Print only (KFW lines).

    GLM53_FW_REPS  calls a trace (default 10)

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_dsa_firstwave_sdpa.py -s
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

from models.demos.blackhole.glm53_flash.tests._trace import is_device_timeout, traced  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    DEVICE_PARAMS_TRACE,
    HIFI4_FP32,
    MESH_SHAPE,
    put,
)

REP = dict(dims=(None, None))
PAGE, RANK, HEADS, DK, MAX_SEQ, HALF = 32, 512, 16, 256, 2048, 512
REPS = int(os.environ.get("GLM53_FW_REPS", "10"))


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_firstwave_sdpa(mesh_device):
    keep = {k: os.environ.get(k) for k in ("TT_SDPA_PAIR_FIT", "TT_SDPA_KV_SHARED")}
    os.environ.setdefault("TT_SDPA_PAIR_FIT", "1")  # (unless set; the arms set PAIR_FIT each)
    os.environ.setdefault("TT_SDPA_KV_SHARED", "1")
    try:
        _probe(mesh_device)
    except Exception as err:  # noqa: BLE001
        if is_device_timeout(err):
            raise
        print(f"KFW FAILED: {err!r}"[:3000], flush=True)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _chips(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def _same(a, b) -> bool:
    return all(
        x.shape == y.shape and torch.equal(x.contiguous().view(torch.int16), y.contiguous().view(torch.int16))
        for x, y in zip(a, b)
    )


def _timed(mesh, fn) -> float:
    out = fn()
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(out)
    with traced(mesh) as trace:
        outs = [fn() for _ in range(REPS)]
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    s = (time.perf_counter() - t0) / REPS
    ttnn.release_trace(mesh, trace)
    for o in outs:
        ttnn.deallocate(o)
    return s


def _probe(mesh):
    grid = mesh.compute_with_storage_grid_size()
    g = torch.Generator().manual_seed(41)
    sdpa = dict(scale=(DK * 1.0) ** -0.5, compute_kernel_config=HIFI4_FP32)

    def prog(qc, kc):
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=False
        )

    def arms(name, calls, ref_name):
        """calls: label -> (pair_fit, fn); the first label is the model's form."""
        ref = None
        for label, (pair_fit, fn) in calls.items():
            os.environ["TT_SDPA_PAIR_FIT"] = pair_fit
            mesh.clear_program_cache()
            try:
                out = fn()
                got = _chips(out)
                ttnn.deallocate(out)
                if ref is None:
                    ref = got
                us = _timed(mesh, fn) * 1e6
                print(f"KFW {name} {label:24s} (PAIR_FIT {pair_fit}): {us:7.1f} us; the same bits as {ref_name}: "
                      f"{_same(ref, got)}", flush=True)  # fmt: skip
            except Exception as err:  # noqa: BLE001
                if is_device_timeout(err):
                    raise
                print(f"KFW {name} {label} FAILED {type(err).__name__}: {str(err).splitlines()[0][:160]}", flush=True)
        os.environ["TT_SDPA_PAIR_FIT"] = "1"
        mesh.clear_program_cache()

    # (A) the first half: q, k, v [1, 16, 512, 256], causal
    q = put(mesh, (torch.randn(1, HEADS, HALF, DK, generator=g) * 0.5).to(torch.bfloat16), **REP)
    k = put(mesh, (torch.randn(1, HEADS, HALF, DK, generator=g) * 0.5).to(torch.bfloat16), **REP)
    v = put(mesh, (torch.randn(1, HEADS, HALF, DK, generator=g) * 0.5).to(torch.bfloat16), **REP)
    a = {
        "model (no config)": (
            "1",
            lambda: ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, **sdpa),
        )
    }
    for pf in ("1", "2", "0"):
        for qc in (32, 64, 128, 256):
            a[f"q {qc} k 32 pf{pf}"] = (
                pf,
                lambda qc=qc: ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, is_causal=True, program_config=prog(qc, 32), **sdpa
                ),
            )
    arms("A first half", a, "the model's")
    for x in (q, k, v):
        ttnn.deallocate(x)

    # (B) the later half: q~ [1, 16, 512, 512] over the latent pages from 512, K chunk 128
    users, user, pages = 2, 1, MAX_SEQ // PAGE
    cache = put(mesh, (torch.randn(users * pages + 1, 1, PAGE, RANK, generator=g) * 0.5).to(torch.bfloat16), **REP)
    table = put(
        mesh,
        torch.tensor([[user * pages + p for p in range(pages)]], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        **REP,
    )
    qt = put(mesh, (torch.randn(1, HEADS, HALF, RANK, generator=g) * 0.5).to(torch.bfloat16), **REP)

    def mla(qc):
        return lambda: ttnn.transformer.chunked_flash_mla_prefill(
            qt, cache, RANK, table, HALF, program_config=prog(qc, 128), **sdpa
        )

    b = {"model (q 128 k 128)": ("1", mla(128))}
    for pf in ("1", "2", "0"):
        for qc in (128, 64, 32):
            b[f"q {qc} k 128 pf{pf}"] = (pf, mla(qc))
    arms("B later half", b, "the model's")
    for x in (qt, cache, table):
        ttnn.deallocate(x)
    print("KFW done", flush=True)
