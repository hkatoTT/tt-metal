"""The sequence-parallel prefill's DSA attention (`TtDsa._prefill_sp`: the flexible
`chunked_scaled_dot_product_attention` over a user's latent pages, the segment's start in a device tensor, K chunk
128) by Q chunk: the generic rule's Q chunk is the largest of 128 / 64 / 32 dividing the segment (128 for the serving
waves' segments of 128 and 512; GLM53_DSA_SP_QCHUNK sets the model's, by default 64 at 512 and 32 at 128), so a
segment of 128 queries is 16 work units (one a head) on the 120-core grid, one of 512 is 64.  Each smaller Q chunk
against 128, bit for bit (a query row's online softmax runs over the same K chunks whatever its Q chunk), and each
one's time (trace, replayed) at every segment start of a 2,048-row cache.

    GLM53_QC_T      segment lengths (default "128,512,1024")
    GLM53_QC_CHUNKS Q chunks to try (default "128,64,32")
    GLM53_QC_REPS   calls a trace (default 10)
    GLM53_QC_DUMP / GLM53_QC_COMPARE  path: save the first Q chunk's outputs / compare them bit for bit with a saved
                    run's (another kernel build of the same op)
Prints "QC t=.. start=.. q=..: .. us, same bits as q=128: True/False" and a summary per T (the slowest start's time:
the serving wave's critical row).  Exit 0 whatever the bits (a probe).
Run (a device test): GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_dsa_sp_qchunk.py -s
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

from models.demos.blackhole.glm53_flash.tests._trace import traced as trace_capture  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    DEVICE_PARAMS_TRACE,
    HIFI4_FP32,
    MESH_SHAPE,
    put,
)

REP = dict(dims=(None, None))
PAGE, RANK, HEADS, DK, MAX_SEQ = 32, 512, 16, 256, 2048


def chips(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def same_bits(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return all(
        x.shape == y.shape and torch.equal(x.contiguous().view(torch.int16), y.contiguous().view(torch.int16))
        for x, y in zip(a, b)
    )


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_sp_qchunk(mesh_device):
    ts = [int(x) for x in os.environ.get("GLM53_QC_T", "128,512,1024").split(",")]
    qcs = [int(x) for x in os.environ.get("GLM53_QC_CHUNKS", "128,64,32").split(",")]
    reps = int(os.environ.get("GLM53_QC_REPS", "10"))
    users, user, pages = 2, 1, MAX_SEQ // PAGE
    g = torch.Generator().manual_seed(0)
    cache = (torch.randn(users * pages + 1, 1, PAGE, RANK, generator=g) * 0.5).to(torch.bfloat16)
    d_cache = put(mesh_device, cache, **REP)
    full = put(
        mesh_device,
        torch.tensor([[user * pages + p for p in range(pages)]], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        **REP,
    )
    start_t = put(
        mesh_device, torch.tensor([0], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, **REP
    )
    grid = mesh_device.compute_with_storage_grid_size()
    sdpa = dict(scale=(DK * 1.0) ** -0.5, compute_kernel_config=HIFI4_FP32)

    def set_start(start: int) -> None:
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([start], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
            start_t,
        )

    summary, dumped = {}, {}
    theirs = torch.load(os.environ["GLM53_QC_COMPARE"]) if os.environ.get("GLM53_QC_COMPARE") else None
    for t in ts:
        q = (torch.randn(1, HEADS, t, RANK, generator=g) * 0.5).to(torch.bfloat16)
        d_q = put(mesh_device, q, **REP)
        starts = list(range(0, MAX_SEQ - t + 1, t))
        ref = {}
        for qc in qcs:
            if t % qc:
                continue
            program = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=128, exp_approx_mode=False
            )

            def run():
                return ttnn.transformer.chunked_scaled_dot_product_attention(
                    d_q, d_cache, d_cache, full, chunk_start_idx_tensor=start_t, program_config=program, **sdpa
                )

            for start in starts:
                set_start(start)
                out = run()  # (compiled before the capture; the values)
                got = chips(out)
                ttnn.deallocate(out)
                if qc == qcs[0]:
                    ref[start] = got
                    same = True
                    dumped[(t, start)] = got[0]  # (chip 0: every chip computes the same, the inputs replicated)
                    if theirs is not None:
                        other = theirs.get((t, start))
                        ok = other is not None and torch.equal(got[0].view(torch.int16), other.view(torch.int16))
                        print(f"QC COMPARE t={t} start={start}: same bits as the saved run: {ok}", flush=True)
                else:
                    same = same_bits(ref[start], got)
                with trace_capture(mesh_device) as trace:
                    outs = [run() for _ in range(reps)]
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                us = (time.perf_counter() - t0) / reps * 1e6
                ttnn.release_trace(mesh_device, trace)
                for o in outs:
                    ttnn.deallocate(o)
                summary.setdefault((t, qc), []).append((start, us, same))
                print(f"QC t={t} start={start} q={qc}: {us:.1f} us, same bits as q={qcs[0]}: {same}", flush=True)
        ttnn.deallocate(d_q)
    if os.environ.get("GLM53_QC_DUMP"):
        torch.save(dumped, os.environ["GLM53_QC_DUMP"])
    for (t, qc), rows in sorted(summary.items()):
        worst = max(rows, key=lambda r: r[1])
        mean = sum(r[1] for r in rows) / len(rows)
        print(
            f"QC SUMMARY t={t} q={qc}: slowest start {worst[0]} {worst[1]:.1f} us, mean over starts {mean:.1f} us, "
            f"all same bits {all(r[2] for r in rows)}",
            flush=True,
        )
