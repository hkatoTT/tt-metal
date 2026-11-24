"""The KDA prefill's chunk preparation (`ttnn.experimental.kda.prepare_chunk_recurrence`, as `TtKda._terms` calls it: q /
k / v [1, T, 2048] bf16 from the conv, the gate in [0, 1] [1, T, 2048] bf16, the projection [1, T, 6432] with the beta
logits at column 6,144, 16 heads of 128) on random inputs: its device time per call at each T (a trace), and with
GLM53_KDA_ZONES=1 two eager calls with the profiler on (for the zones a measurement adds to its kernels).  KPREP lines;
an error fails the test.

    GLM53_PREP_T   the chunk's rows (default 512,1024)

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_kda_prepare_chunk.py -s
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

from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, put  # noqa: E402

HEADS, DH, PROJ, BETA_AT, LOWER = 16, 128, 6432, 6144, -5.0
REPS = 20


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_kda_prepare_chunk(mesh_device):
    mesh = mesh_device
    zones = os.environ.get("GLM53_KDA_ZONES") == "1"
    config = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )  # fmt: skip
    zero = ttnn.from_torch(
        torch.zeros(1, dtype=torch.int64), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )  # fmt: skip
    g = torch.Generator().manual_seed(7)
    for t in (int(x) for x in os.environ.get("GLM53_PREP_T", "512,1024").split(",")):
        q, k, v = (put(mesh, torch.randn(1, t, HEADS * DH, generator=g), dims=(None, None)) for _ in range(3))
        gate = put(mesh, torch.sigmoid(torch.randn(1, t, HEADS * DH, generator=g)), dims=(None, None))
        proj = put(mesh, torch.randn(1, t, PROJ, generator=g), dims=(None, None))

        def call():
            return ttnn.experimental.kda.prepare_chunk_recurrence(
                q, k, v, gate, proj, HEADS, compute_kernel_config=config, output_bf16_mask=0, actual_start=zero,
                gate_scale=LOWER, beta_logits_column_offset=BETA_AT,
            )  # fmt: skip

        def free(out):
            for x in out:
                ttnn.deallocate(x)

        free(call())  # (compiled)
        ttnn.synchronize_device(mesh)
        if zones:
            for _ in range(2):
                free(call())
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            print(f"KPREP T {t}: two eager calls profiled", flush=True)
        else:
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                for _ in range(REPS):
                    free(call())
            finally:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                ts.append((time.perf_counter() - t0) / REPS)
            ttnn.release_trace(mesh, trace)
            print(f"KPREP T {t}: {sorted(ts)[1] * 1e6:.1f} us a call (the median of 3 replays of {REPS})", flush=True)
        for x in (q, k, v, gate, proj):
            ttnn.deallocate(x)
