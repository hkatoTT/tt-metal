"""The device sampling after the LM head (ttnn/sampling.py) against its CPU distribution: random
logits in the greedy head's layout (chip 4 r + c the ids k w ..), rows with different settings (greedy, temperature
1 / top-p 0.95, temperature 0.7 / top-k 10, temperature 1.5 / top-p 0.9), `CandidateSampler.sample` drawn many times
with new seeds; each row's empirical distribution against `reference_distribution` (total variation, beside the
same-size draws on the CPU as the noise floor), greedy rows the exact argmax, every chip the same ids, the time a call.

    GLM53_SAMPLING_ROWS (32), GLM53_SAMPLING_DRAWS (400), GLM53_SAMPLING_ROWOWN=1 (DFlash2's `StepSampling.draw`:
    each mesh row draws its own n / 8 rows, the ids picked by the exact split matmul; its greedy flags 0)
Prints only (a study's test: no failure on a difference).  Run (a device test): GLM53_DEVICE_TEST=1 pytest ... -s
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

from models.demos.blackhole.glm53_flash.ttnn.common import COLS, DEVICE_PARAMS, MESH_SHAPE, ROWS, put  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.sampling import (  # noqa: E402
    CandidateSampler,
    SamplingInputs,
    StepSampling,
    reference_distribution,
)

VOCAB = 154_880
SETTINGS = [  # (k, p, temperature): 0 k = no top-k
    (1, 1.0, 0.0),
    (0, 0.95, 1.0),
    (10, 1.0, 0.7),
    (0, 0.9, 1.5),
]


def _write(mesh, values: torch.Tensor, device_tensor, dtype) -> None:
    host = ttnn.from_torch(
        values, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )
    ttnn.copy_host_to_device_tensor(host, device_tensor)


def _tv(a: torch.Tensor, b: torch.Tensor) -> float:
    return 0.5 * float((a - b).abs().sum())


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_sampling_device(mesh_device):
    mesh = mesh_device
    n = int(os.environ.get("GLM53_SAMPLING_ROWS", "32"))
    draws = int(os.environ.get("GLM53_SAMPLING_DRAWS", "400"))
    chips = ROWS * COLS
    w = -(-VOCAB // (chips * 32)) * 32
    g = torch.Generator().manual_seed(0)
    logits = (torch.randn(n, VOCAB, generator=g) * 2.5).to(torch.bfloat16)  # (a few dozen likely tokens a row)
    full = torch.full((n, chips * w), -1e30, dtype=torch.bfloat16)
    full[:, :VOCAB] = logits
    per_chip = full.view(n, ROWS, COLS * w).permute(1, 0, 2).unsqueeze(1).contiguous()  # [8, 1, n, 4 w]
    part = put(mesh, per_chip, dims=(0, 3), dtype=ttnn.bfloat16)  # chip (r, c): [1, 1, n, w], ids (4 r + c) w ..
    sampler = CandidateSampler(mesh, w, [max(0, min(w, VOCAB - k * w)) for k in range(chips)])
    sampler.prepare(n)
    inputs = SamplingInputs.alloc(mesh, n)
    rows = [SETTINGS[r % len(SETTINGS)] for r in range(n)]
    host = SamplingInputs.host_values([s[0] for s in rows], [s[1] for s in rows], [s[2] for s in rows], range(n))
    _write(mesh, host["k"], inputs.k, ttnn.uint32)
    _write(mesh, host["p"], inputs.p, ttnn.bfloat16)
    _write(mesh, host["inv_temp"], inputs.inv_temp, ttnn.bfloat16)

    rowown = os.environ.get("GLM53_SAMPLING_ROWOWN") == "1"
    if rowown:  # (each mesh row draws its b = n / 8 rows of the 8 b, the parameters a row's)
        b = n // ROWS

        class _Head:
            mesh_ids_width, mesh_valid = w, [max(0, min(w, VOCAB - k * w)) for k in range(chips)]

        step = StepSampling(mesh, _Head, b)
        vals = step.host_rows(
            torch.tensor([s[0] for s in rows]).view(ROWS, b), torch.tensor([s[1] for s in rows]).view(ROWS, b),
            torch.tensor([s[2] for s in rows]).view(ROWS, b), torch.arange(n).view(ROWS, b), 0,
        )  # fmt: skip
        vals["greedy"][:] = 0.0  # (every row drawn: the greedy rows by k 1)
        step.write(vals)
        zero_greedy = put(mesh, torch.zeros(1, 1, n, 1), dims=(None, None), dtype=ttnn.float32)
        pos = put(
            mesh,
            torch.zeros(ROWS, b, dtype=torch.int32),
            dims=(0, None),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    counts = torch.zeros(n, VOCAB)
    same_chips = True
    t_calls = []
    for d in range(draws):
        _write(
            mesh,
            torch.tensor([d * 7919 + r * 104729 + 1 for r in range(n)], dtype=torch.int32),
            inputs.seeds,
            ttnn.uint32,
        )
        t0 = time.perf_counter()
        if rowown:
            from models.demos.blackhole.glm53_flash.ttnn.common import write_rows

            write_rows(mesh, torch.full((ROWS, b), d * 7919, dtype=torch.int32), pos, ttnn.int32)
            seeds = step.seeds(pos)
            out = step.draw(part, zero_greedy, seeds)  # [1, 1, n, 1] fp32
            ttnn.deallocate(seeds)
        else:
            out = sampler.sample(part, inputs)
        ttnn.synchronize_device(mesh)
        t_calls.append(time.perf_counter() - t0)
        chips_out = [ttnn.to_torch(t).flatten().round().to(torch.int64) for t in ttnn.get_device_tensors(out)]
        ttnn.deallocate(out)
        if d < 3 or d == draws - 1:
            same_chips &= all(torch.equal(chips_out[0], c) for c in chips_out[1:])
        ids = chips_out[0][:n]
        if int(ids.max()) >= VOCAB:
            print(f"draw {d}: an id past the vocabulary {int(ids.max())}", flush=True)
            ids = ids.clamp(max=VOCAB - 1)
        counts[torch.arange(n), ids] += 1
        if d == 0:
            print(
                f"first draw ids {ids[:8].tolist()} (CPU argmax {logits[:8].float().argmax(-1).tolist()})", flush=True
            )

    emp = counts / draws
    by_setting = {}
    for r in range(n):
        k, p, t = rows[r]
        ref = reference_distribution(logits[r], k, p, t)
        tv = _tv(emp[r], ref)
        cpu = torch.zeros(VOCAB)
        cpu.index_add_(0, torch.multinomial(ref, draws, replacement=True, generator=g), torch.ones(draws))
        floor = _tv(cpu / draws, ref)
        outside = float(emp[r][ref == 0].sum())  # (draws the CPU distribution never makes)
        by_setting.setdefault((k, p, t), []).append((tv, floor, outside))
    for key, vals in by_setting.items():
        tv = sum(v[0] for v in vals) / len(vals)
        floor = sum(v[1] for v in vals) / len(vals)
        outside = max(v[2] for v in vals)
        print(
            f"k {key[0]}, p {key[1]}, temperature {key[2]}: total variation {tv:.4f} (CPU draws of the same size "
            f"{floor:.4f}), share outside the CPU's support {outside:.4f}, {len(vals)} rows",
            flush=True,
        )
    greedy_rows = [r for r in range(n) if rows[r][2] == 0.0]
    exact = sum(
        int(counts[r].argmax()) == int(logits[r].float().argmax()) and counts[r].max() == draws for r in greedy_rows
    )
    print(
        f"greedy rows: {exact} / {len(greedy_rows)} always the argmax; every chip the same ids: {same_chips}",
        flush=True,
    )
    t_calls = sorted(t_calls[2:]) or t_calls
    print(f"a sample() call (eager, {n} tokens): median {1e3 * t_calls[len(t_calls) // 2]:.2f} ms", flush=True)
