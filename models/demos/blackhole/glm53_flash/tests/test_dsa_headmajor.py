# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The dense latent cache's decode attention with the heads first (ttnn's paged MLA decode with heads_first: its reader
puts each token's q tile together from the heads' rows and its writer sends the output rows back; GLM53_DSA_HEADMAJOR)
against the transposed form (q~ [1, 16, B, 512] transposed to [1, B, 16, 512], the op, its output transposed back:
`TtDsa._step_latent` with GLM53_DSA_HEADMAJOR=0), bit for bit, and both timed in a trace: the transposes and the op
against the op alone with the heads first.

Cases "B:T" (GLM53_HM_CASES): B tokens a mesh row, T a user (B / T users; user u's token j is virtual user j B / T + u
on user u's pages at consecutive positions, as the verify step's).  Default 8:8, 32:8, 64:4 (8 / 32 / 128 users at
K = 7 / 7 / 3: timed), 2:1, 16:1, 40:8, 96:3 (other shapes, padding rows and 3 tile rows: bits only).  Positions: in
the first chunk of 128 (the speed test's, test_dflash2_speed.py from position 0: one core of a token's works), near the
cache's end (4 chunks: four cores of a token's and their tree) and random (1-4 chunks, a token's row of either
parity).  The output's rows past B are checked to be zero.  Arms (GLM53_HM_ARMS, "q:o" pairs): the heads whose Q rows
and output rows the reader (NoC 0) moves, the writer (NoC 1) the rest (TT_SDPA_HF_Q_READER / TT_SDPA_HF_OUT_READER;
0:8 the op's default: Q by the writer while the reader reads K), "q:o:p" with the rows issued by plain calls
(TT_SDPA_HF_LEAN=0) in place of the command buffer's state; each arm's bits on the timed cases and its time there
(first chunk and end), the others' (GLM53_HM_ARMS_OTHER) bits on the rest.  The output in DRAM too for the default arm
at GLM53_HM_OUT_DRAM (8, 40).  The cache 512 positions (16 pages a user).

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_dsa_headmajor.py -s
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
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, put  # noqa: E402

HEADS, RANK, PAGE, PAGES = 16, 512, 32, 16  # one chip's heads; the latent; a user's cache of 512 positions
SCALE = 256**-0.5
REP = dict(dims=(None, None))
REPS = int(os.environ.get("GLM53_HM_REPS", "50"))
TIMED = {8, 32, 64}


def cases() -> list[tuple[int, int]]:
    spec = os.environ.get("GLM53_HM_CASES", "8:8,32:8,64:4,2:1,16:1,40:8,96:3")
    return [tuple(int(x) for x in c.split(":")) for c in spec.split(",") if c]


def arms(timed: bool) -> list[tuple[str, dict]]:
    """[(name, the op's environment)]"""
    spec = os.environ.get("GLM53_HM_ARMS", "0:8,8:8,16:8,0:0,0:16,16:16,0:8:p")
    if not timed:
        spec = os.environ.get("GLM53_HM_ARMS_OTHER", "0:8,8:8,16:16,0:8:p")
    out = []
    for a in (a for a in spec.split(",") if a):
        q, o, *plain = a.split(":")
        env = {"TT_SDPA_HF_Q_READER": q, "TT_SDPA_HF_OUT_READER": o, "TT_SDPA_HF_LEAN": "0" if plain else "1"}
        out.append((a, env))
    return out


def traced(mesh, fn, reps: int = REPS) -> float:
    """fn() -> the tensors it made (deallocated in the trace after each rep); seconds a rep of the replay."""
    for t in fn():
        ttnn.deallocate(t)
    ttnn.synchronize_device(mesh)
    with trace_capture(mesh) as trace:
        for _ in range(reps):
            for t in fn():
                ttnn.deallocate(t)
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
    seconds = (time.perf_counter() - t0) / reps
    ttnn.release_trace(mesh, trace)
    return seconds


def chips(t) -> list[torch.Tensor]:
    """The first and the last chip's padded tiles' values, untilized [.., padded rows, cols] bf16."""
    devs = ttnn.get_device_tensors(t)
    return [devs[i].cpu().to_torch_with_padded_shape() for i in (0, len(devs) - 1)]


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_headmajor(mesh_device):
    mesh = mesh_device
    grid = mesh.compute_with_storage_grid_size()
    program = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid, q_chunk_size=32, k_chunk_size=128, exp_approx_mode=False
    )
    i32 = dict(dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    out_dram = {int(x) for x in os.environ.get("GLM53_HM_OUT_DRAM", "8,40").split(",") if x}
    bad = 0
    for b, t in cases():
        users = b // t
        g = torch.Generator().manual_seed(1000 + b)
        cache = put(mesh, torch.randn(users * PAGES + 1, 1, PAGE, RANK, generator=g), **REP)
        owner = torch.arange(b) % users  # virtual user v = j users + u: user u's token j
        table = put(mesh, (owner[:, None] * PAGES + torch.arange(PAGES)).int(), **REP, **i32)
        q_host = torch.randn(1, HEADS, b, RANK, generator=g)
        q_dram = put(mesh, q_host, **REP)  # [1, 16, B, 512] (the model's q~ in DRAM)
        q_l1 = ttnn.to_memory_config(q_dram, ttnn.L1_MEMORY_CONFIG)
        for where in ("start", "end", "random"):
            if where == "start":  # the speed test's: every user's tokens at 16 .. 16 + T - 1 (the first chunk)
                base = torch.full((users,), 16)
            elif where == "end":  # user u's tokens at 480 .. 480 + T - 1 (4 chunks)
                base = torch.full((users,), PAGES * PAGE - 32)
            else:
                base = torch.randint(0, PAGES * PAGE - t, (users,), generator=g)
            timed = b in TIMED and where != "random"
            pos = put(mesh, (base[owner] + torch.arange(b) // users).int(), **REP, **i32)
            attend = dict(
                head_dim_v=RANK,
                page_table_tensor=table,
                is_causal=True,
                cur_pos_tensor=pos,
                scale=SCALE,
                program_config=program,
            )

            def legacy():
                q_t = ttnn.transpose(q_dram, 1, 2)  # [1, B, 16, 512]
                out = ttnn.transformer.paged_flash_multi_latent_attention_decode(q_t, cache, **attend)
                ttnn.deallocate(q_t)
                back = ttnn.transpose(out, 1, 2)  # [1, 16, B, 512]
                ttnn.deallocate(out)
                return [back]

            def heads_first(memory=ttnn.L1_MEMORY_CONFIG):
                return [
                    ttnn.transformer.paged_flash_multi_latent_attention_decode(
                        q_l1, cache, **attend, memory_config=memory, heads_first=True
                    )
                ]

            def check(got) -> str:
                nonlocal bad
                host = chips(got)
                ttnn.deallocate(got)
                ok = all(same_bits(h[:, :, :b], r) for h, r in zip(host, ref_host))
                pad_zero = all(bool((h[:, :, b:] == 0).all()) for h in host)
                bad += (not ok) + (not pad_zero)
                note = "the same bits" if ok else "DIFFERING"
                if host[0].shape[2] > b:
                    note += ", padding zero" if pad_zero else ", PADDING NOT ZERO"
                return note

            ref = legacy()[0]
            ref_host = [r[:, :, :b] for r in chips(ref)]
            ttnn.deallocate(ref)
            if timed:  # the transposed form: the transposes and the op, and the op alone
                q_t = ttnn.transpose(q_dram, 1, 2)

                def op_alone():
                    return [ttnn.transformer.paged_flash_multi_latent_attention_decode(q_t, cache, **attend)]

                chain, alone = traced(mesh, legacy), traced(mesh, op_alone)
                ttnn.deallocate(q_t)
                print(
                    f"HMTIME B={b} ({where}): transposes + op {chain * 1e6:.1f} us, the op alone {alone * 1e6:.1f} us "
                    f"(the transposes {(chain - alone) * 1e6:.1f} us)",
                    flush=True,
                )
            line = []
            for name, env in arms(timed):
                os.environ.update(env)
                try:
                    note = check(heads_first()[0])
                    if timed:
                        sec = traced(mesh, heads_first)
                        note += (
                            f", {sec * 1e6:.1f} us ({(sec - alone) * 1e6:+.1f} on the op alone, "
                            f"{(sec - chain) * 1e6:+.1f} on the transposes + op)"
                        )
                    line.append(f"{name} {note}")
                finally:
                    for k in env:
                        os.environ.pop(k)
            if b in out_dram and where == "random":
                line.append(f"0:8 out in DRAM {check(heads_first(ttnn.DRAM_MEMORY_CONFIG)[0])}")
            print(f"HM B={b} T={t} ({where}) heads first [reader's Q:out heads]: " + "; ".join(line), flush=True)
            ttnn.deallocate(pos)
        for x in (cache, table, q_dram, q_l1):
            ttnn.deallocate(x)
    print(f"HM RESULT: {'all the same bits, padding zero' if not bad else f'{bad} DIFFERING'}", flush=True)
    assert bad == 0
