"""verify_accept.accept_device against accept_host on planted acceptances (no model): random tokens and greedy ids,
each user's drafts made equal to the greedy ids before them up to a random n, at 1 / 4 / 16 users a row x K + 1 =
2 / 4 / 6 / 8 tokens.

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_verify_accept.py -s
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

from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS, put  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.verify_accept import (  # noqa: E402
    accept_device,
    accept_host,
    accepted_to_host,
)


class _Io:
    pass


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_verify_accept(mesh_device):
    bad = []
    for u in (1, 4, 16):
        for n_tok in (2, 4, 6, 8):
            k = n_tok - 1
            g = torch.Generator().manual_seed(10 * u + n_tok)
            greedy = torch.randint(0, 154880, (ROWS, n_tok, u), generator=g)
            tokens = torch.randint(0, 154880, (ROWS, n_tok, u), generator=g)
            want_n = torch.randint(0, k + 1, (ROWS, u), generator=g)
            for j in range(1, n_tok):  # drafts 1 .. n equal the greedy id before them
                on = j <= want_n
                tokens[:, j] = torch.where(on, greedy[:, j - 1], tokens[:, j])
            tokens, greedy = tokens.reshape(ROWS, -1), greedy.reshape(ROWS, -1)
            n_h, bonus_h = accept_host(tokens, greedy, k)
            io = _Io()
            io.drafts, io.users = k, u
            io.ids = put(
                mesh_device, tokens.to(torch.int32), dims=(0, None), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            others = torch.full((ROWS, 1, ROWS, 1), 1.0e9)
            others[torch.arange(ROWS), 0, torch.arange(ROWS), 0] = 0.0
            io.others = put(mesh_device, others, dims=(0, None), dtype=ttnn.float32)
            ids = put(mesh_device, greedy.float().reshape(1, 1, -1, 1), dims=(None, None), dtype=ttnn.float32)
            out = accept_device(mesh_device, ids, io)
            n_d, bonus_d = accepted_to_host(out, u)
            ttnn.deallocate(out)
            a = time.perf_counter()
            for _ in range(5):
                ttnn.deallocate(accept_device(mesh_device, ids, io))
            ttnn.synchronize_device(mesh_device)
            us = (time.perf_counter() - a) / 5 * 1e6
            same = torch.equal(n_d, n_h) and torch.equal(bonus_d, bonus_h) and torch.equal(n_h, want_n)
            if not same:
                bad.append((u, n_tok))
            print(f"ACCEPT users={u} T={n_tok}: device = host: {same} (eager {us:.0f} us)", flush=True)
            for t in (io.ids, io.others, ids):
                ttnn.deallocate(t)
    assert not bad, bad
