# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The decode's pool scores (`dsa_indexer.pool_scores`) at long caches in the forms of a KV cache study: as
today (each query reads its user's pages, every block of the cache), with a user's T queries reading its pages once
(`group`, SCORES_GROUP), reading only the visible blocks (`visible`), and both.  For U users a mesh row x T tokens
(the verify step's virtual users: query n is user n mod U's) at caches of 32K / 128K tokens (GLM53_KVPS_POOLS, pools a user), the positions at
the cache's end (every block seen) and at its half: the value-free forms bit for bit against today's (a difference:
the test fails), and every form's traced time.

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_kv_pool_scores.py -s
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.tests.test_dsa_indexer_ops import first, traced  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import dsa_indexer  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, put  # noqa: E402

REP = dict(dims=(None, None))
HEADS, DIM, KPOOL = 32, 128, 4


def shapes() -> list[tuple[int, int]]:
    return [tuple(int(v) for v in x.split("x")) for x in os.environ.get("GLM53_KVPS_UT", "1x8,4x8,16x4").split(",")]


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_kv_pool_scores(mesh_device):
    g = torch.Generator().manual_seed(7)
    bad = []
    for n_pools in [int(v) for v in os.environ.get("GLM53_KVPS_POOLS", "8192,32768").split(",")]:
        pp = n_pools // 32  # pages a user
        users_max = max(u for u, _ in shapes())
        keys = torch.randn(users_max * pp + 1, 1, 32, DIM, generator=g).to(torch.bfloat16)
        caches = {"bf16": put(mesh_device, keys, **REP)}
        for users, t in shapes():
            n = users * t
            q = torch.randn(1, 1, n, HEADS * DIM, generator=g).to(torch.bfloat16)
            hw = (torch.randn(1, 1, n, HEADS, generator=g) / 64).to(torch.bfloat16)
            d_q, d_hw = put(mesh_device, q, layout=ttnn.ROW_MAJOR_LAYOUT, **REP), put(mesh_device, hw, **REP)
            for where in ("end", "half"):
                last = n_pools * KPOOL - t if where == "end" else n_pools * KPOOL // 2
                params = torch.zeros(1, 1, n, dsa_indexer.PARAMS, dtype=torch.int64)
                for i in range(n):
                    u, s = i % users, i // users
                    params[0, 0, i, 0] = last - 3 * u + s  # (the users a little apart)
                    params[0, 0, i, 2] = u * pp
                d_p = put(mesh_device, params.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, **REP)
                label = f"{users} x {t} queries, {n_pools} pools ({n_pools * KPOOL // 1024}K), at the {where}"
                base = None
                for form, group, visible in (
                    ("today", False, False),
                    ("group", True, False),
                    ("visible", False, True),
                    ("group+visible", True, True),
                ):
                    dsa_indexer.SCORES_GROUP = group
                    run = lambda v=visible: dsa_indexer.pool_scores(  # noqa: E731
                        mesh_device, d_q, caches["bf16"], d_hw, d_p, n_pools, False, group=(users, t), visible=v
                    )
                    got = first(run()).float().reshape(n, n_pools)
                    same = True
                    if base is None:
                        base = got
                    else:
                        same = torch.equal(torch.nan_to_num(got, neginf=-1e30), torch.nan_to_num(base, neginf=-1e30))
                        if not same:
                            bad.append(f"{label}: {form}")
                    us = traced(mesh_device, run) * 1e6
                    print(f"  RESULT {label}: {form:14s} {us:8.1f} us  bit for bit {same}", flush=True)
            for t_ in (d_q, d_hw):
                ttnn.deallocate(t_)
        for c in caches.values():
            ttnn.deallocate(c)
    dsa_indexer.SCORES_GROUP = False
    assert not bad, bad
