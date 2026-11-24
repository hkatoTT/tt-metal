"""`input_pack.InputPack` (a decode step's small inputs in one host write, `unpack` copying them to
their tensors on the device) against writing each tensor: tensors shaped as a verify step's at 1 and 16 users a row
(ids [1, 2U] uint32, cur_pos [2U] int32, two groups' positions [U] int32, the indexer's params [1, 1, 2U, 16] uint32,
keep's flags [1, U] uint32, rows [1, 1, 1, 32] uint32), random values each step: every tensor bit for bit after the
pack's write and its unpack, eagerly and replayed in a trace (the values written between replays), and the time.

Run (a device test): GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_input_pack.py -s
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
    MESH_SHAPE,
    ROWS,
    put,
    write_rows,
)
from models.demos.blackhole.glm53_flash.ttnn.input_pack import InputPack  # noqa: E402


def rows_of(t):  # each mesh row's values (its first column chip), [ROWS, n]
    chips = ttnn.get_device_tensors(t)
    cols = len(chips) // ROWS
    return torch.stack([ttnn.to_torch(chips[r * cols]).flatten().to(torch.int64) & 0xFFFFFFFF for r in range(ROWS)])


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_input_pack(mesh_device):
    g = torch.Generator().manual_seed(3)
    for users in (1, 16):
        b = 2 * users
        rm = dict(dims=(0, None), layout=ttnn.ROW_MAJOR_LAYOUT)
        shapes = {  # name: (host shape [ROWS, ..], dtype)
            "ids": ((ROWS, b), ttnn.uint32),
            "cur_pos": ((ROWS * b,), ttnn.int32),
            "group0": ((ROWS * users,), ttnn.int32),
            "group1": ((ROWS * users,), ttnn.int32),
            "params": ((ROWS, 1, b, 16), ttnn.uint32),
            "flags": ((ROWS, users), ttnn.uint32),
            "rows": ((ROWS, 1, 1, 32), ttnn.uint32),
        }
        tensors = {
            k: put(mesh_device, torch.zeros(s, dtype=torch.int32), dtype=d, **rm) for k, (s, d) in shapes.items()
        }
        pack = InputPack(mesh_device, list(tensors.values()))

        def fill():
            values = {
                k: torch.randint(-(2**31), 2**31 - 1, s, generator=g, dtype=torch.int64)
                for k, (s, _) in shapes.items()
            }
            for k, v in values.items():
                pack.set(tensors[k], v.reshape(ROWS, -1) if v.dim() == 1 else v)
            pack.write()
            return values

        def check(values, how):
            bad = []
            for k, v in values.items():
                want = v.reshape(ROWS, -1) & 0xFFFFFFFF
                if not torch.equal(rows_of(tensors[k]), want):
                    bad.append(k)
            result = "bit for bit" if not bad else f"differ: {bad}"
            print(f"input pack, {users} users a row, {how}: {result}", flush=True)
            assert not bad

        values = fill()
        pack.unpack()
        check(values, "eager")
        # the same through the separate writes (the values each field would take): equal to the pack's
        for k, v in values.items():
            write_rows(mesh_device, v, tensors[k], shapes[k][1])
        check(values, "written apart")
        # traced: the unpack captured, the values written between replays
        pack.unpack()
        ttnn.synchronize_device(mesh_device)
        with trace_capture(mesh_device) as trace:
            pack.unpack()
        for step in range(3):
            values = fill()
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            check(values, f"replay {step}")
        t0 = time.perf_counter()
        for _ in range(20):
            pack.write()
        ttnn.synchronize_device(mesh_device)
        t1 = time.perf_counter()
        for k, v in values.items():
            write_rows(mesh_device, v, tensors[k], shapes[k][1])
        ttnn.synchronize_device(mesh_device)
        t2 = time.perf_counter()
        print(
            f"input pack, {users} users a row: the pack's write {(t1 - t0) / 20 * 1e3:.3f} ms, the {len(shapes)} "
            f"writes apart {(t2 - t1) * 1e3:.3f} ms ({pack.width * 4} bytes a row)",
            flush=True,
        )
        ttnn.release_trace(mesh_device, trace)
