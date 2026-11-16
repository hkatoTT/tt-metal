"""The traced prefill's packed inputs (`ttnn/prefill_pack.py`, `common.FixedInputs` with a pack) against the buffers
`place` writes one by one, byte for byte on every chip: the one-row form (a first chunk of 1,024 and a later one, the
commit's picks), the 8-row form of 512 (one request, three requests) and a pair of its waves (`sp_pair_inputs`), on a
long cache (the indexer's tables) and a dense one.  Then the host's time of the two ways for the pair.  KPACK lines; an
exception is printed and fails the test.

    GLM53_PACK_LAYERS   layers (default 5: KDA, dense MLP, DSA + MoE, KDA + MoE)
    GLM53_PACK_CACHES   "8192,2048" (default): the caches
    GLM53_PACK_FIELDS   1: each packed field's pages

    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_prefill_pack.py -s
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

from models.demos.blackhole.glm53_flash.checkpoint import ShapeOnlyCheckpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import prefill_pack  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    DEVICE_PARAMS_TRACE,
    MESH_SHAPE,
    ROWS,
    FixedInputs,
    SpGroup,
)
from models.demos.blackhole.glm53_flash.ttnn.kda import TtKda  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_prefill_pack(mesh_device):
    """Runs `_study` (the module's docstring); an exception is printed, then raised (the test fails)."""
    try:
        _study(mesh_device)
    except Exception as err:  # noqa: BLE001
        import traceback

        print(f"PREFILL_PACK FAILED: {err!r}\n{traceback.format_exc()}"[:4000], flush=True)
        raise


def _chips(t) -> list:
    return [ttnn.to_torch(c) for c in ttnn.get_device_tensors(t)]


def _study(mesh):
    layers = int(os.environ.get("GLM53_PACK_LAYERS", "5"))
    differ = []  # (the buffers whose bytes differ: the test fails at the end)
    for cache in (int(c) for c in os.environ.get("GLM53_PACK_CACHES", "8192,2048").split(",")):
        model = TtGlm53Model(mesh, ShapeOnlyCheckpoint(), num_layers=layers, max_seq=cache)
        states = model.empty_state(2)
        users = 2
        g = torch.Generator().manual_seed(17)
        vocab = model.config.vocab_size

        def ids(t):
            return torch.randint(0, vocab, (ROWS, t), generator=g)

        h = 512
        n_sp = min(ROWS, cache // h)
        one = [SpGroup(tuple(range(n_sp)), (h,) * (n_sp - 1) + (h - 37,), 1, 0, True)]
        three = [
            SpGroup((0, 1, 2), (h, h, 300), 0, 0, True),
            SpGroup((3, 4, 5), (h, h, h), 1, h * 3 if cache >= h * 6 else 0, False),
            SpGroup((6, 7), (h, 77), 0, 0, True),
        ]
        forms = {
            "row first 1024": lambda place: model.prefill_inputs(
                place, ids(1024), 0, states, [r % users for r in range(ROWS)], valid=[1024 - 9 * r for r in range(ROWS)],
                last=True, traced=True,
            ),
            "row later 1024": lambda place: model.prefill_inputs(
                place, ids(1024), 1024, states, [None if r == 3 else r % users for r in range(ROWS)],
                rows=[r != 3 for r in range(ROWS)], last=False, traced=True,
            ),
            "commit": lambda place: TtKda.commit_picks(place, users, [None if r % 3 else 1 for r in range(ROWS)]),
            "sp one 512": lambda place: model.sp_inputs(place, ids(h), states, one, traced=True),
            "sp three 512": lambda place: model.sp_inputs(place, ids(h), states, three, traced=True),
        }  # fmt: skip
        if cache >= 2 * ROWS * h:
            a = [SpGroup(tuple(range(ROWS)), (h,) * ROWS, 1, 0, False)]
            b = [SpGroup(tuple(range(ROWS)), (h,) * (ROWS - 1) + (h - 200,), 1, ROWS * h, True)]
            forms["sp pair"] = lambda place: model.sp_pair_inputs(place, ids(2 * h), states, a, b, traced=True)
        for name, make in forms.items():
            if name.startswith("row later") and cache < 2048:
                continue
            state = g.get_state()
            direct = FixedInputs(mesh)
            make(direct.place)  # (the buffers, made)
            g.set_state(state)
            make(direct.place)  # (written again, as a replay writes them)
            packed = FixedInputs(mesh)
            g.set_state(state)
            packed.begin()
            make(packed.place)  # (made, the layout measured)
            packed.end()
            packed.make_pack()
            g.set_state(state)
            t0 = time.perf_counter()
            packed.begin()
            make(packed.place)
            layout = packed.end()
            host_s = time.perf_counter() - t0
            packed.unpack(layout)
            ttnn.synchronize_device(mesh)
            bad = []
            for key, buf in direct.buffers.items():
                if not all(torch.equal(x, y) for x, y in zip(_chips(buf), _chips(packed.buffers[key]))):
                    bad.append(key)
            pages = sum(n for _, _, _, n, _, _ in layout)
            small = sum(n for _, _, page, n, _, _ in layout if page < 1024)
            times = {}
            for batch in (prefill_pack.BATCH_BYTES, 0):  # (the default, then a page a read)
                prefill_pack.BATCH_BYTES, keep = batch, prefill_pack.BATCH_BYTES
                packed.unpack(layout)  # (its program compiled)
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                for _ in range(20):
                    packed.unpack(layout)
                ttnn.synchronize_device(mesh)
                times[batch] = (time.perf_counter() - t0) / 20 * 1e3
                prefill_pack.BATCH_BYTES = keep
            unpack = ", ".join(f"batch {b} {ms:.3f} ms" for b, ms in times.items())
            print(f"KPACK cache {cache} {name}: {len(direct.buffers)} buffers, {len(layout)} packed "
                  f"({packed.pack.width * 4} bytes a chip, {pages} pages, {small} under 1 KB), the same bytes on every "
                  f"chip: {not bad} {bad}; host {host_s * 1e3:.1f} ms, the unpack (20 in a row, eager, with the "
                  f"dispatch) {unpack}", flush=True)  # fmt: skip
            if os.environ.get("GLM53_PACK_FIELDS") == "1":
                for fname, _, page, n, slot, at in layout:
                    print(f"KPACK   {fname}: {n} pages of {page} bytes (slot {slot}, at {at})", flush=True)
            if name == "sp pair":
                g.set_state(state)
                t0 = time.perf_counter()
                make(direct.place)
                ttnn.synchronize_device(mesh)
                print(f"KPACK cache {cache} {name}: one write a buffer {(time.perf_counter() - t0) * 1e3:.1f} ms",
                      flush=True)  # fmt: skip
            direct.deallocate()
            packed.deallocate()
            differ += [f"cache {cache} {name}: {key}" for key in bad]
    assert not differ, f"not the same bytes: {differ}"
