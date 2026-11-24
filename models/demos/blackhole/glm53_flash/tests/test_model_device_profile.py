"""Per-program device time of the decode step, from the device profiler.

Run with TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
TTNN_OP_PROFILER=1 TT_METAL_PROFILER_DIR=<dir>: every program's device time and op name land in
<dir>/.logs/cpp_device_perf_report.csv (one row per program and chip), written at every read.  A truncated model
(GLM53_PROFILE_LAYERS, default 5: layers 0-2 KDA + dense MLP, 3 DSA + MoE, 4 KDA + MoE) runs:
  - eager: after warm-up steps, one step per batch, the profiler read after every stage so its buffers do not
    overflow (the kernel times are the device's; the gaps between programs are the host's dispatch);
  - traced (GLM53_PROFILE_TRACE=1, the default): one replay of the captured step, read once (the gaps are the device's
    own dispatch; raise TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT above the step's program count).
  - prefill (GLM53_PROFILE_PREFILL, chunk lengths): one eager chunk from GLM53_PROFILE_START (default 0; a multiple of
    the chunk: a later chunk, e.g. 8 x 2,048's second wave) on every mesh row, after two
    warm-up chunks: first timed without reads (the host clock, and how long the host took to issue it), then read
    after every stage as above.
The report's row count after every read goes to <dir>/stages.json; tools/profile_report.py attributes the rows to stages.

    GLM53_PROFILE_USERS = comma list of users per row (default 1,8,32; empty: no decode)
    GLM53_PROFILE_PREFILL = comma list of prefill chunk lengths (default none)
    GLM53_PROFILE_PREFILL_SP = 1: those chunks as one request's wave over the rows
    GLM53_PROFILE_PREFILL_PAIR = 1 (with _SP and _TRACE): also a pair of those waves (1 x 8K's form), traced
    GLM53_PROFILE_PREFILL_ROWS = n: requests on the first n rows only (default 8)
    GLM53_PROFILE_DRAFTS = 2: the verify step with two drafts a user (default 1).
    GLM53_PROFILE_SPEC = comma list of users per row: MTP's verify step (`forward_spec`) traced, one
        replay read once (the MTP layer loaded; tools/profile_report.py lists its programs by op)
Run (a device test): GLM53_DEVICE_TEST=1 pytest ... -s
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, ShapeOnlyCheckpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS, SpGroup  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

WARMUP = 3


def tokens_at(t: int, users: int, vocab: int) -> torch.Tensor:
    return torch.randint(0, vocab, (ROWS, users), generator=torch.Generator().manual_seed(1000 + t))


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_model_device_profile(mesh_device):
    out_dir = Path(os.environ["TT_METAL_PROFILER_DIR"])
    report = out_dir / ".logs" / "cpp_device_perf_report.csv"

    def rows() -> int:
        if not report.exists():
            return 0
        with open(report) as f:
            return sum(1 for _ in f)

    layers = int(os.environ.get("GLM53_PROFILE_LAYERS", "5"))
    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1"  # random weights made on the device (same timing)
    checkpoint = ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint()
    # GLM53_PROFILE_MAX_SEQ: the cache (past 2,051 tokens the DSA layers take the indexer), GLM53_PROFILE_START: the
    # position the steps start at
    max_seq = int(os.environ.get("GLM53_PROFILE_MAX_SEQ", "512"))
    spec_users = [int(x) for x in os.environ.get("GLM53_PROFILE_SPEC", "").split(",") if x]
    model = TtGlm53Model(mesh_device, checkpoint, num_layers=layers, max_seq=max_seq, mtp=bool(spec_users) or None)
    # the prefill's way back as one op (moe.COL_ROW_RS): its buffers for the profiled chunks, as the generator takes them
    # (no generator here: without them the MoE takes the ttnn ops one after the other)
    prefill_chunks = [int(x) for x in os.environ.get("GLM53_PROFILE_PREFILL", "").split(",") if x]
    from models.demos.blackhole.glm53_flash.ttnn import moe as moe_module

    if prefill_chunks and moe_module.COL_ROW_RS:
        from models.demos.blackhole.glm53_flash.ttnn import col_row_rs
        from models.demos.blackhole.glm53_flash.ttnn import model as model_module
        from models.demos.blackhole.glm53_flash.ttnn.generator import SP_MAX

        col_row_rs.prealloc(mesh_device, ts=col_row_rs.ts_for(prefill_chunks, SP_MAX if model_module.SP_PAIR else None))
    start = int(os.environ.get("GLM53_PROFILE_START", "0"))
    vocab = model.config.vocab_size
    kinds = [f"{layer.kind}+{'moe' if model.config.is_moe(i) else 'dense'}" for i, layer in enumerate(model.layers)]
    print(f"profiling layers {kinds}", flush=True)
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)  # the load's programs (tilize, typecast) are read and set aside

    stages = []
    for users in [int(x) for x in os.environ.get("GLM53_PROFILE_USERS", "1,8,32").split(",") if x]:
        pos = lambda t: torch.full((ROWS, users), start + t)  # noqa: E731
        states = model.empty_state(users)
        io = model.alloc_io(users)
        for t in range(WARMUP):
            model.write_inputs(io, tokens_at(t, users, vocab), pos(t))
            ttnn.deallocate(model.forward(io, states))
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
        # GLM53_PROFILE_EAGER=0: no eager step read stage by stage (45 layers: hundreds of reads), the traced one only
        eager = os.environ.get("GLM53_PROFILE_EAGER", "1") == "1"
        stages.append({"users": users, "mode": "eager", "stage": "start", "rows": rows()})

        layer = iter(range(layers))

        def after(name):
            # "<i> <part>" inside layer i; the layer's own mark (its kind) closes its last part, the ffn_hc.write
            ttnn.ReadDeviceProfiler(mesh_device)
            if name[0].isdigit() or name in ("embed", "head"):
                label = name
            else:
                label = f"{next(layer)} ffn_hc.write"
            stages.append({"users": users, "mode": "eager", "stage": label, "rows": rows()})

        if eager:
            model.write_inputs(io, tokens_at(WARMUP, users, vocab), pos(WARMUP))
            ttnn.deallocate(model.forward(io, states, after_stage=after))

        if os.environ.get("GLM53_PROFILE_TRACE", "1") == "1":
            model.write_inputs(io, tokens_at(WARMUP + 1, users, vocab), pos(WARMUP + 1))
            trace, ids = model.capture(io, states, greedy=True)
            ttnn.synchronize_device(mesh_device)
            ttnn.ReadDeviceProfiler(mesh_device)  # the capture's compile run
            stages.append({"users": users, "mode": "trace", "stage": "start", "rows": rows()})
            model.write_inputs(io, tokens_at(WARMUP + 2, users, vocab), pos(WARMUP + 2))
            a = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            seconds = time.perf_counter() - a
            ttnn.ReadDeviceProfiler(mesh_device)
            stages.append({"users": users, "mode": "trace", "stage": "step", "rows": rows(), "seconds": seconds})
            print(f"users {users}: traced step {seconds * 1e3:.1f} ms (profiler on)", flush=True)
            ttnn.release_trace(mesh_device, trace)
        for st in states:
            st.deallocate()
        print(f"users {users}: profiled", flush=True)

    drafts = int(os.environ.get("GLM53_PROFILE_DRAFTS", "1"))  # MTP's drafts a user (2: three tokens a user)
    for users in spec_users:  # MTP's verify step, traced (its first run eager: the capture's compile)
        states = model.empty_state(users, spec=drafts)
        sio = model.alloc_spec_io(users, drafts=drafts)

        def spec_inputs(t, users=users):
            n = drafts + 1
            tokens = torch.cat([tokens_at(t + 100 * j, users, vocab) for j in range(n)], dim=1)
            positions = torch.cat([torch.full((ROWS, users), start + n * t + j) for j in range(n)], 1)
            return tokens, positions, torch.full((ROWS, users), drafts)

        for t in range(WARMUP):
            model.write_spec_inputs(sio, *spec_inputs(t))
            for x in model.forward_spec(sio, states):
                ttnn.deallocate(x)
        model.write_spec_inputs(sio, *spec_inputs(WARMUP))
        trace, *_ = model.capture_spec(sio, states)
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
        stages.append({"users": users, "mode": "spec", "stage": "start", "rows": rows()})
        model.write_spec_inputs(sio, *spec_inputs(WARMUP + 1))
        a = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        seconds = time.perf_counter() - a
        ttnn.ReadDeviceProfiler(mesh_device)
        stages.append({"users": users, "mode": "spec", "stage": "step", "rows": rows(), "seconds": seconds})
        print(f"users {users}: traced verify step {seconds * 1e3:.1f} ms (profiler on)", flush=True)
        ttnn.release_trace(mesh_device, trace)
        for st in states:
            st.deallocate()

    # GLM53_PROFILE_VERIFY: comma list of users:T, the verify step of drafts from elsewhere (`forward_verify`, T
    # tokens a user, DFlash2's) traced (its first runs eager: the capture's compile)
    for cfg in [c for c in os.environ.get("GLM53_PROFILE_VERIFY", "").split(",") if c]:
        users, n = (int(x) for x in cfg.split(":"))
        states = model.empty_state(users, spec=n - 1)
        vio = model.alloc_spec_io(users, drafts=n - 1, mtp=False)

        def verify_inputs(t, users=users, n=n):
            tokens = torch.cat([tokens_at(t + 100 * j, users, vocab) for j in range(n)], dim=1)
            positions = torch.cat([torch.full((ROWS, users), start + n * t + j) for j in range(n)], 1)
            return tokens, positions, torch.full((ROWS, users), n - 1)

        for t in range(WARMUP):
            model.write_spec_inputs(vio, *verify_inputs(t))
            ttnn.deallocate(model.forward_verify(vio, states))
        model.write_spec_inputs(vio, *verify_inputs(WARMUP))
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            out = model.forward_verify(vio, states)
        finally:
            ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
        stages.append({"users": users, "tokens": n, "mode": "verify", "stage": "start", "rows": rows()})
        model.write_spec_inputs(vio, *verify_inputs(WARMUP + 1))
        a = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        seconds = time.perf_counter() - a
        ttnn.ReadDeviceProfiler(mesh_device)
        stages.append(
            {"users": users, "tokens": n, "mode": "verify", "stage": "step", "rows": rows(), "seconds": seconds}
        )
        print(f"users {users} x T {n}: traced verify step {seconds * 1e3:.1f} ms (profiler on)", flush=True)
        ttnn.release_trace(mesh_device, trace)
        ttnn.deallocate(out)
        for st in states:
            st.deallocate()

    # GLM53_PROFILE_PREFILL_SP=1: the chunk as one request's wave over the rows (`prefill_sp`;
    # 8 full segments from position GLM53_PROFILE_START, row 0 first)
    sp = os.environ.get("GLM53_PROFILE_PREFILL_SP") == "1"
    # GLM53_PROFILE_PREFILL_SP_GROUPS=k (with _SP): k requests on 8 / k consecutive rows each (a gathered serving
    # prefill's wave, e.g. 4 x 1K at 512: 2 rows each, within a cache of 2,048), default 1 (one request on all 8)
    sp_groups = int(os.environ.get("GLM53_PROFILE_PREFILL_SP_GROUPS", "1"))

    def sp_wave(chunk):
        size = ROWS // sp_groups
        return [
            SpGroup(tuple(range(g * size, (g + 1) * size)), (chunk,) * size, 0, start, False) for g in range(sp_groups)
        ]

    # GLM53_PROFILE_PREFILL_ROWS=n: the one-row form with requests on the first n rows only (the others padding, as a
    # round of fewer requests than rows: serving's typical prefill), default all 8
    busy = int(os.environ.get("GLM53_PROFILE_PREFILL_ROWS", str(ROWS)))
    for chunk in [int(x) for x in os.environ.get("GLM53_PROFILE_PREFILL", "").split(",") if x]:
        states = model.empty_state(1)
        ids = torch.randint(0, vocab, (ROWS, chunk), generator=torch.Generator().manual_seed(chunk))

        def run_chunk(after=None, ids=ids, chunk=chunk, states=states):
            if sp:
                return model.prefill_sp(ids, states, sp_wave(chunk), logits=False, after_stage=after)
            slots = [0 if r < busy else None for r in range(ROWS)]
            # (from GLM53_PROFILE_START: a later chunk, e.g. 8 x 2,048's second wave of 1,024 at 1,024; the cache's
            # earlier positions hold whatever the runs before left, the times the same)
            return model.prefill(ids, start, states, slots, logits=False, after_stage=after)

        for _ in range(2):  # compile; the mHC matrices are made on first use
            run_chunk()
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
        a = time.perf_counter()
        run_chunk()
        enqueued = time.perf_counter() - a
        ttnn.synchronize_device(mesh_device)
        seconds = time.perf_counter() - a
        ttnn.ReadDeviceProfiler(mesh_device)
        print(f"prefill chunk {chunk}: {seconds * 1e3:.1f} ms, issued in {enqueued * 1e3:.1f} ms", flush=True)
        stage = {"chunk": chunk, "mode": "prefill_sp" if sp else "prefill", "seconds": seconds, "enqueued": enqueued}
        stages.append({**stage, "stage": "start", "rows": rows()})
        layer = iter(range(layers))

        def after_prefill(name):
            ttnn.ReadDeviceProfiler(mesh_device)
            if name[0].isdigit() or name in ("embed", "head"):
                label = name
            else:
                label = f"{next(layer)} ffn_hc.write"
            stages.append({**stage, "stage": label, "rows": rows()})

        if os.environ.get("GLM53_PROFILE_PREFILL_EAGER", "1") == "1":  # (0: only the traced forms below)
            run_chunk(after_prefill)
        # GLM53_PROFILE_PREFILL_TRACE=1 (with GLM53_PROFILE_PREFILL_SP): the same wave through its captured form
        # (`capture_prefills`, `replay_prefill_sp`, the KDA commit apart), one replay read on its own: the programs as
        # serving replays them (no host launch skew between the chips in the collectives' times)
        if sp and os.environ.get("GLM53_PROFILE_PREFILL_TRACE") == "1":
            key = model.prefill_key(start, chunk, False, sp=True)
            captured = model.capture_prefills(states, [(key, start)])[key]
            groups = sp_wave(chunk)
            model.replay_prefill_sp(captured, ids, states, groups, logits=False)
            ttnn.synchronize_device(mesh_device)
            ttnn.ReadDeviceProfiler(mesh_device)
            stages.append({**stage, "mode": "prefill_trace", "stage": "start", "rows": rows()})
            a = time.perf_counter()
            model.replay_prefill_sp(captured, ids, states, groups, logits=False)
            ttnn.synchronize_device(mesh_device)
            seconds = time.perf_counter() - a
            ttnn.ReadDeviceProfiler(mesh_device)
            stages.append({**stage, "mode": "prefill_trace", "stage": "replay", "rows": rows(), "seconds": seconds})
            print(f"prefill chunk {chunk}: replayed in {seconds * 1e3:.1f} ms (profiler on)", flush=True)
            captured.release(mesh_device)
        # GLM53_PROFILE_PREFILL_PAIR=1 (with GLM53_PROFILE_PREFILL_SP and the trace): a pair of the waves (`pair_key`,
        # `replay_prefill_sp_pair`: 1 x 8K's form at chunk 512, the first wave from GLM53_PROFILE_START, the second
        # 8 T later), one replay read on its own
        pair = os.environ.get("GLM53_PROFILE_PREFILL_PAIR") == "1"
        if sp and pair and os.environ.get("GLM53_PROFILE_PREFILL_TRACE") == "1":
            second = start + ROWS * chunk
            key = model.pair_key(start, second, chunk)
            captured = model.capture_prefills(states, [(key, start)])[key]
            groups_a = [SpGroup(tuple(range(ROWS)), (chunk,) * ROWS, 0, start, False)]
            groups_b = [SpGroup(tuple(range(ROWS)), (chunk,) * ROWS, 0, second, False)]
            pair_ids = torch.randint(0, vocab, (ROWS, 2 * chunk), generator=torch.Generator().manual_seed(chunk + 1))
            model.replay_prefill_sp_pair(captured, pair_ids, states, groups_a, groups_b, logits=False)
            ttnn.synchronize_device(mesh_device)
            ttnn.ReadDeviceProfiler(mesh_device)
            pair_stage = {**stage, "chunk": 2 * chunk, "mode": "prefill_trace"}
            stages.append({**pair_stage, "stage": "start", "rows": rows()})
            a = time.perf_counter()
            model.replay_prefill_sp_pair(captured, pair_ids, states, groups_a, groups_b, logits=False)
            ttnn.synchronize_device(mesh_device)
            seconds = time.perf_counter() - a
            ttnn.ReadDeviceProfiler(mesh_device)
            stages.append({**pair_stage, "stage": "replay", "rows": rows(), "seconds": seconds})
            print(f"prefill pair of chunks {chunk}: replayed in {seconds * 1e3:.1f} ms (profiler on)", flush=True)
            captured.release(mesh_device)
        # ... and (without GLM53_PROFILE_PREFILL_SP) the chunk on every row through its captured form likewise
        # (`capture_prefills`, `replay_prefill`, from GLM53_PROFILE_START: 0 a wave's first chunk, a multiple of the
        # chunk a later one's form; the commit apart with GLM53_PREFILL_TRACE_ALL)
        if not sp and os.environ.get("GLM53_PROFILE_PREFILL_TRACE") == "1":
            key = model.prefill_key(start, chunk, True)
            assert key is not None, f"no captured form for a chunk of {chunk} from {start}"
            captured = model.capture_prefills(states, [(key, start)])[key]
            slots = [0 if r < busy else None for r in range(ROWS)]
            model.replay_prefill(captured, ids, states, slots, start=start, logits=False)
            ttnn.synchronize_device(mesh_device)
            ttnn.ReadDeviceProfiler(mesh_device)
            stages.append({**stage, "mode": "prefill_trace", "stage": "start", "rows": rows()})
            a = time.perf_counter()
            model.replay_prefill(captured, ids, states, slots, start=start, logits=False)
            ttnn.synchronize_device(mesh_device)
            seconds = time.perf_counter() - a
            ttnn.ReadDeviceProfiler(mesh_device)
            stages.append({**stage, "mode": "prefill_trace", "stage": "replay", "rows": rows(), "seconds": seconds})
            print(f"prefill chunk {chunk}: replayed in {seconds * 1e3:.1f} ms (profiler on)", flush=True)
            captured.release(mesh_device)
        for st in states:
            st.deallocate()
        print(f"prefill chunk {chunk}: profiled", flush=True)

    meta = {
        "tree": os.popen(f"git -C {Path(__file__).parent} rev-parse --short HEAD").read().strip(),
        "layers": kinds,
        "stages": stages,
    }
    (out_dir / "stages.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out_dir / 'stages.json'} ({rows()} report rows)", flush=True)
