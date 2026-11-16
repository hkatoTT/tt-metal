"""The verify step of drafts from elsewhere (`TtGlm53Model.forward_verify`, DFlash2's): its traced time at
GLM53_VERIFY_USERS users a row (default 1,4,16: 8 / 32 / 128 users) x GLM53_VERIFY_T tokens a user (default 2,4,6,8:
the token and T - 1 drafts), random tokens (the drafts all kept), against the MTP verify step of one draft
(`forward_spec`, GLM53_VERIFY_BASE=1, the default) on the same load.  Synthetic weights (GLM53_SYNTHETIC_WEIGHTS=1)
for speed.

    GLM53_DEVICE_TEST=1 GLM53_SYNTHETIC_WEIGHTS=1 pytest models/demos/blackhole/glm53_flash/tests/test_verify_device.py -s
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
from models.demos.blackhole.glm53_flash.paths import data_dir, out_dir  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

MAX_SEQ = int(os.environ.get("GLM53_TRACE_MAX_SEQ", "512"))


def release(states) -> None:
    for st in states:
        free = getattr(st, "deallocate", None)
        if free is not None:
            free()


def timed_steps(mesh, model, io, k: int, u: int, run, g, start: int, steps: int) -> tuple:
    """Capture `run` (after one eager run), then replay it `steps` times, the inputs written before each: -> (the
    median step seconds over steps 3 .., the trace's outputs' first read back)."""
    vocab = model.config.vocab_size
    keep = torch.full((ROWS, u), k)
    # GLM53_VERIFY_KEEP=random: each user's kept drafts drawn every step (0 .. k, as a draft model's acceptance; the
    # KDA ops then take the rejected users' history and state back), else all kept (the default)
    random_keep = os.environ.get("GLM53_VERIFY_KEEP", "all") == "random"

    # GLM53_VERIFY_LIVE_FRAC (read at each call): the share of each row's seats with a request (the rest choose no
    # expert, as the generator's empty seats: moe_route.LiveTokens); unset: all live
    frac = os.environ.get("GLM53_VERIFY_LIVE_FRAC")
    live = None
    if frac is not None:
        seats = round(float(frac) * u)
        live = (torch.arange((k + 1) * u) % u < seats).repeat(ROWS, 1)

    def write(t):
        tokens = torch.randint(0, vocab, (ROWS, (k + 1) * u), generator=g)
        base = (start + t) % (MAX_SEQ - k - 1)  # (long runs wrap: the stress)
        positions = torch.cat([torch.full((ROWS, u), base + j) for j in range(k + 1)], dim=1)
        kept = torch.randint(0, k + 1, (ROWS, u), generator=g) if random_keep else keep
        model.write_spec_inputs(io, tokens, positions, kept, live=live)

    write(0)
    first = run()
    for x in first if isinstance(first, tuple) else (first,):
        ttnn.deallocate(x)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        outs = run()
    finally:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
    outs = outs if isinstance(outs, tuple) else (outs,)
    ttnn.synchronize_device(mesh)
    times, writes, replays = [], [], []
    # GLM53_VERIFY_TIME_WRITES=1: also the inputs' write (to the device's end) and the replay alone, timed apart
    apart = os.environ.get("GLM53_VERIFY_TIME_WRITES") == "1"
    for t in range(1, steps + 1):
        a = time.perf_counter()
        write((k + 1) * t)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        times.append(time.perf_counter() - a)
        if apart:
            a = time.perf_counter()
            write((k + 1) * t)
            ttnn.synchronize_device(mesh)
            b = time.perf_counter()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            writes.append(b - a)
            replays.append(time.perf_counter() - b)
    ttnn.release_trace(mesh, trace)
    for x in outs:
        ttnn.deallocate(x)
    rest = sorted(times[2:])
    if apart:
        w, r = sorted(writes[2:]), sorted(replays[2:])
        print(f"KTIMES users {ROWS * u} T {k + 1}: step {rest[len(rest) // 2] * 1e3:.2f} ms = the inputs' write "
              f"{w[len(w) // 2] * 1e3:.2f} ms (to the device's end) + the replay {r[len(r) // 2] * 1e3:.2f} ms (medians, "
              f"timed apart)", flush=True)  # fmt: skip
    return rest[len(rest) // 2]


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_verify_device(mesh_device):
    """Runs `_test_verify_device` (its docstring), an exception printed and the test ending normally (a
    study that reports, run among others)."""
    try:
        _test_verify_device(mesh_device)
    except Exception as err:  # noqa: BLE001
        print(f"VERIFY_DEVICE FAILED: {err!r}"[:3000], flush=True)


def verify_arms() -> list:
    """GLM53_VERIFY_ARMS -> [(name, [(NAME, value) ..])] (";"-separated "name@NAME=value+NAME=value": environment
    variables the model or a program factory reads at each call, each arm timed in turn on the one load, the program
    cache cleared before and after it); unset: one arm, the defaults."""
    out = []
    for a in os.environ.get("GLM53_VERIFY_ARMS", "").split(";"):
        name, _, env = a.partition("@")
        out.append((name, [tuple(p.split("=", 1)) for p in env.split("+") if p]))
    return out


def with_arm(mesh, env: list, run):
    """`run()` with these environment variables set (the program cache cleared so that the programs are made again)."""
    if not env:
        return run()
    undo = [(k, os.environ.get(k)) for k, _ in env]
    for k, v in env:
        os.environ[k] = v
    mesh.clear_program_cache()
    try:
        return run()
    finally:
        for k, v in reversed(undo):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        mesh.clear_program_cache()


def _test_verify_device(mesh_device):
    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1"
    base = os.environ.get("GLM53_VERIFY_BASE", "1") == "1"
    steps = int(os.environ.get("GLM53_TRACE_SPEED_STEPS", "12"))
    start = int(os.environ.get("GLM53_TRACE_START", "0"))
    t0 = time.time()
    checkpoint = ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint()
    model = TtGlm53Model(mesh_device, checkpoint, max_seq=MAX_SEQ, mtp=base)
    result = {"synthetic_weights": synthetic, "load_seconds": round(time.time() - t0), "speed": {}}
    print(f"loaded in {result['load_seconds']}s", flush=True)
    # GLM53_VERIFY_CASES ("users:T,.." per mesh row, e.g. "32:4,64:2,16:8"; T = 1: the MTP verify step of one draft,
    # `forward_spec`) in place of GLM53_VERIFY_USERS x GLM53_VERIFY_T (and the MTP step per users, GLM53_VERIFY_BASE)
    cases = [tuple(int(x) for x in c.split(":")) for c in os.environ.get("GLM53_VERIFY_CASES", "").split(",") if c]
    if not cases:
        for u in [int(x) for x in os.environ.get("GLM53_VERIFY_USERS", "1,4,16").split(",")]:
            cases += [(u, 1)] if base else []
            cases += [(u, int(n)) for n in os.environ.get("GLM53_VERIFY_T", "2,4,6,8").split(",")]
    # GLM53_VERIFY_ARMS_BITS=1: before each case's timing, one eager step from fresh states on fixed inputs, its ids and
    # logits compared with the first arm's (bit for bit, or the max |difference| and the argmax agreement)
    bits = os.environ.get("GLM53_VERIFY_ARMS_BITS") == "1"
    first_out: dict = {}

    def eager_out(u, n):
        k = n - 1
        g = torch.Generator().manual_seed(5000 + 10 * u + n)
        vocab = model.config.vocab_size
        states = model.empty_state(u, spec=max(k, 1))
        if n == 1:
            io = model.alloc_spec_io(u, drafts=1)
            toks = torch.randint(0, vocab, (ROWS, 2 * u), generator=g)
            pos = torch.full((ROWS, u), start + 3)
            model.write_spec_inputs(io, toks, torch.cat([pos, pos + 1], dim=1), torch.ones(ROWS, u, dtype=torch.int64))
            outs = model.forward_spec(io, states, logits=True)
            ids, lg = outs[0], outs[2]
            got = (ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).float(), model.logits_to_host(lg, 2 * u).float())
        else:
            io = model.alloc_spec_io(u, drafts=k, mtp=False)
            toks = torch.randint(0, vocab, (ROWS, n * u), generator=g)
            pos = torch.full((ROWS, u), start + 3)
            model.write_spec_inputs(
                io, toks, torch.cat([pos + j for j in range(n)], dim=1), torch.full((ROWS, u), k, dtype=torch.int64)
            )
            outs = model.forward_verify(io, states, logits=True)
            got = (
                ttnn.to_torch(ttnn.get_device_tensors(outs[0])[0]).float(),
                model.logits_to_host(outs[1], n * u).float(),
            )
        for t in outs:
            if t is not None:
                ttnn.deallocate(t)
        release(states)
        return got

    for arm, env in verify_arms():
        tag = f"_{arm}" if arm else ""

        def run_cases(tag=tag, arm=arm):
            for u, n in cases:
                users = ROWS * u
                line = result["speed"].setdefault(f"{users}_users", {})
                label = f" [{arm}]" if arm else ""
                if bits:
                    ids, lg = eager_out(u, n)
                    if (u, n) not in first_out:
                        first_out[(u, n)] = (arm, ids, lg)
                    else:
                        a0, ids0, lg0 = first_out[(u, n)]
                        same = torch.equal(lg.view(torch.int32), lg0.view(torch.int32)) and torch.equal(ids, ids0)
                        d = float((lg - lg0).abs().max()) / float(lg0.abs().max())
                        agree = float((lg.argmax(-1) == lg0.argmax(-1)).float().mean())
                        print(
                            f"VBITS users={users} T={n} [{arm}] against [{a0}]: "
                            + ("bit for bit" if same else f"DIFFERENT, max rel {d:.2e}, argmax agree {agree:.4f}"),
                            flush=True,
                        )
                if n == 1:  # the MTP verify step of one draft (today's), for the ratio
                    g = torch.Generator().manual_seed(3000 + u)
                    states = model.empty_state(u, spec=1)
                    io = model.alloc_spec_io(u, drafts=1)
                    step = timed_steps(
                        mesh_device, model, io, 1, u, lambda: model.forward_spec(io, states), g, start, steps
                    )
                    line[f"mtp_k1{tag}_step_s"] = round(step, 5)
                    print(f"VERIFY users={users} MTP K=1 (forward_spec){label}: {step * 1e3:.2f} ms", flush=True)
                    release(states)
                    continue
                k = n - 1
                g = torch.Generator().manual_seed(4000 + 10 * u + n)
                states = model.empty_state(u, spec=k)
                io = model.alloc_spec_io(u, drafts=k, mtp=False)
                step = timed_steps(
                    mesh_device, model, io, k, u, lambda: model.forward_verify(io, states), g, start, steps
                )
                line[f"T{n}{tag}_step_s"] = round(step, 5)
                mtp = line.get(f"mtp_k1{tag}_step_s")
                ratio = f", {step / mtp:.2f}x the MTP K=1 step" if mtp else ""
                print(
                    f"VERIFY users={users} T={n}{label}: {step * 1e3:.2f} ms ({users * n / step:.0f} verified "
                    f"tokens/s){ratio}",
                    flush=True,
                )
                release(states)

        with_arm(mesh_device, env, run_cases)
    path = out_dir() / f"verify-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {path}", flush=True)


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_verify_correct(mesh_device):
    """Runs `_test_verify_correct` (its docstring), an exception printed and the test ending normally (a
    study that reports, run among others)."""
    try:
        _test_verify_correct(mesh_device)
    except Exception as err:  # noqa: BLE001
        print(f"VERIFY_CORRECT FAILED: {err!r}"[:3000], flush=True)


def _test_verify_correct(mesh_device):
    """forward_verify against the plain decode (eager, real weights unless GLM53_SYNTHETIC_WEIGHTS=1): each user's
    greedy stream G of GLM53_VERIFY_LEN positions (default 40) from a random first token by the plain decode, then the
    verify steps of T tokens (GLM53_VERIFY_T, default 2,4,8) from the same token: the drafts G's next tokens (each with
    probability q, else a random token; q in GLM53_VERIFY_Q, default 0.8,0: the record's tokens / random), the drafts
    accepted on the host (verify_accept.accept_host), the next step from the bonus token n + 1 positions on with KDA's
    keep n.  Every committed position's greedy id (the accepted drafts' and the bonus's) must equal G's next token: the
    verify step's ids, and the states it goes on from, the plain decode's.  GLM53_VERIFY_USERS (default 1,4).

    GLM53_VERIFY_REF "plain" (the default): G from the plain decode at U users a row; "wide": G from the plain decode
    at T U users a row (user i's stream in users i, U + i, ..: the plain step's ops in the forms of T U tokens a row,
    the verify step's: an op's form changes its sums' order at some tokens a row, e.g. the MoE's return past 32 tokens
    a mesh), with the plain U stream's agreement with it printed; "both": each.  GLM53_VERIFY_MTP=1: the
    verify step is MTP's (`forward_spec`, one draft: T = 2 only, the model loaded with its MTP layer; its own drafts
    not used, the host's given as above).  Prints only (VCORRECT lines, a VRESULT line at the end; no
    assert)."""
    from models.demos.blackhole.glm53_flash.ttnn.verify_accept import accept_host

    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1"
    length = int(os.environ.get("GLM53_VERIFY_LEN", "40"))
    refs = {"plain": ["plain"], "wide": ["wide"], "both": ["plain", "wide"]}[
        os.environ.get("GLM53_VERIFY_REF", "plain")
    ]
    mtp = os.environ.get("GLM53_VERIFY_MTP", "0") == "1"
    with_margin = os.environ.get("GLM53_VERIFY_MARGIN", "0") == "1"
    margins = {}  # width -> [ROWS, U, length + 1]: the plain stream's top-1 minus top-2 logit at each position
    t0 = time.time()
    model = TtGlm53Model(
        mesh_device, ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint(), max_seq=MAX_SEQ, mtp=mtp
    )
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    vocab = model.config.vocab_size
    bad = []

    def plain_stream(first, width):
        """first [ROWS, U] -> the plain greedy streams [ROWS, U, length + 1] at `width` users a row (U | width: user i's
        stream in users i, U + i, ..; the copies' agreement printed)."""
        u = first.shape[1]
        rep = width // u
        stream = torch.zeros(ROWS, width, length + 1, dtype=torch.int64)
        stream[:, :, 0] = first.repeat(1, rep)
        margin = torch.zeros(ROWS, width, length + 1)
        states, io = model.empty_state(width), model.alloc_io(width)
        a = time.time()
        for pos in range(length):
            model.write_inputs(io, stream[:, :, pos], torch.full((ROWS, width), pos))
            if with_margin:  # (the full logits: the argmax on the host, and its margin over the second)
                logits = model.forward(io, states)
                top = model.logits_to_host(logits, width).topk(2, dim=-1)
                ttnn.deallocate(logits)
                stream[:, :, pos + 1] = top.indices[..., 0]
                margin[:, :, pos + 1] = top.values[..., 0] - top.values[..., 1]
            else:
                ids = model.forward(io, states, greedy=True)
                stream[:, :, pos + 1] = model.ids_to_host(ids, width)
                ttnn.deallocate(ids)
        release(states)
        margins[width] = margin.reshape(ROWS, rep, u, length + 1)[:, 0]
        copies = stream.reshape(ROWS, rep, u, length + 1)
        same = bool((copies == copies[:, :1]).all())
        print(
            f"  the plain greedy stream of {length} positions at {ROWS * width} users ({rep} copies a user) in "
            f"{time.time() - a:.0f}s, the copies the same: {same}",
            flush=True,
        )
        return copies[:, 0]

    def agreement(s1, s2) -> str:
        """Two streams [ROWS, U, L + 1]: the users that differ and their first differing positions."""
        diff = s1 != s2
        firsts = [int(d.nonzero()[0]) for d in diff.reshape(-1, diff.shape[-1]) if bool(d.any())]
        return f"{len(firsts)} / {diff.shape[0] * diff.shape[1]} users differ (first at positions {sorted(firsts)})"

    for u in [int(x) for x in os.environ.get("GLM53_VERIFY_USERS", "1,4").split(",")]:
        g = torch.Generator().manual_seed(7000 + u)
        first = torch.randint(0, vocab, (ROWS, u), generator=g)
        print(f"users={ROWS * u}:", flush=True)
        plain = plain_stream(first, u)
        wide = {}
        for n in [int(x) for x in os.environ.get("GLM53_VERIFY_T", "2,4,8").split(",")]:
            if "wide" in refs and n * u not in wide:
                wide[n * u] = plain_stream(first, n * u) if n > 1 else plain
                print(
                    f"  plain at {ROWS * u} users against {ROWS * n * u}: {agreement(plain, wide[n * u])}", flush=True
                )
            for ref in refs:
                stream = plain if ref == "plain" else wide[n * u]
                stream_margin = margins[u if ref == "plain" or n == 1 else n * u]
                for q in [float(x) for x in os.environ.get("GLM53_VERIFY_Q", "0.8,0").split(",")]:
                    k = n - 1
                    if mtp and n != 2:
                        continue
                    states = model.empty_state(u, spec=k)
                    vio = model.alloc_spec_io(u, drafts=k, mtp=mtp)
                    pos = torch.zeros(ROWS, u, dtype=torch.int64)
                    cur = stream[:, :, 0].clone()
                    keep = torch.full((ROWS, u), k)
                    alive = torch.ones(ROWS, u, dtype=torch.bool)
                    left = torch.full((ROWS, u), -1, dtype=torch.int64)
                    compared = matched = steps = 0
                    accepted = []
                    while bool(((pos + n + 1 <= length) & alive).any()):
                        idx = (pos[:, :, None] + torch.arange(1, n)).clamp(max=length)
                        oracle = stream.gather(2, idx)
                        rand = torch.randint(0, vocab, oracle.shape, generator=g)
                        drafts = torch.where(torch.rand(oracle.shape, generator=g) < q, oracle, rand)
                        tokens = torch.cat([cur[:, None, :], drafts.permute(0, 2, 1)], dim=1).reshape(ROWS, n * u)
                        positions = (pos[:, None, :] + torch.arange(n)[None, :, None]).clamp(max=MAX_SEQ - 1)
                        model.write_spec_inputs(vio, tokens, positions.reshape(ROWS, n * u), keep)
                        if mtp:  # (its next drafts not used)
                            out, mtp_drafts = model.forward_spec(vio, states)
                            ttnn.deallocate(mtp_drafts)
                        else:
                            out = model.forward_verify(vio, states)
                        greedy = model.ids_to_host(out, n * u)
                        ttnn.deallocate(out)
                        acc, bonus = accept_host(tokens, greedy, k)
                        gr = greedy.reshape(ROWS, n, u)
                        for r in range(ROWS):
                            for i in range(u):
                                if not alive[r, i] or pos[r, i] + acc[r, i] + 1 > length:
                                    continue
                                for j in range(int(acc[r, i]) + 1):  # the committed positions p .. p + n
                                    want = stream[r, i, pos[r, i] + j + 1]
                                    compared += 1
                                    if gr[r, j, i] == want:
                                        matched += 1
                                    else:
                                        alive[r, i] = False  # (its stream leaves G: not compared further)
                                        left[r, i] = pos[r, i] + j + 1
                                        break
                        accepted.append(acc.float().mean().item())
                        cur, keep = bonus, acc
                        pos = pos + acc + 1
                        steps += 1
                    release(states)
                    ok = matched == compared
                    if not ok:
                        bad.append((u, n, q, ref))
                    print(
                        f"VCORRECT{' MTP' if mtp else ''} users={ROWS * u} T={n} "
                        f"q={q} ref={ref}: {steps} steps, "
                        f"drafts accepted a step "
                        f"{sum(accepted) / max(len(accepted), 1):.2f}, committed positions equal to the plain decode "
                        f"{matched} / {compared}, users that left G {int((~alive).sum())} (at positions "
                        f"{sorted(int(x) for x in left[~alive])})",
                        flush=True,
                    )
                    if with_margin and bool((~alive).any()):  # G's margin where each user left it
                        out = [
                            f"{int(left[r, i])}: {float(stream_margin[r, i, left[r, i]]):.3f}"
                            for r in range(ROWS)
                            for i in range(u)
                            if not alive[r, i]
                        ]
                        print(
                            f"  VCORRECT margins (position: G's top-1 minus top-2 logit) {', '.join(out)}", flush=True
                        )
    print(f"VRESULT {'all equal' if not bad else 'DIFFER: ' + str(bad)}", flush=True)


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_verify_first_step(mesh_device):
    """Runs `_test_verify_first_step` (its docstring), an exception printed and the test ending normally (a
    study that reports, run among others)."""
    try:
        _test_verify_first_step(mesh_device)
    except Exception as err:  # noqa: BLE001
        print(f"VERIFY_FIRST_STEP FAILED: {err!r}"[:3000], flush=True)


def _test_verify_first_step(mesh_device):
    """The verify step's logits at every one of its T tokens against the plain decode's over the same tokens one
    position at a time, per mesh row and token (eager; synthetic weights unless GLM53_SYNTHETIC_WEIGHTS=0): the max
    |difference| relative to the logits' max, and whether the argmax agrees, for GLM53_VERIFY_CASES users:T (default
    1:4,1:8,4:2,2:4) and models of GLM53_VERIFY_LAYERS layers (default 45).  Step 0 from the empty state (positions
    0 .. T - 1), then for each a in GLM53_VERIFY_KEEPS (default none; e.g. "0,3": a <= T - 1) a second step with keep a
    (positions a + 1 .., the plain decode over step 0's first a + 1 tokens and the new ones): where an 8-token row of
    one user parts from the plain decode.  GLM53_VERIFY_WIDE=1 (the default): also against the plain decode
    at the verify step's T U tokens a row (user i's tokens in users i, U + i, ..: the ops in the verify step's forms),
    with the plain decode's own spread between the two widths.  Prints only (a failure is read from the log: no
    assert)."""
    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS", "1") == "1"
    cases = [
        tuple(int(x) for x in c.split(":")) for c in os.environ.get("GLM53_VERIFY_CASES", "1:4,1:8,4:2,2:4").split(",")
    ]
    keeps = [int(x) for x in os.environ.get("GLM53_VERIFY_KEEPS", "").split(",") if x != ""]
    wide = os.environ.get("GLM53_VERIFY_WIDE", "1") == "1"

    def plain_run(model, u, seq, width=None):
        """seq [ROWS, L, U] tokens at positions 0 .. L - 1 through the plain decode at `width` users a row (default U;
        U | width, user i's tokens in users i, U + i, ..) -> logits [ROWS, L, U, V] (the first copies')."""
        width = width or u
        states, io = model.empty_state(width), model.alloc_io(width)
        out = []
        for j in range(seq.shape[1]):
            model.write_inputs(io, seq[:, j].repeat(1, width // u), torch.full((ROWS, width), j))
            lg = model.forward(io, states)
            out.append(model.logits_to_host(lg, width)[:, :u])
            ttnn.deallocate(lg)
        release(states)
        return torch.stack(out, dim=1)

    def verify_run(model, vio, states, u, n, toks, start, keep):
        """toks [ROWS, n, U] at positions start .. start + n - 1, keep [ROWS, U] -> logits [ROWS, n, U, V]."""
        positions = (start + torch.arange(n)).repeat_interleave(u)[None].expand(ROWS, -1)
        model.write_spec_inputs(vio, toks.reshape(ROWS, n * u), positions, keep)
        ids, logits = model.forward_verify(vio, states, logits=True)
        ver = model.logits_to_host(logits, n * u).reshape(ROWS, n, u, -1)
        for t in (ids, logits):
            ttnn.deallocate(t)
        return ver

    def report(tag, ver, plain):
        d = (ver - plain).abs().amax(dim=(2, 3)) / plain.abs().amax(dim=(2, 3))  # [ROWS, n]
        same = (ver.argmax(-1) == plain.argmax(-1)).all(dim=2)  # [ROWS, n]
        worst = float(d.max())
        print(f"{tag}: max rel {worst:.4f}, argmax the same {int(same.sum())} / {same.numel()}", flush=True)
        for r in range(ROWS):
            print(
                f"  row {r}: rel per token {[round(float(x), 4) for x in d[r]]}, argmax {''.join('=' if x else 'X' for x in same[r].tolist())}",
                flush=True,
            )

    for layers in [int(x) for x in os.environ.get("GLM53_VERIFY_LAYERS", "45").split(",")]:
        t0 = time.time()
        model = TtGlm53Model(
            mesh_device,
            ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint(),
            num_layers=layers,
            max_seq=MAX_SEQ,
            mtp=False,
        )
        print(f"layers={layers}: loaded in {time.time() - t0:.0f}s", flush=True)
        vocab = model.config.vocab_size
        for u, n in cases:
            g = torch.Generator().manual_seed(9000 + 10 * u + n)
            toks0 = torch.randint(0, vocab, (ROWS, n, u), generator=g)
            a0 = time.time()
            tag = f"VFIRST layers={layers} users={ROWS * u} T={n}"
            plain = plain_run(model, u, toks0)
            plain_w = plain_run(model, u, toks0, n * u) if wide and n > 1 else None
            if plain_w is not None:  # (the plain decode's own spread between the two widths)
                report(f"{tag} step 0 plain at {u} against {n * u} a row", plain, plain_w)
            seconds = {}  # keep a -> (the second step's tokens, the plain decode's logits there, at T U a row)
            for a in keeps:
                if a > n - 1:
                    continue
                toks1 = torch.randint(0, vocab, (ROWS, n, u), generator=g)
                seq = torch.cat([toks0[:, : a + 1], toks1], dim=1)
                p1 = plain_run(model, u, seq)[:, a + 1 :]
                p1w = plain_run(model, u, seq, n * u)[:, a + 1 :] if plain_w is not None else None
                if p1w is not None:
                    report(f"{tag} step 1 keep {a} plain at {u} against {n * u} a row", p1, p1w)
                seconds[a] = (toks1, p1, p1w)
            vio = model.alloc_spec_io(u, drafts=n - 1, mtp=False)
            states = model.empty_state(u, spec=n - 1)
            ver = verify_run(model, vio, states, u, n, toks0, 0, torch.full((ROWS, u), n - 1))
            release(states)
            report(f"{tag} step 0", ver, plain)
            if plain_w is not None:
                report(f"{tag} step 0 against {n * u} a row", ver, plain_w)
            for a, (toks1, p1, p1w) in seconds.items():
                states = model.empty_state(u, spec=n - 1)
                verify_run(model, vio, states, u, n, toks0, 0, torch.full((ROWS, u), n - 1))
                ver = verify_run(model, vio, states, u, n, toks1, a + 1, torch.full((ROWS, u), a))
                release(states)
                report(f"{tag} step 1 keep {a}", ver, p1)
                if p1w is not None:
                    report(f"{tag} step 1 keep {a} against {n * u} a row", ver, p1w)
            print(f"  (case {u}:{n} in {time.time() - a0:.0f}s)", flush=True)
        del model


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_verify_chunk_speed(mesh_device):
    """Runs `_test_verify_chunk_speed` (its docstring), an exception printed and the test ending normally (a
    study that reports, run among others)."""
    try:
        _test_verify_chunk_speed(mesh_device)
    except Exception as err:  # noqa: BLE001
        print(f"VERIFY_CHUNK_SPEED FAILED: {err!r}"[:3000], flush=True)


def _test_verify_chunk_speed(mesh_device):
    """The KDA verify step's (the chunked form) settings on one load: the verify step
    (`forward_verify`) traced, in turn GLM53_VERIFY_REPEATS times (default 2), for GLM53_VERIFY_CASES users-a-row:T
    (default 1:8,4:8,16:4: 8 / 32 users K = 7, 128 users K = 3).  The keeps: GLM53_VERIFY_KEEP (all, the default, or
    random: each user's accepted drafts drawn every step).  GLM53_VERIFY_SPEED_FORMS (default chunk,chunk_dram_nosplit;
    chunk: the model's, the hand-offs in L1 and decode_prepare's split_channels; a "_dram" part: the hand-offs in DRAM,
    kda.HANDOFF_L1 False; a "_nosplit" part: kda.PREP_SPLIT False), each against the first.  VCHUNK lines.
    """
    from models.demos.blackhole.glm53_flash.ttnn import kda as kda_mod

    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1"
    steps = int(os.environ.get("GLM53_TRACE_SPEED_STEPS", "12"))
    start = int(os.environ.get("GLM53_TRACE_START", "0"))
    repeats = int(os.environ.get("GLM53_VERIFY_REPEATS", "2"))
    cases = [
        tuple(int(x) for x in c.split(":")) for c in os.environ.get("GLM53_VERIFY_CASES", "1:8,4:8,16:4").split(",")
    ]
    keep = os.environ.get("GLM53_VERIFY_KEEP", "all")
    t0 = time.time()
    checkpoint = ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint()
    model = TtGlm53Model(mesh_device, checkpoint, max_seq=MAX_SEQ, mtp=False)
    print(f"loaded in {round(time.time() - t0)}s", flush=True)
    result = {}
    defaults = (kda_mod.HANDOFF_L1, kda_mod.PREP_SPLIT)
    for u, n in cases:
        k = n - 1
        for rep in range(repeats):
            for form in os.environ.get("GLM53_VERIFY_SPEED_FORMS", "chunk,chunk_dram_nosplit").split(","):
                kda_mod.HANDOFF_L1 = "dram" not in form.split("_")
                kda_mod.PREP_SPLIT = "nosplit" not in form.split("_")
                g = torch.Generator().manual_seed(5000 + 10 * u + n)
                states = model.empty_state(u, spec=k)
                io = model.alloc_spec_io(u, drafts=k, mtp=False)
                step = timed_steps(
                    mesh_device, model, io, k, u, lambda: model.forward_verify(io, states), g, start, steps
                )
                result.setdefault(f"{ROWS * u}x{n}", {}).setdefault(form, []).append(round(step * 1e3, 3))
                print(f"VCHUNK users={ROWS * u} T={n} keep={keep} rep {rep} {form}: {step * 1e3:.2f} ms", flush=True)
                release(states)
    kda_mod.HANDOFF_L1, kda_mod.PREP_SPLIT = defaults
    for key, forms in result.items():
        mean = {f: sum(v) / len(v) for f, v in forms.items()}
        names = list(mean)
        print(f"VCHUNK {key} keep={keep}: " + ", ".join(f"{f} {mean[f]:.2f} ms" for f in names) + " ("
              + ", ".join(f"{f} {mean[f] - mean[names[0]]:+.2f}" for f in names[1:]) + ")", flush=True)  # fmt: skip


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_verify_kld(mesh_device):
    """Runs `_test_verify_kld` (its docstring), an exception printed and the test ending normally (a
    study that reports, run among others)."""
    try:
        _test_verify_kld(mesh_device)
    except Exception as err:  # noqa: BLE001
        print(f"VERIFY_KLD FAILED: {err!r}"[:3000], flush=True)


def _test_verify_kld(mesh_device):
    """Teacher forcing through the verify step (real weights; the chunked KDA form): the 8 acceptance records' prompts
    by the prefill (one a row), then their generated tokens in traced verify steps of T = K + 1 (GLM53_VERIFY_T, default
    8: the token and the record's next K as the drafts), each step accepting a random number of the drafts (the same
    draws for every run).  At the accepted positions the logits against the CPU decode path's reference (as
    test_model_device_accuracy's teacher): KLD, top-1; with two runs (GLM53_VERIFY_FORMS, e.g. chunk,chunk:3) their
    paired KLD difference with a bootstrap 95 % interval.  GLM53_VERIFY_DIRTY=n (default 0): before that, other
    records' prompts in the same seats and n eager verify steps from them (the seats reused, as a server's: the prefill
    must start each request afresh); a run "chunk:3" in GLM53_VERIFY_FORMS sets it for that run only.  The saved rows
    after the prefill are checked against the rows that do nothing (VKLD rows lines).  VKLD lines (no assert)."""
    from models.demos.blackhole.glm53_flash.ttnn import dsa as dsa_mod
    from models.demos.blackhole.glm53_flash.ttnn import kda as kda_mod
    from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator

    n = int(os.environ.get("GLM53_VERIFY_T", "8"))
    k = n - 1
    forms = os.environ.get("GLM53_VERIFY_FORMS", "chunk").split(",")
    records_dir = Path(__file__).parents[1] / "tools" / "acceptance" / "greedy-v1"
    records = [json.loads(f.read_text()) for f in sorted(records_dir.glob("*.json"))]
    # (tools/run_cpu_reference.py --teacher-decode's logits)
    ref = torch.load(data_dir("teacher-decode-v1", "bf16_decode_logits.pt"))
    offsets, total = [], 0  # each record's first row in the reference (records in file order)
    for r in records:
        offsets.append(total)
        total += len(r["generated_token_ids"])
    t0 = time.time()
    model = TtGlm53Model(mesh_device, Glm53Checkpoint(), max_seq=1024, mtp=False)
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    dsa_mod.PREFILL_SP = False  # (one request a row)
    slots = records[:ROWS]
    vocab = model.config.vocab_size
    results = {}
    for entry in forms:
        form, _, dirty_s = entry.partition(":")
        dirty = int(dirty_s) if dirty_s else int(os.environ.get("GLM53_VERIFY_DIRTY", "0"))
        gen = Glm53Generator(model, users_per_row=1, chunk=(128, 512))
        for st in gen.states:
            st.deallocate()
        gen.states = model.empty_state(1, spec=k)
        sio = model.alloc_spec_io(1, drafts=k, mtp=False)
        if dirty:  # (other requests first in the same seats, a few verify steps on)
            others = [records[(r + 1) % len(records)] for r in range(ROWS)]
            gen.prefill_many([(r, s["prompt_token_ids"]) for r, s in enumerate(others)], chunk=512, traced=False)
            mess = torch.Generator().manual_seed(5 + n)
            at = [len(s["prompt_token_ids"]) for s in others]
            keep = torch.full((ROWS, 1), k)
            for _ in range(dirty):
                toks = torch.tensor([[(others[r]["generated_token_ids"] + [0] * n)[at[r] - len(others[r]["prompt_token_ids"]) + j]
                                      for j in range(n)] for r in range(ROWS)])  # fmt: skip
                poss = torch.tensor([[at[r] + j for j in range(n)] for r in range(ROWS)])
                model.write_spec_inputs(sio, toks, poss, keep.float())
                ids = model.forward_verify(sio, gen.states)
                ttnn.deallocate(ids)
                acc = torch.randint(0, k + 1, (ROWS,), generator=mess)
                at = [at[r] + int(acc[r]) + 1 for r in range(ROWS)]
                keep = acc.view(ROWS, 1)
            print(f"VKLD T={n} {entry}: {dirty} verify steps of other requests in the seats first", flush=True)
        gen.prefill_many([(r, s["prompt_token_ids"]) for r, s in enumerate(slots)], chunk=512, traced=False)
        layers = [st for st in gen.states if isinstance(st, kda_mod.TtKdaState) and st.spec_rows is not None]
        if layers:  # (every KDA layer's saved rows, on every chip, those that do nothing: as made here)
            pristine = kda_mod.chunk_saved_host(1, layers[0].recurrent.shape[1], k + 1)
            same = [
                all(bool((ttnn.to_torch(r).float() == pristine).all()) for r in ttnn.get_device_tensors(st.spec_rows))
                for st in layers
            ]  # fmt: skip
            print(f"VKLD rows T={n} {entry}: after the prefill, the saved rows do nothing in {sum(same)} / {len(same)} "
                  f"KDA layers", flush=True)  # fmt: skip
        seqs = [s["prompt_token_ids"] + s["generated_token_ids"] for s in slots]
        pos = [len(s["prompt_token_ids"]) for s in slots]  # the step's first position (its token: seqs[r][pos])
        ends = [len(q) - 1 for q in seqs]  # the last position whose logits are compared (it predicts nothing past)
        draws = torch.Generator().manual_seed(77 + n)
        keep = torch.full((ROWS, 1), k)
        trace = outs = None
        kld, top1, where = [], [], []
        steps = 0
        while any(p <= e for p, e in zip(pos, ends)):
            toks = torch.tensor([[seqs[r][min(pos[r] + j, ends[r])] for j in range(n)] for r in range(ROWS)])
            poss = torch.tensor([[min(pos[r] + j, 1023) for j in range(n)] for r in range(ROWS)])
            model.write_spec_inputs(sio, toks, poss, keep.float())
            if trace is None:  # (one eager run, then the capture: the eager one's logits used)
                ids, lg = model.forward_verify(sio, gen.states, logits=True)
                lp = model.logits_to_host(lg, n).float()
                ttnn.deallocate(ids)
                ttnn.deallocate(lg)
            else:
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
                lp = model.logits_to_host(outs[1], n).float()
            lp = torch.log_softmax(lp[..., :vocab], dim=-1)  # [ROWS, T, V]
            acc = torch.randint(0, k + 1, (ROWS,), generator=draws)
            for r in range(ROWS):
                for j in range(int(acc[r]) + 1):
                    q = pos[r] + j
                    kk = q - (len(slots[r]["prompt_token_ids"]) - 1)
                    if q > ends[r] or not 0 <= kk < len(slots[r]["generated_token_ids"]):
                        continue
                    row = offsets[r] + kk
                    lc = torch.log_softmax(ref[row].float()[:vocab], dim=-1)
                    kld.append(float((lc.exp() * (lc - lp[r, j])).sum()))
                    top1.append(float(lp[r, j].argmax() == lc.argmax()))
                    where.append(row)
            for r in range(ROWS):
                pos[r] += int(acc[r]) + 1
            keep = acc.view(ROWS, 1)
            steps += 1
            if trace is None:
                trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                try:
                    outs = model.forward_verify(sio, gen.states, logits=True)
                finally:
                    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
                # (the capture ran nothing: the next step replays it)
        if trace is not None:
            ttnn.release_trace(mesh_device, trace)
            for t in outs:
                ttnn.deallocate(t)
        gen.release()
        order = sorted(range(len(where)), key=lambda i: where[i])
        results[entry] = (torch.tensor([kld[i] for i in order]), torch.tensor([top1[i] for i in order]),
                         [where[i] for i in order])  # fmt: skip
        print(f"VKLD T={n} {entry}: {len(kld)} positions in {steps} steps, mean KLD {torch.tensor(kld).mean():.5f}, "
              f"top-1 {torch.tensor(top1).mean():.4f}", flush=True)  # fmt: skip
    if len(results) == 2:
        (ka, _, wa), (kb, _, wb) = results[forms[0]], results[forms[1]]
        if wa == wb:
            d = kb - ka
            g = torch.Generator().manual_seed(1)
            boots = torch.stack([d[torch.randint(0, len(d), (len(d),), generator=g)].mean() for _ in range(2000)])
            lo, hi = boots.quantile(0.025).item(), boots.quantile(0.975).item()
            print(f"VKLD T={n} {forms[1]} - {forms[0]}: mean {d.mean():+.6f}, 95 % [{lo:+.6f}, {hi:+.6f}] "
                  f"(contains 0: {lo <= 0 <= hi})", flush=True)  # fmt: skip
        else:
            print(f"VKLD T={n}: the forms' positions differ ({len(wa)} / {len(wb)})", flush=True)


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_verify_reuse_bits(mesh_device):
    """Runs `_test_verify_reuse_bits` (its docstring), an exception printed and the test ending normally (a
    study that reports, run among others)."""
    try:
        _test_verify_reuse_bits(mesh_device)
    except Exception as err:  # noqa: BLE001
        print(f"VERIFY_REUSE_BITS FAILED: {err!r}"[:3000], flush=True)


def _test_verify_reuse_bits(mesh_device):
    """A seat reused, bit for bit (synthetic weights unless GLM53_SYNTHETIC_WEIGHTS=0): random
    prompts by the prefill (one a row, eager), then GLM53_VERIFY_REUSE_STEPS (default 4) eager verify steps of random
    drafts and keep codes: their logits; then other prompts and verify steps in the same seats, then the first prompts
    and the same steps again: the logits must be the same bits. For each of GLM53_VERIFY_FORMS (default
    chunk,chunk:dram+nosplit: the model's settings, then options after ":" joined by "+": dram (the hand-offs in DRAM,
    kda.HANDOFF_L1 False), nosplit (kda.PREP_SPLIT False)), T = GLM53_VERIFY_T (default 8); and each form's first
    run's logits against the first form's (the forms of the same values: bit for bit).  VREUSE lines (no assert)."""
    from models.demos.blackhole.glm53_flash.ttnn import dsa as dsa_mod
    from models.demos.blackhole.glm53_flash.ttnn import kda as kda_mod
    from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator

    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS", "1") == "1"
    n = int(os.environ.get("GLM53_VERIFY_T", "8"))
    k = n - 1
    steps = int(os.environ.get("GLM53_VERIFY_REUSE_STEPS", "4"))
    forms = os.environ.get("GLM53_VERIFY_FORMS", "chunk,chunk:dram+nosplit").split(",")
    model = TtGlm53Model(
        mesh_device, ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint(), max_seq=1024, mtp=False
    )
    dsa_mod.PREFILL_SP = False
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(31)
    prompts = [[torch.randint(0, vocab, (40 + 7 * r,), generator=g).tolist() for r in range(ROWS)] for _ in range(2)]
    plan = [(torch.randint(0, vocab, (ROWS, n), generator=g), torch.randint(0, k + 1, (ROWS,), generator=g))
            for _ in range(steps)]  # fmt: skip
    defaults = (kda_mod.HANDOFF_L1, kda_mod.PREP_SPLIT)

    def snap(gen):
        """The first KDA layer's state tensors (chip 0) -> {name: host tensor}."""
        st = next(x for x in gen.states if isinstance(x, kda_mod.TtKdaState))
        out = {}
        for name in ("recurrent", "spec_rows", "spec_saved", "spec_inert"):
            t = getattr(st, name)
            if t is not None:
                out[name] = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().clone()
        if st.spec_rows is not None:  # (against the rows that do nothing as made on the host)
            pristine = kda_mod.chunk_saved_host(1, st.recurrent.shape[1], k + 1)
            for name in ("spec_rows", "spec_inert"):
                d = (out[name] - pristine).abs()
                bad = (d.reshape(-1, 32, 32).amax(dim=(1, 2)) > 0).nonzero().flatten().tolist()  # (tiles)
                nt = kda_mod.chunk_slot_tiles(k + 1)
                print(f"  VREUSE {name} against the host's rows that do nothing: max |diff| {float(d.max()):.3g}, "
                      f"tiles that differ {len(bad)} / {d.numel() // 1024} (units {sorted({t // nt for t in bad})[:12]}, "
                      f"tiles in a unit {sorted({t % nt for t in bad})})", flush=True)  # fmt: skip
        for i, h in enumerate(st.history):
            out[f"history{i}"] = ttnn.to_torch(ttnn.get_device_tensors(h)[0]).float().clone()
        return out

    snaps = []

    def run(gen, sio, which):
        """Prompts `which` into the seats, then the planned verify steps -> their logits [steps, ROWS, T, V]."""
        gen.prefill_many([(r, p) for r, p in enumerate(prompts[which])], chunk=512, traced=False)
        snaps.append(("after the prefill", snap(gen)))
        pos = [len(p) for p in prompts[which]]
        keep = torch.full((ROWS, 1), k)
        out = []
        for i, (toks, acc) in enumerate(plan):
            if i == 1:
                snaps.append(("after step 0", snap(gen)))
            poss = torch.tensor([[pos[r] + j for j in range(n)] for r in range(ROWS)])
            model.write_spec_inputs(sio, toks, poss, keep.float())
            ids, lg = model.forward_verify(sio, gen.states, logits=True)
            out.append(model.logits_to_host(lg, n).float()[..., :vocab].clone())
            for t in (ids, lg):
                ttnn.deallocate(t)
            pos = [pos[r] + int(acc[r]) + 1 for r in range(ROWS)]
            keep = acc.view(ROWS, 1)
        return torch.stack(out)

    firsts = {}
    for entry in forms:
        form, _, opt = entry.partition(":")
        opts = set(opt.split("+")) if opt else set()
        kda_mod.HANDOFF_L1 = "dram" not in opts
        kda_mod.PREP_SPLIT = "nosplit" not in opts
        gen = Glm53Generator(model, users_per_row=1, chunk=(128, 512))
        for st in gen.states:
            st.deallocate()
        gen.states = model.empty_state(1, spec=k)
        sio = model.alloc_spec_io(1, drafts=k, mtp=False)
        snaps.clear()
        first = run(gen, sio, 0)
        run(gen, sio, 1)
        again = run(gen, sio, 0)
        for (where, a), (_, b) in zip(snaps[0:2], snaps[4:6]):
            diff = {key: float((a[key] - b[key]).abs().max()) for key in a if key in b}
            print(f"VREUSE T={n} {entry}: the first KDA layer {where}, first run vs again: max |diff| {diff}",
                  flush=True)  # fmt: skip
        firsts[entry] = first
        base = next(iter(firsts))
        if base != entry:
            print(f"VREUSE T={n} {entry} against {base}: the first run's logits the same bits "
                  f"{[bool(torch.equal(first[s], firsts[base][s])) for s in range(steps)]}", flush=True)  # fmt: skip
        same = [bool((first[s] == again[s]).all()) for s in range(steps)]
        worst = [float((first[s] - again[s]).abs().max()) for s in range(steps)]
        print(f"VREUSE T={n} {entry}: the first prompts again after others in the seats, per step the same bits {same}, "
              f"max |diff| {[round(w, 4) for w in worst]}", flush=True)  # fmt: skip
        gen.release()
    kda_mod.HANDOFF_L1, kda_mod.PREP_SPLIT = defaults
