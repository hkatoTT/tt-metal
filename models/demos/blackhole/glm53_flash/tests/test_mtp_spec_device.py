"""MTP self-speculative decode on the device (`TtGlm53Model.forward_spec`): the verify step's speed, and
its greedy output against the plain decode step's and the acceptance records.

    GLM53_MTP_MODE = speed (default; greedy+speed: both on one load): the traced verify step at GLM53_MTP_USERS users a row (default 1,4,16: 8 / 32 /
        128 users), random tokens, with the ids read back; tok/s at the acceptance rate GLM53_MTP_ACCEPT (default
        0.898, the CPU reference's, tools/mtp_acceptance.py).  With GLM53_SYNTHETIC_WEIGHTS=1 for speed experiments;
        GLM53_MTP_PLAIN=1 also times the plain greedy step at the same users.  GLM53_MTP_DRAFTS=2: two drafts a user
        (3 tokens a user; tok/s with the second's acceptance GLM53_MTP_ACCEPT2, default 0.806, the CPU's).
    GLM53_MTP_MODE = greedy: real weights, 2 users a row, the 11 greedy records (slot k the record k mod 11), each
        prompt fed through the steps themselves (no prefill: the plain step one token a step, the verify step two, the
        next prompt token given in place of the draft and of the MTP layer's input) and the record's generation length
        generated; the verify step's tokens against the plain step's (the same model, the same lockstep inputs) and
        the records, and the acceptance rate of the drafts.
Results: out_dir()/mtp-<mode>-<stamp>.json.  Run (a device test):
    GLM53_DEVICE_TEST=1 GLM53_MTP=1 pytest models/demos/blackhole/glm53_flash/tests/test_mtp_spec_device.py -s
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
from models.demos.blackhole.glm53_flash.paths import out_dir  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

RECORDS = Path(__file__).resolve().parents[1] / "tools" / "acceptance" / "greedy-v1"
MAX_SEQ = int(os.environ.get("GLM53_TRACE_MAX_SEQ", "512"))


def capture(mesh, run):
    """`run()` once eagerly (the step at the inputs written: its outputs read back by the caller), then captured."""
    first = run()
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = run()
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    return trace, first, outs


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_mtp_spec_device(mesh_device):
    mode = os.environ.get("GLM53_MTP_MODE", "speed")
    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1"
    result = {"tree": os.popen(f"git -C {Path(__file__).parent} rev-parse --short HEAD").read().strip(), "mode": mode}
    result["synthetic_weights"] = synthetic
    t0 = time.time()
    checkpoint = ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint()
    model = TtGlm53Model(mesh_device, checkpoint, max_seq=MAX_SEQ, mtp=True)
    result["load_seconds"] = round(time.time() - t0)
    print(f"loaded in {result['load_seconds']}s", flush=True)
    if "greedy" in mode:
        configs = os.environ.get("GLM53_MTP_GREEDY", "")  # users:drafts,.. (default: 2 users a row, GLM53_MTP_DRAFTS)
        if configs:
            for c in configs.split(","):
                users, k = (int(x) for x in c.split(":"))
                result[f"greedy_{users}x{k}"] = sub = {}
                greedy(mesh_device, model, sub, users=users, k_drafts=k)
        else:
            greedy(mesh_device, model, result)
    if "speed" in mode:  # (greedy+speed: both on one load)
        speed(mesh_device, model, result)
    path = out_dir() / f"mtp-{mode}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {path}", flush=True)


def speed(mesh, model, result: dict) -> None:
    """The verify step's replay time (inputs written, replayed, synchronized; then both ids read back), the median
    over steps 3-12, random tokens at positions 1..12 (the drafts all kept)."""
    accept = float(os.environ.get("GLM53_MTP_ACCEPT", "0.898"))
    accept2 = float(os.environ.get("GLM53_MTP_ACCEPT2", "0.806"))
    k = int(os.environ.get("GLM53_MTP_DRAFTS", "1"))
    start = int(os.environ.get("GLM53_TRACE_START", "0"))  # the context position the steps start at (serving: ~1K)
    per_step = 1 + accept + (accept * accept2 if k > 1 else 0.0)  # tokens a user a step
    vocab = model.config.vocab_size
    result["speed"] = {"acceptance_assumed": accept, "drafts": k, "tokens_a_step": per_step}
    if k > 1:
        result["speed"]["acceptance2_assumed"] = accept2
    for u in [int(x) for x in os.environ.get("GLM53_MTP_USERS", "1,4,16").split(",")]:
        g = torch.Generator().manual_seed(2000 + u)
        states = model.empty_state(u, spec=k)  # the MTP layer's last
        io = model.alloc_spec_io(u, drafts=k)
        keep = torch.full((ROWS, u), k)

        def write(t):
            tokens = torch.randint(0, vocab, (ROWS, (k + 1) * u), generator=g)
            positions = torch.cat([torch.full((ROWS, u), start + t + j) for j in range(k + 1)], dim=1)
            model.write_spec_inputs(io, tokens, positions, keep)

        write(0)
        t1 = time.time()
        trace, first, outs = capture(mesh, lambda: model.forward_spec(io, states))
        for x in first:
            ttnn.deallocate(x)
        ttnn.synchronize_device(mesh)
        t2 = time.time()
        device_only, with_readback, parts = [], [], []
        # GLM53_MTP_TIME_WRITES=1: the inputs' writes timed apart (host enqueue, then synchronized), then the replay alone
        time_writes = os.environ.get("GLM53_MTP_TIME_WRITES") == "1"
        for t in range(1, int(os.environ.get("GLM53_TRACE_SPEED_STEPS", "12")) + 1):
            a = time.perf_counter()
            write((k + 1) * t)
            if time_writes:
                w1 = time.perf_counter()
                ttnn.synchronize_device(mesh)
                w2 = time.perf_counter()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            b = time.perf_counter()
            if time_writes:
                parts.append((w1 - a, w2 - a, b - w2))
            for x in outs:
                model.ids_to_host(x, (k + 1) * u)
            c = time.perf_counter()
            device_only.append(b - a)
            with_readback.append(c - a)
        med = lambda v: sorted(v[2:])[len(v[2:]) // 2]  # noqa: E731
        users = ROWS * u
        step, step_rb = med(device_only), med(with_readback)
        result["speed"][f"{users}_users"] = {
            "step_s": round(step, 4),
            "step_with_ids_readback_s": round(step_rb, 4),
            "tok_s_total": round(users * per_step / step, 2),
            "tok_s_total_with_ids_readback": round(users * per_step / step_rb, 2),
            "compile_and_capture_s": round(t2 - t1, 1),
        }
        print(f"verify step, {users} users: {result['speed'][f'{users}_users']}", flush=True)
        if parts:
            enq, done, replay = (med([x[i] for x in parts]) for i in range(3))
            print(
                f"verify step, {users} users: inputs enqueued {enq * 1e3:.2f} ms, written (synchronized) "
                f"{done * 1e3:.2f} ms, the replay alone {replay * 1e3:.2f} ms",
                flush=True,
            )
        ttnn.release_trace(mesh, trace)
        for st in states:
            st.deallocate()
        if os.environ.get("GLM53_MTP_PLAIN", "0") == "1":  # the plain greedy step at the same users, for the ratio
            states = model.empty_state(u)
            io = model.alloc_io(u)
            model.write_inputs(io, torch.randint(0, vocab, (ROWS, u), generator=g), torch.full((ROWS, u), start))
            trace, first, ids = capture(mesh, lambda: model.forward(io, states, greedy=True))
            ttnn.deallocate(first)
            times, parts = [], []
            for t in range(1, 13):
                tokens = torch.randint(0, vocab, (ROWS, u), generator=g)
                a = time.perf_counter()
                model.write_inputs(io, tokens, torch.full((ROWS, u), start + t))
                if time_writes:  # (GLM53_MTP_TIME_WRITES: the writes timed apart, as the verify step's)
                    w1 = time.perf_counter()
                    ttnn.synchronize_device(mesh)
                    w2 = time.perf_counter()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                times.append(time.perf_counter() - a)
                if time_writes:
                    parts.append((w1 - a, w2 - a, time.perf_counter() - w2))
            plain = med(times)
            if parts:
                enq, done, replay = (med([x[i] for x in parts]) for i in range(3))
                print(
                    f"plain step, {users} users: inputs enqueued {enq * 1e3:.2f} ms, written (synchronized) "
                    f"{done * 1e3:.2f} ms, the replay alone {replay * 1e3:.2f} ms",
                    flush=True,
                )
            result["speed"][f"{users}_users"]["plain_step_s"] = round(plain, 4)
            result["speed"][f"{users}_users"]["plain_tok_s_total"] = round(users / plain, 2)
            print(f"plain step, {users} users: {plain * 1e3:.1f} ms, {users / plain:.1f} tok/s", flush=True)
            ttnn.release_trace(mesh, trace)
            for st in states:
                st.deallocate()


def greedy(mesh, model, result: dict, users: int = 2, k_drafts: int | None = None) -> None:
    records = [json.loads(p.read_text()) for p in sorted(RECORDS.glob("*.json"))]
    names = [p.stem for p in sorted(RECORDS.glob("*.json"))]
    slots = ROWS * users
    prompts = [records[k % len(records)]["prompt_token_ids"] for k in range(slots)]
    wants = [records[k % len(records)]["generated_token_ids"] for k in range(slots)]
    lengths = [len(w) for w in wants]

    # the plain step: one token a step, each slot its prompt, then its own greedy tokens (lockstep positions)
    t0 = time.time()
    plain = [[] for _ in range(slots)]
    states = model.empty_state(users)  # (with the MTP layer's, unused here)
    io = model.alloc_io(users)

    def plain_inputs(i):
        # a slot done keeps feeding its last token (its outputs are no longer taken)
        tok = [
            prompts[k][i] if i < len(prompts[k]) else plain[k][min(i - len(prompts[k]), len(plain[k]) - 1)]
            for k in range(slots)
        ]
        return torch.tensor(tok).reshape(ROWS, users), torch.full((ROWS, users), i)

    def plain_take(i, got):
        for k in range(slots):
            if i + 1 >= len(prompts[k]) and len(plain[k]) < lengths[k]:
                plain[k].append(int(got.reshape(-1)[k]))

    steps = max(len(p) + n for p, n in zip(prompts, lengths)) - 1
    model.write_inputs(io, *plain_inputs(0))
    trace, first, ids = capture(mesh, lambda: model.forward(io, states, greedy=True))
    plain_take(0, model.ids_to_host(first, users))
    ttnn.deallocate(first)
    for i in range(1, steps):
        model.write_inputs(io, *plain_inputs(i))
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        plain_take(i, model.ids_to_host(ids, users))
    ttnn.release_trace(mesh, trace)
    for st in states:
        st.deallocate()
    result["plain_steps"] = steps
    result["plain_seconds"] = round(time.time() - t0, 1)

    def prefix(a, b):
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return n

    if k_drafts is None:
        k_drafts = int(os.environ.get("GLM53_MTP_DRAFTS", "1"))
    result["users_a_row"], result["drafts"] = users, k_drafts
    for phase in ("fed", "prefilled"):  # prompts through the verify step itself / through the prefill
        if phase == "prefilled" and (os.environ.get("GLM53_MTP_PREFILL", "1") != "1" or k_drafts > 1):
            continue  # (two drafts: no chained first drafts after a prefill yet)
        result[phase] = spec_phase(mesh, model, users, prompts, lengths, prefilled=phase == "prefilled", k=k_drafts)
        out, gen_steps, accepted, gen_tokens = (result[phase].pop(x) for x in ("out", "gen_steps", "accepted", "gen"))
        tried2, accepted2 = (result[phase].pop(x) for x in ("tried2", "accepted2"))
        rows = []
        for k in range(slots):
            o = out[k][: lengths[k]]
            rows.append(
                {
                    "slot": k,
                    "record": names[k % len(records)],
                    "plain_vs_record": prefix(plain[k], wants[k]),
                    "spec_vs_record": prefix(o, wants[k]),
                    "spec_vs_plain": prefix(o, plain[k]),
                    "length": lengths[k],
                    "gen_steps": gen_steps[k],
                    "accepted": accepted[k],
                }
            )
            print(
                f"{users}x{k_drafts} {phase} slot {k:2d} {rows[-1]['record']:10s}: plain/record {rows[-1]['plain_vs_record']:2d}, "
                f"spec/record {rows[-1]['spec_vs_record']:2d}, spec/plain {rows[-1]['spec_vs_plain']:2d} of "
                f"{lengths[k]}; drafts accepted {accepted[k]} / {gen_steps[k]}",
                flush=True,
            )
        r = result[phase]
        r["slots"] = rows
        first_slots = rows[: len(records)]
        r["records_full_plain"] = sum(x["plain_vs_record"] == x["length"] for x in first_slots)
        r["records_full_spec"] = sum(x["spec_vs_record"] == x["length"] for x in first_slots)
        r["spec_equals_plain"] = sum(x["spec_vs_plain"] == x["length"] for x in rows)
        r["acceptance"] = round(sum(accepted) / max(1, sum(gen_steps)), 4)
        r["tokens_per_generation_step"] = round(sum(gen_tokens) / max(1, sum(gen_steps)), 3)
        second = ""
        if k_drafts > 1:  # the second draft, where the first was accepted
            r["acceptance2"] = round(sum(accepted2) / max(1, sum(tried2)), 4)
            second = f", the second draft {r['acceptance2']} (given the first)"
        print(
            f"{users}x{k_drafts} {phase}: records in full: plain {r['records_full_plain']} / {len(records)}, spec "
            f"{r['records_full_spec']} / {len(records)}; spec equals plain in {r['spec_equals_plain']} / "
            f"{slots} slots; acceptance {r['acceptance']}{second}, {r['tokens_per_generation_step']} tokens a "
            f"generation step",
            flush=True,
        )
    assert result["fed"]["acceptance"] > 0.5, "the drafts are mostly rejected: the MTP layer's inputs are likely wrong"


def spec_phase(mesh, model, users: int, prompts: list, lengths: list, prefilled: bool, k: int = 1) -> dict:
    """The verify step (`k` drafts a user) until every slot has its generation length.  The prompts go through the
    verify step itself (k + 1 tokens a step, the next prompt tokens in place of the drafts and of the MTP layer's
    inputs), or with `prefilled` (one draft) through the prefill (eager, chunks of 128, the MTP layer's cache with
    them) and the MTP layer's step at each prompt's last position (`mtp_first_drafts`), the first generated token from
    the prefill's logits.  A step's drafts are checked in turn against the next tokens (the prompt's, or the step's
    own greedy ids): with a accepted it takes them and the next token after them, and the MTP layer's drafts from that
    slot go on (keep code a)."""
    t0 = time.time()
    slots = ROWS * users
    n = k + 1
    states = model.empty_state(users, spec=k)
    sio = model.alloc_spec_io(users, drafts=k)
    pos = [0] * slots
    tok = [p[0] for p in prompts]
    drafts = [list(p[1 : 1 + k]) for p in prompts]
    keep = [k] * slots
    out = [[] for _ in range(slots)]
    gen_steps, accepted, gen_tokens = [0] * slots, [0] * slots, [0] * slots
    tried2, accepted2 = [0] * slots, [0] * slots
    if prefilled:
        chunk = 128
        for u in range(users):  # a wave per user: one request per mesh row
            ks = [r * users + u for r in range(ROWS)]  # (not k: the drafts)
            longest = max(len(prompts[x]) for x in ks)
            for start in range(0, longest, chunk):
                ids = torch.zeros(ROWS, chunk, dtype=torch.int64)
                nxt = torch.zeros(ROWS, chunk, dtype=torch.int64)
                rows, valid, ending = [False] * ROWS, [chunk] * ROWS, {}
                for r, slot in enumerate(ks):
                    prompt = prompts[slot]
                    m = min(chunk, len(prompt) - start)  # (not n: the tokens a user a step)
                    if m <= 0:
                        continue
                    rows[r], valid[r] = True, m
                    ids[r, :m] = torch.tensor(prompt[start : start + m])
                    ahead = prompt[start + 1 : start + m + 1]
                    nxt[r, : len(ahead)] = torch.tensor(ahead)
                    if start + m == len(prompt):
                        ending[r] = slot
                last = start + chunk >= longest
                logits = model.prefill(
                    ids,
                    start,
                    states,
                    [u] * ROWS,
                    rows=rows,
                    valid=valid,
                    logits=bool(ending),
                    last=last,
                    next_tokens=nxt,
                )
                if not ending:
                    continue
                first = torch.full((ROWS,), -1, dtype=torch.int64)
                where = [None] * ROWS
                for r, slot in ending.items():
                    first[r] = int(logits[r].argmax())
                    where[r] = (u, len(prompts[slot]) - 1)
                d = model.mtp_first_drafts(states, first, where, users)
                for r, s in ending.items():
                    pos[s], tok[s], drafts[s] = len(prompts[s]), int(first[r]), [int(d[r])]
                    out[s].append(int(first[r]))

    def spec_inputs():
        tokens = torch.zeros(slots, n, dtype=torch.int64)
        positions = torch.zeros(slots, n, dtype=torch.int64)
        forced = torch.full((slots, n), -1, dtype=torch.int64)
        for s in range(slots):
            p, prompt = pos[s], prompts[s]
            tokens[s] = torch.tensor([tok[s], *drafts[s]])
            positions[s] = torch.arange(p, p + n)
            for j in range(n):  # in the prompt the MTP layer's next tokens are the prompt's
                if p + j + 1 <= len(prompt) - 1:
                    forced[s, j] = prompt[p + j + 1]
        # [slots, n] -> [ROWS, nU]: a row's tokens first (its users in order), then its first drafts, ..
        row = lambda t: t.reshape(ROWS, users, n).transpose(1, 2).reshape(ROWS, n * users)  # noqa: E731
        return row(tokens), row(positions), torch.tensor(keep, dtype=torch.float32).reshape(ROWS, users), row(forced)

    def spec_take(g_ids, *d_ids):
        col = lambda t: t.reshape(ROWS, n, users).transpose(1, 2).reshape(slots, n)  # noqa: E731
        g, levels = col(g_ids), [col(d) for d in d_ids]
        for s in range(slots):
            p, prompt = pos[s], prompts[s]
            last = len(prompt) - 1
            # the next token after each of the step's tokens: the prompt's, or the step's own greedy id
            truth = [prompt[p + j + 1] if p + j + 1 <= last else int(g[s, j]) for j in range(n)]
            a = 0
            while a < k and drafts[s][a] == truth[a]:
                a += 1
            if p >= last and len(out[s]) < lengths[s]:  # a generation step (every draft the MTP layer's)
                gen_steps[s] += 1
                accepted[s] += a >= 1
                tried2[s] += k > 1 and a >= 1
                accepted2[s] += k > 1 and a >= 2
                gen_tokens[s] += a + 1
            for j, t in enumerate([*drafts[s][:a], truth[a]]):  # at positions p + 1 .. p + a + 1
                if p + 1 + j > last:
                    out[s].append(t)
            pos[s], tok[s], keep[s] = p + a + 1, truth[a], a
            nxt = [int(level[s, a]) for level in levels]  # the MTP layer's drafts from slot a, by level
            q = pos[s]
            drafts[s] = [prompt[q + 1 + i] if q + 1 + i <= last else nxt[i] for i in range(k)]

    tokens, positions, keep_t, forced = spec_inputs()
    model.write_spec_inputs(sio, tokens, positions, keep_t, forced=forced)
    trace, first, outs = capture(mesh, lambda: model.forward_spec(sio, states))
    spec_take(*[model.ids_to_host(x, n * users) for x in first])
    for x in first:
        ttnn.deallocate(x)
    spec_steps = 1
    while any(len(o) < m for o, m in zip(out, lengths)):
        tokens, positions, keep_t, forced = spec_inputs()
        if int(positions.max()) + 1 >= MAX_SEQ:
            break
        model.write_spec_inputs(sio, tokens, positions, keep_t, forced=forced)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        spec_take(*[model.ids_to_host(x, n * users) for x in outs])
        spec_steps += 1
    ttnn.release_trace(mesh, trace)
    for st in states:
        st.deallocate()
    return {
        "steps": spec_steps,
        "seconds": round(time.time() - t0, 1),
        "out": out,
        "gen_steps": gen_steps,
        "accepted": accepted,
        "gen": gen_tokens,
        "tried2": tried2,
        "accepted2": accepted2,
    }
