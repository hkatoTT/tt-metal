"""The KDA verify states' ping-pong (GLM53_KDA_SPEC_PINGPONG, serving's run-ahead) in the whole model:
DFlash2's verify step (`forward_verify`, T tokens a user) taken back after a speculative launch gives the same bits as
never launching it.

Synthetic weights (seeded per tensor).  Two state sets on one load: A without the ping-pong (in place, as the default)
and B with it.  Each step s: A runs the real inputs; B, from step 1 on (but step 4), first runs a speculative step (as
if the last step's drafts were all accepted: keep K, positions on, other tokens), `kda_spec_launched`, then
`kda_spec_undo`, and runs the real inputs (`kda_spec_launched` again).  Before step 3 (after B's undo) and step 4 (no
speculative step: the prefill follows a step kept) a prefill writes a new request into seat 1 / 2 of every mesh row of
both (positions from its prompt on, keep K): the two parities' commits.  After the prefill of step 4, `kda_spec_undo`
must refuse.  Each step's ids and logits of A and B are compared bit for bit, and at the end the KDA states (B's
current set).

Two phases: eager (forward_verify and the prefill eager), then traced on new states: B's prefill and its commit
replayed (`capture_prefills`: the commit captured once per parity; A's prefill eager, the same bits), A's verify
step one trace, B's one trace per parity.

    GLM53_PP_USERS (4), GLM53_PP_T (8), GLM53_PP_STEPS (8), GLM53_PP_MAX_SEQ (512), GLM53_PP_POS (0: the steps' first
    position; past 2,051 on a long cache the DSA indexer's pools), GLM53_PP_PROMPT (128), GLM53_PP_PHASES (eager,traced)
    GLM53_PP_SPEC: the speculative step's inputs: "ahead" (default: as if the last step's drafts were all accepted,
    every seat's tokens other; its DSA rows up to 2T past the last step's start: on a long cache the indexer's 16-row
    recent ring loses rows a real step reads, DSA's, not KDA's) or "same" (the real positions and keeps, the real
    tokens but seat 0 of every row's: serving's case, a speculative step taken back for other seats' sake)
    GLM53_PP_PRINT_ONLY=1: print the comparison only (no failure)
Run (a device test): GLM53_DEVICE_TEST=1 GLM53_SYNTHETIC_WEIGHTS=1 pytest
    models/demos/blackhole/glm53_flash/tests/test_kda_spec_pingpong_model.py -s
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
from models.demos.blackhole.glm53_flash.config import KDA  # noqa: E402
from models.demos.blackhole.glm53_flash.tests._trace import traced as trace_capture  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import model as model_module  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.is_floating_point():
        width = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return torch.equal(a.contiguous().view(width), b.contiguous().view(width))
    return torch.equal(a, b)


def chips(t, which=(0, 31)) -> list:
    devs = ttnn.get_device_tensors(t)
    return [ttnn.to_torch(devs[i]).clone() for i in which if i < len(devs)]


def kda_fields(model, states) -> list:
    """Each KDA layer's (history 0-2, recurrent, spec_saved, spec_rows) as the fields hold them (B: its current set)."""
    out = []
    for st, layer in zip(states, model.layers):
        if layer.kind == KDA:
            out.append([*st.history, st.recurrent, st.spec_saved, st.spec_rows])
    return out


def free_states(states) -> None:
    for st in states:
        free = getattr(st, "deallocate", None)
        if free is not None:
            free()


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_kda_spec_pingpong_model(mesh_device, expect_error):
    t0 = time.time()
    users = int(os.environ.get("GLM53_PP_USERS", "4"))
    t_tok = int(os.environ.get("GLM53_PP_T", "8"))
    k = t_tok - 1
    steps = int(os.environ.get("GLM53_PP_STEPS", "8"))
    max_seq = int(os.environ.get("GLM53_PP_MAX_SEQ", "512"))
    pos0 = int(os.environ.get("GLM53_PP_POS", "0"))
    plen = int(os.environ.get("GLM53_PP_PROMPT", "128"))
    phases = os.environ.get("GLM53_PP_PHASES", "eager,traced").split(",")
    spec_kind = os.environ.get("GLM53_PP_SPEC", "ahead")
    model = TtGlm53Model(mesh_device, ShapeOnlyCheckpoint(), max_seq=max_seq)
    vocab = model.config.vocab_size
    print(f"loaded in {time.time() - t0:.1f} s ({users} users a row, T = {t_tok}, cache {max_seq})", flush=True)
    io = model.alloc_spec_io(users, drafts=k, mtp=False)
    prefill_at = {3: 1, 4: 2}  # step -> the seat a prefill writes before it (every mesh row)
    spec_at = {s for s in range(1, steps) if s != 4}  # B's speculative step first (then undone)

    def make(pp: bool) -> list:
        os.environ["GLM53_KDA_SPEC_PINGPONG"] = "1" if pp else "0"
        try:
            return model.empty_state(users, spec=k)
        finally:
            os.environ["GLM53_KDA_SPEC_PINGPONG"] = "0"

    def positions(p):
        return torch.cat([p + j for j in range(t_tok)], dim=1)

    def scenario(phase: str, run_a, run_b, prefill_a, prefill_b, states_b) -> list:
        """The steps; run_*(tokens, positions, keep) -> (ids, logits) on the host; prefill_*(tokens, slot) -> logits.
        -> the differing (step, what)."""
        g = torch.Generator().manual_seed(31)
        pos = torch.full((ROWS, users), pos0, dtype=torch.int64)
        last = pos.clone()  # the last step's first positions (a speculative step assumes its drafts all accepted)
        keep = torch.full((ROWS, users), k, dtype=torch.int64)
        bad = []
        for s in range(steps):
            toks = torch.randint(0, vocab, (ROWS, t_tok * users), generator=g)
            if s in spec_at:  # B: the speculative step, launched, then taken back
                wrong = torch.randint(0, vocab, (ROWS, t_tok * users), generator=g)
                if spec_kind == "same":  # (the real inputs but seat 0's tokens: token j of user u at j U + u)
                    other = toks.clone()
                    other[:, ::users] = wrong[:, ::users]
                    run_b(other, positions(pos), keep)
                else:
                    run_b(wrong, positions(last + t_tok), torch.full((ROWS, users), k, dtype=torch.int64))
                model.kda_spec_launched(states_b)
                model.kda_spec_undo(states_b)
            if s in prefill_at:
                seat = prefill_at[s]
                prompt = torch.randint(0, vocab, (ROWS, plen), generator=g)
                la, lb = prefill_a(prompt, seat), prefill_b(prompt, seat)
                if not same_bits(la, lb):
                    bad.append((s, "prefill logits"))
                pos[:, seat] = plen
                keep[:, seat] = k
                if s not in spec_at:  # the prefill's commit followed a step kept: that step can't be taken back now
                    with expect_error(RuntimeError, "no verify step to take back"):
                        model.kda_spec_undo(states_b)
            ia, lga = run_a(toks, positions(pos), keep)
            ib, lgb = run_b(toks, positions(pos), keep)
            model.kda_spec_launched(states_b)
            same_ids, same_lg = same_bits(ia, ib), same_bits(lga, lgb)
            print(
                f"KPPM {phase} step {s}: parity {model.kda_spec_parity(states_b)}, spec {s in spec_at}, prefill "
                f"{prefill_at.get(s)}, ids {'same' if same_ids else 'DIFFERENT'}, logits "
                f"{'same' if same_lg else 'DIFFERENT max ' + str(float((lga - lgb).abs().max()))}",
                flush=True,
            )
            if not same_ids:
                bad.append((s, "ids"))
            if not same_lg:
                bad.append((s, "logits"))
            last = pos.clone()
            keep = torch.randint(0, t_tok, (ROWS, users), generator=g)
            pos = pos + keep + 1
        return bad

    heads = next(layer.attention.heads for layer in model.layers if layer.kind == KDA)
    sp = (1536 * t_tok + 64 + 4095) // 4096  # (a unit's slot in the rows, in 32-row pages)

    def unread(x):  # each unit's slot's beta row past its T values: bytes the op leaves as they were in L1, never read
        x = x.clone()
        for u in range(heads * 4):
            x[:, :, (u * sp + t_tok // 4) * 32, t_tok:16] = 0
        return x

    def compare_states(phase, a, b) -> list:
        bad = []
        names = ["history0", "history1", "history2", "recurrent", "spec_saved", "spec_rows"]
        for layer, (fa, fb) in enumerate(zip(kda_fields(model, a), kda_fields(model, b))):
            for name, ta, tb in zip(names, fa, fb):
                for chip, (xa, xb) in enumerate(zip(chips(ta), chips(tb))):
                    if name == "spec_rows":
                        xa, xb = unread(xa), unread(xb)
                    if not same_bits(xa, xb):
                        bad.append((layer, name, chip))
        print(f"KPPM {phase} states: {'all the same' if not bad else f'DIFFERENT {bad[:12]} ({len(bad)})'}", flush=True)
        return bad

    failures = []
    if "eager" in phases:
        sa, sb = make(False), make(True)
        assert model.kda_spec_parity(sa) is None and model.kda_spec_parity(sb) == 0

        def run(states):
            def go(toks, p, keep):
                model.write_spec_inputs(io, toks, p, keep)
                ids, lg = model.forward_verify(io, states, logits=True)
                out = (
                    ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).float().clone(),
                    model.logits_to_host(lg, t_tok * users).float().clone(),
                )
                ttnn.deallocate(ids)
                ttnn.deallocate(lg)
                return out

            return go

        def prefill(states):
            return lambda prompt, seat: model.prefill(prompt, 0, states, [seat] * ROWS).float().clone()

        t1 = time.time()
        bad = scenario("eager", run(sa), run(sb), prefill(sa), prefill(sb), sb)
        bad += compare_states("eager", sa, sb)
        print(f"KPPM eager: {'all the same' if not bad else f'DIFFERENT {bad}'} ({time.time() - t1:.1f} s)", flush=True)
        failures += [("eager", *b) for b in bad]
        free_states(sa)
        free_states(sb)

    if "traced" in phases:
        if "eager" not in phases:  # the verify step's device constants before any capture
            sc = make(True)
            zero = torch.zeros(ROWS, users, dtype=torch.int64)
            model.write_spec_inputs(io, torch.zeros(ROWS, t_tok * users, dtype=torch.int64), positions(zero), zero + k)
            for t in model.forward_verify(io, sc, logits=True):
                ttnn.deallocate(t)
            free_states(sc)
        sa, sb = make(False), make(True)
        # A's prefill eager (traced and eager the same bits); first user 0 of every row as capture_prefills'
        # eager runs leave B's (a zero prompt of the chunk's length, committed)
        model.prefill(torch.zeros(ROWS, plen, dtype=torch.int64), 0, sa, [0] * ROWS)
        specs = [(key, s) for key, s in model.prefill_trace_specs(plen, plen) if not key.sp and key.pair < 0]
        print(f"prefill captures: {[tuple(key) for key, _ in specs]}", flush=True)
        cap_b = model.capture_prefills(sb, specs)  # (the commit once a parity)
        key = next(key for key, _ in specs if key != model_module.COMMIT)
        assert cap_b[model_module.COMMIT].other is not None

        def capture(states):
            with trace_capture(mesh_device) as trace:
                ids, lg = model.forward_verify(io, states, logits=True)
            return trace, ids, lg

        used = model_module.trace_bytes(mesh_device)
        tr_a = capture(sa)
        size_a = model_module.trace_bytes(mesh_device) - used
        tr_b = {}
        for p in (0, 1):
            model.kda_spec_set_parity(sb, p)
            tr_b[p] = capture(sb)
        model.kda_spec_set_parity(sb, 0)
        now = model_module.trace_bytes(mesh_device)
        print(
            f"verify traces: A {size_a / 2**20:.1f} MiB, B two {(now - used - size_a) / 2**20:.1f} MiB; the region "
            f"holds {now / 2**20:.1f} MiB",
            flush=True,
        )

        def replay(tr):
            trace, ids, lg = tr
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
            return (
                ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).float().clone(),
                model.logits_to_host(lg, t_tok * users).float().clone(),
            )

        def run_a(toks, p, keep):
            model.write_spec_inputs(io, toks, p, keep)
            return replay(tr_a)

        def run_b(toks, p, keep):
            model.write_spec_inputs(io, toks, p, keep)
            return replay(tr_b[model.kda_spec_parity(sb)])

        def prefill_a(prompt, seat):
            return model.prefill(prompt, 0, sa, [seat] * ROWS).float().clone()

        def prefill_b(prompt, seat):
            slots = [seat] * ROWS
            lg = model.replay_prefill(cap_b[key], prompt, sb, slots, valid=plen)
            model.replay_commit(cap_b[model_module.COMMIT], sb, slots)
            return lg.float().clone()

        t1 = time.time()
        bad = scenario("traced", run_a, run_b, prefill_a, prefill_b, sb)
        bad += compare_states("traced", sa, sb)
        print(
            f"KPPM traced: {'all the same' if not bad else f'DIFFERENT {bad}'} ({time.time() - t1:.1f} s)", flush=True
        )
        failures += [("traced", *b) for b in bad]
        for tr in (tr_a, tr_b[0], tr_b[1]):
            ttnn.release_trace(mesh_device, tr[0])
        for one in cap_b.values():
            one.release(mesh_device)
        free_states(sa)
        free_states(sb)
    print(f"KPPM RESULT ({spec_kind}): {'all the same' if not failures else f'DIFFERENT {failures}'}", flush=True)
    if os.environ.get("GLM53_PP_PRINT_ONLY") != "1":
        assert not failures, failures
