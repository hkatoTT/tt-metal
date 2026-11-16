"""The model's outputs, saved for a bit-for-bit comparison between two trees or settings (e.g. a change that only
removes unused forms must leave every default output the same).

Synthetic weights (seeded per tensor: the same in every run), the serving generator, a long cache (the DSA indexer):
  - the prefill, eager and then traced (chunks of 512): one request a row (unequal lengths) and one long request over
    the 8 rows; the logits after each prompt;
  - then greedy decode steps of every slot (traced): the first step's logits and GLM53_BITS_STEPS steps' ids (compared
    at the slots with a request only: the others choose no expert, moe_route.LiveTokens, and their values are not used).

    GLM53_BITS_OUT=path.pt   save them
    GLM53_BITS_REF=path.pt   compare them with an earlier run's, bit for bit (the test fails on a difference)
    GLM53_BITS_USERS (4), GLM53_BITS_MAX_SEQ (4096), GLM53_BITS_STEPS (8), GLM53_BITS_CHUNK (512: the prefill's chunk;
    1024 takes the waves of 1,024 tokens a row, the long request's in SP pairs)
    GLM53_BITS_PRINT_ONLY=1  print the comparison only (no failure: a study's run ends cleanly)
    GLM53_BITS_SPEC=N        then N of MTP's verify steps (one draft, eager) from fresh states: their ids, drafts and
                             logits too (the model's MTP layer)
    GLM53_BITS_SPEC_DRAFTS=2 ... with two drafts (MTP's K = 2): the ids and both levels' drafts (no logits)
    GLM53_BITS_VERIFY_T=T    then max(N, 4) verify steps of T tokens a user (DFlash2's `forward_verify`, eager) from
                             fresh states, random keeps: their ids and logits
    GLM53_BITS_VERIFY_POS=P  the MTP and T-token verify steps from position P (default 0; the fresh states' rows
                             before it zero): past 2,051 on a long cache the indexer chooses among the pools, so the
                             pool keys the steps write reach the ids and logits
    GLM53_BITS_REPEATS=N     then the traced prefills and the decode steps N times more, each against its first run
                             in this process, bit for bit (replays that differ: counted, the test fails), and whether
                             the ids change: each prompt's first token and the greedy steps' (the requests' slots)
Run (a device test): GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_model_bits.py -s
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
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal bit for bit (NaN payloads and -0.0 too)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.is_floating_point():
        width = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return torch.equal(a.contiguous().view(width), b.contiguous().view(width))
    return torch.equal(a, b)


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_model_bits(mesh_device):
    torch.set_num_threads(16)
    users = int(os.environ.get("GLM53_BITS_USERS", "4"))
    max_seq = int(os.environ.get("GLM53_BITS_MAX_SEQ", "4096"))
    steps = int(os.environ.get("GLM53_BITS_STEPS", "8"))
    verify_pos = int(os.environ.get("GLM53_BITS_VERIFY_POS", "0"))
    chunk = int(os.environ.get("GLM53_BITS_CHUNK", "512"))
    t0 = time.time()
    model = TtGlm53Model(mesh_device, ShapeOnlyCheckpoint(), max_seq=max_seq)
    gen = Glm53Generator(model, users_per_row=users, chunk=(128, chunk))
    print(f"loaded in {time.time() - t0:.1f} s ({users} users a row, cache {max_seq})", flush=True)
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(17)
    prompt = lambda n: torch.randint(0, vocab, (n,), generator=g).tolist()  # noqa: E731
    # one request a row (slot users r + r mod users: row r), then one long request over the rows (row 3's slot)
    rows = [(users * r + r % users, prompt(300 + 97 * r)) for r in range(ROWS)]
    long = [(users * 3 + (3 + 1) % users, prompt(min(max_seq - 64, 3000)))]
    out = {}
    for label, traced in (("eager", False), ("traced", True)):
        if traced:
            t1 = time.time()
            seconds = gen.warmup(greedy=True, logits=True, chunks=[chunk], traced=(chunk,))
            print(f"warm-up and captures {time.time() - t1:.1f} s: {seconds}", flush=True)
        for name, reqs in (("rows", rows), ("long", long)):
            t1 = time.time()
            got = gen.prefill_many(reqs, chunk=chunk, traced=traced)
            out[f"prefill_{name}_{label}"] = torch.stack([x.clone().float() for x in got])
            print(f"prefill {name}, {label}: {time.time() - t1:.1f} s, counts {dict(gen.counts)}", flush=True)
    # decode from the traced prefills: every slot with a request, from its prompt's end
    lengths = {slot: len(p) for slot, p in rows + long}
    positions = torch.full((ROWS * users,), -1, dtype=torch.int64)
    tokens = torch.zeros(ROWS * users, dtype=torch.int64)
    finals = list(out["prefill_rows_traced"]) + list(out["prefill_long_traced"])
    last = {slot: x for (slot, _), x in zip(rows + long, finals)}
    for slot, n in lengths.items():
        positions[slot] = n
        tokens[slot] = int(torch.argmax(last[slot].reshape(-1)[:vocab]))
    logits = gen.decode(tokens, positions, greedy=False)
    out["decode_logits"] = logits.clone().float()
    ids = []
    for _ in range(steps):
        got = gen.decode(tokens, positions, greedy=True)
        ids.append(got.clone())
        tokens = torch.where(positions >= 0, got, tokens)
        positions = torch.where(positions >= 0, positions + 1, positions)
    out["decode_ids"] = torch.stack(ids)
    repeats = int(os.environ.get("GLM53_BITS_REPEATS", "0"))
    differ, ids_changed = {}, {}
    active = sorted(lengths)  # (the other slots' states move on with every step: not compared)
    for i in range(repeats):  # the same replays again (the caches rewritten with the same values)
        for name, reqs in (("rows", rows), ("long", long)):
            got = torch.stack([x.clone().float() for x in gen.prefill_many(reqs, chunk=chunk, traced=True)])
            want = out[f"prefill_{name}_traced"]
            if not same_bits(got, want):
                d = (got - want).abs()
                differ.setdefault(f"prefill_{name}_traced", []).append((i, float(d.max())))
                top = lambda t: t.reshape(len(reqs), -1)[:, :vocab].argmax(-1)  # noqa: E731
                if not torch.equal(top(got), top(want)):
                    ids_changed.setdefault(f"first_token_{name}", []).append(i)
        tokens0 = torch.zeros(ROWS * users, dtype=torch.int64)
        pos0 = torch.full((ROWS * users,), -1, dtype=torch.int64)
        for slot, n in lengths.items():
            pos0[slot] = n
            tokens0[slot] = int(torch.argmax(last[slot].reshape(-1)[:vocab]))
        got = gen.decode(tokens0, pos0, greedy=False).clone().float()[active]
        if not same_bits(got, out["decode_logits"][active]):
            d = (got - out["decode_logits"][active]).abs()
            differ.setdefault("decode_logits", []).append((i, float(d.max())))
        for step in range(steps):  # the greedy steps again, from the same start
            got_ids = gen.decode(tokens0, pos0, greedy=True).clone()
            if not torch.equal(got_ids[active], out["decode_ids"][step][active]):
                ids_changed.setdefault("decode_ids", []).append((i, step))
                break
            tokens0 = torch.where(pos0 >= 0, got_ids, tokens0)
            pos0 = torch.where(pos0 >= 0, pos0 + 1, pos0)
    if repeats:
        print(f"{repeats} repeats: replays differing from the first {differ or 'none'}", flush=True)
        print(f"{repeats} repeats: ids changed {ids_changed or 'none'}", flush=True)
    gen.release()
    spec_steps = int(os.environ.get("GLM53_BITS_SPEC", "0"))
    spec_drafts = int(os.environ.get("GLM53_BITS_SPEC_DRAFTS", "1"))
    if spec_steps and model.mtp is not None and spec_drafts == 2:
        # MTP's verify step with two drafts (eager, from fresh states): the ids and both levels' drafts each step
        gs = torch.Generator().manual_seed(29)
        sstates = model.empty_state(users, spec=2)
        sio = model.alloc_spec_io(users, drafts=2)
        pos = torch.full((ROWS, users), verify_pos, dtype=torch.int64)
        keep = torch.full((ROWS, users), 2, dtype=torch.int64)
        got = {"spec2_ids": [], "spec2_drafts": [], "spec2_drafts2": []}
        for _ in range(spec_steps):
            toks = torch.randint(0, vocab, (ROWS, 3 * users), generator=gs)
            model.write_spec_inputs(sio, toks, torch.cat([pos, pos + 1, pos + 2], dim=1), keep)
            outs = model.forward_spec(sio, sstates)
            for key, t in zip(("spec2_ids", "spec2_drafts", "spec2_drafts2"), outs):
                got[key].append(ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().clone())
                ttnn.deallocate(t)
            keep = torch.randint(0, 3, (ROWS, users), generator=gs)
            pos = pos + keep + 1
        for st in sstates:
            free = getattr(st, "deallocate", None)
            if free is not None:
                free()
        for k, v in got.items():
            out[k] = torch.stack(v)
        print(f"MTP verify steps (two drafts): {spec_steps}, to positions {int(pos.max())}", flush=True)
    elif spec_steps and model.mtp is not None:
        # MTP's verify step (one draft, eager, `forward_spec`) from fresh states on the same long cache: positions 0 ..,
        # random tokens and keeps (the same in every run); the ids, the next drafts and both logits each step
        gs = torch.Generator().manual_seed(23)
        sstates = model.empty_state(users, spec=1)
        sio = model.alloc_spec_io(users, drafts=1)
        pos = torch.full((ROWS, users), verify_pos, dtype=torch.int64)
        keep = torch.ones(ROWS, users, dtype=torch.int64)
        got = {"spec_ids": [], "spec_drafts": [], "spec_logits": [], "spec_draft_logits": []}
        t1 = time.time()
        for _ in range(spec_steps):
            toks = torch.randint(0, vocab, (ROWS, 2 * users), generator=gs)
            model.write_spec_inputs(sio, toks, torch.cat([pos, pos + 1], dim=1), keep)
            s_ids, s_drafts, s_lg, s_dlg = model.forward_spec(sio, sstates, logits=True)
            got["spec_ids"].append(ttnn.to_torch(ttnn.get_device_tensors(s_ids)[0]).float().clone())
            got["spec_drafts"].append(ttnn.to_torch(ttnn.get_device_tensors(s_drafts)[0]).float().clone())
            got["spec_logits"].append(model.logits_to_host(s_lg, 2 * users).float().clone())
            got["spec_draft_logits"].append(model.logits_to_host(s_dlg, 2 * users).float().clone())
            for t in (s_ids, s_drafts, s_lg, s_dlg):
                ttnn.deallocate(t)
            keep = torch.randint(0, 2, (ROWS, users), generator=gs)
            pos = pos + keep + 1
        for st in sstates:
            free = getattr(st, "deallocate", None)
            if free is not None:
                free()
        for k, v in got.items():
            out[k] = torch.stack(v)
        print(f"MTP verify steps: {spec_steps} in {time.time() - t1:.1f} s, to positions {int(pos.max())}", flush=True)
    verify_t = int(os.environ.get("GLM53_BITS_VERIFY_T", "0"))
    if verify_t:
        # the verify step of T tokens a user (`forward_verify`, DFlash2's, eager) from fresh states: positions 0 ..,
        # random tokens and keeps (the drafts the last step accepted, 0 .. T - 1); the ids and logits each step
        gv = torch.Generator().manual_seed(31)
        vstates = model.empty_state(users, spec=verify_t - 1)
        vio = model.alloc_spec_io(users, drafts=verify_t - 1, mtp=False)
        pos = torch.full((ROWS, users), verify_pos, dtype=torch.int64)
        keep = torch.full((ROWS, users), verify_t - 1, dtype=torch.int64)
        got = {"verify_ids": [], "verify_logits": []}
        for _ in range(max(spec_steps, 4)):
            toks = torch.randint(0, vocab, (ROWS, verify_t * users), generator=gv)
            model.write_spec_inputs(vio, toks, torch.cat([pos + j for j in range(verify_t)], dim=1), keep)
            v_ids, v_lg = model.forward_verify(vio, vstates, logits=True)
            got["verify_ids"].append(ttnn.to_torch(ttnn.get_device_tensors(v_ids)[0]).float().clone())
            got["verify_logits"].append(model.logits_to_host(v_lg, verify_t * users).float().clone())
            for t in (v_ids, v_lg):
                ttnn.deallocate(t)
            keep = torch.randint(0, verify_t, (ROWS, users), generator=gv)
            pos = pos + keep + 1
        for st in vstates:
            free = getattr(st, "deallocate", None)
            if free is not None:
                free()
        for k, v in got.items():
            out[k] = torch.stack(v)
        print(f"verify steps (T = {verify_t}): to positions {int(pos.max())}", flush=True)
    for k, v in out.items():
        print(f"{k}: {tuple(v.shape)}, finite {bool(torch.isfinite(v.float()).all())}", flush=True)

    path = os.environ.get("GLM53_BITS_OUT")
    if path:
        torch.save(out, path)
        print(f"saved {path}", flush=True)
    ref_path = os.environ.get("GLM53_BITS_REF")
    if ref_path:
        ref = torch.load(ref_path)
        bad = []
        for k, v in ref.items():
            if k.startswith("decode_") and k in out:  # the slots with a request
                v = v[active] if k == "decode_logits" else v[:, active]
                out[k] = out[k][active] if k == "decode_logits" else out[k][:, active]
            same = k in out and same_bits(out[k], v)
            if not same:
                d = (out[k].float() - v.float()).abs() if k in out and out[k].shape == v.shape else None
                worst = float(d[torch.isfinite(d)].max()) if d is not None and torch.isfinite(d).any() else None
                bad.append((k, worst))
            extra = ""
            if not same and k in out and out[k].shape == v.shape:  # how far: logits' max |d| / max |ref|, ids equal
                a, b = out[k].float(), v.float()
                if k.endswith("logits"):
                    rel = float((a - b).abs().max() / b.abs().max().clamp_min(1e-9))
                    top = float((a.argmax(-1) == b.argmax(-1)).float().mean())
                    extra = f" (max |d| / max |ref| {rel:.3e}, argmax equal {top:.3f})"
                else:
                    extra = f" (equal {float((a == b).float().mean()):.3f})"
            print(f"{k}: {'bit for bit' if same else 'DIFFERENT'} against {ref_path}{extra}", flush=True)
        print(f"BITS RESULT: {'all bit for bit' if not bad else f'DIFFERENT {bad}'}", flush=True)
        if os.environ.get("GLM53_BITS_PRINT_ONLY") != "1":  # (print only: a study's run ends cleanly)
            assert not bad, bad
    if os.environ.get("GLM53_BITS_PRINT_ONLY") == "1":
        print(f"BITS REPEATS: {'all the same' if not differ else f'DIFFERENT {differ}'}", flush=True)
        return
    assert not differ, differ
