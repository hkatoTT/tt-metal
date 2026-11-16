"""The long cache's decode with the split sparse attention (`dsa.SPLIT_K`, `TtDsa._split_attend`) against
sparse_sdpa, through the generator as serving runs it: a truncated model (GLM53_SPLIT_LAYERS, default 5: layer 3 the
DSA one) with the MTP layer, synthetic weights, the long cache (GLM53_SPLIT_CACHE, default 9,728), every slot
prefilled with its own random prompt of GLM53_SPLIT_LEN (default 8,192) tokens, then GLM53_SPLIT_STEPS (default 16)
plain greedy decode steps (traced, logits) and as many verify steps (MTP, one draft, traced).  Each form in turn (the
traces released and captured again, the slots prefilled again): the first step's logits (max |d| against the
largest |logit|, the top-1 per slot) and how many tokens each slot's text keeps equal.

    GLM53_SPLIT_FORMS = the forms in turn, comma separated, each a +-joined set of switches (default "0,K8"):
        0 (sparse_sdpa, the rest off), K<n> (dsa.SPLIT_K = n), P (dsa_indexer.SELECT_PARTS), W (dsa.TRIM_WRITES); the
        first form is the reference (the inputs always packed)
    GLM53_SPLIT_TF=1: the other forms teacher forced: fed the reference's tokens (the plain step) and its inputs (the
        verify step) at every step, each step's choices compared with the reference's: the plain step's flips with the
        reference's margin between the two tokens (its logits), against the margins of its top two at every step
    GLM53_SPLIT_DUMP=path: every form's outputs saved (the plain step's logits' and the prefills' sha1, the texts, the
        verify step's ids) to compare two runs; the form D: the modules' defaults as loaded
    GLM53_SPLIT_SNAP=k: slot k's MTP layer state (its row-major latents, rkey / rgate page, pool pages) and first draft
        after the verify part's prefill and after its first verify steps, compared with the first form's (a slot used
        again: what it carries over); GLM53_SPLIT_RESET=mtp / all: the MTP layer's (or every layer's) state zeroed
        before the verify part's prefill; GLM53_SPLIT_SP_CHECK=1: at the end, slot 0's prompt alone over the rows and on
        its row: the kept hidden (MTP's first drafts' input) and the first draft compared
Run (a device test): GLM53_DEVICE_TEST=1 GLM53_MTP=1 GLM53_SYNTHETIC_WEIGHTS=1 pytest ... -s
"""

from __future__ import annotations

import hashlib
import os
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.checkpoint import ShapeOnlyCheckpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import dsa as dsa_module  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import dsa_indexer as dsa_indexer_module  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import COLS, DEVICE_PARAMS_TRACE, MESH_SHAPE  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

USERS = 4


@pytest.mark.timeout(1400)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_dsa_split_model(mesh_device):
    layers = int(os.environ.get("GLM53_SPLIT_LAYERS", "5"))
    cache = int(os.environ.get("GLM53_SPLIT_CACHE", "9728"))
    n = int(os.environ.get("GLM53_SPLIT_LEN", "8192"))
    steps = int(os.environ.get("GLM53_SPLIT_STEPS", "16"))
    forms = os.environ.get("GLM53_SPLIT_FORMS", "0,K8").split(",")
    forced = os.environ.get("GLM53_SPLIT_TF") == "1"

    def apply(form: str) -> None:
        if form == "D":  # (the modules' defaults as loaded)
            return
        sw = set(form.split("+"))
        dsa_module.SPLIT_K = next((int(x[1:]) for x in sw if x.startswith("K")), 0)
        dsa_indexer_module.SELECT_PARTS = "P" in sw
        dsa_module.TRIM_WRITES = "W" in sw

    model = TtGlm53Model(mesh_device, ShapeOnlyCheckpoint(), num_layers=layers, max_seq=cache, mtp=True)
    gen = Glm53Generator(model, USERS, chunk=(128, 512), spec=1)
    slots = gen.slots
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(11)
    prompts = [torch.randint(0, vocab, (n,), generator=g).tolist() for _ in range(slots)]
    results = {}  # by the form's index (a form may come twice: the test's own control)
    dump, hashes, prefill_sha = {}, [], []  # GLM53_SPLIT_DUMP
    snap_slot = os.environ.get("GLM53_SPLIT_SNAP")
    reset = os.environ.get("GLM53_SPLIT_RESET", "")
    snaps = {}

    def snapshot(fi, tag, first_tokens=None):  # slot snap_slot's MTP layer state, on its row's first chip
        if snap_slot is None:
            return
        r, u = divmod(int(snap_slot), USERS)
        st = gen.states[len(model.layers)]
        chip = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[r * COLS])  # noqa: E731
        pp = (st.pools.shape[0] - 1) // USERS
        rm = chip(st.rm).reshape(-1, st.rm.shape[-1])
        snaps.setdefault(fi, {})[tag] = dict(
            rm=rm[u * cache : u * cache + n + 2 * steps + 4].clone(),
            rkey=chip(st.rkey)[u].clone(),
            rgate=chip(st.rgate)[u].clone(),
            pools=chip(st.pools)[u * pp : (u + 1) * pp].clone(),
            first=gen.first.get(int(snap_slot)),
            first_all=dict(gen.first),
            tokens=list(first_tokens) if first_tokens is not None else None,
            fio_hidden=chip(model._first[USERS].hidden).float().clone() if USERS in model._first else None,
            kept=chip(model._prefill_hidden[0]).float().clone() if getattr(model, "_prefill_hidden", None) else None,
        )
        if fi:
            a, b = snaps[0][tag], snaps[fi][tag]
            if a["tokens"] is not None:
                drafts = [k for k in a["first_all"] if a["first_all"][k] != b["first_all"].get(k)]
                tok = [k for k in range(len(a["tokens"])) if a["tokens"][k] != b["tokens"][k]]
                dh = (a["fio_hidden"] - b["fio_hidden"]).abs().max().item() if a["fio_hidden"] is not None else None
                dk = (a["kept"] - b["kept"]).abs().max().item() if a["kept"] is not None else None
                print(
                    f"  {tag}: the slots whose first draft differs {drafts}; first tokens differ at {tok}; slot "
                    f"{snap_slot}'s row: FirstIO.hidden max |d| {dh}, the kept prefill hidden max |d| {dk}",
                    flush=True,
                )
            rows = (a["rm"] != b["rm"]).any(-1).nonzero().flatten().tolist()
            pr = (a["pools"] != b["pools"]).any(-1).reshape(-1).nonzero().flatten().tolist()
            kr = (a["rkey"] != b["rkey"]).any(-1).reshape(-1).nonzero().flatten().tolist()
            gr = (a["rgate"] != b["rgate"]).any(-1).reshape(-1).nonzero().flatten().tolist()
            print(
                f"  slot {snap_slot} {tag}: first draft {a['first']} / {b['first']}; latent rows differing (positions; "
                f"the prompt {n}) {rows[:12]}{'..' if len(rows) > 12 else ''} ({len(rows)}); rkey rows {kr}, rgate "
                f"rows {gr}; pool rows (page * 32 + row) {pr[:12]} ({len(pr)})",
                flush=True,
            )

    def zero_states():
        picked = gen.states[len(model.layers) :] if reset == "mtp" else gen.states
        for st in picked:
            for name in ("keys", "values", "rm", "pools", "rkey", "rgate", "recurrent"):
                t = getattr(st, name, None)
                if t is not None:
                    ttnn.copy(ttnn.zeros_like(t), t)

    for fi, form in enumerate(forms):
        apply(form)
        if results:
            gen.release()
        t0 = time.time()
        gen.warmup(greedy=False, logits=True, traced=(128, 512))
        warm = time.time() - t0
        # the plain decode (logits)
        gen.first.clear()
        pre = gen.prefill_many(list(enumerate(prompts)))
        prefill_sha.append(hashlib.sha1(torch.stack([x.float() for x in pre]).numpy().tobytes()).hexdigest())
        first = [int(lg.argmax()) for lg in pre]
        snapshot(fi, "after the plain part's prefill", first)
        base = results[0] if results and forced else None
        if base is not None:
            first = [t[0] for t in base["plain"]]
        toks, pos = list(first), [n] * slots
        plain, logits0, flips, top2, logit_d = [[t] for t in first], None, [], [], 0.0
        all_lg = []
        t1 = time.time()
        for step in range(steps):
            lg = gen.decode(torch.tensor(toks), torch.tensor(pos), greedy=False).float()  # [slots, V]
            hashes.append(hashlib.sha1(lg.numpy().tobytes()).hexdigest())
            if step == 0:
                logits0 = lg
            mine = lg.argmax(-1).tolist()
            if base is None:
                if forced:
                    all_lg.append(lg)
                toks = mine
            else:  # teacher forced: the reference's next tokens; its logits at this step for the flips
                ref = base["all_lg"][step]
                logit_d = max(logit_d, (lg - ref).abs().max().item())
                theirs = [base["plain"][k][step + 1] for k in range(slots)]
                for k in range(slots):
                    v2 = ref[k].topk(2).values
                    top2.append((v2[0] - v2[1]).item())
                    if mine[k] != theirs[k]:
                        flips.append((step, k, (ref[k, theirs[k]] - ref[k, mine[k]]).item(), ref[k].std().item()))
                toks = theirs
            pos = [p + 1 for p in pos]
            for k in range(slots):
                plain[k].append(mine[k] if base is None else toks[k])
        plain_s = (time.time() - t1) / steps
        if base is not None:
            gaps = torch.tensor(top2)
            print(
                f"form {form} teacher forced, the plain step: {len(flips)} flips of {steps * slots} choices, logits "
                f"max |d| {logit_d:.2e}; the reference's top-2 margins: median {gaps.median():.2e}, 1st percentile "
                f"{gaps.quantile(0.01):.2e}",
                flush=True,
            )
            for step, k, m, sd in flips:
                rank = (gaps < m).float().mean().item()
                print(
                    f"  flip at step {step} slot {k}: the reference's margin {m:.2e} (logits std {sd:.2e}; "
                    f"{rank:.3f} of all top-2 margins smaller)",
                    flush=True,
                )
        # the verify step (one draft)
        gen.first.clear()
        if reset:
            zero_states()
        pre = gen.prefill_many(list(enumerate(prompts)))
        prefill_sha.append(hashlib.sha1(torch.stack([x.float() for x in pre]).numpy().tobytes()).hexdigest())
        first = [int(lg.argmax()) for lg in pre]
        snapshot(fi, "after the prefill", first)
        out, tok = [[f] for f in first], list(first)
        draft, keep, pos = [gen.first[k] for k in range(slots)], [1] * slots, [n] * slots
        accepted, v_inputs, v_outputs, v_flips = 0, [], [], 0
        t1 = time.time()
        for step in range(steps):
            tokens = torch.tensor([[tok[k], draft[k]] for k in range(slots)])
            positions = torch.tensor([[pos[k], pos[k] + 1] for k in range(slots)])
            keep_t = torch.tensor(keep)
            if base is not None:  # teacher forced: the reference's inputs
                tokens, positions, keep_t = base["v_inputs"][step]
            ids, drafts = gen.verify(tokens, positions, keep_t)
            if step < 2:
                snapshot(fi, f"after verify step {step}")
            v_inputs.append((tokens, positions, keep_t))
            v_outputs.append((ids.clone(), drafts.clone()))
            if base is not None:
                r_ids, r_drafts = base["v_outputs"][step]
                v_flips += int((ids != r_ids).sum()) + int((drafts != r_drafts).sum())
                for name, a_, b_ in (("ids", ids, r_ids), ("drafts", drafts, r_drafts)):
                    for k, j in (a_ != b_).nonzero().tolist():
                        print(
                            f"  verify step {step}: slot {k} {name}[{j}] {int(a_[k, j])} (reference "
                            f"{int(b_[k, j])})",
                            flush=True,
                        )
            for k in range(slots):
                gt, gd = int(ids[k, 0]), int(ids[k, 1])
                a = int(draft[k] == gt)
                accepted += a
                out[k] += [draft[k], gd] if a else [gt]
                keep[k], pos[k], tok[k] = a, pos[k] + a + 1, gd if a else gt
                draft[k] = int(drafts[k, a])
        verify_s = (time.time() - t1) / steps
        if base is not None:
            print(
                f"form {form} teacher forced, the verify step: {v_flips} of {steps * slots * 4} ids (the greedy ids "
                f"after the token and the draft, the MTP drafts from each) differ from the reference's",
                flush=True,
            )
        dump.setdefault("forms", {})[fi] = dict(
            form=form,
            plain=plain,
            verify=out,
            plain_logits_sha=list(hashes),
            v_outputs=[(a.clone(), b.clone()) for a, b in v_outputs],
            prefill_sha=list(prefill_sha),
        )
        hashes.clear()
        prefill_sha.clear()
        results[fi] = dict(
            plain=plain,
            verify=out,
            logits0=logits0,
            accepted=accepted / (steps * slots),
            all_lg=all_lg,
            v_inputs=v_inputs,
            v_outputs=v_outputs,
        )
        print(
            f"form {form}: warm-up {warm:.0f} s, plain step {plain_s * 1e3:.1f} ms (host clock), verify step "
            f"{verify_s * 1e3:.1f} ms, acceptance {accepted / (steps * slots):.3f}",
            flush=True,
        )

    def same(a, b):
        m = 0
        while m < min(len(a), len(b)) and a[m] == b[m]:
            m += 1
        return m

    base = results[0]
    for fi, form in list(enumerate(forms))[1:]:
        r = results[fi]
        d = (r["logits0"] - base["logits0"]).abs().max().item()
        top = base["logits0"].abs().max().item()
        top1 = (r["logits0"].argmax(-1) == base["logits0"].argmax(-1)).float().mean().item()
        p_same = [same(a, b) for a, b in zip(r["plain"], base["plain"])]
        v_same = [same(a, b) for a, b in zip(r["verify"], base["verify"])]
        print(
            f"form {form} against {forms[0]}: first step logits max |d| {d:.4f} ({d / top:.2e} of the largest), "
            f"top-1 the same {top1:.3f}; plain text equal for {sorted(p_same)}, verify text equal for "
            f"{sorted(v_same)}",
            flush=True,
        )
    if os.environ.get("GLM53_SPLIT_SP_CHECK") == "1":  # the kept hidden after a sequence-parallel prefill (replayed)
        got = {}
        for sp in (True, False):
            gen.sp = sp
            gen.first.clear()
            lg = gen.prefill_many([(0, prompts[0])])
            hid = ttnn.to_torch(ttnn.get_device_tensors(model._prefill_hidden[0])[0]).float().flatten()
            got[sp] = (int(lg[0].argmax()), gen.first.get(0), hid)
            dump.setdefault("sp_check", {})[sp] = (
                int(lg[0].argmax()),
                gen.first.get(0),
                hashlib.sha1(lg[0].float().numpy().tobytes()).hexdigest(),
            )
        (t1, d1, h1), (t0, d0, h0) = got[True], got[False]
        rel = ((h1 - h0).abs().max() / h0.abs().max()).item()
        print(
            f"slot 0 alone over the rows (sequence parallel) / on its row: first token {t1} / {t0}, first draft "
            f"{d1} / {d0}, the kept hidden's max |d| {rel:.2e} of its largest",
            flush=True,
        )
        gen.sp = True
    if os.environ.get("GLM53_SPLIT_DUMP"):  # every form's outputs (to compare two runs bit for bit)
        torch.save(dump, os.environ["GLM53_SPLIT_DUMP"])
        print(f"wrote {os.environ['GLM53_SPLIT_DUMP']}", flush=True)
    gen.release()
