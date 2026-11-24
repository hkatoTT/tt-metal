"""Long contexts on the 8x4 mesh (the DSA indexer): a passkey retrieval per mesh row.

Each row's prompt is filler text of a given length with a passphrase sentence at a given depth, ending in a question
whose answer is the passphrase; the generator prefills the 8 prompts in one wave (chunks of 512) and decodes greedily.
The test reports, per row, whether the passphrase comes back, and the prefill and decode times.  Environment:
    GLM53_DEVICE_TEST=1 (required), GLM53_WEIGHT_DTYPE (bfp8 as the other real-weight tests),
    GLM53_LONG_CACHE (8,192 tokens), GLM53_LONG_LENGTHS (8 prompt lengths, one a row), GLM53_LONG_TOKENS (24),
    GLM53_SYNTHETIC_WEIGHTS=1 (random weights: the flow only), GLM53_LONG_WARMUP_TOKENS (the warm-up's prompt, default
    the whole cache; the wave then compiles what the warm-up did not), GLM53_LONG_FORMS (default one form, today's):
    the DSA switches in turn as tests/test_dsa_split_model.py (0, K<n>, P, W joined by +; D: the modules' own
    defaults; a part "module.NAME=value" sets that switch for the form, e.g. D+dsa.PREFILL_SPARSE_KCHUNK=256) on
    MTP's verify
    step (the model with its MTP layer, one draft): the first form decodes on its own outputs (its passkeys), the others
    teacher forced (fed its inputs each step), each step eager with the logits; each id that differs from the first
    form's (the model's after the token and after the draft, the MTP layer's from each) with the first form's margin
    between the two ids in its logits there and its top-2 margin.  GLM53_LONG_REF (a reference file of
    tools/sp_reference.py --tokens N, e.g. data_dir("sp-v2", "teacher.pt")): with the forms, its prompts instead
    (prompt r % 3 on row r) and every form teacher forced on the reference's tokens through MTP's verify step (the token
    and the next as its draft, kept), the KLD of each position's logits from the reference's and the top-1 agreement
    With the forms, each teacher-forced form also reports the KLD of the model's logits (after the token and after the
    draft) from the first form's at the same inputs and their top-1 agreement (a KV cache precision study: the first
    form bf16, the others its cache rounded, e.g. D+dsa.KV_ROUND_LATENT=8, or stored in FP8, D+dsa.KV_LATENT_FP8=1, or
    the pools in BFP, D+dsa.POOLS_DTYPE=bfp8: the generator's states made again); GLM53_LONG_OWN=1 also runs each later
    form on its own outputs (its passkeys and acceptance), GLM53_LONG_OWN=last the last form only,
    GLM53_LONG_OWN_FORMS=i,j those forms (0-based).  A form's "module.NAME=value" of a string switch sets the string
    (e.g. dsa_indexer.FP8_K_IN=bf16).  GLM53_LONG_SAVE=prefix: each form's model logits at the first form's inputs
    saved to prefix-<i>.pt ([steps, rows, 2 (after the token, after the draft), V]); GLM53_LONG_BUDGET (seconds from
    the start): a form not begun by then is skipped.
Run: pytest models/demos/blackhole/glm53_flash/tests/test_model_device_long.py -s
"""

from __future__ import annotations

import gc
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

from models.demos.blackhole.glm53_flash.checkpoint import (  # noqa: E402
    Glm53Checkpoint,
    ShapeOnlyCheckpoint,
    checkpoint_root,
)
from models.demos.blackhole.glm53_flash.paths import out_dir  # noqa: E402
from models.demos.blackhole.glm53_flash.tools.passkey import DEPTHS, WORDS, prompt_ids  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_model_device_long(mesh_device):
    from tokenizers import Tokenizer

    cache = int(os.environ.get("GLM53_LONG_CACHE", "8192"))
    lengths = [
        int(x) for x in os.environ.get("GLM53_LONG_LENGTHS", "1000,2500,3500,4500,5500,6500,7500,8100").split(",")
    ]
    steps = int(os.environ.get("GLM53_LONG_TOKENS", "24"))
    assert len(lengths) == ROWS and max(lengths) + steps <= cache
    tok = Tokenizer.from_file(str(Path(checkpoint_root()) / "tokenizer.json"))
    g = torch.Generator().manual_seed(5)
    answers = [f"{WORDS[r]}-{int(torch.randint(1000, 9999, (1,), generator=g))}" for r in range(ROWS)]
    prompts = [prompt_ids(tok, n, DEPTHS[r], *answers[r].split("-")) for r, n in enumerate(lengths)]
    ref_path = os.environ.get("GLM53_LONG_REF")
    reference = None
    if ref_path:  # the reference's prompts, prompt r % k on row r
        from models.demos.blackhole.glm53_flash.tools.passkey import sp_prompts

        reference = torch.load(ref_path)
        if "prompts" in reference:  # (tools/sp_reference.py --lengths: its own prompts)
            ref_answers, ref_prompts = reference["answers"], reference["prompts"]
        else:
            ref_answers, ref_prompts = sp_prompts(tok)
        assert [len(p) for p in ref_prompts] == reference["lengths"], "the reference's prompts differ"
        k = len(ref_prompts)
        answers = [ref_answers[r % k] for r in range(ROWS)]
        prompts = [ref_prompts[r % k] for r in range(ROWS)]
    print(f"prompt lengths {[len(p) for p in prompts]}", flush=True)
    result = {"cache": cache, "lengths": [len(p) for p in prompts], "depths": DEPTHS, "answers": answers}

    t0 = time.time()
    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1"  # the flow only (the text is noise)
    forms = [f for f in os.environ.get("GLM53_LONG_FORMS", "").split(",") if f]
    model = TtGlm53Model(
        mesh_device, ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint(), max_seq=cache, mtp=bool(forms) or None
    )
    result["load_seconds"] = round(time.time() - t0)
    print(f"loaded in {result['load_seconds']} s", flush=True)
    gen = Glm53Generator(model, 1, chunk=(512,), spec=1 if forms else 0)
    warm = int(os.environ.get("GLM53_LONG_WARMUP_TOKENS", "0")) or None  # default: the whole cache
    if forms:
        result["forms"] = {}
        base = None
        from models.demos.blackhole.glm53_flash.ttnn import dsa as dsa_module

        # (the generator's states made with these: the long cache's latents' and pools' storage forms)
        store_now = (dsa_module.KV_LATENT_FP8, dsa_module.POOLS_DTYPE, dsa_module.SHARD_KV)
        budget = float(os.environ.get("GLM53_LONG_BUDGET", "0")) or None
        save = os.environ.get("GLM53_LONG_SAVE")
        own_forms = {int(x) for x in os.environ.get("GLM53_LONG_OWN_FORMS", "").split(",") if x}
        for i, form in enumerate(forms):
            if budget is not None and time.time() - t0 > budget:
                print(
                    f"form {form}: skipped, {time.time() - t0:.0f} s past the start (budget {budget:.0f} s)", flush=True
                )
                continue
            if i:
                gen.release()
            _apply(form, gen.states)
            # (SHARD_KV: the cache sharded over the row's 4 chips)
            store = (dsa_module.KV_LATENT_FP8, dsa_module.POOLS_DTYPE, dsa_module.SHARD_KV)
            if store != store_now:  # a form storing the long cache's latents or pools otherwise (read when the states
                # are made, `TtDsa.empty_state`): the generator's states made again, the old ones freed first
                gen.states = None
                gc.collect()
                gen.states = model.empty_state(gen.users, spec=gen.spec or gen.dk)
                store_now = store
                print(
                    f"form {form}: the states made again (latents in {'FP8' if store[0] else 'bf16'}, pools in "
                    f"{store[1]}, the latents split over the row's chips from {store[2]})",
                    flush=True,
                )
            gen.warmup(greedy=True, logits=False, chunks=(512,), tokens=warm, traced=())
            if reference is not None:
                got = _reference_form(gen, prompts, reference)
                print(
                    f"form {form} against the reference: mean KLD {got['kld']:.5f} (by prompt "
                    f"{' / '.join(f'{x:.5f}' for x in got['kld_by_prompt'])}), top-1 {got['top1']:.4f}, "
                    f"{got['positions']} positions",
                    flush=True,
                )
                result["forms"][form] = got
                continue
            got = _verify_form(gen, tok, prompts, answers, steps, base)
            if save:  # the model's logits [steps, rows, 2, V] (bf16 when that holds them exactly)
                lg = torch.stack([x[:, 0] for x in got["logits"]])
                if torch.equal(lg, lg.bfloat16().float()):
                    lg = lg.bfloat16()
                torch.save({"form": form, "logits": lg}, f"{save}-{i}.pt")
                print(f"form {form}: logits {tuple(lg.shape)} {lg.dtype} saved to {save}-{i}.pt", flush=True)
            if base is None:
                print(
                    f"form {form}: passkeys {got['found']} of {ROWS} (its own verify steps), acceptance "
                    f"{got['accepted']:.3f}",
                    flush=True,
                )
            else:
                ms = [d["margin"] for d in got["diffs"]]
                print(
                    f"form {form} teacher forced: {len(ms)} of {got['n_ids']} ids differ, logits max |d| model "
                    f"{got['logit_d'][0]:.3e} / MTP {got['logit_d'][1]:.3e} (largest |logit| {got['top']:.1f}); "
                    f"margins at the differing ids: max {max(ms, default=0.0):.4f}, over 0.05: "
                    f"{sum(m > 0.05 for m in ms)}; the first form's top-2 margins: median {got['gaps'][0]:.3f}, "
                    f"5th percentile {got['gaps'][1]:.4f}",
                    flush=True,
                )
                for d in got["diffs"]:
                    print(
                        f"  step {d['step']} row {d['row']} {d['which']}: first form {d['ref_id']} / this "
                        f"{d['id']}, margin {d['margin']:.4f} (its top-2 margin {d['top2']:.4f})",
                        flush=True,
                    )
            result["forms"][form] = {k: v for k, v in got.items() if k not in ("logits", "inputs", "outputs")}
            if base is None:
                base = got
            else:
                print(
                    f"form {form} against the first form: mean KLD {got['kld']:.6f} (by row "
                    f"{' / '.join(f'{x:.5f}' for x in got['kld_by_row'])}; 95 % over rows {got['kld_ci'][0]:.6f} .. "
                    f"{got['kld_ci'][1]:.6f}), p99 {got['kld_p99']:.5f}, top-1 {got['top1']:.4f}, {got['positions']} "
                    f"positions",
                    flush=True,
                )
                own_env = os.environ.get("GLM53_LONG_OWN", "0")
                if own_env == "1" or (own_env == "last" and i == len(forms) - 1) or i in own_forms:
                    own = _verify_form(gen, tok, prompts, answers, steps, None)
                    print(
                        f"form {form} on its own outputs: passkeys {own['found']} of {ROWS}, acceptance "
                        f"{own['accepted']:.3f}",
                        flush=True,
                    )
                    result["forms"][form]["own"] = {k: own[k] for k in ("found", "accepted", "texts")}
        gen.release()
        path = out_dir() / f"long-forms-{cache}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
        path.write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {path}", flush=True)
        return
    result["warmup_seconds"] = gen.warmup(greedy=True, logits=False, chunks=(512,), tokens=warm, traced=())
    print(f"warm-up {result['warmup_seconds']}", flush=True)
    dram = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)  # (one chip's banks; every chip the same)
    banks = int(dram.num_banks)
    result["dram_gib"] = [
        round(int(dram.total_bytes_allocated_per_bank) * banks / 2**30, 2),
        round(int(dram.total_bytes_free_per_bank) * banks / 2**30, 2),
    ]
    print(
        f"cache {cache}: DRAM a chip {result['dram_gib'][0]} GiB allocated, {result['dram_gib'][1]} GiB free",
        flush=True,
    )

    t1 = time.time()
    logits = gen.prefill_many([(r, p) for r, p in enumerate(prompts)], chunk=512, traced=False)
    result["prefill_seconds"] = round(time.time() - t1, 2)
    print(f"prefill of {sum(len(p) for p in prompts)} tokens in one wave: {result['prefill_seconds']} s", flush=True)
    tokens = torch.tensor([int(lg.argmax()) for lg in logits], dtype=torch.int64)
    positions = torch.tensor([len(p) for p in prompts], dtype=torch.int64)
    generated = [[int(t)] for t in tokens]
    vocab = model.config.vocab_size
    t2 = time.time()
    for _ in range(steps - 1):
        tokens = gen.decode(tokens, positions, greedy=True)
        assert bool(((tokens >= 0) & (tokens < vocab)).all()), "decode ids out of the vocabulary"
        positions += 1
        for r in range(ROWS):
            generated[r].append(int(tokens[r]))
    result["decode_seconds"] = round(time.time() - t2, 2)
    found = 0
    rows = []
    for r in range(ROWS):
        text = tok.decode(generated[r])
        hit = answers[r] in text.replace(" ", "")
        found += hit
        rows.append({"length": len(prompts[r]), "depth": DEPTHS[r], "answer": answers[r], "text": text, "found": hit})
        print(
            f"row {r}: {len(prompts[r])} tokens, passphrase at {DEPTHS[r]:.2f}: {'found' if hit else 'MISSED'} "
            f"({answers[r]}) -> {text!r}",
            flush=True,
        )
    result["rows"] = rows
    result["found"] = found
    print(
        f"passkey retrieval: {found} of {ROWS} rows; decode {steps - 1} steps in {result['decode_seconds']} s",
        flush=True,
    )
    gen.release()
    path = out_dir() / f"long-{cache}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {path}", flush=True)


_DEFAULTS = {}  # the switches' values before the first form (D)
_UNDO: list = []  # the restores of the current form's module switches ("module.NAME=value")


def _apply(form: str, states=None) -> None:
    """The DSA switches of a form (as tests/test_dsa_split_model.py; D or none of them: the modules' defaults)."""
    from models.demos.blackhole.glm53_flash.ttnn import dsa as dsa_module
    from models.demos.blackhole.glm53_flash.ttnn import dsa_indexer as dsa_indexer_module

    dsa_names = ("SPLIT_K", "TRIM_WRITES")
    if not _DEFAULTS:
        _DEFAULTS.update({n: getattr(dsa_module, n) for n in dsa_names})
        _DEFAULTS["SELECT_PARTS"] = dsa_indexer_module.SELECT_PARTS
    sw = set(form.split("+"))
    # module switches by name (`common.apply_setting`: "module.NAME=value", e.g. dsa.PREFILL_SPARSE_KCHUNK=256), the
    # previous form's undone first
    from models.demos.blackhole.glm53_flash.ttnn.common import apply_setting

    while _UNDO:
        _UNDO.pop()()
    for x in sorted(x for x in sw if "=" in x):
        _UNDO.append(_setting(x, apply_setting))
    sw = {x for x in sw if "=" not in x}
    if sw <= {"D"}:
        for n in dsa_names:
            setattr(dsa_module, n, _DEFAULTS[n])
        dsa_indexer_module.SELECT_PARTS = _DEFAULTS["SELECT_PARTS"]
        return
    dsa_module.SPLIT_K = next((int(x[1:]) for x in sw if x.startswith("K")), 0)
    dsa_indexer_module.SELECT_PARTS = "P" in sw
    dsa_module.TRIM_WRITES = "W" in sw


def _setting(x: str, apply_setting):
    """`common.apply_setting` of one "module.NAME=value"; a string switch (e.g. dsa_indexer.FP8_K_IN=bf16) set as the
    string -> its restore."""
    import importlib

    target, value = x.split("=")
    mod_name, name = target.rsplit(".", 1)
    mod = importlib.import_module(f"models.demos.blackhole.glm53_flash.ttnn.{mod_name}")
    keep = getattr(mod, name)
    if isinstance(keep, str):
        setattr(mod, name, value)
        return lambda: setattr(mod, name, keep)
    return apply_setting(x)


WHICH = ("model after the token", "model after the draft", "MTP from the token", "MTP from the draft")


def _verify_form(gen, tok, prompts, answers, steps: int, base) -> dict:
    """GLM53_LONG_FORMS: the prompts prefilled, then `steps` verify steps (one draft) eager with the logits: on its own
    outputs (`base` None: the committed text, its passkeys), or fed `base`'s inputs, each id compared with base's."""
    model = gen.model
    out = {}
    logits = gen.prefill_many([(r, p) for r, p in enumerate(prompts)], chunk=512, traced=False)
    tok_now = [int(lg.argmax()) for lg in logits]
    draft = [gen.first[r] for r in range(ROWS)]
    keep, pos = [1] * ROWS, [len(p) for p in prompts]
    text = [[t] for t in tok_now]
    inputs, outputs, all_lg, diffs, gaps, accepted = [], [], [], [], [], 0
    klds, agree = [], []  # (a teacher-forced form: its model logits against base's)
    logit_d = [0.0, 0.0]
    for step in range(steps):
        if base is None:
            tokens = torch.tensor([[tok_now[r], draft[r]] for r in range(ROWS)])
            positions = torch.tensor([[pos[r], pos[r] + 1] for r in range(ROWS)])
            keep_t = torch.tensor(keep)
        else:
            tokens, positions, keep_t = base["inputs"][step]
        inputs.append((tokens, positions, keep_t))
        model.write_spec_inputs(gen.sio, tokens, positions, keep_t.view(ROWS, 1).float())
        ids_d, drafts_d, lg_m, lg_d = model.forward_spec(gen.sio, gen.states, logits=True)
        ids, drafts = model.ids_to_host(ids_d, 2), model.ids_to_host(drafts_d, 2)  # [ROWS, 2]: at t, at d
        lg = torch.stack([model.logits_to_host(lg_m, 2), model.logits_to_host(lg_d, 2)], dim=1).float()  # [R, 2, 2, V]
        for t in (ids_d, drafts_d, lg_m, lg_d):
            ttnn.deallocate(t)
        mine = torch.stack([ids, drafts], dim=1)  # [ROWS, 2 (model, MTP), 2 (at t, at d)]
        outputs.append(mine)
        all_lg.append(lg)
        # the device's ids against the argmax of its own logits (the pairing of ids and logits rows)
        own_ok = int((lg.argmax(-1) == mine).sum())
        if base is not None:
            ref, ref_ids = base["logits"][step], base["outputs"][step]
            for h in range(2):
                logit_d[h] = max(logit_d[h], (lg[:, h] - ref[:, h]).abs().max().item())
            lp = torch.log_softmax(ref[:, 0], -1)  # the model's logits [ROWS, 2 (at t, at d), V]
            kl = (lp.exp() * (lp - torch.log_softmax(lg[:, 0], -1))).sum(-1)  # [ROWS, 2]
            klds.append(kl)
            agree.append(ref[:, 0].argmax(-1) == lg[:, 0].argmax(-1))
            d = (lg - ref).abs().amax(-1)  # [ROWS, 2, 2]
            print(
                f"  step {step}: logits max |d| by row (model t, model d, MTP t, MTP d): "
                + " ".join("/".join(f"{d[r, h, j]:.2g}" for h in range(2) for j in range(2)) for r in range(ROWS))
                + f"; ids = argmax of its logits {own_ok} / {ROWS * 4}",
                flush=True,
            )
            for r in range(ROWS):
                for h in range(2):
                    for j in range(2):
                        v2 = ref[r, h, j].topk(2).values
                        gaps.append((v2[0] - v2[1]).item())
                        a, b = int(ref_ids[r, h, j]), int(mine[r, h, j])
                        if a != b:
                            diffs.append(
                                dict(
                                    step=step,
                                    row=r,
                                    which=WHICH[2 * h + j],
                                    ref_id=a,
                                    id=b,
                                    margin=(ref[r, h, j, a] - ref[r, h, j, b]).item(),
                                    top2=gaps[-1],
                                )
                            )
        else:
            print(
                f"  step {step} (first form): ids = argmax of its logits {own_ok} / {ROWS * 4}, keep "
                f"{keep_t.tolist()}",
                flush=True,
            )
            for r in range(ROWS):
                gt, gd = int(ids[r, 0]), int(ids[r, 1])
                a = int(draft[r] == gt)
                accepted += a
                text[r] += [draft[r], gd] if a else [gt]
                keep[r], pos[r], tok_now[r] = a, pos[r] + a + 1, gd if a else gt
                draft[r] = int(drafts[r, a])
    out["inputs"], out["outputs"], out["logits"] = inputs, outputs, all_lg
    if base is None:
        texts = [tok.decode(t) for t in text]
        out["texts"] = texts
        out["found"] = sum(answers[r] in texts[r].replace(" ", "") for r in range(ROWS))
        out["accepted"] = accepted / (steps * ROWS)
    else:
        g = torch.tensor(gaps)
        out["diffs"], out["n_ids"], out["logit_d"] = diffs, steps * ROWS * 4, logit_d
        out["gaps"] = (g.median().item(), g.quantile(0.05).item())
        out["top"] = max(x.abs().max().item() for x in base["logits"])
        kl = torch.stack(klds)  # [steps, ROWS, 2]
        by_row = kl.mean(dim=(0, 2))
        gen_ = torch.Generator().manual_seed(0)
        pick = torch.randint(0, ROWS, (10000, ROWS), generator=gen_)
        boots = by_row[pick].mean(1)
        out["kld"], out["kld_by_row"] = kl.mean().item(), by_row.tolist()
        out["kld_ci"] = [boots.quantile(0.025).item(), boots.quantile(0.975).item()]
        out["kld_p99"], out["positions"] = kl.flatten().quantile(0.99).item(), kl.numel()
        out["top1"] = torch.stack(agree).float().mean().item()
        out["kld_steps"] = kl.tolist()
    return out


def _reference_form(gen, prompts, reference) -> dict:
    """GLM53_LONG_REF: the prompts prefilled, then the reference's tokens fed through the verify step two at a time
    (the token at p, the next as the draft at p + 1, kept), eager with the logits: each position's KLD from the
    reference's logits (after the prompt: the prefill's) and whether its top-1 is the reference's."""
    model = gen.model
    ref_tokens, ref_logits = reference["tokens"], reference["teacher"].float()  # [k][N], [k, N, V]
    k, n = ref_logits.shape[:2]
    pick = lambda r: r % k  # noqa: E731
    klds, top1, by_prompt = [], [], [[] for _ in range(k)]

    def score(r, j, lg):  # the device's logits [V] at reference position j of row r's prompt
        ref = ref_logits[pick(r), j]
        lp, lq = torch.log_softmax(ref, -1), torch.log_softmax(lg.float()[: ref.shape[-1]], -1)
        kld = float((lp.exp() * (lp - lq)).sum())
        klds.append(kld)
        by_prompt[pick(r)].append(kld)
        top1.append(int(lg.argmax()) == int(ref.argmax()))

    logits = gen.prefill_many([(r, p) for r, p in enumerate(prompts)], chunk=512, traced=False)
    for r in range(ROWS):
        score(r, 0, logits[r])
    for i in range((n - 1) // 2):  # feeding tokens 2i, 2i + 1: the logits for positions 2i + 1, 2i + 2
        tokens = torch.tensor([[ref_tokens[pick(r)][2 * i], ref_tokens[pick(r)][2 * i + 1]] for r in range(ROWS)])
        positions = torch.tensor([[len(prompts[r]) + 2 * i, len(prompts[r]) + 2 * i + 1] for r in range(ROWS)])
        model.write_spec_inputs(gen.sio, tokens, positions, torch.ones(ROWS, 1))
        ids_d, drafts_d, lg_m, lg_d = model.forward_spec(gen.sio, gen.states, logits=True)
        lg = model.logits_to_host(lg_m, 2)  # [ROWS, 2, V]
        for t in (ids_d, drafts_d, lg_m, lg_d):
            ttnn.deallocate(t)
        for r in range(ROWS):
            score(r, 2 * i + 1, lg[r, 0])
            score(r, 2 * i + 2, lg[r, 1])
    return dict(
        kld=sum(klds) / len(klds),
        kld_by_prompt=[sum(v) / len(v) for v in by_prompt],
        top1=sum(top1) / len(top1),
        positions=len(klds),
    )
