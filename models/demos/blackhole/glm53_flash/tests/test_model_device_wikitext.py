"""KL divergence on wikitext-2 test against the BF16 CPU reference, as llama.cpp's --kl-divergence measures it.

llama.cpp's --kl-divergence (usually 48 chunks of 1,024) scores the second half of each chunk: the logits at positions ctx / 2 .. ctx - 2
against the reference model's.  The reference here is tools/wikitext_reference.py's (`tt/`, BF16, LM head FP32): the
first chunks of the same text.  On the device each chunk takes a slot (chunk c on mesh row c % 8, user c // 8), its
first half is prefilled (the serving generator, chunks of 512) and its second half fed by the plain decode step
(teacher forcing, the captured logits step), so the scored logits come from decode steps after a prefill, as served.
GLM53_WT_MODE=decode instead feeds every position by decode steps (no prefill).  GLM53_WT_MODE=prefill_all scores the
prefill itself: each chunk's first ctx - 1 tokens prefilled eagerly (the generator's own choice of chunks among
GLM53_WT_CHUNK_SET, default 128,512,1024 as the serving adapter's), the logits at the same scored positions read from
the prefill (`TtGlm53Model.logits_all`: every position's, each chip's quarter of the vocabulary) and scored against the
same reference; GLM53_WT_PREFILL_FORM = sp (default: one chunk at a time, which the generator puts over the 8 rows)
or rows (all 8 at once, one a mesh row), or a comma list (each form in turn within each setting).  Each position from
GLM53_WT_DIGEST_FROM (default: the first scored) gets a fingerprint of its logits saved (fp64 sum, sum of squares,
argmax, max, logsumexp; and each chunk's SHA-1 of its scored logits' bytes), to tell runs apart bit for bit.
GLM53_WT_TAPS = chunks (e.g. 0,2,3; prefill_all only): no scoring; the mean of the streams entering every layer
(`model.taps`) at positions 0..63 and 512..575 of those chunks saved (out_dir()/wikitext-taps-<form>-<stamp>.pt), for
a comparison with a CPU reference's; with GLM53_WT_ROUTER=1 also every MoE layer's router logits at
the same positions (the device's own routing choices).

    GLM53_WT_REF = the reference (default data_dir("wikitext", "reference-8x1024.pt"), tools/wikitext_reference.py's)
    GLM53_WT_CHUNKS = the first n of its chunks (default all; 8 a mesh row's user, so 16 take 2 users a row)
    GLM53_TIME_BUDGET = seconds from the start after which the steps stop (default 690; a partial result is marked)
    GLM53_WT_ROTATE = k: chunk c on mesh row (c + k) % 8 (default 0: whether a chunk's KLD follows its row)
    GLM53_WT_PREFILL_CHUNK = the prefill's chunk length (default 512: the first half in one chunk; 256 or 128: later
        chunks too, the dense cache's later-chunk attention)
    GLM53_WT_MAX_SEQ = the model's cache (default 2,048: the dense DSA; 4,096: the long cache's indexer)
    GLM53_QS_SETTINGS = ";"-separated in-process switches (`common.apply_setting`, "" the defaults), each measured in
        turn on the one load
Prints and saves (out_dir()/wikitext-<stamp>.json, the per-position KLD and top-1 in the .pt beside it):
the mean KLD (natural log, KL(reference || device)) with its standard error over chunks, the 99th percentile, the
share of positions whose top-1 is the reference's, and the perplexity ratio (device / reference) on the text's next
tokens.  The model's precision switches (GLM53_EXPERT_DTYPE, GLM53_EXPERT_LAYERS, ...) apply as usual.
GLM53_SYNTHETIC_WEIGHTS=1: random weights, the flow only.  Run (a device test): GLM53_DEVICE_TEST=1
GLM53_WEIGHT_DTYPE=bfp8 pytest ... -s
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, ShapeOnlyCheckpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.paths import data_dir, out_dir  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_model_device_wikitext(mesh_device):
    start = time.time()
    budget = float(os.environ.get("GLM53_TIME_BUDGET", "690"))
    torch.set_num_threads(16)
    ref = torch.load(os.environ.get("GLM53_WT_REF") or data_dir("wikitext", "reference-8x1024.pt"))
    tokens, ref_logits, first, ctx = ref["tokens"], ref["logits"], ref["first"], ref["ctx"]
    n = int(os.environ.get("GLM53_WT_CHUNKS", str(tokens.shape[0])))
    tokens, ref_logits = tokens[:n], ref_logits[:n]
    users = -(-n // ROWS)
    mode = os.environ.get("GLM53_WT_MODE", "prefill")
    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1"  # (the flow only)
    checkpoint = ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint()
    # GLM53_WT_MAX_SEQ: the cache (past 2,051 tokens the DSA layers take the indexer even for short contexts, e.g. for
    # the indexer's top-k)
    max_seq = int(os.environ.get("GLM53_WT_MAX_SEQ", str(max(2048, 2 * ctx))))
    model = TtGlm53Model(mesh_device, checkpoint, max_seq=max_seq)
    load = time.time() - start
    print(f"loaded in {load:.0f} s; {n} chunks of {ctx}, {users} users a row, mode {mode}", flush=True)
    if mode == "prefill_all":
        _prefill_all(mesh_device, model, tokens, ref_logits, first, ctx, n, users, start, load)
        return
    prefill_chunk = int(os.environ.get("GLM53_WT_PREFILL_CHUNK", "512"))  # (the generator's, warmed up)
    gen = Glm53Generator(model, users, chunk=prefill_chunk)
    t1 = time.time()
    warm = gen.warmup(greedy=False, logits=True, chunks=(prefill_chunk,), tokens=first if mode == "prefill" else 512)
    print(f"warm-up {time.time() - t1:.0f} s: {warm}", flush=True)
    rotate = int(os.environ.get("GLM53_WT_ROTATE", "0"))  # chunk c on mesh row (c + rotate) % 8 (a row's check)
    slots = [((c + rotate) % ROWS) * users + c // ROWS for c in range(n)]
    # GLM53_QS_SETTINGS: ";"-separated in-process switches (`common.apply_setting`, "" the defaults), each measured in
    # turn on the one load (the prefill again from position 0, then the decode); the time budget is shared
    from models.demos.blackhole.glm53_flash.ttnn.common import apply_setting

    finite = True
    for setting_spec in os.environ.get("GLM53_QS_SETTINGS", "").split(";"):
        restore = apply_setting(setting_spec)
        print(f"== setting {setting_spec or 'defaults'}", flush=True)
        start_t = first
        if mode == "prefill":
            gen.prefill_many(
                [(s, tokens[c, :first].tolist()) for c, s in enumerate(slots)], chunk=prefill_chunk, traced=False
            )
        else:
            start_t = 0
        kld = torch.full((n, ctx - 1 - first), float("nan"))
        top1 = torch.zeros(n, ctx - 1 - first)
        nll_dev = torch.zeros(n, ctx - 1 - first)
        nll_ref = torch.zeros(n, ctx - 1 - first)
        steps = 0
        partial = False
        for t in range(start_t, ctx - 1):
            if time.time() - start > budget:
                partial = True
                print(f"time budget ({budget:.0f} s) reached at position {t}", flush=True)
                break
            tok = torch.zeros(gen.slots, dtype=torch.int64)
            pos = torch.full((gen.slots,), -1, dtype=torch.int64)
            for c, s in enumerate(slots):
                tok[s], pos[s] = int(tokens[c, t]), t
            logits = gen.decode(tok, pos, greedy=False)
            steps += 1
            if t < first:
                continue
            j = t - first
            dev = torch.log_softmax(logits[slots].float(), -1)  # [n, V]
            rf = torch.log_softmax(ref_logits[:, j].float(), -1)
            kld[:, j] = (rf.exp() * (rf - dev)).sum(-1)
            top1[:, j] = (dev.argmax(-1) == rf.argmax(-1)).float()
            nxt = tokens[:, t + 1].long()
            nll_dev[:, j] = -dev.gather(1, nxt[:, None])[:, 0]
            nll_ref[:, j] = -rf.gather(1, nxt[:, None])[:, 0]
        done = ~torch.isnan(kld[0])
        k, tp = kld[:, done], top1[:, done]
        per_chunk = k.mean(1)
        result = {
            "setting": setting_spec,
            "prefill_chunk": prefill_chunk,
            "chunks": n,
            "ctx": ctx,
            "mode": mode,
            "rotate": rotate,
            "positions": int(k.numel()),
            "partial": partial,
            "mean_kld": k.mean().item(),
            "kld_stderr_by_chunk": (per_chunk.std() / n**0.5).item() if n > 1 else None,
            "p99_kld": k.flatten().quantile(0.99).item(),
            "same_top1": tp.mean().item(),
            "ppl_ratio": torch.exp(nll_dev[:, done].mean() - nll_ref[:, done].mean()).item(),
            "kld_by_chunk": per_chunk.tolist(),
            "load_seconds": round(load),
            "seconds": round(time.time() - start),
            "precision": {
                k_: os.environ.get(k_) for k_ in os.environ if k_.startswith(("GLM53_EXPERT", "GLM53_WEIGHT"))
            },
        }
        print(
            f"wikitext KLD {result['mean_kld']:.4f} (+/- {result['kld_stderr_by_chunk'] or 0:.4f} by chunk), p99 "
            f"{result['p99_kld']:.3f}, same top-1 {result['same_top1']:.2%}, PPL ratio {result['ppl_ratio']:.4f}; "
            f"{result['positions']} positions{' (partial)' if partial else ''}, {steps} decode steps",
            flush=True,
        )
        path = out_dir() / f"wikitext-{n}x{ctx}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
        path.write_text(json.dumps(result, indent=1) + "\n")
        torch.save({"kld": kld, "top1": top1}, path.with_suffix(".pt"))
        print(f"wrote {path}", flush=True)
        restore()
        finite = finite and bool(torch.isfinite(k).all())
    gen.release()
    assert finite


def _router_windows(model, captured, chunks, windows, owner):
    """The captured router logits (each wave's MoE layers in order) -> {"layers", "logits": {chunk: [MoE layers,
    window positions, E]}}."""
    moe_layers = sorted({i for i, _ in captured})
    per_wave = len(moe_layers)
    waves = model.logits_all
    assert len(captured) == per_wave * len(waves), (len(captured), per_wave, len(waves))
    width = sum(b - a for a, b in windows)
    e = captured[0][1].shape[-1]
    got = {c: torch.full((per_wave, width, e), float("nan")) for c in chunks}
    for w, (_, _, segs) in enumerate(waves):
        for r, off, gpos, cnt, own in segs:
            c = owner[own]
            if c not in got:
                continue
            col = 0
            for a, b in windows:
                aa, bb = max(a, gpos), min(b, gpos + cnt)
                if aa < bb:
                    for j in range(per_wave):
                        layer, rows = captured[w * per_wave + j]
                        assert layer == moe_layers[j], (layer, moe_layers[j])
                        got[c][j, col + aa - a : col + bb - a] = rows[r, off + aa - gpos : off + bb - gpos]
                col += b - a
    return {"layers": moe_layers, "logits": got}


def _save_taps(model, chunks, windows, owner, setting_spec, form, n, ctx, router=None):
    """The taps of a prefill_all setting (`model.taps`: each wave's layers in order, [1, 1, T, D] a chip, the mesh
    rows' own tokens) -> [layers, window positions, D] fp32 per tapped chunk, saved; `model.taps` released."""
    from models.demos.blackhole.glm53_flash.ttnn.common import COLS

    taps, model.taps, model.tap_layers = model.taps, None, frozenset()
    layers = len(model.layers)
    waves = model.logits_all
    assert len(taps) == layers * len(waves), (len(taps), layers, len(waves))
    d = model.config.hidden_size
    width = sum(b - a for a, b in windows)
    got = {c: torch.full((layers, width, d), float("nan")) for c in chunks}
    for w, (_, _, segs) in enumerate(waves):
        for r, off, gpos, cnt, own in segs:
            c = owner[own]
            if c not in got:
                continue
            parts, col = [], 0
            for a, b in windows:
                aa, bb = max(a, gpos), min(b, gpos + cnt)
                if aa < bb:
                    parts.append((aa, bb, col + aa - a))
                col += b - a
            if not parts:
                continue
            for layer in range(layers):
                x = ttnn.to_torch(ttnn.get_device_tensors(taps[w * layers + layer])[r * COLS]).reshape(-1, d)
                for aa, bb, dst in parts:
                    got[c][layer, dst : dst + bb - aa] = x[off + aa - gpos : off + bb - gpos].float()
    for t in taps:
        ttnn.deallocate(t)
    missing = {c: int(torch.isnan(g[:, :, 0]).sum()) for c, g in got.items()}
    assert not any(missing.values()), f"taps: positions without a tap {missing}"
    path = out_dir() / f"wikitext-taps-{form}-{n}x{ctx}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.pt"
    torch.save({"setting": setting_spec, "form": form, "windows": windows, "taps": got, "router": router}, path)
    print(f"wrote {path}: taps of chunks {chunks}, {layers} layers, positions {windows}", flush=True)


def _prefill_all(mesh, model, tokens, ref_logits, first: int, ctx: int, n: int, users: int, start: float, load: float):
    """GLM53_WT_MODE=prefill_all: each setting's eager prefill of every chunk's first ctx - 1 tokens, the logits at
    positions first .. ctx - 2 from the prefill itself (`model.logits_all`), scored as the decode mode scores them."""
    from models.demos.blackhole.glm53_flash.ttnn.common import COLS, apply_setting

    forms = [f for f in os.environ.get("GLM53_WT_PREFILL_FORM", "sp").split(",") if f]
    chunk_set = tuple(int(x) for x in os.environ.get("GLM53_WT_CHUNK_SET", "128,512,1024").split(","))
    gen = Glm53Generator(model, users, chunk=chunk_set)
    slots = [(c % ROWS) * users + c // ROWS for c in range(n)]
    owner = {slot // users: c for c, slot in enumerate(slots)} if users == 1 else None
    assert owner is not None, "prefill_all: one chunk a mesh row (GLM53_WT_CHUNKS <= 8)"
    vocab = model.config.vocab_size
    lo, hi = first, ctx - 1  # the scored positions (the logits at t predict token t + 1)
    dlo = int(os.environ.get("GLM53_WT_DIGEST_FROM", str(first)))  # (fingerprinted from here)
    finite = True
    runs = [(spec, form) for spec in os.environ.get("GLM53_QS_SETTINGS", "").split(";") for form in forms]
    # GLM53_WT_TAPS (chunks, e.g. 0,2,3): the mean of the streams entering every layer (`model.taps`) at positions
    # 0 .. 63 and 512 .. 575 of those chunks, saved (out_dir()/wikitext-taps-<form>-<stamp>.pt) in place of the scoring
    tap_chunks = [int(x) for x in os.environ.get("GLM53_WT_TAPS", "").split(",") if x]
    tap_windows = [(0, 64), (512, 576)]
    # GLM53_WT_ROUTER=1 (with GLM53_WT_TAPS): every MoE layer's router logits ([T, E] fp32, a row's chip 0) at the same
    # positions, saved with the taps (the device's own routing choices)
    router = bool(tap_chunks) and os.environ.get("GLM53_WT_ROUTER") == "1"
    captured_logits = []  # (layer, [ROWS, T, E]) in call order
    if router:
        from models.demos.blackhole.glm53_flash.ttnn.common import COLS
        from models.demos.blackhole.glm53_flash.ttnn.moe import TtMoe

        layer_of = {id(layer.mlp): i for i, layer in enumerate(model.layers) if isinstance(layer.mlp, TtMoe)}
        original = TtMoe.router_logits

        def capture(moe, x_all):
            out = original(moe, x_all)
            i = layer_of.get(id(moe))
            if i is not None and model.taps is not None:  # (only during these prefills)
                chips = ttnn.get_device_tensors(out)
                t, e = out.shape[-2], out.shape[-1]
                rows = [ttnn.to_torch(chips[r * COLS]).reshape(t, e) for r in range(ROWS)]
                captured_logits.append((i, torch.stack(rows)))
            return out

        TtMoe.router_logits = capture
    for setting_spec, form in runs:
        restore = apply_setting(setting_spec)
        print(f"== setting {setting_spec or 'defaults'} (prefill_all, form {form}, chunks {chunk_set})", flush=True)
        t0 = time.time()
        model.logits_all = []
        if tap_chunks:
            model.taps, model.tap_layers = [], frozenset(range(len(model.layers)))
        requests = [(slots[c], tokens[c, : ctx - 1].tolist()) for c in range(n)]
        if form == "rows":
            gen.prefill_many(requests, traced=False)
        else:
            for req in requests:
                gen.prefill_many([req], traced=False)
        t_prefill = time.time() - t0
        if tap_chunks:
            logits_of = _router_windows(model, captured_logits, tap_chunks, tap_windows, owner) if router else None
            captured_logits.clear()
            _save_taps(model, tap_chunks, tap_windows, owner, setting_spec, form, n, ctx, logits_of)
            for out, _, _ in model.logits_all:
                ttnn.deallocate(out)
            model.logits_all = None
            restore()
            time.sleep(1)  # (distinct stamps)
            continue
        dev = torch.full((n, hi - lo, vocab), float("nan"))
        digest = torch.full((n, hi - dlo, 5), float("nan"), dtype=torch.float64)
        waves = []
        for out, t_row, segs in model.logits_all:
            chips = ttnn.get_device_tensors(out)
            rows_read = {}
            for r, off, gpos, cnt, own in segs:
                a, b = max(gpos, min(lo, dlo)), min(gpos + cnt, hi)
                if a >= b:
                    continue
                if r not in rows_read:  # the row's 4 chips: their quarters of the vocabulary, in order
                    quarters = [ttnn.to_torch(chips[r * COLS + c]).reshape(t_row, -1).float() for c in range(COLS)]
                    rows_read[r] = torch.cat(quarters, dim=1)[:, :vocab]
                block = rows_read[r][off + a - gpos : off + b - gpos]  # positions a .. b - 1
                c = owner[own]
                da = max(a, dlo)
                if da < b:  # fingerprints: sum, sum of squares (fp64 sums of the fp32 values), argmax, max, logsumexp
                    x = block[da - a :]
                    digest[c, da - dlo : b - dlo] = torch.stack(
                        [
                            x.sum(-1, dtype=torch.float64),
                            (x * x).sum(-1, dtype=torch.float64),
                            x.argmax(-1).double(),
                            x.max(-1).values.double(),
                            torch.logsumexp(x, -1).double(),
                        ],
                        dim=-1,
                    )
                sa = max(a, lo)
                if sa < b:
                    dev[c, sa - lo : b - lo] = block[sa - a :]
            waves.append((t_row, len(segs)))
            ttnn.deallocate(out)
        model.logits_all = None
        missing = int(torch.isnan(dev[:, :, 0]).sum())
        assert missing == 0, f"prefill_all: {missing} scored positions without logits"
        sha = [hashlib.sha1(dev[c].numpy().tobytes()).hexdigest() for c in range(n)]
        kld = torch.zeros(n, hi - lo)
        top1 = torch.zeros(n, hi - lo)
        nll_dev = torch.zeros(n, hi - lo)
        nll_ref = torch.zeros(n, hi - lo)
        for c in range(n):
            d = torch.log_softmax(dev[c], -1)
            rf = torch.log_softmax(ref_logits[c].float(), -1)
            kld[c] = (rf.exp() * (rf - d)).sum(-1)
            top1[c] = (d.argmax(-1) == rf.argmax(-1)).float()
            nxt = tokens[c, lo + 1 : hi + 1].long()
            nll_dev[c] = -d.gather(1, nxt[:, None])[:, 0]
            nll_ref[c] = -rf.gather(1, nxt[:, None])[:, 0]
        per_chunk = kld.mean(1)
        result = {
            "setting": setting_spec,
            "mode": "prefill_all",
            "form": form,
            "chunk_set": list(chunk_set),
            "waves": waves,
            "chunks": n,
            "ctx": ctx,
            "positions": int(kld.numel()),
            "mean_kld": kld.mean().item(),
            "kld_stderr_by_chunk": (per_chunk.std() / n**0.5).item() if n > 1 else None,
            "p99_kld": kld.flatten().quantile(0.99).item(),
            "same_top1": top1.mean().item(),
            "ppl_ratio": torch.exp(nll_dev.mean() - nll_ref.mean()).item(),
            "kld_by_chunk": per_chunk.tolist(),
            "logits_sha1_by_chunk": sha,
            "digest_from": dlo,
            "load_seconds": round(load),
            "prefill_seconds": round(t_prefill, 1),
            "seconds": round(time.time() - start),
        }
        print(
            f"wikitext (prefill_all, {form}) KLD {result['mean_kld']:.5f} (+/- {result['kld_stderr_by_chunk'] or 0:.5f} "
            f"by chunk), p99 {result['p99_kld']:.3f}, same top-1 {result['same_top1']:.2%}, PPL ratio "
            f"{result['ppl_ratio']:.4f}; {result['positions']} positions, waves (tokens a row, segments) {waves}, "
            f"prefill {t_prefill:.1f} s",
            flush=True,
        )
        path = out_dir() / f"wikitext-prefill-{form}-{n}x{ctx}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
        path.write_text(json.dumps(result, indent=1) + "\n")
        torch.save({"kld": kld, "top1": top1, "digest": digest, "digest_from": dlo}, path.with_suffix(".pt"))
        print(f"wrote {path}; logits SHA-1 by chunk {[h[:10] for h in sha]}", flush=True)
        del dev
        restore()
        finite = finite and bool(torch.isfinite(kld).all())
        time.sleep(1)  # (distinct stamps)
    gen.release()
    if router:
        TtMoe.router_logits = original
    assert finite
