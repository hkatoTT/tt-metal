"""The KV cache in low precision, CPU side (a study of the caches' precision): the formats' rounding and the hooks the
CPU reference (`tt/`) calls where the device would store a cache value, plus the study that scores a rounded run
against a saved bf16 reference.

Formats (`round_kv`, the value as the cache would hold it, back in the input's dtype; groups of 16 along the last dim,
the 16 values a BFP tile face row shares an exponent over: 16 consecutive features of one token / pool / position):
    fp8     e4m3, round half to even, saturating at 448 (torch's float8_e4m3fn; the latent's |x| <= 26 never saturates)
    fp8t    e4m3 truncated toward zero (ttnn.typecast bf16 -> fp8_e4m3 on BH, measured)
    fp8a    e4m3 of the bf16 nudged by 8 in its bits, truncated (the device's write form: ties away from zero)
    fp8b8 / fp8ab8   fp8 / fp8a, then BFP8 (upstream sparse_sdpa's fp8 kv tilized into bfp8_b tiles)
    bfp8 / bfp4   the host packer's block floating point (tools/bfp_study.bfp_round: the group's largest exponent,
            round half to even, saturate)
    bfp8a / bfp4a   ... ties away from zero (ttnn.typecast's BFP8 on BH, measured)
    bfp4t   BFP4 truncated toward zero (ttnn.typecast's BFP4 on BH, measured)
The device's own rounding was checked against these on BH.

Hooks (read at each call, empty or unset: off, the reference as it was):
    GLM53_REF_KV_LATENT  the DSA latent (tt/dsa.py: every token's normed kv latent, the keys and values' source)
    GLM53_REF_KV_DRAFT   the DFlash2 drafter's K / V (tt/dflash2.py: the context's and the block's)

The study (one form a run, the hooks from the environment; the logits through tools/bfp_study.teacher_forced):
    python -m models.demos.blackhole.glm53_flash.tools.kv_round wikitext <reference-8x1024.pt> <out.json> [chunks]
    python -m models.demos.blackhole.glm53_flash.tools.kv_round long <teacher-2600.pt> <out.json>  (+ out.pt: logits)
    python -m models.demos.blackhole.glm53_flash.tools.kv_round compare <a.pt> <b.pt>  (two long runs' logits)
-> the mean KLD from the reference's logits (and per chunk / prompt, the per-position values), the top-1 agreement,
   for wikitext the mean NLL of the text's next tokens (both), and the hooks' settings.
"""

from __future__ import annotations

import os

import torch

GROUP = 16
FP8_MAX = 448.0
_BITS = {"bfp8": 7, "bfp4": 3}


def fp8_e4m3(x: torch.Tensor, mode: str = "rne") -> torch.Tensor:
    """e4m3 (bias 7, 3 mantissa bits, subnormals from 2^-9) of x (fp32 result): "rne" round half to even with
    saturation at 448, "trunc" toward zero (ttnn.typecast's), "rna" a bf16 x nudged by 8 in its bits then
    truncated (the device's write: ties away from zero in the normal range, truncation among the subnormals)."""
    xf = x.float()
    if mode == "rne":
        return xf.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    if mode == "rna":
        bits = x.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
        xf = ((bits + 8) & 0xFFFF).to(torch.int16).view(torch.bfloat16).float()
    a = xf.abs()
    e = torch.floor(torch.log2(a.clamp_min(2.0**-9))).clamp(min=-6)
    step = torch.pow(2.0, e - 3)
    q = torch.minimum(torch.trunc(a / step) * step, torch.tensor(FP8_MAX))
    return torch.where(a > 0, q, torch.zeros_like(q)) * torch.sign(xf)


def _bfp(x: torch.Tensor, fmt: str, rounding) -> torch.Tensor:
    bits = _BITS[fmt]
    g = x.float().reshape(-1, GROUP)
    amax = g.abs().amax(dim=-1, keepdim=True)
    e = torch.frexp(amax).exponent.to(torch.float32) - 1
    step = torch.pow(2.0, e - (bits - 1))
    lim = float(2**bits - 1)
    q = torch.clamp(rounding(g / step), -lim, lim) * step
    return torch.where(amax > 0, q, torch.zeros_like(q)).reshape(x.shape)


def bfp_round(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """BFP8 / BFP4 as the host packer (tools/bfp_study.bfp_round), fp32 result."""
    return _bfp(x, fmt, torch.round)


def bfp_rna(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """BFP8 / BFP4 rounded to nearest, ties away from zero (ttnn.typecast's BFP8), fp32 result."""
    return _bfp(x, fmt, lambda v: torch.sign(v) * torch.floor(v.abs() + 0.5))


def bfp_trunc(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """BFP8 / BFP4 with the mantissas truncated toward zero (a candidate for the device packer), fp32 result."""
    return _bfp(x, fmt, torch.trunc)


def round_kv(x: torch.Tensor, fmt: str | None) -> torch.Tensor:
    """x as a cache in format `fmt` holds it, in x's dtype (fmt None / "" / "bf16": x itself)."""
    if not fmt or fmt == "bf16":
        return x
    if fmt == "fp8":
        y = fp8_e4m3(x)
    elif fmt == "fp8t":
        y = fp8_e4m3(x, "trunc")
    elif fmt == "fp8a":
        y = fp8_e4m3(x, "rna")
    elif fmt == "fp8b8":
        y = bfp_round(fp8_e4m3(x), "bfp8")
    elif fmt == "fp8ab8":
        y = bfp_rna(fp8_e4m3(x, "rna"), "bfp8")
    elif fmt in _BITS:
        y = bfp_round(x, fmt)
    elif fmt in ("bfp8a", "bfp4a"):
        y = bfp_rna(x, fmt[:-1])
    elif fmt == "bfp4t":
        y = bfp_trunc(x, "bfp4")
    else:
        raise ValueError(f"KV format {fmt!r}: one of fp8, fp8t, fp8a, fp8b8, fp8ab8, bfp8, bfp8a, bfp4, bfp4a, bfp4t")
    return y.to(x.dtype)


def _hook(name: str, x: torch.Tensor) -> torch.Tensor:
    return round_kv(x, os.environ.get(name, ""))


def latent(x: torch.Tensor) -> torch.Tensor:
    return _hook("GLM53_REF_KV_LATENT", x)


def draft(x: torch.Tensor) -> torch.Tensor:
    return _hook("GLM53_REF_KV_DRAFT", x)


def settings() -> dict:
    return {k: os.environ.get(k, "") for k in ("GLM53_REF_KV_LATENT", "GLM53_REF_KV_DRAFT")}


# -- the study -----------------------------------------------------------------------------------------------------
def _score(ref: torch.Tensor, got: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per position KLD(ref || got) and top-1 agreement, ref / got [N, V] fp32."""
    lp = torch.log_softmax(ref.float(), -1)
    lq = torch.log_softmax(got.float()[:, : ref.shape[-1]], -1)
    kld = (lp.exp() * (lp - lq)).sum(-1)
    return kld, (ref.argmax(-1) == got.argmax(-1))


def _boot(groups: list[torch.Tensor], n: int = 10000) -> list[float]:
    """95 % interval of the mean over positions, resampling the groups (chunks / prompts) with their positions."""
    g = torch.Generator().manual_seed(0)
    sums = torch.tensor([float(x.sum()) for x in groups], dtype=torch.float64)
    cnts = torch.tensor([x.numel() for x in groups], dtype=torch.float64)
    k = torch.randint(0, len(groups), (n, len(groups)), generator=g)
    means = sums[k].sum(1) / cnts[k].sum(1)
    return [float(means.quantile(0.025)), float(means.quantile(0.975))]


def main() -> int:
    import json
    import sys
    import time
    from pathlib import Path

    from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint
    from models.demos.blackhole.glm53_flash.config import Glm53Config
    from models.demos.blackhole.glm53_flash.tools.bfp_study import teacher_forced

    mode, ref_path, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if mode == "compare":  # two long runs' saved logits: KLD(ref_path's || out's) per position, top-1 (no model)
        a, b = torch.load(ref_path), torch.load(out)
        k = [_score(a[i], b[i]) for i in range(a.shape[0])]
        kl = torch.cat([x[0] for x in k])
        print(
            f"KV round compare {ref_path.name} -> {out.name}: KLD {kl.mean():.6f} (max {kl.max():.5f}), top-1 "
            f"{torch.cat([x[1] for x in k]).float().mean():.4f}, {kl.numel()} positions",
            flush=True,
        )
        return 0
    torch.set_num_threads(int(os.environ.get("WT_THREADS", "24")))
    ref = torch.load(ref_path, weights_only=False)
    ckpt = Glm53Checkpoint()
    config = Glm53Config.from_dict(ckpt.config)
    t0 = time.time()
    result = {"mode": mode, "reference": str(ref_path), "hooks": settings(), "threads": torch.get_num_threads()}
    if mode == "wikitext":
        n = int(sys.argv[4]) if len(sys.argv) > 4 else ref["tokens"].shape[0]
        tokens, first, ctx = ref["tokens"][:n], ref["first"], ref["ctx"]
        got = teacher_forced(ckpt, config, [row.tolist() for row in tokens], [slice(first, ctx - 1)] * n)
        got = got.view(n, ctx - 1 - first, -1)
        klds, top1, nll_ref, nll_got = [], [], [], []
        for c in range(n):
            k, t = _score(ref["logits"][c], got[c])
            nxt = tokens[c, first + 1 : ctx].long()
            nll_ref.append(-torch.log_softmax(ref["logits"][c].float(), -1).gather(1, nxt[:, None]).squeeze(1))
            nll_got.append(-torch.log_softmax(got[c].float(), -1).gather(1, nxt[:, None]).squeeze(1))
            klds.append(k)
            top1.append(t)
        groups = klds
        result.update(
            chunks=n,
            nll_ref=float(torch.cat(nll_ref).mean()),
            nll=float(torch.cat(nll_got).mean()),
            nll_diff_ci=_boot([a - b for a, b in zip(nll_got, nll_ref)]),
        )
    elif mode == "long":
        prompts, toks, teacher = ref["prompts"], ref["tokens"], ref["teacher"]
        k, n = teacher.shape[:2]
        records = [list(prompts[i]) + list(toks[i][: n - 1]) for i in range(k)]
        slices = [slice(len(prompts[i]) - 1, len(prompts[i]) + n - 1) for i in range(k)]
        got = teacher_forced(ckpt, config, records, slices).view(k, n, -1)
        torch.save(got.to(torch.float32), out.with_suffix(".pt"))  # (small: a form against another's, `compare`)
        groups, top1 = [], []
        for i in range(k):
            kk, t = _score(teacher[i], got[i])
            groups.append(kk)
            top1.append(t)
        result.update(prompts=k, lengths=[len(p) for p in prompts])
    else:
        raise SystemExit(f"mode {mode}: wikitext or long")
    allk = torch.cat(groups)
    result.update(
        seconds=round(time.time() - t0),
        positions=int(allk.numel()),
        kld=float(allk.mean()),
        kld_ci=_boot(groups),
        kld_by_group=[float(x.mean()) for x in groups],
        kld_p99=float(allk.quantile(0.99)),
        top1=float(torch.cat(top1).float().mean()),
        kld_positions=[x.tolist() for x in groups],
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result) + "\n")
    print(
        f"KV round {mode} {result['hooks']}: KLD {result['kld']:.5f} {result['kld_ci']}, top-1 {result['top1']:.4f}, "
        f"{result['positions']} positions, {result['seconds']} s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
