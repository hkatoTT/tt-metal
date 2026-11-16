"""One long prompt over the 8 mesh rows (sequence parallel) against the same prompt on its own
row, through the serving generator (real weights).

Three passkey prompts (tests/test_model_device_long.py's filler; about 600, 1,000 and 1,900 tokens) for slots on
three rows (2, 7, 4), each prefilled alone, then decoded together greedily (24 tokens), in passes:
- rows: on the slot's own row (`prefill_many` without the sequence-parallel form), chunks of 512, eager
- sp: over the rows in segments of 512 (one wave), eager
- sp128: over the rows in segments of 128 (the 1,900-token prompt in two waves), eager
- sp-all: the three at once over the rows (`sp_plan`: blocks of 3, 2 and 3 rows; the 1,900-token prompt on 2 rows
  in two waves), eager
- sp-trace, sp-all-trace (GLM53_PREFILL_SP_TRACE=1, the default): sp and sp-all through their captured forms
For each pass and prompt: the KLD of the logits after the prompt from the rows pass's, top-1, the generated tokens'
first divergence from the rows pass's, and whether the passphrase comes back.  The prompts prefilled after one
another and decoded together check that a prompt over the rows leaves the other rows' users as they were.
    GLM53_DEVICE_TEST=1 GLM53_WEIGHT_DTYPE=bfp8 pytest ... -s
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

from models.demos.blackhole.glm53_flash.checkpoint import (  # noqa: E402
    Glm53Checkpoint,
    ShapeOnlyCheckpoint,
    checkpoint_root,
)
from models.demos.blackhole.glm53_flash.paths import data_dir, out_dir  # noqa: E402
from models.demos.blackhole.glm53_flash.tools.passkey import SP_CASES, sp_prompts  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

REFERENCE = data_dir("sp-v1", "logits.pt")  # tools/sp_reference.py (the CPU reference)
# GLM53_SP_TEACHER=1: every pass teacher-forced along the CPU reference's own greedy continuation
# (`tools/sp_reference.py --tokens 24`), the KLD at each position against the reference and the rows pass
TEACHER = data_dir("sp-v2", "teacher.pt")


def kld(p_logits, q_logits) -> float:
    p, q = (torch.log_softmax(x.float(), dim=-1) for x in (p_logits, q_logits))
    return float((p.exp() * (p - q)).sum())


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_model_device_prefill_sp(mesh_device):
    from tokenizers import Tokenizer

    steps, users, cache = 24, 2, 2048
    traced = os.environ.get("GLM53_PREFILL_SP_TRACE", "1") == "1"
    tok = Tokenizer.from_file(str(Path(checkpoint_root()) / "tokenizer.json"))
    answers, prompts = sp_prompts(tok)
    slots = [row * users + user for *_, row, user in SP_CASES]
    print(f"prompt lengths {[len(p) for p in prompts]}, slots {slots}", flush=True)

    t0 = time.time()
    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1"
    model = TtGlm53Model(mesh_device, ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint(), max_seq=cache)
    assert model.sp_ready, "dsa.PREFILL_SP is needed"
    gen = Glm53Generator(model, users, chunk=(128, 512))
    result = {"lengths": [len(p) for p in prompts], "answers": answers, "load_seconds": round(time.time() - t0)}
    teacher = torch.load(TEACHER) if os.environ.get("GLM53_SP_TEACHER") == "1" else None
    if teacher is not None:
        assert teacher["lengths"] == [len(p) for p in prompts], "the teacher file is of other prompts"
        steps = len(teacher["tokens"][0])
    result["warmup_seconds"] = gen.warmup(greedy=True, logits=teacher is not None, traced=(128, 512) if traced else ())
    print(f"loaded in {result['load_seconds']} s; warm-up {result['warmup_seconds']}", flush=True)

    passes = [
        ("rows", False, 512, False, False),
        ("sp", True, 512, False, False),
        ("sp128", True, 128, False, False),
        ("sp-all", True, 512, False, True),
    ]
    if traced:
        passes += [("sp-trace", True, 512, True, False), ("sp-all-trace", True, 512, True, True)]
    # GLM53_SP_PASSES: the passes by name, a name + "+ybf16" with the prefill MoE's Y in bf16 (moe_prefill.Y_BF16), +
    # "+ybfp8" in bfp8 (moe_prefill.Y_BFP8; eager passes only: a traced pass replays the warm-up's captures)
    if os.environ.get("GLM53_SP_PASSES"):
        known = {p[0]: p[1:] for p in passes}
        passes = [(n, *known[n.split("+")[0]]) for n in os.environ["GLM53_SP_PASSES"].split(",")]
    from models.demos.blackhole.glm53_flash.ttnn import moe_prefill

    y_bf16, y_bfp8 = moe_prefill.Y_BF16, moe_prefill.Y_BFP8
    vocab = model.config.vocab_size
    out: dict = {}
    for name, sp, chunk, use_trace, together in passes:
        gen.sp = sp
        moe_prefill.Y_BF16 = y_bf16 or "ybf16" in name.split("+")[1:]
        moe_prefill.Y_BFP8 = y_bfp8 or "ybfp8" in name.split("+")[1:]
        logits, seconds = [], []
        if together:
            t1 = time.time()
            logits = gen.prefill_many(list(zip(slots, prompts)), chunk=chunk, traced=use_trace)
            seconds = [round(time.time() - t1, 3)] * len(prompts)
        for slot, prompt in zip(slots, prompts) if not together else ():
            t1 = time.time()
            logits.append(gen.prefill_many([(slot, prompt)], chunk=chunk, traced=use_trace)[0])
            seconds.append(round(time.time() - t1, 3))
        tokens = torch.zeros(gen.slots, dtype=torch.int64)
        positions = torch.full((gen.slots,), -1, dtype=torch.int64)
        for slot, prompt, lg in zip(slots, prompts, logits):
            tokens[slot], positions[slot] = int(lg.argmax()), len(prompt)
        if teacher is not None:  # the reference's tokens in, the logits at every position out
            forced = [[lg.float()] for lg in logits]
            for step in range(1, steps):
                for i, s in enumerate(slots):
                    tokens[s] = teacher["tokens"][i][step - 1]
                full = gen.decode(tokens, positions, greedy=False)
                positions = torch.where(positions >= 0, positions + 1, positions)
                for i, s in enumerate(slots):
                    forced[i].append(full[s].float())
            out[name] = {"logits": logits, "forced": [torch.stack(f) for f in forced], "seconds": seconds}
            continue
        generated = [[int(tokens[s])] for s in slots]
        for _ in range(steps - 1):
            tokens = gen.decode(tokens, positions, greedy=True)
            assert bool(((tokens >= 0) & (tokens < vocab)).all()), "decode ids out of the vocabulary"
            positions = torch.where(positions >= 0, positions + 1, positions)
            for i, s in enumerate(slots):
                generated[i].append(int(tokens[s]))
        out[name] = {"logits": logits, "generated": generated, "seconds": seconds}

    moe_prefill.Y_BF16, moe_prefill.Y_BFP8 = y_bf16, y_bfp8
    if teacher is not None:  # the KLD per position against the reference and the rows pass, then done
        summary = {}
        for name, entry in out.items():
            vs_cpu = [[kld(teacher["teacher"][i][j], entry["forced"][i][j]) for j in range(steps)] for i in range(3)]
            vs_rows = [
                [kld(out["rows"]["forced"][i][j], entry["forced"][i][j]) for j in range(steps)] for i in range(3)
            ]
            top1 = [
                sum(int(entry["forced"][i][j].argmax()) == teacher["tokens"][i][j] for j in range(steps)) / steps
                for i in range(3)
            ]
            mean = lambda rows: sum(map(sum, rows)) / (3 * steps)  # noqa: E731
            summary[name] = {
                "kld_vs_cpu": mean(vs_cpu),
                "kld_vs_rows": mean(vs_rows),
                "per_prompt_vs_cpu": [sum(r) / steps for r in vs_cpu],
                "top1_vs_cpu": top1,
                "per_position_vs_cpu": vs_cpu,
            }
            print(
                f"{name}: teacher-forced {steps} positions x 3 prompts: KLD vs the CPU reference {mean(vs_cpu):.5f} "
                f"(per prompt {[round(sum(r) / steps, 5) for r in vs_cpu]}), vs rows {mean(vs_rows):.5f}, top-1 "
                f"{[round(x, 3) for x in top1]}",
                flush=True,
            )
            twin = name.split("+")[0]
            if twin != name and twin in out:  # against the same pass without the flags
                vs_twin = [
                    [kld(out[twin]["forced"][i][j], entry["forced"][i][j]) for j in range(steps)] for i in range(3)
                ]
                same1 = sum(
                    int(out[twin]["forced"][i][j].argmax()) == int(entry["forced"][i][j].argmax())
                    for i in range(3)
                    for j in range(steps)
                )
                summary[name]["kld_vs_twin"] = mean(vs_twin)
                print(
                    f"{name}: against {twin}: KLD {mean(vs_twin):.6f} (per prompt "
                    f"{[round(sum(r) / steps, 6) for r in vs_twin]}), top-1 the same at {same1} of {3 * steps}",
                    flush=True,
                )
        result["teacher"] = summary
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        (out_dir() / f"prefill-sp-teacher-{stamp}.json").write_text(json.dumps(result, indent=1))
        gen.release()
        return

    base = out["rows"]
    reference = torch.load(REFERENCE) if REFERENCE.exists() else None
    if reference is not None and reference["lengths"] != [len(p) for p in prompts]:
        reference = None
    rows = {}
    for name, entry in out.items():
        per = []
        for i, (prompt, answer) in enumerate(zip(prompts, answers)):
            same = [a == b for a, b in zip(entry["generated"][i], base["generated"][i])]
            text = tok.decode(entry["generated"][i])
            per.append(
                {
                    "kld_vs_rows": kld(base["logits"][i], entry["logits"][i]),
                    "kld_vs_cpu": kld(reference["logits"][i], entry["logits"][i]) if reference is not None else None,
                    "top1_same": int(entry["logits"][i].argmax()) == int(base["logits"][i].argmax()),
                    "first_divergence": same.index(False) if False in same else steps,
                    "found": answer in text.replace(" ", ""),
                    "text": text,
                    "prefill_seconds": entry["seconds"][i],
                }
            )
            vs_cpu = per[-1]["kld_vs_cpu"]
            print(
                f"{name}: {len(prompt)} tokens: KLD vs rows {per[-1]['kld_vs_rows']:.5f}"
                + (f", vs the CPU reference {vs_cpu:.5f}" if vs_cpu is not None else "")
                + ", top-1 same "
                f"{per[-1]['top1_same']}, generated as rows' for {per[-1]['first_divergence']} of {steps}, passphrase "
                f"{'found' if per[-1]['found'] else 'MISSED'}, prefill {entry['seconds'][i]} s -> {text!r}",
                flush=True,
            )
        rows[name] = per
    result["passes"] = rows
    result["counts"] = gen.counts
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    (out_dir() / f"prefill-sp-{stamp}.json").write_text(json.dumps(result, indent=1))
    torch.save(
        {name: torch.stack(entry["logits"]) for name, entry in out.items()}, out_dir() / f"prefill-sp-{stamp}.pt"
    )
    gen.release()
    for name, per in rows.items():
        for p in per:
            assert p["kld_vs_rows"] < 0.05, f"{name}: the logits after the prompt differ from the rows pass's"
