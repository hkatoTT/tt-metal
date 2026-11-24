"""Whole-model chunked prefill, then the traced greedy decode, against the acceptance records (real BF16 weights).

    GLM53_PREFILL_CHUNKS = comma list of chunk lengths, one full pass each (default 32,128)
    GLM53_PREFILL_USERS = users per row (default 2)
    GLM53_PREFILL_BATCHED = 1 (default): one more pass in the longest chunk with every prompt given at once
        (`prefill_many`: one request per mesh row per chunk); its logits must equal that chunk's one-by-one pass bit
        for bit (the rows compute independently); with several chunk lengths, one more with every prompt at once in
        the generator's choice of chunks per prompt (`Glm53Generator.schedule`, "auto"); 0: skip them
    GLM53_PREFILL_TRACE = comma list of chunk lengths whose one-chunk prefill is captured as a trace (default 128 when
        it is one of the chunks; empty for none): one more pass per length with every prompt at once in that chunk,
        the waves of one chunk replaying the trace, the others eager; its logits must equal that chunk's one-by-one
        (eager) pass bit for bit
    GLM53_TIME_BUDGET = seconds from the test's start after which the chunk lengths not begun are skipped (default
        1200)

Through the serving generator (ttnn/generator.py): the prefill is warmed up for both chunk lengths, the one-chunk
prefill traces and then the greedy decode trace are captured (on throwaway states: a prefill from position 0 starts its slot afresh).  Slot (row r, user
u) takes record (r U + u) mod 11; its prompt goes in chunk by chunk on its own row (the other rows keep their state),
the logits after the prompt give the first token, and the trace generates the rest greedily, every slot at its own
position.  Each pass reuses the slots of the one before.
- the first token's logits (the prefill's output) against the CPU bf16 references: KLD and top-1 per slot
- greedy: the first divergence from the record per slot
- seconds per prefill chunk (eager)
Results: out_dir()/prefill-<stamp>.json.  Run (a device test): GLM53_DEVICE_TEST=1 pytest ... -s
"""

from __future__ import annotations

import faulthandler
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

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.paths import data_dir, out_dir  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

RECORDS = Path(__file__).parents[1] / "tools" / "acceptance" / "greedy-v1"
REFERENCES = {
    # tools/run_cpu_reference.py --teacher-decode, and tools/bfp_study.py e2e's bf16 logits
    "cpu_decode_path": data_dir("teacher-decode-v1", "bf16_decode_logits.pt"),
    "cpu_prefill_path": data_dir("bfp-study-v1", "bf16_logits.pt"),
}
PASS_WATCHDOG_S = 300  # a pass (16 prefills and 64 decode steps) takes under a minute


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_model_device_prefill(mesh_device):
    torch.set_num_threads(16)
    chunks = [int(c) for c in os.environ.get("GLM53_PREFILL_CHUNKS", "32,128").split(",")]
    users = int(os.environ.get("GLM53_PREFILL_USERS", "2"))
    records = [json.loads(f.read_text()) for f in sorted(RECORDS.glob("*.json"))]
    offsets, n = [], 0  # each record's first row in the reference logits (records in file order)
    for r in records:
        offsets.append(n)
        n += len(r["generated_token_ids"])
    refs = {name: torch.load(path) for name, path in REFERENCES.items() if path.exists()}
    slots = [(row, u, (row * users + u) % len(records)) for row in range(ROWS) for u in range(users)]
    result = {
        "users": ROWS * users,
        "tree": os.popen(f"git -C {Path(__file__).parent} rev-parse --short HEAD").read().strip(),
    }

    t0 = time.time()
    model = TtGlm53Model(mesh_device, Glm53Checkpoint(), max_seq=512)
    result["load_seconds"] = round(time.time() - t0)
    print(f"loaded in {result['load_seconds']}s", flush=True)

    traced = [int(c) for c in os.environ.get("GLM53_PREFILL_TRACE", "128").split(",") if c and int(c) in chunks]
    gen = Glm53Generator(model, users, chunk=tuple(chunks))
    result["warmup_seconds"] = gen.warmup(greedy=True, logits=False, chunks=chunks, tokens=256, traced=tuple(traced))
    print(f"warm-up {result['warmup_seconds']}", flush=True)

    budget = float(os.environ.get("GLM53_TIME_BUDGET", "1200"))
    passes = []  # (chunk, batched, traced): a traced pass right after its chunk's one-by-one pass
    for chunk in chunks:
        passes += [(chunk, False, False)] + ([(chunk, True, True)] if chunk in traced else [])
    if os.environ.get("GLM53_PREFILL_BATCHED", "1") == "1":
        passes.append((max(chunks), True, False))
        if len(chunks) > 1:
            passes.append((None, True, False))
    done, serial_logits, mismatched = [], {}, {}
    # GLM53_QS_SETTINGS: ";"-separated in-process switches (`common.apply_setting`: "module.NAME=value[+..]", "" the
    # defaults), each pass run under each in turn on the one load (labels "<pass>@<setting>"; a setting's own one-by-one
    # logits for its batched pass's check)
    from models.demos.blackhole.glm53_flash.ttnn.common import apply_setting

    settings = os.environ.get("GLM53_QS_SETTINGS", "").split(";")
    runs = [(st, p) for st in settings for p in passes]
    current, restore = None, None
    for setting_spec, (chunk, batched, use_trace) in runs:
        if setting_spec != current:
            if restore is not None:
                restore()
            restore, current = apply_setting(setting_spec), setting_spec
            serial_logits = {}
        label = (f"{chunk or 'auto'}_batched" if batched else f"{chunk}") + ("_traced" if use_trace else "")
        label += f"@{setting_spec}" if setting_spec else ""
        if time.time() - t0 > budget:
            print(f"time budget ({budget:.0f} s) reached: skipping chunk {label}", flush=True)
            continue
        faulthandler.dump_traceback_later(PASS_WATCHDOG_S, exit=True)  # a hung device: stacks, then exit
        times, first, kld = [], {}, {name: [] for name in refs}
        counts = dict(gen.counts)
        if batched:
            t1 = time.time()
            prompts = [(row * users + u, records[rec]["prompt_token_ids"]) for row, u, rec in slots]
            outs = gen.prefill_many(prompts, chunk=chunk, traced=use_trace)
            times.append({"tokens": sum(len(p) for _, p in prompts), "seconds": time.time() - t1})
            got = {(row, u): logits for (row, u, _), logits in zip(slots, outs)}
            if chunk in serial_logits:
                mismatched[label] = [k for k, v in got.items() if not torch.equal(v, serial_logits[chunk][k])]
        else:
            got = {}
            for row, u, rec in slots:
                t1 = time.time()
                prompt = records[rec]["prompt_token_ids"]
                got[(row, u)] = gen.prefill(row * users + u, prompt, chunk=chunk, traced=False)
                times.append({"tokens": len(prompt), "seconds": time.time() - t1})
            serial_logits[chunk] = got
        waves = {k: gen.counts[k] - counts[k] for k in counts}
        for row, u, rec in slots:
            logits = got[(row, u)]
            first[(row, u)] = int(logits.argmax())
            logp = torch.log_softmax(logits.float(), -1)
            for name, ref in refs.items():
                cpu = torch.log_softmax(ref[offsets[rec]].float(), -1)
                kld[name].append(
                    {"kld": (cpu.exp() * (cpu - logp)).sum().item(), "top1": int(cpu.argmax()) == first[(row, u)]}
                )
        print(
            f"chunk {label}: {len(slots)} prompts, {sum(t['tokens'] for t in times)} tokens in "
            f"{sum(t['seconds'] for t in times):.1f} s; per call {[round(t['seconds'], 2) for t in times]}; {waves}",
            flush=True,
        )
        if label in mismatched:
            bad = mismatched[label]
            print(f"chunk {label}: logits equal to the one-by-one pass bit for bit: {not bad} {bad}", flush=True)

        tokens = torch.zeros(ROWS * users, dtype=torch.int64)
        positions = torch.zeros(ROWS * users, dtype=torch.int64)
        for row, u, rec in slots:
            tokens[row * users + u] = first[(row, u)]
            positions[row * users + u] = len(records[rec]["prompt_token_ids"])
        generated = {(row, u): [first[(row, u)]] for row, u, _ in slots}
        steps = max(len(r["generated_token_ids"]) for r in records) - 1
        vocab = model.config.vocab_size
        for _ in range(steps):
            tokens = gen.decode(tokens, positions, greedy=True)
            # garbage ids would make the next step's embedding read past its table (a hang, not a failure)
            assert bool(((tokens >= 0) & (tokens < vocab)).all()), f"chunk {label}: decode ids out of the vocabulary"
            positions += 1
            for row, u, _ in slots:
                generated[(row, u)].append(int(tokens[row * users + u]))

        per_record = {}
        for row, u, rec in slots:
            want = records[rec]["generated_token_ids"]
            got = generated[(row, u)][: len(want)]
            div = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
            per_record.setdefault(records[rec]["id"], []).append(len(got) if div is None else div)
        summary = {
            "prefill_seconds": times,
            "first_token": {
                name: {
                    "mean_kld": sum(v["kld"] for v in vals) / len(vals),
                    "max_kld": max(v["kld"] for v in vals),
                    "top1_agreement": sum(v["top1"] for v in vals) / len(vals),
                    # each prompt's (slot order), for paired comparisons between runs
                    "per_prompt_kld": [round(v["kld"], 7) for v in vals],
                }
                for name, vals in kld.items()
            },
            "greedy": {},
        }
        for rec in records:
            firsts = per_record.get(rec["id"], [])
            full = sum(d == len(rec["generated_token_ids"]) for d in firsts)
            summary["greedy"][rec["id"]] = {"slots": len(firsts), "all_match": full, "first_divergence": sorted(firsts)}
            print(
                f"chunk {label} greedy {rec['id']}: {full}/{len(firsts)} slots match all "
                f"{len(rec['generated_token_ids'])}; first divergences {sorted(firsts)}",
                flush=True,
            )
        for name, v in summary["first_token"].items():
            print(f"chunk {label} first token vs {name}: {v}", flush=True)
        summary["waves"] = waves
        if label in mismatched:
            summary["bitwise_equal_to_one_by_one"] = not mismatched[label]
            summary["mismatched_slots"] = mismatched[label]
        result[f"chunk_{label}"] = summary
        done.append(label)
        faulthandler.cancel_dump_traceback_later()

    if restore is not None:
        restore()
    gen.release()
    path = out_dir() / f"prefill-{users}x8-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {path}", flush=True)
    for label in done:
        for name, v in result[f"chunk_{label}"]["first_token"].items():
            assert v["mean_kld"] < 0.01, f"chunk {label}: the prefill's logits are off ({name}: {v})"
    bad = {label: keys for label, keys in mismatched.items() if keys}
    assert not bad, f"prefill_many's logits differ from the one-by-one prefill: {bad}"
