"""Whole-model accuracy on the traced decode path, for optimization milestones (real BF16 weights).

    GLM53_ACC = comma list of: teacher, greedy, free (default: teacher,greedy)
    GLM53_ACC_USERS = users per row (default 2; 32 = the 256-user batch)
    GLM53_ACC_FREE_TOKENS = tokens generated per user in the free part (default 256)
    GLM53_TIME_BUDGET = seconds from the test's start after which a part stops early, marked truncated, and the
        parts not begun are skipped (default 1200)

The 11 acceptance records (tools/acceptance/greedy-v1) are dealt to the 8 x users slots in turn; every step runs the
captured trace (as the speed tests) and reads the logits back.
- teacher: the records' tokens fed one at a time; at the 659 positions that predict the generated tokens, against the
  CPU bf16 references (the decode path, as the device runs, and the prefill path): mean / p99 KLD, top-1 agreement,
  and the share of positions where the reference's top token is in the device's top 5 / top 100.
- greedy: each slot's prompt, then its own greedy tokens; the first divergence from the record per slot.
- free: each slot's prompt, then GLM53_ACC_FREE_TOKENS greedy tokens; flags degenerate output (more than 10 % of the
  tokens equal to the previous one, or a 4-gram repeated over more than half of the text) and keeps the decoded text
  of one slot per record for reading.
Results: out_dir()/accuracy-<users>x8-<stamp>.json (the teacher's per-position KLD and top-1 in the .pt
next to it).  Run (a device test): GLM53_DEVICE_TEST=1 pytest ... -s
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

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, checkpoint_root  # noqa: E402
from models.demos.blackhole.glm53_flash.paths import data_dir, out_dir  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

RECORDS = Path(__file__).parents[1] / "tools" / "acceptance" / "greedy-v1"
REFERENCES = {
    # tools/run_cpu_reference.py --teacher-decode, and tools/bfp_study.py e2e's bf16 logits
    "cpu_decode_path": data_dir("teacher-decode-v1", "bf16_decode_logits.pt"),
    "cpu_prefill_path": data_dir("bfp-study-v1", "bf16_logits.pt"),
}


class TracedSteps:
    """One decode step per call on the captured trace; step 0 is the capture's compile run (its logits unused)."""

    def __init__(self, model, users: int):
        self.model, self.users = model, users
        self.states = model.empty_state(users)
        self.io = model.alloc_io(users)
        self.trace = self.out = None

    def __call__(self, tokens: torch.Tensor, t: int, read: bool = True):
        """tokens [ROWS, users] at position t -> logits [ROWS * users, V] fp32 (None at t = 0, or unread: `read`
        False runs the step without reading its logits back, the 512-user batch's ~5 s a step)."""
        self.model.write_inputs(self.io, tokens, torch.full((ROWS, self.users), t))
        if self.trace is None:
            self.trace, self.out = self.model.capture(self.io, self.states, greedy=False)
            return None
        if not read:  # (blocking: the next step's host inputs are written only after this one ran, as a read-back does)
            ttnn.execute_trace(self.model.mesh, self.trace, cq_id=0, blocking=True)
            return None
        return self.model.replay(self.trace, self.out, self.users).reshape(ROWS * self.users, -1)

    def close(self):
        ttnn.release_trace(self.model.mesh, self.trace)
        for st in self.states:
            st.deallocate()


BUDGET = float(os.environ.get("GLM53_TIME_BUDGET", "1200"))
# GLM53_ACC_MAX_STEPS: the steps a part runs at most (0: all), to compare batches at the same positions when a large one
# would not finish in one run (the 256-user batch reads 256 rows of logits back a step)
MAX_STEPS = int(os.environ.get("GLM53_ACC_MAX_STEPS", "0"))
START = time.time()


def over_budget() -> bool:
    return time.time() - START > BUDGET


def run(model, slots, users: int, feed, steps_after_prompt: int, on_logits):
    """Feed each slot's prompt, then `feed(slot, k, generated)` tokens; `on_logits(t, logits)` sees every step.
    Stops early past the time budget (the slots' lists are then shorter)."""
    longest = max(len(r["prompt_token_ids"]) for r in slots)
    if min(len(r["prompt_token_ids"]) for r in slots) < 2:
        raise ValueError("a prompt of one token would need the logits of the trace's compile run")
    step = TracedSteps(model, users)
    generated = [[] for _ in slots]
    first = min(len(r["prompt_token_ids"]) for r in slots) - 1  # before it no slot needs the logits
    for t in range(longest + steps_after_prompt):
        if MAX_STEPS and t >= MAX_STEPS:
            print(f"step limit ({MAX_STEPS}) reached: stopping this part", flush=True)
            break
        if over_budget():
            print(f"time budget ({BUDGET:.0f} s) reached at step {t}: stopping this part", flush=True)
            break
        tokens = torch.zeros(ROWS * users, dtype=torch.int64)
        for s, r in enumerate(slots):
            prompt = r["prompt_token_ids"]
            tokens[s] = prompt[t] if t < len(prompt) else feed(s, t - len(prompt), generated[s])
        logits = step(tokens.view(ROWS, users), t, read=t >= first)
        if logits is None:
            continue
        for s, r in enumerate(slots):
            if t >= len(r["prompt_token_ids"]) - 1:
                generated[s].append(int(logits[s].argmax()))
        on_logits(t, logits)
    step.close()
    return generated


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_model_device_accuracy(mesh_device):
    global START
    START = time.time()
    torch.set_num_threads(16)
    wanted = set(os.environ.get("GLM53_ACC", "teacher,greedy").split(","))
    records = [json.loads(f.read_text()) for f in sorted(RECORDS.glob("*.json"))]
    t0 = time.time()
    model = TtGlm53Model(mesh_device, Glm53Checkpoint(), max_seq=512)
    load_seconds = round(time.time() - t0)
    print(f"loaded in {load_seconds}s", flush=True)
    # GLM53_ACC_USERS: users a row, or a comma list of them measured one after the other on the one load
    # GLM53_ACC_AR_ROWS: comma list of common.TP_AR_ASYNC_ROWS values (32 / 64: the row all-reduce at 64 tokens a row
    # by ttnn.all_reduce / all_reduce_async), each measured at each users on the one load
    ar_rows = [int(v) for v in os.environ.get("GLM53_ACC_AR_ROWS", "").split(",") if v] or [None]
    for users in (int(v) for v in os.environ.get("GLM53_ACC_USERS", "2").split(",")):
        for rows in ar_rows:
            if rows is not None:
                from models.demos.blackhole.glm53_flash.ttnn import common

                for st in common._TP_AR_ASYNC.values():  # (made again at the next call, on the same L1)
                    for b in st.bufs:
                        ttnn.deallocate(b)
                common._TP_AR_ASYNC.clear()
                common.TP_AR_ASYNC_ROWS = rows
                print(
                    f"== the row all-reduce up to {rows} tokens a row by all_reduce_async, {users} users a row",
                    flush=True,
                )
            evaluate(model, users, records, wanted, load_seconds)


def evaluate(model, users: int, records: list, wanted: set, load_seconds: int) -> None:
    """The parts of GLM53_ACC at `users` a row, results to out_dir()."""
    slots = [records[i % len(records)] for i in range(ROWS * users)]
    record_of = [i % len(records) for i in range(ROWS * users)]
    kda_state = "bf16"  # (the KDA states')
    moe_k = next((layer.mlp.k for layer in model.layers if hasattr(getattr(layer, "mlp", None), "k")), None)
    result = {
        "users": ROWS * users,
        "moe_topk": moe_k,
        "tree": os.popen(f"git -C {Path(__file__).parent} rev-parse --short HEAD").read().strip(),
        "load_seconds": load_seconds,
        "kda_state": kda_state,
    }

    if "teacher" in wanted and over_budget():
        result.setdefault("skipped_over_budget", []).append("teacher")
        print("time budget reached: skipping teacher", flush=True)
    elif "teacher" in wanted:
        t1 = time.time()
        refs = {name: torch.load(path) for name, path in REFERENCES.items() if path.exists()}
        offsets, n = [], 0  # each record's first row in the reference logits (records in file order)
        for r in records:
            offsets.append(n)
            n += len(r["generated_token_ids"])
        sums = {name: {"kld": [], "top1": [], "top5": [], "top100": []} for name in refs}
        where = []  # each position's row in the reference logits, for paired comparisons between runs

        def on_logits(t, logits):
            rows, refrows = [], []
            for s, r in enumerate(slots):
                k = t - (len(r["prompt_token_ids"]) - 1)
                if 0 <= k < len(r["generated_token_ids"]):
                    rows.append(s)
                    refrows.append(offsets[record_of[s]] + k)
            if not rows:
                return
            where.append(torch.tensor(refrows))
            dev = logits[rows].float()
            logp_dev = torch.log_softmax(dev, -1)
            top100 = dev.topk(100, dim=-1).indices
            for name, ref in refs.items():
                cpu = ref[refrows].float()
                logp_cpu = torch.log_softmax(cpu, -1)
                best = cpu.argmax(-1, keepdim=True)
                sums[name]["kld"].append((logp_cpu.exp() * (logp_cpu - logp_dev)).sum(-1))
                sums[name]["top1"].append((top100[:, :1] == best).any(-1).float())
                sums[name]["top5"].append((top100[:, :5] == best).any(-1).float())
                sums[name]["top100"].append((top100 == best).any(-1).float())

        def feed(s, k, generated):
            rec = slots[s]["generated_token_ids"]
            return rec[min(k, len(rec) - 1)]

        run(model, slots, users, feed, max(len(r["generated_token_ids"]) for r in slots), on_logits)
        result["teacher"] = {}
        for name, v in sums.items():
            kld = torch.cat(v["kld"])
            result["teacher"][name] = {
                "positions": len(kld),
                "mean_kld": kld.mean().item(),
                "p99_kld": kld.quantile(0.99).item(),
                **{f"{k}_agreement": torch.cat(v[k]).mean().item() for k in ("top1", "top5", "top100")},
            }
            print(f"teacher-forced vs {name}: {result['teacher'][name]}", flush=True)
        result["teacher_seconds"] = round(time.time() - t1)
        per_position = {name: {k: torch.cat(v[k]) for k in ("kld", "top1")} for name, v in sums.items()}
        per_position["reference_row"] = torch.cat(where)

    if "greedy" in wanted and over_budget():
        result.setdefault("skipped_over_budget", []).append("greedy")
        print("time budget reached: skipping greedy", flush=True)
    elif "greedy" in wanted:
        t1 = time.time()
        gen = max(len(r["generated_token_ids"]) for r in slots)
        greedy = run(model, slots, users, lambda s, k, g: g[k], gen - 1, lambda t, logits: None)
        per_record = {}
        for s, r in enumerate(slots):
            want = r["generated_token_ids"]
            got = greedy[s][: len(want)]
            div = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
            per_record.setdefault(r["id"], []).append(len(got) if div is None else div)  # shorter if cut by the budget
        result["greedy"] = {}
        for r in records:
            firsts = per_record[r["id"]]
            full = sum(d == len(r["generated_token_ids"]) for d in firsts)
            result["greedy"][r["id"]] = {"slots": len(firsts), "all_match": full, "first_divergence": sorted(firsts)}
            print(
                f"greedy {r['id']}: {full}/{len(firsts)} slots match all {len(r['generated_token_ids'])}; first "
                f"divergences {sorted(firsts)}",
                flush=True,
            )
        result["greedy_seconds"] = round(time.time() - t1)

    if "free" in wanted and over_budget():
        result.setdefault("skipped_over_budget", []).append("free")
        print("time budget reached: skipping free", flush=True)
    elif "free" in wanted:
        t1 = time.time()
        steps = int(os.environ.get("GLM53_ACC_FREE_TOKENS", "256"))
        free = run(model, slots, users, lambda s, k, g: g[k], steps - 1, lambda t, logits: None)
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(checkpoint_root())
        except Exception as e:  # noqa: BLE001  (the text is for reading only)
            print(f"no tokenizer ({e}); token ids only", flush=True)
            tok = None
        flagged, texts = [], {}
        for s, r in enumerate(slots):
            ids = free[s][:steps]
            repeats = sum(a == b for a, b in zip(ids, ids[1:])) / max(1, len(ids) - 1)
            grams = [tuple(ids[i : i + 4]) for i in range(len(ids) - 3)]
            top_gram = max((grams.count(g) for g in set(grams)), default=0) * 4 / max(1, len(ids))
            if repeats > 0.10 or top_gram > 0.5:
                flagged.append({"slot": s, "record": r["id"], "adjacent_repeats": repeats, "top_4gram_cover": top_gram})
            if r["id"] not in texts and tok is not None:
                texts[r["id"]] = tok.decode(ids)
        result["free"] = {"tokens": steps, "slots": len(slots), "flagged": flagged, "texts": texts}
        print(f"free generation: {len(flagged)} of {len(slots)} slots flagged as degenerate", flush=True)
        result["free_seconds"] = round(time.time() - t1)

    result["over_budget"] = over_budget()
    path = out_dir() / f"accuracy-{users}x8-{kda_state}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    if "teacher" in result:
        torch.save(per_position, path.with_suffix(".pt"))
    print(f"wrote {path}", flush=True)
