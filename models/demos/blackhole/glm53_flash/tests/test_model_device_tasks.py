"""Task accuracy (GSM8K) and the generations' log-probability under the target: the common yardstick of the decode
forms that may change the text (the looser acceptance of the drafts, GLM53_ACCEPT_RELAX; lower precisions).

The first GLM53_TASKS_N problems of tools/gsm8k_prompts.py's file (GLM53_TASKS_FILE), in the prompt form
GLM53_TASKS_FORM (low: the template's "Reasoning Effort: Low", the model answers at once).  The forms in
GLM53_TASKS_MODES run in turn on one load, each through the serving generator (`Glm53Generator`, U =
GLM53_TASKS_USERS users a mesh row, 32 seats at 4), the problems arriving into the freed seats (the prefill one
request a row in chunks of 128: `dsa.PREFILL_SP` off, so that a prompt's prefill rounds the same in every form; a
one-chunk wave replayed from its trace, the same bits), each generating until an end token or GLM53_TASKS_MAX_NEW
tokens:
  greedy          the plain greedy decode (the traced step)
  wide:C          the plain greedy decode with C copies of each problem a row (U C users a row: the tokens a row of a
                  verify step with K = C - 1 drafts, whose roundings it shares; DFlash2's exact
                  acceptance must give its tokens), on the first GLM53_TASKS_WIDE_N problems (one wave)
  dflashK[:gap]   DFlash2 with K drafts a step (the drafter's block 4 for K <= 3, else 8, as the vLLM adapter), a draft
                  also taken when its logit is within `gap` nats of the target's top (the device's relaxed head,
                  `TtLmHead.greedy_mesh`'s relax); no gap or "exact": the exact acceptance
  mtp[:gap]       MTP's verify step with one draft (the model loaded with its MTP layer when a form asks for it)
A form with a gap turns the relaxed head on for the load (`prepare_relax`, before any capture); the exact forms then
run it with a gap < 0 (the same ids, bit for bit).  The model's precision switches (GLM53_EXPERT_DTYPE, ...) apply as
usual.

Per form: each answer (the last \\boxed{} / number, `gsm8k_prompts.correct`) against GSM8K's -> the accuracy; the
tokens a verify step; then teacher forcing (`TtGlm53Model.prefill_score`: the prompt and its generation prefilled in
chunks of 512, one a row): each generated token's log-probability under the target and its logit's gap below the top
(0: the greedy token there) -> the mean log-probability a token, the share of tokens not the top (gap > 0, and > 0.1
nats); against the first form: the same answers, the generations equal / their first difference.
Results: out_dir()/tasks-<stamp>.json (each form's generations' ids too).

    GLM53_TASKS_FROM (0: the first problem; GLM53_TASKS_N from it), GLM53_TASKS_MAX_SEQ (1024: the cache),
    GLM53_TASKS_SET (gsm8k; humaneval: tools/humaneval_tasks.py's, scored on the CPU afterwards by its `score`; the
    other sets' files, tools/*_tasks.py / aime24_prompts.py, through GLM53_TASKS_FILE: AIME scored here by the GSM8K
    checker, the others afterwards by their tools' `score`;
    records: the 11 greedy records' prompts, tools/acceptance/greedy-v1, no answers: the
    tokens a step and the log-probability on varied text), GLM53_TASKS_RECORDS (0; 1: the records too, after the
    problems, up to GLM53_TASKS_RECORDS_NEW (256) tokens each, reported apart),
    GLM53_TASKS_MODES (greedy,dflash7; a form "dflash7:1@NAME=value+.." with environment variables: ";" between
    the forms when a value holds ","; GLM53_KV_LATENT_FP8 / GLM53_KV_ROUND_LATENT / GLM53_DSA_POOLS_DTYPE there read
    again for the form's states, the DSA's long cache: GLM53_TASKS_MAX_SEQ past 2,048),
    GLM53_TASKS_N (32), GLM53_TASKS_USERS (4), GLM53_TASKS_MAX_NEW (512),
    GLM53_TASKS_SAMPLING ("temperature:top_p:top_k" with GLM53_DFLASH2_SAMPLING=1, DFlash2 forms: every seat draws;
    the first token after the prompt stays the prefill's greedy one),
    GLM53_TASKS_FORM (low), GLM53_TASKS_SCORE (1: teacher forcing; GLM53_TASKS_RESCORE=a.json,.. earlier runs'
    generations scored by this load, GLM53_TASKS_RESCORE_EXACT=1 their exact forms too, GLM53_TASKS_RESCORE_ONLY=1
    no forms run), GLM53_TASKS_WIDE_N (8 x users),
    GLM53_TASKS_PREFILL_TRACED (1: the one-chunk prefill waves replayed from traces, as served; 0: eager),
    GLM53_TASKS_REFILL_ROWS (4: a refill once that many rows have a free seat, or fewer problems are left, or nothing
    runs; 6: about -8 % of the generating time on GSM8K at 32 seats),
    GLM53_PREFILL_IDS (model.PREFILL_IDS, 1: each new problem's first token by the device's argmax, 8 ids read a
    wave instead of the rows' logits; 0: the host's argmax of the logits; the same tokens),
    GLM53_TASKS_RESCORE (earlier runs' JSON files, comma-separated: their generations scored again first, before any
    capture; the same problems),
    GLM53_TIME_BUDGET (seconds from the start: a form not begun is skipped, a running one stops, marked partial;
    default 720)
Run (a device test, real weights):
    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_model_device_tasks.py -s
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
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
from models.demos.blackhole.glm53_flash.tools.gsm8k_prompts import OUT as TASKS_FILE  # noqa: E402
from models.demos.blackhole.glm53_flash.tools.gsm8k_prompts import extract_answer  # noqa: E402
from models.demos.blackhole.glm53_flash.tools.humaneval_tasks import OUT as HUMANEVAL_FILE  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn import dsa as dsa_mod  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import DEVICE_PARAMS_TRACE, MESH_SHAPE, ROWS  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.dflash2 import TtDFlash2, load_weights  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

MAX_SEQ = 1024
REFILL_ROWS = int(os.environ.get("GLM53_TASKS_REFILL_ROWS", "4"))  # (`Seats.refill`)
# GLM53_TASKS_SHORT_FIRST=1: the queue's shortest prompts first (a continuation of long generations: the problems that
# have run least take the seats first, instead of the longest prefills sharing their waves)
SHORT_FIRST = os.environ.get("GLM53_TASKS_SHORT_FIRST", "0") == "1"
CHUNKS = (128,)  # the prompts (<= 159 tokens) in chunks of 128: the same chunks whatever their wave
# GLM53_TASKS_CHUNKS: other prefill chunk lengths (each dividing the next, `Glm53Generator`), e.g. 128,1024 for the long
# prompts of a continuation (GLM53_TASKS_FILE items with "prev_gen"); GLM53_TASKS_TRACED_CHUNKS: the ones captured as
# traces (the others eager)
CHUNKS = tuple(int(c) for c in os.environ.get("GLM53_TASKS_CHUNKS", ",".join(map(str, CHUNKS))).split(","))
TRACED_CHUNKS = tuple(
    int(c) for c in os.environ.get("GLM53_TASKS_TRACED_CHUNKS", ",".join(map(str, CHUNKS))).split(",")
)
# GLM53_TASKS_WARMUP_TOKENS: the warm-up's eager prefill this long (default the whole cache: at a long cache it takes
# ~2.4 s per 1K tokens and chunk length: a 132K cache's took over 15 minutes); the run then compiles the
# chunk starts past it on first use (eager prefill only: GLM53_TASKS_PREFILL_TRACED=0, as the long test)
WARMUP_TOKENS = int(os.environ.get("GLM53_TASKS_WARMUP_TOKENS", "0")) or None
SCORE_CHUNK = 512
EOS = {154820, 154827, 154829}  # generation_config's


def parse_mode(text: str) -> dict:
    """ "greedy" / "wide:8" / "dflash7" / "dflash3:0.5" / "mtp:exact" -> {kind, k, copies, gap (None: exact)}; a suffix
    "@NAME=value+NAME=value" sets environment variables a program factory reads for that form (e.g. the progressive
    form's GLM53_MOE_PROG_F: a new generator, the program cache cleared)."""
    full = text
    text, _, env_spec = text.partition("@")
    env = tuple(tuple(p.split("=", 1)) for p in env_spec.split("+") if p)
    out = _parse_mode(text)
    out["name"], out["env"] = full, env
    return out


def _parse_mode(text: str) -> dict:
    name, _, arg = text.partition(":")
    gap = None if arg in ("", "exact") or name in ("greedy", "wide") else float(arg)
    if name == "greedy":
        return dict(name=text, kind="greedy", k=0, copies=1, gap=None)
    if name == "wide":
        return dict(name=text, kind="wide", k=0, copies=int(arg), gap=None)
    if name == "mtp":
        return dict(name=text, kind="mtp", k=1, copies=1, gap=gap)
    if name.startswith("dflash"):
        return dict(name=text, kind="dflash", k=int(name[len("dflash") :]), copies=1, gap=gap)
    raise ValueError(f"GLM53_TASKS_MODES: {text}")


class Seats:
    """The problems queued into a generator's seats: a free seat takes the next problem (a refill when REFILL_ROWS
    rows have a free seat, or nothing runs), one prefill call per refill (one request a row a wave; the longest prompts
    first, so that the few past one chunk share their waves and the others' waves replay the one-chunk trace); each
    new problem's first token its greedy id (`prefill_many(greedy_ids=True)`)."""

    def __init__(self, gen, prompts: list, users: int, traced: bool):
        self.gen, self.prompts, self.users, self.traced = gen, prompts, users, traced
        self.queue = sorted(range(len(prompts)), key=lambda i: len(prompts[i]) if SHORT_FIRST else -len(prompts[i]))
        self.owner = [None] * gen.slots

    def refill(self) -> list:
        free = [s for s in range(self.gen.slots) if self.owner[s] is None]
        rows = {}
        for s in free:
            rows.setdefault(s // self.users, s)  # one free seat a row
        busy = any(o is not None for o in self.owner)
        if not self.queue or (busy and len(rows) < min(REFILL_ROWS, len(self.queue))):
            return []
        arriving = []
        for s in rows.values():
            if not self.queue:
                break
            i = self.queue.pop(0)
            self.owner[s] = i
            arriving.append((s, i))
        firsts = self.gen.prefill_many([(s, self.prompts[i]) for s, i in arriving], traced=self.traced, greedy_ids=True)
        return [(s, i, first) for (s, i), first in zip(arriving, firsts)]

    def busy(self) -> bool:
        return bool(self.queue) or any(o is not None for o in self.owner)


def finished(seq: list, max_new: int) -> bool:
    return len(seq) >= max_new or any(t in EOS for t in seq)


def trim(seq: list, max_new: int) -> list:
    seq = seq[:max_new]
    for j, t in enumerate(seq):
        if t in EOS:
            return seq[: j + 1]
    return seq


def run_plain(gen, prompts, limits, deadline, traced) -> dict:
    users = gen.users
    seats = Seats(gen, prompts, users, traced)
    out = [None] * len(prompts)
    tok = torch.zeros(gen.slots, dtype=torch.int64)
    pos = torch.full((gen.slots,), -1, dtype=torch.int64)
    steps = 0
    while seats.busy() and time.time() < deadline:
        for s, i, first in seats.refill():
            out[i] = [first]
            tok[s], pos[s] = first, len(prompts[i])
        ids = gen.decode(tok, pos, greedy=True)
        steps += 1
        for s in range(gen.slots):
            i = seats.owner[s]
            if i is None:
                continue
            if finished(out[i], limits[i]):  # (it ended at its prefill)
                seats.owner[s], pos[s] = None, -1
                continue
            out[i].append(int(ids[s]))
            tok[s], pos[s] = ids[s], pos[s] + 1
            if finished(out[i], limits[i]):
                seats.owner[s], pos[s] = None, -1
    return dict(out=out, steps=steps, accepted=None)


def run_wide(gen, prompts, limits, copies, deadline, traced) -> dict:
    """One wave: problem (row r, user u) in the row's seats j U + u for j < copies (U = gen.users / copies)."""
    users = gen.users // copies
    n = min(len(prompts), ROWS * users)
    seat = lambda i, j: (i % ROWS) * gen.users + j * users + i // ROWS  # noqa: E731
    logits = gen.prefill_many([(seat(i, j), prompts[i]) for i in range(n) for j in range(copies)], traced=traced)
    vocab = gen.model.config.vocab_size
    out = [[int(logits[i * copies].reshape(-1)[:vocab].argmax())] for i in range(n)]
    tok = torch.zeros(gen.slots, dtype=torch.int64)
    pos = torch.full((gen.slots,), -1, dtype=torch.int64)
    for i in range(n):
        for j in range(copies):
            tok[seat(i, j)], pos[seat(i, j)] = out[i][0], len(prompts[i])
    steps, copies_differ = 0, 0
    while any(not finished(o, limits[i]) for i, o in enumerate(out)) and time.time() < deadline:
        ids = gen.decode(tok, pos, greedy=True)
        steps += 1
        for i in range(n):
            if finished(out[i], limits[i]):
                continue
            got = [int(ids[seat(i, j)]) for j in range(copies)]
            copies_differ += len(set(got)) > 1
            out[i].append(got[0])
            for j in range(copies):
                tok[seat(i, j)], pos[seat(i, j)] = got[0], pos[seat(i, j)] + 1
    return dict(out=out + [None] * (len(prompts) - n), steps=steps, accepted=None, copies_differ=copies_differ)


def run_dflash(gen, prompts, limits, k, deadline, traced) -> dict:
    seats = Seats(gen, prompts, gen.users, traced)
    slots = gen.slots
    out = [None] * len(prompts)
    accepted = [[] for _ in prompts]
    changed = [0] * len(prompts)  # (the relaxed head: accepted drafts not the target's argmax, the device's count)
    bonus = torch.zeros(slots, dtype=torch.int64)
    drafts = torch.zeros(slots, k, dtype=torch.int64)
    pos = torch.full((slots,), -1, dtype=torch.int64)
    keep = torch.full((slots,), k, dtype=torch.int64)
    steps = 0
    # each step: [the seats busy at its verify, the problems prefilled before it, the refill's s, the verify's s]
    step_log = []
    while seats.busy() and time.time() < deadline:
        t_a = time.perf_counter()
        arrived = seats.refill()
        for s, i, first in arrived:
            out[i] = [first]
            bonus[s], pos[s], keep[s] = first, len(prompts[i]), k
            drafts[s] = torch.tensor(gen.first.pop(s))
        toks = torch.cat([bonus[:, None], drafts], dim=1)
        poss = torch.where(pos[:, None] >= 0, pos[:, None] + torch.arange(k + 1)[None, :], -1)
        t_b = time.perf_counter()
        n, b, d = gen.dflash_verify(toks, poss, keep)
        step_log.append(
            [
                sum(o is not None for o in seats.owner),
                len(arrived),
                round(t_b - t_a, 5),
                round(time.perf_counter() - t_b, 5),
            ]
        )
        steps += 1
        for s in range(slots):
            i = seats.owner[s]
            if i is None:
                continue
            if finished(out[i], limits[i]):
                seats.owner[s], pos[s] = None, -1
                continue
            ns = int(n[s])
            out[i] += drafts[s, :ns].tolist() + [int(b[s])]
            accepted[i].append(ns)
            if gen.dflash_changed is not None:
                changed[i] += int(gen.dflash_changed[s])
            bonus[s], drafts[s], keep[s], pos[s] = b[s], d[s], ns, pos[s] + ns + 1
            if finished(out[i], limits[i]):
                seats.owner[s], pos[s] = None, -1
    return dict(out=out, steps=steps, accepted=accepted, changed=changed, step_log=step_log)


def run_mtp(gen, prompts, limits, deadline, traced) -> dict:
    seats = Seats(gen, prompts, gen.users, traced)
    slots = gen.slots
    out = [None] * len(prompts)
    accepted = [[] for _ in prompts]
    tok = torch.zeros(slots, dtype=torch.int64)
    draft = torch.zeros(slots, dtype=torch.int64)
    pos = torch.full((slots,), -1, dtype=torch.int64)
    keep = torch.full((slots,), 1, dtype=torch.int64)
    steps = 0
    while seats.busy() and time.time() < deadline:
        for s, i, first in seats.refill():
            out[i] = [first]
            tok[s], pos[s], keep[s] = first, len(prompts[i]), 1
            draft[s] = int(gen.first.pop(s))
        tokens = torch.stack([tok, draft], dim=1)
        positions = torch.where(pos[:, None] >= 0, pos[:, None] + torch.arange(2)[None, :], -1)
        ids, levels = gen.verify(tokens, positions, keep)
        steps += 1
        for s in range(slots):
            i = seats.owner[s]
            if i is None:
                continue
            if finished(out[i], limits[i]):
                seats.owner[s], pos[s] = None, -1
                continue
            a = int(int(draft[s]) == int(ids[s, 0]))  # (the relaxed head hands back a near draft as its id)
            out[i] += [int(draft[s])] * a + [int(ids[s, a])]
            accepted[i].append(a)
            keep[s], pos[s], tok[s], draft[s] = a, pos[s] + a + 1, ids[s, a], levels[s, a]
            if finished(out[i], limits[i]):
                seats.owner[s], pos[s] = None, -1
    return dict(out=out, steps=steps, accepted=accepted)


def score_key(prompt: list, gen: list) -> str:
    return hashlib.sha1(torch.tensor([len(prompt)] + prompt + gen, dtype=torch.int32).numpy().tobytes()).hexdigest()


def score_cache_path() -> Path:
    """The teacher forcing's scores of a sequence depend on the tree and the model's settings only: kept in a file per
    (tree, its HEAD, the GLM53_ / MOE_ / MHC_ settings but this test's own and the acceptance's)."""
    home = os.environ.get("TT_METAL_HOME", "")
    head = subprocess.run(["git", "-C", home, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    # (not the MTP layer's presence: the model's own logits do not depend on it)
    own = (
        "GLM53_TASKS_",
        "GLM53_TIME_BUDGET",
        "GLM53_ACCEPT_RELAX",
        "GLM53_DFLASH2_",
        "GLM53_DEVICE_TEST",
        "GLM53_MTP",
    )
    env = sorted(
        (k, v) for k, v in os.environ.items() if k.startswith(("GLM53_", "MOE_", "MHC_")) and not k.startswith(own)
    )
    sig = hashlib.sha1(json.dumps([home, head, env]).encode()).hexdigest()[:16]
    return out_dir() / f"tasks-score-cache-{sig}.pt"


def score(model, states, prompts, gens, cache: dict | None = None) -> list:
    """Teacher forcing: each (prompt, generation) prefilled in chunks of SCORE_CHUNK, one a row -> per generation
    [n, 3] float64 (each generated token's log-probability, gap below the top, the top's log-probability); the
    sequences already in `cache` (score_key -> scores) taken from it, the others added."""
    cache = {} if cache is None else cache
    out = [None] * len(gens)
    keys = {i: score_key(prompts[i], g) for i, g in enumerate(gens) if g}
    for i, k in keys.items():
        if k in cache:
            out[i] = cache[k]
    items = [i for i in keys if out[i] is None]
    for w0 in range(0, len(items), ROWS):
        wave = items[w0 : w0 + ROWS]
        seqs = [prompts[i] + gens[i] for i in wave]
        chunks = -(-max(len(s) for s in seqs) // SCORE_CHUNK)
        slots = [0 if r < len(wave) else None for r in range(ROWS)]
        got = [[] for _ in wave]
        for c in range(chunks):
            start, t = c * SCORE_CHUNK, SCORE_CHUNK
            tokens = torch.zeros(ROWS, t, dtype=torch.int64)
            nxt = torch.full((ROWS, t), -1, dtype=torch.int64)
            rows, valid = [False] * ROWS, [t] * ROWS
            for r, seq in enumerate(seqs):
                k = min(t, len(seq) - start)
                if k <= 0:
                    continue
                rows[r], valid[r] = True, k
                tokens[r, :k] = torch.tensor(seq[start : start + k])
                ahead = seq[start + 1 : start + k + 1]
                nxt[r, : len(ahead)] = torch.tensor(ahead)
            sc = model.prefill_score(tokens, start, states, slots, nxt, rows=rows, valid=valid, last=c == chunks - 1)
            for r, (i, seq) in enumerate(zip(wave, seqs)):
                lo, hi = max(start, len(prompts[i]) - 1), min(start + t, len(seq) - 1)
                if lo < hi:
                    got[r].append(sc[r, lo - start : hi - start])
        for r, i in enumerate(wave):
            out[i] = cache[keys[i]] = torch.cat(got[r])
    return out


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_model_device_tasks(mesh_device):
    t_start = time.time()
    budget = float(os.environ.get("GLM53_TIME_BUDGET", "720"))
    modes_spec = os.environ.get(
        "GLM53_TASKS_MODES", "greedy,dflash7"
    )  # (";"-separated when a form's settings hold ",")
    modes = [parse_mode(m) for m in modes_spec.split(";" if ";" in modes_spec else ",")]
    if os.environ.get("GLM53_TASKS_RESCORE_ONLY") == "1":  # (GLM53_TASKS_RESCORE's teacher forcing only: no forms)
        modes = []
    n = int(os.environ.get("GLM53_TASKS_N", "32"))
    users = int(os.environ.get("GLM53_TASKS_USERS", "4"))
    max_new = int(os.environ.get("GLM53_TASKS_MAX_NEW", "512"))
    form = os.environ.get("GLM53_TASKS_FORM", "low")
    do_score = os.environ.get("GLM53_TASKS_SCORE", "1") == "1"
    # the prompts' one-chunk waves replayed from the generator's prefill traces (the same bits as eager)
    traced = os.environ.get("GLM53_TASKS_PREFILL_TRACED", "1") == "1"
    wide_n = int(os.environ.get("GLM53_TASKS_WIDE_N", str(ROWS * users)))
    task_set = os.environ.get("GLM53_TASKS_SET", "gsm8k")
    first = int(os.environ.get("GLM53_TASKS_FROM", "0"))  # (the first problem: a run's share of a set)
    max_seq = int(os.environ.get("GLM53_TASKS_MAX_SEQ", str(MAX_SEQ)))
    if task_set == "records":
        records = Path(__file__).resolve().parents[1] / "tools" / "acceptance" / "greedy-v1"
        items = [json.loads(f.read_text()) for f in sorted(records.glob("*.json"))][:n]
        items = [{"idx": r["id"], "answer": None, "prompt_ids": {form: r["prompt_token_ids"]}} for r in items]
    else:
        default = HUMANEVAL_FILE if task_set == "humaneval" else TASKS_FILE
        doc = json.loads(Path(os.environ.get("GLM53_TASKS_FILE", str(default))).read_text())
        items = doc["items"][first : first + n]
    for x in items:
        x["set"] = task_set
    if os.environ.get("GLM53_TASKS_RECORDS", "0") == "1" and task_set != "records":
        records = Path(__file__).resolve().parents[1] / "tools" / "acceptance" / "greedy-v1"
        for f in sorted(records.glob("*.json")):
            r = json.loads(f.read_text())
            items.append(
                {"idx": r["id"], "answer": None, "prompt_ids": {form: r["prompt_token_ids"]}, "set": "records"}
            )
    # a continuation (an item's "prev_gen": the tokens an earlier run generated, tools/tasks_continue.py): prefilled after
    # the prompt, the generation goes on up to GLM53_TASKS_MAX_NEW tokens in all (the prefill rounds the earlier tokens
    # a little differently from the decode that made them, as a server restarted mid-answer)
    prevs = [list(x.get("prev_gen") or []) for x in items]
    prompts = [x["prompt_ids"][form] + prev for x, prev in zip(items, prevs)]
    rec_new = int(os.environ.get("GLM53_TASKS_RECORDS_NEW", "256"))
    limits = [rec_new if x["set"] == "records" and task_set != "records" else max_new for x in items]
    limits = [max(1, lim - len(prev)) for lim, prev in zip(limits, prevs)]
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(checkpoint_root()) / "tokenizer.json"))
    # (one request a row: a prompt's prefill the same in every form; GLM53_TASKS_PREFILL_SP=1 keeps the serving's
    # sequence-parallel prefill and its extra DSA user, for the memory of the serving's form)
    dsa_mod.PREFILL_SP = os.environ.get("GLM53_TASKS_PREFILL_SP") == "1"
    mtp = any(m["kind"] == "mtp" for m in modes)
    # GLM53_SYNTHETIC_WEIGHTS=1: random weights (the timing and memory of a form only; the text is noise)
    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1"
    model = TtGlm53Model(
        mesh_device, ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint(), max_seq=max_seq, mtp=mtp
    )
    relaxed = any(m["gap"] is not None for m in modes)
    if relaxed:  # (its constants before any capture; the exact forms then at a gap < 0)
        model.lm_head.prepare_relax(-1.0)
    if do_score:
        model.lm_head.prepare_scoring()
        score_states = model.empty_state(1)
        cache_path = score_cache_path()
        cache = torch.load(cache_path) if cache_path.exists() else {}
        # (scores made after a capture, before the warm-up below: garbage; dropped)
        bad = [
            k
            for k, v in cache.items()
            if not bool(torch.isfinite(v).all()) or float((v[:, 1] > 0).float().mean()) > 0.3
        ]
        for k in bad:
            del cache[k]
        print(f"teacher forcing's cache {cache_path}: {len(cache)} sequences ({len(bad)} bad dropped)", flush=True)
        # the scoring's prefill (chunks of 512 from 0 and 512) once now: the device constants it makes on first use
        # must come before any capture (made after one, a replay overwrote them: NaN scores)
        t1 = time.time()
        dummy = [0] * 100  # (every chunk start the scored sequences may take, up to the cache's end)
        score(model, score_states, [dummy], [[1] * max(SCORE_CHUNK + 64, max_seq - 164)])
        print(f"teacher forcing warmed up in {time.time() - t1:.0f} s", flush=True)
        rescored = {}
        # GLM53_TASKS_RESCORE_EXACT=1: the exact forms too (runs made with GLM53_TASKS_SCORE=0 have no scores of
        # their own); the files' forms taken while the time budget lasts
        rescore_exact = os.environ.get("GLM53_TASKS_RESCORE_EXACT") == "1"
        for entry in [p_ for p_ in os.environ.get("GLM53_TASKS_RESCORE", "").split(",") if p_]:  # earlier runs' scores
            path, _, only = entry.partition("#")  # ("file.json#form|form": those forms only, by name before any "@")
            only = set(only.split("|")) if only else None
            for r in json.loads(Path(path).read_text())["results"]:
                if only is not None and r["mode"].partition("@")[0] not in only:
                    continue
                if r["gap"] is None and not rescore_exact:  # (a file's first form, scored before any replay after
                    continue  # its captures: sound)
                if time.time() > t_start + budget - 60:
                    print(f"rescore: {Path(path).name} {r['mode']} skipped (the time budget)", flush=True)
                    continue
                done_r = [int(k) for k in r["generations"]]
                fresh = {}  # (scored again, not taken from the cache: entries scored after a capture may be wrong)
                sc = score(
                    model, score_states, prompts, [r["generations"].get(str(i)) for i in range(len(items))], fresh
                )
                cache.update(fresh)
                for name in dict.fromkeys(x["set"] for x in items):
                    mine = [i for i in done_r if items[i]["set"] == name]
                    if not mine:
                        continue
                    allsc = torch.cat([sc[i] for i in mine])
                    rescored[f"{Path(path).name} {r['mode']} {name}"] = st = {
                        "mean_logprob": allsc[:, 0].mean().item(),
                        "not_top": (allsc[:, 1] > 0).float().mean().item(),
                        "not_top_01": (allsc[:, 1] > 0.1).float().mean().item(),
                        "tokens": int(allsc.shape[0]),
                    }
                    print(
                        f"rescored {Path(path).name} {r['mode']} [{name}]: mean log-prob {st['mean_logprob']:.4f} a token, "
                        f"not the top {st['not_top']:.4f} (> 0.1 nats {st['not_top_01']:.4f}), {st['tokens']} tokens",
                        flush=True,
                    )
            torch.save(cache, cache_path)
    drafter_weights = load_weights() if any(m["kind"] == "dflash" for m in modes) else None
    print(
        f"loaded in {time.time() - t_start:.0f} s (MTP layer {mtp}, relaxed head {relaxed}); {len(items)} problems, "
        f"form {form}, {users} users a row, up to {max_new} tokens; forms {[m['name'] for m in modes]}",
        flush=True,
    )

    results = []
    current = None  # (key, generator, drafter)
    env_now, env_undo = (), []  # (the forms' "@" environment variables: set now, their earlier values)
    first_out = None
    for mode in modes:
        left = t_start + budget - time.time()
        if left < 60:
            print(f"{mode['name']}: skipped, {left:.0f} s of the time budget left", flush=True)
            continue
        key = (mode["kind"], mode["k"], mode["copies"], mode["env"])
        if current is None or current[0] != key:
            if current is not None:
                current[1].release()
                current = None
                gc.collect()
            if mode["env"] != env_now:  # (the form's environment: the earlier form's undone, the programs made again)
                for k_, v_ in env_undo:
                    if v_ is None:
                        os.environ.pop(k_, None)
                    else:
                        os.environ[k_] = v_
                env_undo = [(k_, os.environ.get(k_)) for k_, _ in mode["env"]]
                for k_, v_ in mode["env"]:
                    os.environ[k_] = v_
                env_now = mode["env"]
                # the DSA's KV cache switches (read at import) read again for the form: the generator's states made
                # below take them (GLM53_KV_LATENT_FP8 the latents stored in FP8, GLM53_KV_ROUND_LATENT rounded)
                dsa_mod.KV_LATENT_FP8 = os.environ.get("GLM53_KV_LATENT_FP8", "1") == "1"
                dsa_mod.KV_ROUND_LATENT = int(os.environ.get("GLM53_KV_ROUND_LATENT", "0"))
                dsa_mod.POOLS_DTYPE = os.environ.get("GLM53_DSA_POOLS_DTYPE", "bfp4")  # (the pools' storage form)
                mesh_device.clear_program_cache()
            t0 = time.time()
            drafter = None
            if mode["kind"] == "dflash":
                cfg, w, mask = drafter_weights
                drafter = TtDFlash2(
                    mesh_device, cfg, w, mask, users=users, max_ctx=max_seq + 8, block=4 if mode["k"] <= 3 else 8,
                    head_weight=model.lm_head.mesh_weight, embedding=model.embed,
                )  # fmt: skip
                gen = Glm53Generator(model, users, chunk=CHUNKS, dflash=(drafter, mode["k"]))
            elif mode["kind"] == "mtp":
                gen = Glm53Generator(model, users, chunk=CHUNKS, spec=1)
            else:
                gen = Glm53Generator(model, users * mode["copies"], chunk=CHUNKS)
            warm = gen.warmup(
                greedy=mode["kind"] in ("greedy", "wide"),
                logits=False,
                tokens=WARMUP_TOKENS,
                traced=TRACED_CHUNKS if traced else (),
            )
            current = (key, gen, drafter)
            print(f"{mode['name']}: generator ready in {time.time() - t0:.0f} s {warm}", flush=True)
            dram = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)  # (one chip's banks; every chip the same)
            banks = int(dram.num_banks)
            print(
                f"{mode['name']}: cache {gen.max_seq}, DRAM a chip {int(dram.total_bytes_allocated_per_bank) * banks / 2**30:.2f} "
                f"GiB allocated, {int(dram.total_bytes_free_per_bank) * banks / 2**30:.2f} GiB free",
                flush=True,
            )
        gen = current[1]
        # GLM53_TASKS_SAMPLING="temperature:top_p:top_k" (a form's, with GLM53_DFLASH2_SAMPLING=1):
        # every seat draws with these (seeds by seat; GLM53_TASKS_SAMPLING_SEED shifts them: another run's draws), else
        # greedy
        spec_sampling = os.environ.get("GLM53_TASKS_SAMPLING")
        seed_shift = int(os.environ.get("GLM53_TASKS_SAMPLING_SEED", "0")) * 1000003
        if spec_sampling and getattr(gen.dstep, "sampling", None) is not None:
            t_, p_, k_ = (float(x) for x in spec_sampling.split(":"))
            shape = (ROWS, users)
            gen.dstep.set_sampling(
                torch.full(shape, int(k_)), torch.full(shape, p_), torch.full(shape, t_),
                torch.arange(ROWS * users).view(shape) * 7919 + 12345 + seed_shift,
            )  # fmt: skip
            print(f"{mode['name']}: sampling temperature {t_}, top-p {p_}, top-k {int(k_)} (0: none)", flush=True)
        if relaxed:
            model.lm_head.set_relax(-1.0 if mode["gap"] is None else mode["gap"])
        deadline = t_start + budget - (60 if do_score else 20)  # (the teacher forcing after it: ~20-40 s)
        t0 = time.time()
        if mode["kind"] == "greedy":
            run = run_plain(gen, prompts, limits, deadline, traced)
        elif mode["kind"] == "wide":
            run = run_wide(gen, prompts[:wide_n], limits[:wide_n], mode["copies"], deadline, traced)
            run["out"] += [None] * (len(prompts) - len(run["out"]))
        elif mode["kind"] == "dflash":
            run = run_dflash(gen, prompts, limits, mode["k"], deadline, traced)
        else:
            run = run_mtp(gen, prompts, limits, deadline, traced)
        gen_seconds = time.time() - t0
        outs = [trim(o, limits[i]) if o is not None else None for i, o in enumerate(run["out"])]
        done = [i for i, o in enumerate(outs) if o is not None and finished(o, limits[i])]
        partial = len(done) < sum(o is not None for o in outs) or (mode["kind"] != "wide" and len(done) < len(items))
        texts = {i: tokenizer.decode(prevs[i] + outs[i], skip_special_tokens=True) for i in done}
        answers = {i: extract_answer(texts[i]) for i in done}
        correct = {
            i: answers[i] is not None and items[i]["answer"] is not None and abs(answers[i] - items[i]["answer"]) < 1e-6
            for i in done
        }
        sc = None
        if do_score and done:
            t1 = time.time()
            before = len(cache)
            sc = score(model, score_states, prompts, [outs[i] if i in done else None for i in range(len(items))], cache)
            score_seconds = round(time.time() - t1, 1)
            if len(cache) > before:
                torch.save(cache, cache_path)
            scored_new = len(cache) - before
        rec = {
            "mode": mode["name"],
            "gap": mode["gap"],
            "partial": partial,
            "steps": run["steps"],
            "generate_seconds": round(gen_seconds, 1),
            "copies_differ": run.get("copies_differ"),
            "sets": {},
        }
        if run.get("step_log") is not None:
            rec["step_log"] = run["step_log"]
        if sc is not None:
            rec["score_seconds"], rec["scored_new"] = score_seconds, scored_new
        if first_out is not None:
            base_name, base, base_answers = first_out
        for name in dict.fromkeys(x["set"] for x in items):
            mine = [i for i in done if items[i]["set"] == name]
            if not mine:
                continue
            acc = [run["accepted"][i] for i in mine] if run["accepted"] else None
            st = {
                "problems": len(mine),
                "correct": sum(correct[i] for i in mine),
                "accuracy": sum(correct[i] for i in mine) / len(mine),
                "tokens_a_step": sum(sum(a + 1 for a in x) for x in acc) / max(1, sum(len(x) for x in acc))
                if acc
                else 1.0,
                "generated_tokens": sum(len(outs[i]) for i in mine),
                "ended_by_eos": sum(outs[i][-1] in EOS for i in mine),
            }
            if acc:  # the drafts a step accepted: 0 .. K, the steps of each
                hist = [0] * (mode["k"] + 1)
                for x in acc:
                    for a in x:
                        hist[min(a, mode["k"])] += 1
                st["accepted_hist"] = hist
            if run.get("changed") is not None and mode["gap"] is not None:  # (the device's count, steps overshooting an
                # end included)
                st["not_greedy_taken"] = sum(run["changed"][i] for i in mine) / max(1, st["generated_tokens"])
            if sc is not None:
                allsc = torch.cat([sc[i] for i in mine])
                st.update(
                    mean_logprob=allsc[:, 0].mean().item(),
                    not_top=(allsc[:, 1] > 0).float().mean().item(),
                    not_top_01=(allsc[:, 1] > 0.1).float().mean().item(),
                    top_logprob=allsc[:, 2].mean().item(),
                    per_problem_logprob={str(items[i]["idx"]): sc[i][:, 0].mean().item() for i in mine},
                )
            if first_out is not None:
                both = [i for i in mine if base[i] is not None and finished(base[i], limits[i])]
                st["vs"] = {
                    "form": base_name,
                    "problems": len(both),
                    "same_generation": sum(outs[i] == base[i] for i in both),
                    "same_answer": sum(answers[i] == base_answers[i] for i in both),
                    "first_differences": [
                        next((j for j in range(min(len(outs[i]), len(base[i]))) if outs[i][j] != base[i][j]), None)
                        for i in both
                        if outs[i] != base[i]
                    ],
                }
            rec["sets"][name] = st
            line = (
                f"{mode['name']} [{name}]: accuracy {st['accuracy']:.3f} ({st['correct']} / {st['problems']}"
                f"{', partial' if partial else ''}), {st['tokens_a_step']:.3f} tokens a step, {st['generated_tokens']} "
                f"tokens ({st['ended_by_eos']} ended); {rec['generate_seconds']} s generating"
            )
            if "accepted_hist" in st:
                total = max(1, sum(st["accepted_hist"]))
                line += (
                    f"; accepted a step 0..{mode['k']}: {st['accepted_hist']} of {total} steps ("
                    + " ".join(f"{c / total:.3f}" for c in st["accepted_hist"])
                    + ")"
                )
            if "not_greedy_taken" in st:
                line += f"; drafts taken not the argmax {st['not_greedy_taken']:.4f} a token"
            if sc is not None:
                line += (
                    f"; teacher forcing: mean log-prob {st['mean_logprob']:.4f} a token, not the top {st['not_top']:.4f}"
                    f" (> 0.1 nats {st['not_top_01']:.4f}), {score_seconds} s ({scored_new} new sequences)"
                )
            if "vs" in st:
                v = st["vs"]
                line += (
                    f"; against {v['form']}: {v['same_generation']} / {v['problems']} the same generations, "
                    f"{v['same_answer']} the same answers, first differences "
                    f"{sorted(x for x in v['first_differences'] if x is not None)[:12]}"
                )
            print(line, flush=True)
        if first_out is None:
            first_out = (mode["name"], outs, answers)
        rec["answers"] = {str(i): answers[i] for i in done}
        rec["generations"] = {str(i): prevs[i] + outs[i] for i in done}
        # (unfinished at the time budget: the tokens so far, to be continued by another run)
        rec["unfinished"] = {str(i): prevs[i] + o for i, o in enumerate(outs) if o is not None and i not in done}
        if run["accepted"]:
            rec["accepted"] = {str(i): run["accepted"][i] for i in done}
        results.append(rec)
        if mode is modes[0] and done:
            i0 = done[0]
            print(f"  problem {items[i0]['idx']} ({len(outs[i0])} tokens): {texts[i0][-400:]!r}", flush=True)
    if current is not None:
        current[1].release()
    path = out_dir() / f"tasks-{task_set}-{form}-{first}+{n}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(
        json.dumps(
            {
                "problems": n,
                "first": first,
                "problem_ids": [x["idx"] for x in items],  # (the generations' keys: an index into this list)
                "set": task_set,
                "form": form,
                "users": users,
                "max_new": max_new,
                "precision": {k: v for k, v in os.environ.items() if k.startswith(("GLM53_EXPERT", "GLM53_WEIGHT"))},
                "results": results,
                "rescored": rescored if do_score else {},
                "seconds": round(time.time() - t_start),
            }
        )
        + "\n"
    )
    print(f"wrote {path}; {time.time() - t_start:.0f} s", flush=True)
    if not results:  # (the time budget spent before any form: nothing measured, not a failure of the model)
        print("TASKS: no form ran (the time budget)", flush=True)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_prefill_forms_same(mesh_device):
    """Whether the tasks test's prefill gives each prompt the same logits whatever its form (synthetic weights,
    seeded: the same in every run): 8 prompts of 60-128 tokens one a row in chunks of 128, eager (a) and again with the
    rows shuffled (b), alone on its row (c: one request a wave), and replayed from the one-chunk trace (d, after the
    warm-up's capture).  (Two runs, eager with the queue in order and traced with the queue by length, once generated
    differently from the first token on 98 of 211 problems.)  Prints the comparisons (bit for bit); fails on none."""
    from models.demos.blackhole.glm53_flash.checkpoint import ShapeOnlyCheckpoint

    dsa_mod.PREFILL_SP = False
    model = TtGlm53Model(mesh_device, ShapeOnlyCheckpoint(), max_seq=MAX_SEQ, mtp=False)
    gen = Glm53Generator(model, 4, chunk=CHUNKS)
    g = torch.Generator().manual_seed(23)
    vocab = model.config.vocab_size
    prompts = [torch.randint(0, vocab, (60 + 9 * r,), generator=g).tolist() for r in range(ROWS)]

    def run(order, traced=False):
        got = gen.prefill_many([(row * 4 + 1, prompts[i]) for row, i in enumerate(order)], traced=traced)
        return {i: x.clone().float() for i, x in zip(order, got)}

    def same(a, b):
        return [i for i in a if not torch.equal(a[i].view(torch.int32), b[i].view(torch.int32))]

    order = list(range(ROWS))
    a = run(order)
    b = run(order[::-1])
    c = {}
    for i in order:
        c[i] = gen.prefill_many([(i * 4 + 2, prompts[i])], traced=False)[0].clone().float()
    gen.warmup(greedy=True, logits=False, traced=CHUNKS)
    d = run(order, traced=True)
    print(f"traced waves {gen.counts}", flush=True)
    for label, other in (("rows shuffled", b), ("one a wave", c), ("traced", d)):
        diff = same(a, other)
        worst = max((float((a[i] - other[i]).abs().max()) for i in diff), default=0.0)
        flips = sum(int(a[i].argmax() != other[i].argmax()) for i in diff)
        print(
            f"PREFILL FORMS eager against {label}: {ROWS - len(diff)} / {ROWS} prompts bit for bit (max |diff| "
            f"{worst:.3g}, top-1 flips {flips})",
            flush=True,
        )
    gen.release()


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_seat_reuse_same(mesh_device):
    """Whether a prompt prefilled into a used seat gives what it gives in a fresh one (synthetic weights, seeded): the
    prompt P into seat 1 (row 0's user 1) of a fresh generator, its prefill logits and 4 greedy decode steps' logits
    (the traced step, every other seat idle); then Q there and 24 decode steps (the seat used), P again, the same
    steps; with the prefill eager and from its trace (P <= 128 tokens: a one-chunk wave), and with the other seats of
    the row busy too.  (The tasks test once found first tokens that changed with the seats' history.)  Prints
    the comparisons (bit for bit); fails on none."""
    from models.demos.blackhole.glm53_flash.checkpoint import ShapeOnlyCheckpoint

    dsa_mod.PREFILL_SP = False
    model = TtGlm53Model(mesh_device, ShapeOnlyCheckpoint(), max_seq=MAX_SEQ, mtp=False)
    gen = Glm53Generator(model, 4, chunk=CHUNKS)
    gen.warmup(greedy=False, logits=True, traced=CHUNKS)
    g = torch.Generator().manual_seed(29)
    vocab = model.config.vocab_size
    prompt = lambda n: torch.randint(0, vocab, (n,), generator=g).tolist()  # noqa: E731
    p, q, others = prompt(100), prompt(120), [prompt(70 + 10 * k) for k in range(3)]
    seat = 1

    def run(tokens, traced, steps, busy=False):
        """tokens into the seat (and, busy, the row's other seats their own prompts) -> [prefill logits, steps' logits]."""
        reqs = [(seat, tokens)] + ([(u, o) for u, o in zip((0, 2, 3), others)] if busy else [])
        got = gen.prefill_many(reqs, traced=traced)
        out = [got[0].clone().float()]
        tok = torch.zeros(gen.slots, dtype=torch.int64)
        pos = torch.full((gen.slots,), -1, dtype=torch.int64)
        for (s, t), lg in zip(reqs, got):
            tok[s], pos[s] = int(lg.reshape(-1)[:vocab].argmax()), len(t)
        for _ in range(steps):
            logits = gen.decode(tok, pos, greedy=False)
            out.append(logits[seat].clone().float())
            nxt = logits[:, :vocab].argmax(-1)
            tok = torch.where(pos >= 0, nxt, tok)
            pos = torch.where(pos >= 0, pos + 1, pos)
        return out

    def same(a, b):
        return [j for j, (x, y) in enumerate(zip(a, b)) if not torch.equal(x.view(torch.int32), y.view(torch.int32))]

    fresh = run(p, False, 4)
    for label, traced, busy in (
        ("eager", False, False),
        ("traced", True, False),
        ("eager, the row busy", False, True),
        ("traced, the row busy", True, True),
    ):
        run(q, False, 24, busy=busy)  # (the seat used: another prompt and its decode)
        again = run(p, traced, 4, busy=busy)
        diff = same(fresh, again)
        flips = sum(int(fresh[j].argmax() != again[j].argmax()) for j in diff)
        print(
            f"SEAT REUSE ({label}): prefill + 4 steps against the fresh seat: {5 - len(diff)} / 5 bit for bit "
            f"(differ at {diff}, top-1 flips {flips})",
            flush=True,
        )
    gen.release()


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_prefill_history_real(mesh_device):
    """The tasks test's prefill logits with the real weights: GSM8K's first 8 prompts (<= 128 tokens), one a row, in
    the DFlash2 generator (K = 7, 4 users a row, its prefill traces captured): (A) eager into fresh seats (user 1),
    (B) eager again into the same seats after 16 DFlash2 steps there, (C) from the trace into the same seats, (E) from
    the trace into fresh seats (user 2); then the plain generator's (D) eager into fresh seats.  (The tasks test once
    found first tokens that changed with the seats' history or the prefill's form; synthetic weights: none.)
    Prints the comparisons (bit for bit, and the top-1); fails on none."""
    dsa_mod.PREFILL_SP = False
    model = TtGlm53Model(mesh_device, Glm53Checkpoint(), max_seq=MAX_SEQ, mtp=False)
    doc = json.loads(Path(TASKS_FILE).read_text())
    prompts = [x["prompt_ids"]["low"] for x in doc["items"] if len(x["prompt_ids"]["low"]) <= 128][:ROWS]
    cfg, w, mask = load_weights()
    k = 7
    drafter = TtDFlash2(
        mesh_device, cfg, w, mask, users=4, max_ctx=MAX_SEQ + 8, block=8, head_weight=model.lm_head.mesh_weight,
        embedding=model.embed,
    )  # fmt: skip
    gen = Glm53Generator(model, 4, chunk=CHUNKS, dflash=(drafter, k))
    gen.warmup(greedy=False, logits=False, traced=CHUNKS)
    vocab = model.config.vocab_size

    def pre(user, traced, g=gen):
        got = g.prefill_many([(r * 4 + user, p) for r, p in enumerate(prompts)], traced=traced)
        g.first.clear()
        return [x.clone().float().reshape(-1)[:vocab] for x in got]

    def compare(label, a, b):
        bits = sum(torch.equal(x.view(torch.int32), y.view(torch.int32)) for x, y in zip(a, b))
        top = sum(int(x.argmax() == y.argmax()) for x, y in zip(a, b))
        worst = max(float((x - y).abs().max()) for x, y in zip(a, b))
        print(
            f"PREFILL HISTORY {label}: {bits} / {len(a)} bit for bit, top-1 the same {top} / {len(a)}, max |diff| "
            f"{worst:.4g}",
            flush=True,
        )

    a = pre(1, False)
    # 16 DFlash2 steps in those seats (their first drafts dropped: zeros)
    slots = gen.slots
    toks = torch.zeros(slots, k + 1, dtype=torch.int64)
    poss = torch.full((slots, k + 1), -1, dtype=torch.int64)
    keep = torch.full((slots,), k, dtype=torch.int64)
    for r, p in enumerate(prompts):
        s = r * 4 + 1
        toks[s, 0] = int(a[r].argmax())
        poss[s] = len(p) + torch.arange(k + 1)
    for _ in range(16):
        n, b, d = gen.dflash_verify(toks, poss, keep)
        for r in range(ROWS):
            s = r * 4 + 1
            ns = int(n[s])
            toks[s, 0], toks[s, 1:] = b[s], d[s]
            poss[s] = poss[s, 0] + ns + 1 + torch.arange(k + 1)
            keep[s] = ns
    b_ = pre(1, False)
    c = pre(1, True)
    e = pre(2, True)
    print(f"counts {gen.counts}", flush=True)
    gen.release()
    plain = Glm53Generator(model, 4, chunk=CHUNKS)
    plain.warmup(greedy=True, logits=False)
    d_ = pre(3, False, plain)
    plain.release()
    compare("A eager fresh / B eager after 16 DFlash2 steps in the seat", a, b_)
    compare("A eager fresh / C traced in the used seat", a, c)
    compare("A eager fresh / E traced fresh", a, e)
    compare("A DFlash2 generator eager fresh / D plain generator eager fresh", a, d_)


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_prefill_companions_real(mesh_device):
    """Whether a prompt's prefill logits depend on the other rows' prompts of its wave, with the real weights (the
    plain generator, 4 users a row, eager, chunks of 128): GSM8K's prompts 0 .. 7 one a row (A); prompt 0 with prompts
    8 .. 14 on the other rows (B); prompt 0 alone (C); prompts 0 .. 7 with the rows rotated by 3 (D); each prompt's
    logits against A (bit for bit, top-1, and the top two's gap at A).  (None with the same 8 prompts in every wave;
    the tasks test's waves are of any problems.)  Prints only."""
    dsa_mod.PREFILL_SP = False
    model = TtGlm53Model(mesh_device, Glm53Checkpoint(), max_seq=MAX_SEQ, mtp=False)
    doc = json.loads(Path(TASKS_FILE).read_text())
    prompts = [x["prompt_ids"]["low"] for x in doc["items"] if len(x["prompt_ids"]["low"]) <= 128][:15]
    gen = Glm53Generator(model, 4, chunk=CHUNKS)
    gen.warmup(greedy=True, logits=False)
    vocab = model.config.vocab_size

    def pre(placed):  # {row: prompt index} -> {prompt index: logits}
        got = gen.prefill_many([(row * 4 + 1, prompts[i]) for row, i in placed.items()], traced=False)
        return {i: x.clone().float().reshape(-1)[:vocab] for i, x in zip(placed.values(), got)}

    a = pre({r: r for r in range(ROWS)})
    cases = {
        "B prompt 0 with prompts 8 .. 14": pre({0: 0, **{r: 7 + r for r in range(1, ROWS)}}),
        "C prompt 0 alone": pre({0: 0}),
        "D prompts 0 .. 7 on rows rotated by 3": pre({(r + 3) % ROWS: r for r in range(ROWS)}),
    }
    gaps = {i: float(x.topk(2).values[0] - x.topk(2).values[1]) for i, x in a.items()}
    print(
        f"PREFILL COMPANIONS the top two's gap at A: {', '.join(f'{i}: {g:.3f}' for i, g in gaps.items())}", flush=True
    )
    for label, got in cases.items():
        got = {i: x for i, x in got.items() if i in a}  # (B's other prompts are not in A)
        bits = [i for i in got if torch.equal(got[i].view(torch.int32), a[i].view(torch.int32))]
        top = [i for i in got if int(got[i].argmax()) == int(a[i].argmax())]
        worst = max(float((got[i] - a[i]).abs().max()) for i in got)
        print(
            f"PREFILL COMPANIONS {label}: {len(bits)} / {len(got)} bit for bit, top-1 the same {len(top)} / "
            f"{len(got)}, max |diff| {worst:.4g}",
            flush=True,
        )
    gen.release()
