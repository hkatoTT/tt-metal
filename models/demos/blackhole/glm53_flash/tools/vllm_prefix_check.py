"""Multi-turn conversations through the vLLM OpenAI server (GLM-5.3-Flash adapter), for the prefix store
(GLM53_PREFIX_ENTRIES): each conversation's first prompt is about --first tokens of text (a passkey prompt), and each
next turn's prompt is the last prompt, the last output's token ids and about --new tokens of
new text, so that a turn's prompt starts with the last one exactly (token ids, no chat template).  Greedy, --out-len
tokens a turn (no early stop), streamed with the token ids (`return_token_ids`).  Per turn: the time to its first token,
its time, its output ids.  Run once with the store off and once on against the same server settings, then compare:

    python models/demos/blackhole/glm53_flash/tools/vllm_prefix_check.py --port PORT --out off.json
    python models/demos/blackhole/glm53_flash/tools/vllm_prefix_check.py --compare off.json on.json

The standard library and the checkpoint's tokenizer (`tokenizers`), so it runs in the vLLM env beside the server.
Prints one line a turn and a summary; exits 0 (a measurement, not a pass / fail check).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # (the tt-metal root: tools/passkey.py's package)

from models.demos.blackhole.glm53_flash.tools.passkey import DEPTHS, WORDS, filler, prompt_ids  # noqa: E402

MODEL = "glm-5.3-flash"


def turn(port: int, ids: list, out_len: int) -> dict:
    """One greedy request, streamed -> its first token's and last event's times, its output ids (or the error)."""
    body = dict(
        model=MODEL, prompt=ids, max_tokens=out_len, temperature=0, ignore_eos=True, stream=True, return_token_ids=True
    )
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    rec = dict(tokens_in=len(ids), t0=time.time(), out=[])
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                event = json.loads(line[5:])
                rec.setdefault("id", event.get("id"))  # (the response's id: the engine's request id starts with it)
                for choice in event.get("choices") or []:
                    got = choice.get("token_ids") or []
                    if got and "t_first" not in rec:
                        rec["t_first"] = time.time()
                    rec["out"].extend(int(t) for t in got)
        rec["ok"] = len(rec["out"]) > 0
    except Exception as e:  # (a failed request: reported, not raised)
        rec.update(ok=False, error=repr(e)[:300])
    rec["t1"] = time.time()
    rec["ttft"] = round(rec.get("t_first", rec["t1"]) - rec["t0"], 3)
    return rec


def conversation(port: int, first: list, news: list, out_len: int, recs: list) -> None:
    ids = list(first)
    for k in range(len(news) + 1):
        rec = turn(port, ids, out_len)
        recs.append(rec)
        if not rec["ok"] or k == len(news):
            return
        ids = ids + rec["out"] + news[k]


def compare(a_path: str, b_path: str) -> None:
    a, b = (json.load(open(p)) for p in (a_path, b_path))
    same, diff, first_diff = 0, 0, []
    for c, (ca, cb) in enumerate(zip(a["conversations"], b["conversations"])):
        for k, (ta, tb) in enumerate(zip(ca, cb)):
            if ta.get("out") == tb.get("out"):
                same += 1
            else:
                diff += 1
                at = next((i for i, (x, y) in enumerate(zip(ta["out"], tb["out"])) if x != y), None)
                first_diff.append((c, k, at))
    print(
        f"prefix check compare: {same} turns the same, {diff} different (conversation, turn, first different token: "
        f"{first_diff[:16]}); TTFT medians by turn {a['summary']['ttft_median_by_turn']} -> "
        f"{b['summary']['ttft_median_by_turn']}; output tok/s {a['summary']['out_tok_s']} -> {b['summary']['out_tok_s']}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int)
    p.add_argument("--tokenizer", default=os.path.join(os.environ.get("GLM53_CHECKPOINT", ""), "tokenizer.json"))
    p.add_argument("--convs", type=int, default=8, help="conversations at once")
    p.add_argument("--turns", type=int, default=4)
    p.add_argument("--first", type=int, default=30000, help="the first prompt's tokens")
    p.add_argument("--new", type=int, default=800, help="new tokens a later turn")
    p.add_argument("--out-len", type=int, default=128)
    p.add_argument("--stagger", type=float, default=1.0, help="s between the conversations' starts")
    p.add_argument("--out")
    p.add_argument("--compare", nargs=2, metavar=("OFF", "ON"))
    args = p.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    firsts, news = [], []
    for i in range(args.convs):
        answer = f"{WORDS[i % len(WORDS)]}-{4000 + 731 * i}"
        firsts.append(prompt_ids(tok, args.first, DEPTHS[i % len(DEPTHS)], *answer.split("-")))
        conv = []
        for k in range(args.turns - 1):
            text = "".join(filler(100000 * (i + 1) + 1000 * k + j) for j in range(args.new // 15))
            conv.append(tok.encode(text).ids[: args.new])
        news.append(conv)
    recs = [[] for _ in range(args.convs)]
    threads = [
        threading.Thread(target=conversation, args=(args.port, f, n, args.out_len, r))
        for f, n, r in zip(firsts, news, recs)
    ]
    t_start = time.time()
    for t in threads:
        t.start()
        time.sleep(args.stagger)
    for t in threads:
        t.join()
    span = max(r["t1"] for conv in recs for r in conv) - t_start
    for i, conv in enumerate(recs):
        for k, r in enumerate(conv):
            print(
                f"prefix check conv {i} turn {k}: {r['tokens_in']} tokens, TTFT {r['ttft']:.2f} s, "
                f"{r['t1'] - r['t0']:.2f} s, {len(r['out'])} out{'' if r['ok'] else ' ERROR ' + r.get('error', '')}",
                flush=True,
            )
    by_turn = [
        round(statistics.median(conv[k]["ttft"] for conv in recs if len(conv) > k), 3)
        for k in range(max(len(c) for c in recs))
    ]
    out_tokens = sum(len(r["out"]) for conv in recs for r in conv)
    summary = dict(
        turns_ok=sum(r["ok"] for conv in recs for r in conv),
        turns=sum(len(conv) for conv in recs),
        ttft_median_by_turn=by_turn,
        out_tokens=out_tokens,
        span_s=round(span, 2),
        out_tok_s=round(out_tokens / span, 1) if span > 0 else 0.0,
        errors=[r["error"] for conv in recs for r in conv if r.get("error")][:5],
    )
    print(f"prefix check: {json.dumps(summary)}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(summary=summary, conversations=recs), f)


if __name__ == "__main__":
    main()
