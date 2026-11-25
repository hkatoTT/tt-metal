"""Long prompts arriving while others decode, through the vLLM OpenAI server (GLM-5.3-Flash adapter;
GLM53_VLLM_CHUNKED, vLLM's chunked prefill): short requests stream their tokens (greedy, no early stop) and, once they
decode, passkey prompts of about --long-len tokens come in one after another.  Per short request, the longest gap
between its streamed events while the long prompts are in and before them (a long prompt's whole prefill stops every
decode without chunked prefill; with it, one chunk at a time); per long prompt, its time and whether it answers its
passkey.  Run once with GLM53_VLLM_CHUNKED=0 and once with 1 against the same load.

    python models/demos/blackhole/glm53_flash/tools/vllm_chunked_check.py --port PORT [--out JSON]

The standard library and the checkpoint's tokenizer (`tokenizers`), so it runs in the vLLM env beside the server.
Prints one line a long prompt and a summary; exits 0 (a study's run).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # (the tt-metal root: tools/passkey.py's package)

from models.demos.blackhole.glm53_flash.paths import checkpoint_dir  # noqa: E402
from models.demos.blackhole.glm53_flash.tools.passkey import DEPTHS, WORDS, filler, prompt_ids  # noqa: E402

MODEL = "glm-5.3-flash"


def post(port: int, body: dict, stream: bool):
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=json.dumps(dict(body, model=MODEL, stream=stream)).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=1200)


def short_request(port: int, ids: list, out_len: int, rec: dict) -> None:
    """Greedy, `out_len` tokens (no early stop), streamed: each event's arrival time."""
    body = dict(prompt=ids, max_tokens=out_len, temperature=0, ignore_eos=True, stream_options={"include_usage": True})
    rec["t0"], rec["events"] = time.time(), []
    try:
        with post(port, body, stream=True) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                msg = json.loads(line[5:])
                if msg.get("choices"):
                    rec["events"].append(time.time())
                if msg.get("usage"):
                    rec["tokens"] = msg["usage"].get("completion_tokens")
        rec["ok"] = True
    except Exception as e:  # (a failed request: reported, not raised)
        rec["ok"], rec["error"] = False, repr(e)[:300]
    rec["t1"] = time.time()


def long_request(port: int, ids: list, answer: str, rec: dict) -> None:
    rec["t0"] = time.time()
    try:
        with post(port, dict(prompt=ids, max_tokens=12, temperature=0), stream=False) as r:
            text = json.loads(r.read())["choices"][0]["text"]
        rec.update(ok=True, text=text, found=answer in text.replace(" ", ""))
    except Exception as e:
        rec.update(ok=False, found=False, error=repr(e)[:300])
    rec["t1"] = time.time()


def max_gap(events: list, lo: float, hi: float) -> float:
    """The longest gap between consecutive events whose later event falls in [lo, hi] (s)."""
    gaps = [b - a for a, b in zip(events, events[1:]) if lo <= b <= hi]
    return max(gaps, default=0.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--tokenizer", help="tokenizer.json (default: the checkpoint's, GLM53_CHECKPOINT or HF_MODEL)")
    p.add_argument("--short", type=int, default=24, help="short requests, decoding all along")
    p.add_argument("--short-len", type=int, default=1024)
    p.add_argument("--short-out", type=int, default=1536)
    p.add_argument("--long", type=int, default=4, help="passkey prompts, one after another")
    p.add_argument("--long-len", type=int, default=30000)
    p.add_argument("--delay", type=float, default=20.0, help="s after the short requests start: the first long one")
    p.add_argument("--gap", type=float, default=4.0, help="s between the long prompts' arrivals")
    p.add_argument("--out")
    args = p.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer or str(checkpoint_dir() / "tokenizer.json"))
    shorts = []
    for i in range(args.short):  # (each its own filler: distinct prompts)
        text = "".join(filler(1000 * i + j) for j in range(args.short_len // 20))
        shorts.append(tok.encode(text).ids[: args.short_len])
    longs = []
    for i in range(args.long):
        answer = f"{WORDS[i % len(WORDS)]}-{4000 + 731 * i}"
        longs.append((prompt_ids(tok, args.long_len, DEPTHS[i % len(DEPTHS)], *answer.split("-")), answer))
    srec = [dict() for _ in shorts]
    lrec = [dict(tokens_in=len(ids)) for ids, _ in longs]
    threads = [
        threading.Thread(target=short_request, args=(args.port, ids, args.short_out, r)) for ids, r in zip(shorts, srec)
    ]
    t_start = time.time()
    for t in threads:
        t.start()
    time.sleep(args.delay)
    lthreads = []
    for (ids, answer), r in zip(longs, lrec):
        lthreads.append(threading.Thread(target=long_request, args=(args.port, ids, answer, r)))
        lthreads[-1].start()
        time.sleep(args.gap)
    for t in lthreads + threads:
        t.join()
    lo = min(r["t0"] for r in lrec)
    hi = max(r["t1"] for r in lrec)
    for i, r in enumerate(lrec):
        print(
            f"chunked check long {i}: {r['tokens_in']} tokens, {r['t1'] - r['t0']:.2f} s, passkey "
            f"{'found' if r.get('found') else 'MISSED'} {json.dumps(r.get('text', r.get('error', '')))}",
            flush=True,
        )
    during = [max_gap(r.get("events", []), lo, hi) for r in srec]
    before = [max_gap(r.get("events", []), t_start, lo) for r in srec]
    tokens = sum(int(r.get("tokens") or 0) for r in srec)
    span = max(r["t1"] for r in srec) - t_start
    summary = dict(
        long_found=sum(bool(r.get("found")) for r in lrec),
        long_n=len(lrec),
        long_s=[round(r["t1"] - r["t0"], 2) for r in lrec],
        short_ok=sum(bool(r.get("ok")) for r in srec),
        short_n=len(srec),
        short_max_gap_during_s=round(max(during, default=0.0), 3),
        short_median_max_gap_during_s=round(sorted(during)[len(during) // 2], 3) if during else 0.0,
        short_max_gap_before_s=round(max(before, default=0.0), 3),
        short_tokens=tokens,
        short_tok_s=round(tokens / span, 1) if span > 0 else 0.0,
        errors=[r["error"] for r in srec + lrec if r.get("error")][:5],
    )
    print(f"chunked check: {json.dumps(summary)}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                dict(summary=summary, long=lrec, short=[{k: v for k, v in r.items() if k != "events"} for r in srec]),
                f,
                indent=1,
            )


if __name__ == "__main__":
    main()
