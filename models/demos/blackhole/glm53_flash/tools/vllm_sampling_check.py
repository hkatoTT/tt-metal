"""Sampled requests through the vLLM OpenAI server (GLM-5.3-Flash adapter with DFlash2 and GLM53_VLLM_SAMPLING=1): a
few requests sent at once, each checked for an answer (not refused) and its tokens; the ones the device samples
(temperature / top-p, the server's default top-k) next to ones that take the plugin's "logits" mode (a repetition
penalty, logprobs, a top-k of none), so that one batch mixes the two.

    python models/demos/blackhole/glm53_flash/tools/vllm_sampling_check.py --port PORT [--out JSON]

The standard library only (urllib), so it runs in the vLLM env beside the server.  Prints one line a case and a total;
exits 0 (a study's run).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request

MODEL = "glm-5.3-flash"
PROMPT = "Write two sentences about the sea."
SAMPLED = dict(temperature=1.0, top_p=0.95)  # (the model card's; the server's default top-k)
CASES = [  # name, endpoint, the request's own fields, the path its rows take
    ("greedy", "chat", dict(temperature=0.0), "device"),
    ("sampled-a", "chat", dict(SAMPLED), "device"),
    ("sampled-b", "chat", dict(SAMPLED), "device"),
    ("seeded-a", "chat", dict(SAMPLED, seed=7), "device"),
    ("seeded-b", "chat", dict(SAMPLED, seed=7), "device"),
    ("top-k-none", "chat", dict(SAMPLED, top_k=-1), "logits"),
    ("penalty", "chat", dict(SAMPLED, repetition_penalty=1.1), "logits"),
    ("logprobs", "completions", dict(SAMPLED, logprobs=2), "logits"),
]


def post(port: int, endpoint: str, body: dict) -> dict:
    path = "/v1/chat/completions" if endpoint == "chat" else "/v1/completions"
    req = urllib.request.Request(
        f"http://localhost:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def run(port: int, case: tuple, max_tokens: int) -> dict:
    name, endpoint, fields, path = case
    body = dict(model=MODEL, max_tokens=max_tokens, **fields)
    if endpoint == "chat":
        body["messages"] = [{"role": "user", "content": PROMPT}]
    else:
        body["prompt"] = PROMPT
    t0 = time.time()
    try:
        out = post(port, endpoint, body)
    except urllib.error.HTTPError as e:
        return dict(name=name, path=path, ok=False, error=f"HTTP {e.code}: {e.read().decode()[:300]}")
    except Exception as e:  # (a refused or failed request: reported, not raised)
        return dict(name=name, path=path, ok=False, error=repr(e)[:300])
    choice = out["choices"][0]
    msg = choice.get("message") or {}
    text = msg.get("content") or msg.get("reasoning_content") or choice.get("text") or ""
    res = dict(
        name=name, path=path, ok=True, seconds=round(time.time() - t0, 2), finish=choice.get("finish_reason"),
        tokens=(out.get("usage") or {}).get("completion_tokens"), text=text[:160],
    )  # fmt: skip
    if endpoint == "completions":
        lp = choice.get("logprobs") or {}
        res["logprobs"] = len(lp.get("token_logprobs") or [])
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--out")
    args = p.parse_args()
    with concurrent.futures.ThreadPoolExecutor(len(CASES)) as pool:
        results = list(pool.map(lambda c: run(args.port, c, args.max_tokens), CASES))
    for r in results:
        print(
            f"sampling check {r['name']} ({r['path']}): {json.dumps({k: v for k, v in r.items() if k not in ('name', 'path')}, ensure_ascii=False)}",
            flush=True,
        )
    by = {r["name"]: r for r in results}
    same_seed = by["seeded-a"].get("text") == by["seeded-b"].get("text") if by["seeded-a"]["ok"] else None
    print(
        f"sampling check: {sum(r['ok'] for r in results)} / {len(results)} answered; the two unseeded sampled answers "
        f"{'differ' if by['sampled-a'].get('text') != by['sampled-b'].get('text') else 'are the same'}; the two seeded "
        f"ones {'the same' if same_seed else 'differ'}",
        flush=True,
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
