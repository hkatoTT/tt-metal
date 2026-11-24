"""Sampled requests through the vLLM OpenAI server (GLM-5.3-Flash adapter with DFlash2, sampled requests on:
GLM53_VLLM_SAMPLING): a few requests sent at once, each checked against what the server's settings should do with it;
the ones the device draws (greedy, temperature / top-p with the server's default top-k or their own, seeded) next to
ones the device does not draw (a top-k of none, a repetition penalty, logprobs, one of them streamed), which
GLM53_VLLM_HOST_SAMPLING=refuse (the default) answers with HTTP 400 (a streamed one: an error event in the stream) and
=allow serves in the plugin's "logits" mode, so that one batch mixes the two.  Then one greedy request and /health:
the engine still serves.

    python models/demos/blackhole/glm53_flash/tools/vllm_sampling_check.py --port PORT [--host refuse|allow]
        [--server-top-k 32] [--server-sampling on|off] [--out JSON]

--host: the server's GLM53_VLLM_HOST_SAMPLING; --server-top-k: its default top-k (--override-generation-config
'{"top_k": 32}' as the README starts it; 0: none, a request that sends no top-k then takes the host's path);
--server-sampling off: a server with GLM53_VLLM_SAMPLING=0 (greedy only: the plugin refuses every request but those
that send temperature 0, the server's default temperature being the model's 1.0).  The standard library only
(urllib), so it runs in the vLLM env beside the server.  Prints one line a case and a total; exits 0 (it reports what
it saw; it does not fail).
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
SAMPLED = dict(temperature=1.0, top_p=0.95)  # (the model card's; the top-k the server's default)
CASES = [  # name, endpoint, the request's own fields, the path its rows take ("default": as the server's top-k)
    ("greedy", "chat", dict(temperature=0.0), "device"),
    ("defaults", "chat", dict(), "default"),  # (the server's defaults: the model's temperature 1.0 / top-p 0.95)
    ("sampled-a", "chat", dict(SAMPLED), "default"),
    ("sampled-b", "chat", dict(SAMPLED), "default"),
    ("seeded-a", "chat", dict(SAMPLED, seed=7), "default"),
    ("seeded-b", "chat", dict(SAMPLED, seed=7), "default"),
    ("top-k-32", "chat", dict(SAMPLED, top_k=32), "device"),
    ("top-k-none", "chat", dict(SAMPLED, top_k=-1), "host"),
    ("penalty", "chat", dict(SAMPLED, repetition_penalty=1.1), "host"),
    ("penalty-stream", "chat", dict(SAMPLED, repetition_penalty=1.1, stream=True), "host"),
    ("logprobs", "completions", dict(SAMPLED, logprobs=2), "host"),
]


def url(port: int, path: str) -> str:
    return f"http://localhost:{port}{path}"


def post(port: int, endpoint: str, body: dict) -> tuple:
    """-> (HTTP status, the JSON answer; a streamed answer: its events' JSON list)."""
    path = "/v1/chat/completions" if endpoint == "chat" else "/v1/completions"
    req = urllib.request.Request(
        url(port, path), data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        if not body.get("stream"):
            return r.status, json.loads(r.read())
        events = []
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
        return r.status, events


def run(port: int, case: tuple, max_tokens: int, refuse: bool, default_on_device: bool, sampling: bool = True) -> dict:
    name, endpoint, fields, path = case
    if path == "default":
        path = "device" if default_on_device else "host"
    expect = "refused" if refuse and path == "host" else "answered"
    if not sampling:  # (greedy only: the plugin refuses the rest)
        expect = "answered" if fields.get("temperature") == 0.0 else "refused"
    body = dict(model=MODEL, max_tokens=max_tokens, **fields)
    if endpoint == "chat":
        body["messages"] = [{"role": "user", "content": PROMPT}]
    else:
        body["prompt"] = PROMPT
    res = dict(name=name, path=path, expect=expect)
    t0 = time.time()
    try:
        status, out = post(port, endpoint, body)
    except urllib.error.HTTPError as e:
        return dict(
            res, got="refused", status=e.code, error=e.read().decode()[:400], seconds=round(time.time() - t0, 2)
        )
    except Exception as e:  # (a failed request: reported, not raised)
        return dict(res, got="failed", error=repr(e)[:300], seconds=round(time.time() - t0, 2))
    res.update(status=status, seconds=round(time.time() - t0, 2))
    if body.get("stream"):  # (a refusal in a stream: HTTP 200 and an error event)
        errors = [e["error"] for e in out if "error" in e]
        if errors:
            return dict(res, got="refused", error=json.dumps(errors[0])[:400])
        text = "".join(
            (c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content") or ""
            for e in out
            for c in e.get("choices", [])
        )
        return dict(res, got="answered", events=len(out), text=text[:160])
    choice = out["choices"][0]
    msg = choice.get("message") or {}
    text = msg.get("content") or msg.get("reasoning_content") or choice.get("text") or ""
    res.update(
        got="answered", finish=choice.get("finish_reason"), tokens=(out.get("usage") or {}).get("completion_tokens"),
        text=text[:160],
    )  # fmt: skip
    if endpoint == "completions":
        lp = choice.get("logprobs") or {}
        res["logprobs"] = len(lp.get("token_logprobs") or [])
    return res


def alive(port: int, max_tokens: int) -> dict:
    """After the cases: /health and one greedy request (the engine still serves)."""
    out = dict()
    try:
        with urllib.request.urlopen(url(port, "/health"), timeout=30) as r:
            out["health"] = r.status
    except Exception as e:
        out["health"] = repr(e)[:200]
    r = run(port, ("after", "chat", dict(temperature=0.0), "device"), max_tokens, False, True)
    out["greedy_after"] = r["got"]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument(
        "--host", choices=("refuse", "allow"), default="refuse", help="the server's GLM53_VLLM_HOST_SAMPLING"
    )
    p.add_argument("--server-top-k", type=int, default=32, help="the server's default top-k (0: none)")
    p.add_argument("--server-sampling", choices=("on", "off"), default="on", help="the server's GLM53_VLLM_SAMPLING")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--out")
    args = p.parse_args()
    refuse, on_device, sampling = args.host == "refuse", 1 <= args.server_top_k <= 32, args.server_sampling == "on"
    with concurrent.futures.ThreadPoolExecutor(len(CASES)) as pool:
        results = list(pool.map(lambda c: run(args.port, c, args.max_tokens, refuse, on_device, sampling), CASES))
    for r in results:
        shown = {k: v for k, v in r.items() if k not in ("name", "path")}
        print(f"sampling check {r['name']} ({r['path']}): {json.dumps(shown, ensure_ascii=False)}", flush=True)
    after = alive(args.port, args.max_tokens)
    by = {r["name"]: r for r in results}
    seeded = [by[n].get("text") for n in ("seeded-a", "seeded-b") if by[n]["got"] == "answered"]
    unseeded = [by[n].get("text") for n in ("sampled-a", "sampled-b") if by[n]["got"] == "answered"]
    print(
        f"sampling check (sampling {args.server_sampling}, {args.host}, server top-k {args.server_top_k}): "
        f"{sum(r['got'] == r['expect'] for r in results)} / {len(results)} as expected "
        f"({sum(r['got'] == 'answered' for r in results)} answered, {sum(r['got'] == 'refused' for r in results)} "
        f"refused, {sum(r['got'] == 'failed' for r in results)} failed); the two unseeded sampled answers "
        f"{'differ' if len(set(unseeded)) == 2 else 'are the same' if len(unseeded) == 2 else 'n/a'}; the two seeded "
        f"ones {'the same' if len(seeded) == 2 and seeded[0] == seeded[1] else 'differ' if len(seeded) == 2 else 'n/a'}"
        f"; after: /health {after['health']}, a greedy request {after['greedy_after']}",
        flush=True,
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(cases=results, after=after), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
