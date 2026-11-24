"""Tool calling through the vLLM OpenAI server (GLM-5.3-Flash adapter, `--enable-auto-tool-choice --tool-call-parser
glm47`): hand-made cases sent at once (greedy), each checked for the tool chosen, its arguments (JSON, the schema's
types, the expected values) or for no call; one case also streamed, its tool call rebuilt from the deltas.

    python models/demos/blackhole/glm53_flash/tools/vllm_tool_check.py --port PORT [--out JSON]

The standard library only (urllib), so it runs in the vLLM env beside the server.  Prints one line a case and a total;
exits 0 (a study's run).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.request

MODEL = "glm-5.3-flash"

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name in English, e.g. Tokyo"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
        },
    },
}
FLIGHTS = {
    "type": "function",
    "function": {
        "name": "search_flights",
        "description": "Search flights between two airports on a date.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "IATA code of the departure airport"},
                "destination": {"type": "string", "description": "IATA code of the arrival airport"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "passengers": {"type": "integer"},
                "nonstop": {"type": "boolean"},
            },
            "required": ["origin", "destination", "date", "passengers", "nonstop"],
        },
    },
}
EVENT = {
    "type": "function",
    "function": {
        "name": "create_event",
        "description": "Create a calendar event.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "attendees": {"type": "array", "items": {"type": "string"}, "description": "email addresses"},
                "duration_minutes": {"type": "integer"},
                "location": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "address": {"type": "string"}},
                    "required": ["name", "address"],
                },
            },
            "required": ["title", "attendees", "duration_minutes", "location"],
        },
    },
}
CALC = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    },
}
ALL = [WEATHER, FLIGHTS, EVENT, CALC]


def _user(text):
    return [{"role": "user", "content": text}]


def _calls(want):
    """A checker of the response's tool calls against [(name, {arg: value or a predicate})]."""

    def check(resp):
        msg = resp["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if len(calls) != len(want):
            return False, f"{len(calls)} calls, want {len(want)}; content {str(msg.get('content'))[:120]!r}"
        got = []
        for c in calls:
            try:
                got.append((c["function"]["name"], json.loads(c["function"]["arguments"])))
            except (KeyError, ValueError) as e:
                return False, f"unparsable call {c}: {e}"
        for name, args in want:
            match = [
                g
                for g in got
                if g[0] == name
                and all((v(g[1].get(k)) if callable(v) else _same(g[1].get(k), v)) for k, v in args.items())
            ]
            if not match:
                return False, f"no {name} with {args}: got {got}"
        return True, f"{got} (finish {resp['choices'][0]['finish_reason']})"

    return check


def _same(a, b):
    if isinstance(b, str) and isinstance(a, str):
        return a.strip().lower() == b.lower()
    return type(a) is type(b) and a == b


def _no_call(needle):
    def check(resp):
        msg = resp["choices"][0]["message"]
        text = msg.get("content") or ""
        if msg.get("tool_calls"):
            return False, f"called {msg['tool_calls']}"
        return needle.lower() in text.lower(), f"content {text[:160]!r}"

    return check


WEATHER_CALL = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": "Tokyo", "unit": "celsius"})},
        }
    ],
}

CASES = [
    (
        "pick and enum",
        _user("What's the weather in Tokyo right now, in celsius?"),
        ALL,
        _calls([("get_weather", {"city": "Tokyo", "unit": "celsius"})]),
    ),
    (
        "types: int and bool",
        _user("Find nonstop flights from SFO to NRT on 2026-12-01 for 2 passengers."),
        ALL,
        _calls(
            [
                (
                    "search_flights",
                    {"origin": "SFO", "destination": "NRT", "date": "2026-12-01", "passengers": 2, "nonstop": True},
                )
            ]
        ),
    ),
    (
        "two calls at once",
        _user("Compare the current weather in Paris and in London, both in celsius."),
        ALL,
        _calls(
            [
                ("get_weather", {"city": "Paris", "unit": "celsius"}),
                ("get_weather", {"city": "London", "unit": "celsius"}),
            ]
        ),
    ),
    (
        "nested arguments",
        _user(
            "Create a 45-minute event titled 'Design review' with alice@example.com and bob@example.com at the "
            "'Blue Room', 1 Main St, Springfield."
        ),
        ALL,
        _calls(
            [
                (
                    "create_event",
                    {
                        "title": "Design review",
                        "duration_minutes": 45,
                        "attendees": lambda v: isinstance(v, list)
                        and sorted(v) == ["alice@example.com", "bob@example.com"],
                        "location": lambda v: isinstance(v, dict)
                        and "blue room" in str(v.get("name", "")).lower()
                        and "main st" in str(v.get("address", "")).lower(),
                    },
                )
            ]
        ),
    ),
    ("no tool needed", _user("What is the capital of France? Answer in one word."), ALL, _no_call("Paris")),
    (
        "tool result, then the answer",
        _user("What's the weather in Tokyo right now, in celsius?")
        + [
            WEATHER_CALL,
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "content": json.dumps({"temperature": 18, "condition": "cloudy"}),
            },
        ],
        ALL,
        _no_call("18"),
    ),
    (
        "Japanese request",
        _user("大阪の今の天気を華氏で教えてください。"),
        ALL,
        _calls([("get_weather", {"city": "Osaka", "unit": "fahrenheit"})]),
    ),
    (
        "calculator",
        _user("Use the calculator to compute 1234 * 5678."),
        ALL,
        _calls([("calculator", {"expression": lambda v: isinstance(v, str) and "1234" in v and "5678" in v})]),
    ),
]


def post(port, body, timeout=600):
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if not body.get("stream"):
            return json.loads(r.read())
        calls, finish, text = {}, None, ""
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            choice = json.loads(line[5:])["choices"][0]
            delta = choice.get("delta", {})
            text += delta.get("content") or ""
            for tc in delta.get("tool_calls") or []:
                c = calls.setdefault(tc["index"], {"function": {"name": "", "arguments": ""}})
                f = tc.get("function", {})
                c["function"]["name"] += f.get("name") or ""
                c["function"]["arguments"] += f.get("arguments") or ""
            finish = choice.get("finish_reason") or finish
        message = {"content": text, "tool_calls": [calls[i] for i in sorted(calls)] or None}
        return {"choices": [{"message": message, "finish_reason": finish}]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--out")
    args = p.parse_args()
    jobs = [(name, msgs, tools, check, False) for name, msgs, tools, check in CASES]
    jobs.append(("types: int and bool, streamed",) + CASES[1][1:] + (True,))
    t0 = time.time()

    def run(job):
        name, msgs, tools, check, stream = job
        body = {
            "model": MODEL,
            "messages": msgs,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "stream": stream,
        }
        t = time.time()
        try:
            resp = post(args.port, body)
            ok, why = check(resp)
        except Exception as e:  # (a study: report and go on)
            resp, ok, why = None, False, f"error {e!r}"
        return {"case": name, "ok": ok, "why": why, "seconds": round(time.time() - t, 1), "response": resp}

    with concurrent.futures.ThreadPoolExecutor(len(jobs)) as pool:
        results = list(pool.map(run, jobs))
    for r in results:
        print(f"{'OK  ' if r['ok'] else 'FAIL'} {r['case']} ({r['seconds']} s): {r['why'][:300]}", flush=True)
    print(f"tool calls: {sum(r['ok'] for r in results)} / {len(results)} cases as expected, {time.time() - t0:.0f} s")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
