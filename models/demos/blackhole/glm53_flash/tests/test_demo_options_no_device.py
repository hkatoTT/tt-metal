# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The demo's settings (`demo/demo.py`'s `demo_settings`, the options of `demo/conftest.py`): each pytest option over
its GLM53_DEMO_* environment variable, over the default; the prompts file's forms, --batch_size and --max_seq_len; the
sample prompts file.  No device (the demo module imports ttnn: skipped without it)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.demo.demo import (  # noqa: E402
    DEFAULT_PROMPT,
    MAX_PROMPTS,
    cache_length,
    demo_settings,
    load_prompts,
)

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
SAMPLE = DEMO_DIR / "sample_prompts" / "questions.json"


def options(**given):
    """pytest's getoption as the demo calls it: the value, None where the option was not given."""
    return lambda name: given.get(name.lstrip("-"))


def prompts_file(tmp_path: Path, name: str, items: list) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(items))
    return str(path)


def test_defaults():
    s = demo_settings(environ={})
    assert (s.prompts, s.max_new, s.reasoning, s.k, s.max_seq) == ([DEFAULT_PROMPT], 256, "low", 7, None)


def test_environment():
    environ = {
        "GLM53_DEMO_PROMPT": "Hi",
        "GLM53_DEMO_MAX_NEW": "32",
        "GLM53_DEMO_K": "3",
        "GLM53_DEMO_REASONING": "high",
    }
    s = demo_settings(environ=environ)
    assert (s.prompts, s.max_new, s.reasoning, s.k) == (["Hi"], 32, "high", 3)


def test_options_over_environment(tmp_path):
    first = prompts_file(tmp_path, "a.json", ["a1", "a2"])
    second = prompts_file(tmp_path, "b.json", [{"prompt": "b1"}])
    environ = {"GLM53_DEMO_PROMPTS": first, "GLM53_DEMO_PROMPT": "one", "GLM53_DEMO_MAX_NEW": "32"}
    s = demo_settings(environ=environ)
    assert (s.prompts, s.max_new, s.source) == (["a1", "a2"], 32, first)  # (the file over the one prompt)
    s = demo_settings(options(input_prompts=second, max_generated_tokens=16), environ)
    assert (s.prompts, s.max_new, s.source) == (["b1"], 16, second)


def test_batch_size(tmp_path, expect_error):
    path = prompts_file(tmp_path, "p.json", ["x", "y", "z"])
    assert demo_settings(options(input_prompts=path, batch_size=8), {}).prompts == ["x", "y", "z"] * 2 + ["x", "y"]
    assert demo_settings(options(input_prompts=path, batch_size=2), {}).prompts == ["x", "y"]
    assert demo_settings(options(batch_size=4), {}).prompts == [DEFAULT_PROMPT] * 4
    with expect_error(ValueError, "--batch_size 65"):
        demo_settings(options(batch_size=MAX_PROMPTS + 1), {})
    many = prompts_file(tmp_path, "many.json", [f"q{i}" for i in range(MAX_PROMPTS + 1)])
    with expect_error(ValueError, "65 prompts"):
        demo_settings(options(input_prompts=many), {})
    assert len(demo_settings(options(input_prompts=many, batch_size=MAX_PROMPTS), {}).prompts) == MAX_PROMPTS


def test_prompts_file_forms(tmp_path, expect_error):
    mixed = prompts_file(tmp_path, "mixed.json", ["a", {"prompt": "b", "note": "ignored"}])
    assert load_prompts(mixed) == ["a", "b"]
    for name, items in (("empty.json", []), ("object.json", {"prompt": "a"}), ("item.json", ["a", {"text": "b"}])):
        with expect_error(ValueError, "prompt"):
            load_prompts(prompts_file(tmp_path, name, items))


def test_cache_length(expect_error):
    assert cache_length(100, 256) == 1024  # (at least 1,024)
    assert cache_length(1000, 256) == 1536  # (1,272 rounded up to 512s)
    assert cache_length(100, 256, asked=4000) == 4096
    assert cache_length(100, 256, asked=600) == 1024
    with expect_error(ValueError, "need 1272"):
        cache_length(1000, 256, asked=1024)
    assert demo_settings(options(max_seq_len=8192), {}).max_seq == 8192


def test_sample_prompts():
    prompts = load_prompts(SAMPLE)
    assert 1 <= len(prompts) <= MAX_PROMPTS and all(p.strip() for p in prompts)
    assert demo_settings(options(input_prompts=str(SAMPLE)), {}).prompts == prompts


def test_conftest_options():
    """demo/conftest.py registers the options under the names the other demos use, and reuses one another conftest of
    the same run registered already."""
    spec = importlib.util.spec_from_file_location("glm53_demo_conftest", DEMO_DIR / "conftest.py")
    conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest)

    class Parser:  # (pytest's: a name added twice raises ValueError)
        def __init__(self):
            self.names = {}

        def addoption(self, name, **kwargs):
            if name in self.names:
                raise ValueError(f"option names {{'{name}'}} already added")
            self.names[name] = kwargs

    parser = Parser()
    parser.addoption("--input_prompts", action="store", help="another demo's")
    conftest.pytest_addoption(parser)
    assert set(parser.names) == {"--input_prompts", "--max_generated_tokens", "--batch_size", "--max_seq_len"}
    assert parser.names["--input_prompts"]["help"] == "another demo's"
    assert all(parser.names[n]["type"] is int for n in ("--max_generated_tokens", "--batch_size", "--max_seq_len"))
    other = Path(__file__).resolve().parents[5] / "models" / "tt_transformers" / "demo" / "conftest.py"
    if other.exists():  # (the same names as the shared text demo's options)
        text = other.read_text()
        assert all(f'"{name}"' in text for name in parser.names)
