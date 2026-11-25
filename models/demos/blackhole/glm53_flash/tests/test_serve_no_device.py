# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""`tools/serve.py`, the vLLM server's command with the release's settings: the drafts a step by seats, the trace
region by context length, the flags and the environment it sets, the flags given after '--', the constants it shares
with the model, and the README's copy of the command.  The standard library only: no device, no vLLM."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import pytest

from models.demos.blackhole.glm53_flash.tools import serve

MODEL_DIR = Path(serve.__file__).resolve().parents[1]
ENV = {"TT_METAL_HOME": "/TT_METAL_HOME"}  # (no PYTHONPATH, no GLM53_DFLASH2_K: the defaults)
MiB = 1 << 20


def flags(command: serve.ServeCommand) -> dict:
    """{flag: its value, or None for a switch} of the server's arguments."""
    out, argv = {}, command.argv[1:]
    for i, token in enumerate(argv):
        if token.startswith("--"):
            value = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else None
            out[token] = value
    return out


@pytest.mark.parametrize("seats, k", [(8, 7), (16, 7), (32, 7), (40, 3), (64, 3), (128, 3)])
def test_drafts_by_seats(seats, k):
    f = flags(serve.build("/c", "/d", seats=seats, environ=ENV))
    assert f["--max_num_seqs"] == str(seats)
    assert json.loads(f["--speculative-config"]) == {
        "method": "custom_class",
        "model": "vllm_tt_plugin.model_owned_drafter",
        "num_speculative_tokens": k,
    }


def test_drafts_from_environment():
    f = flags(serve.build("/c", "/d", seats=64, environ={**ENV, "GLM53_DFLASH2_K": "7"}))
    assert json.loads(f["--speculative-config"])["num_speculative_tokens"] == 7


def test_drafts_match_the_adapter(monkeypatch):
    """The same rule as the model's (`glm53_vllm.dflash2_k`), which refuses another num_speculative_tokens."""
    pytest.importorskip("ttnn")
    from models.demos.blackhole.glm53_flash.tools.glm53_vllm import dflash2_k

    monkeypatch.delenv("GLM53_DFLASH2_K", raising=False)
    for seats in range(8, 520, 8):
        assert serve.drafts_per_step(seats, {}) == dflash2_k(seats), seats


@pytest.mark.parametrize(
    "length, size",
    [(512, 768), (2048, 768), (2049, 1024), (4096, 1024), (33792, 1024), (33793, 2048), (132096, 2048)],
)
def test_trace_region(length, size):
    tt = json.loads(flags(serve.build("/c", "/d", max_model_len=length, environ=ENV))["--additional-config"])["tt"]
    assert tt["trace_region_size"] == size * MiB
    tt = json.loads(flags(serve.build("/c", "/d", trace_region=5 << 30, environ=ENV))["--additional-config"])["tt"]
    assert tt["trace_region_size"] == 5 << 30


def test_release_settings():
    command = serve.build("/ckpt", "/drafter", environ=ENV)
    f = flags(command)
    assert f["--model"] == "/ckpt" and f["--served-model-name"] == "glm-5.3-flash"
    assert f["--max_num_seqs"] == "32" and f["--max-model-len"] == "33792" and f["--block_size"] == "64"
    assert "--no-enable-prefix-caching" in f and "--enable-chunked-prefill" in f
    assert f["--max-num-batched-tokens"] == "8192"
    assert f["--scheduler-cls"] == "models.demos.blackhole.glm53_flash.tools.glm53_vllm_scheduler.Glm53Scheduler"
    assert json.loads(f["--hf-overrides"]) == {"architectures": ["TTGlm5NextForConditionalGeneration"]}
    assert json.loads(f["--additional-config"]) == {
        "tt": {
            "l1_small_size": 32768,
            "trace_region_size": 1 << 30,
            "worker_l1_size": 1436288,
            "sample_on_device_mode": "decode_only",
            "prefill_chunk_alignment": 8192,
            "decode_interleave_prefill_steps": 1,
        }
    }
    assert json.loads(f["--override-generation-config"]) == {"top_k": 32}
    assert f["--port"] == "8000"
    assert "--enable-auto-tool-choice" not in f
    assert command.argv[0] == "vllm.entrypoints.openai.api_server"
    assert command.cwd == Path("/TT_METAL_HOME")
    assert command.env == {
        "EXTRA_MODELS_DIR": "/TT_METAL_HOME/models/demos/blackhole/glm53_flash/tools/vllm_bundle",
        "MESH_DEVICE": "(8, 4)",
        "TT_MESH_GRAPH_DESC_PATH": "/TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/"
        "single_bh_galaxy_torus_xy_graph_descriptor.textproto",
        "GLM53_DFLASH2": "1",
        "GLM53_DFLASH2_DRAFTER": "/drafter",
        "PYTHONPATH": "/TT_METAL_HOME",
    }


def test_environment_kept():
    """A PYTHONPATH that has the root already is kept; without TT_METAL_HOME the root is this file's tree."""
    command = serve.build("/c", "/d", environ={**ENV, "PYTHONPATH": "/a:/TT_METAL_HOME"})
    assert "PYTHONPATH" not in command.env
    command = serve.build("/c", "/d", environ={"PYTHONPATH": "/a"})
    assert command.env["TT_METAL_HOME"] == str(serve.ROOT) and command.env["PYTHONPATH"] == f"{serve.ROOT}:/a"


def test_tool_calls():
    f = flags(serve.build("/c", "/d", tool_calls=True, environ=ENV))
    assert "--enable-auto-tool-choice" in f
    assert f["--tool-call-parser"] == "glm47" and f["--reasoning-parser"] == "glm47"


def test_extra_flags():
    """Flags for vLLM go after the script's own; one the script also sets replaces it (any spelling)."""
    extra = ["--port=9000", "--served_model_name", "glm", "--enable-prefix-caching", "--enable-log-requests"]
    command = serve.build("/c", "/d", extra=extra, environ=ENV)
    assert command.argv[-len(extra) :] == extra
    for flag in ("--port", "--served-model-name", "--no-enable-prefix-caching"):
        assert flag not in command.argv
    assert command.replaced == ["--served-model-name", "--no-enable-prefix-caching", "--port"]


@pytest.mark.parametrize("extra", [["--max_num_seqs", "64"], ["--max-model-len=2048"]])
def test_context_flags_are_the_scripts(expect_error, extra):
    with expect_error(ValueError, "give it to this script"):
        serve.build("/c", "/d", extra=extra, environ=ENV)


@pytest.mark.parametrize("seats", [0, 4, 12, 33])
def test_seats_by_rows(expect_error, seats):
    with expect_error(ValueError, "a multiple of 8"):
        serve.build("/c", "/d", seats=seats, environ=ENV)


def test_parse():
    args, extra = serve.parse(["--seats", "64", "--max-model-len", "2048", "--print", "--", "--enable-log-requests"])
    assert (args.seats, args.max_model_len, args.port, args.print, args.tool_calls) == (64, 2048, 8000, True, False)
    assert extra == ["--enable-log-requests"]
    args, extra = serve.parse(["--max_num_seqs", "16", "--uvicorn-log-level", "warning", "--tool-calls"])
    assert args.seats == 16 and args.tool_calls and extra == ["--uvicorn-log-level", "warning"]
    args, extra = serve.parse(["--", "--port", "9000"])  # (after '--': vLLM's, even a name the script has)
    assert args.port == 8000 and extra == ["--port", "9000"]


def test_print(capsys, monkeypatch):
    """`--print`: the command as one shell line (the environment, then `python -m` and the flags), nothing run."""
    monkeypatch.setenv("TT_METAL_HOME", "/TT_METAL_HOME")
    monkeypatch.delenv("GLM53_DFLASH2_K", raising=False)
    monkeypatch.delenv("GLM53_KERNEL_CONFIG_EXTRA_KB", raising=False)
    serve.main(
        ["--print", "--checkpoint", "/ckpt", "--drafter", "/drafter", "--seats", "64", "--", "--enable-log-requests"]
    )
    out = capsys.readouterr().out
    assert out.startswith("cd /TT_METAL_HOME && \\\n") and "  --model /ckpt \\\n" in out
    tokens = shlex.split(out.replace("\\\n", " "))
    assert tokens[:3] == ["cd", "/TT_METAL_HOME", "&&"]
    python = tokens.index("python")
    assert {"MESH_DEVICE=(8, 4)", "GLM53_DFLASH2=1", "GLM53_DFLASH2_DRAFTER=/drafter"} <= set(tokens[3:python])
    command = serve.build("/ckpt", "/drafter", seats=64, extra=["--enable-log-requests"])
    assert tokens[python + 1 :] == ["-m"] + command.argv
    assert json.loads(flags(command)["--speculative-config"])["num_speculative_tokens"] == 3


def test_shared_constants():
    """The constants the script writes out are the model's: the worker L1 (`ttnn/common.py`), the chunked prefill's
    grid (`glm53_vllm.CHUNK_GRID`), the scheduler class and the plugin's model bundle."""
    common = (MODEL_DIR / "ttnn" / "common.py").read_text()
    base = int(re.search(r'DEVICE_PARAMS\["worker_l1_size"\] = (\d+) - \(KERNEL_CONFIG_EXTRA_KB << 10\)', common)[1])
    extra = int(re.search(r'"GLM53_KERNEL_CONFIG_EXTRA_KB", "(\d+)"', common)[1])
    assert serve.worker_l1_size({}) == base - (extra << 10)
    assert serve.worker_l1_size({"GLM53_KERNEL_CONFIG_EXTRA_KB": "0"}) == base
    adapter = (MODEL_DIR / "tools" / "glm53_vllm.py").read_text()
    assert int(re.search(r"^CHUNK_GRID = (\d+)", adapter, re.M)[1]) == serve.CHUNK_TOKENS
    assert "class Glm53Scheduler" in (MODEL_DIR / "tools" / "glm53_vllm_scheduler.py").read_text()
    assert serve.SCHEDULER.rsplit(".", 1)[0] == "models.demos.blackhole.glm53_flash.tools.glm53_vllm_scheduler"
    assert (MODEL_DIR / "tools" / "vllm_bundle").is_dir()
    assert (serve.ROOT / serve.MODEL_DIR) == MODEL_DIR
    if (serve.ROOT / "tt_metal").is_dir():
        assert (serve.ROOT / serve.MESH_GRAPH).is_file()


def test_readme_command():
    """The README's one-line call and the full command it shows are what the script builds (with $TT_METAL_HOME,
    $GLM53_CHECKPOINT and $GLM53_DFLASH2_DRAFTER for the paths)."""
    readme = MODEL_DIR / "README.md"
    if not readme.exists():
        pytest.skip("no README.md")
    blocks = re.findall(r"^```\n(.*?)^```", readme.read_text(), re.M | re.S)
    lines = [line for block in blocks for line in block.splitlines()]
    calls = [line for line in lines if line.startswith("python -m models.demos.blackhole.glm53_flash.tools.serve")]
    shown = [b.rstrip("\n") for b in blocks if b.startswith("cd $TT_METAL_HOME && \\\n")]
    assert len(shown) == 1, "the README shows the full command once"
    assert calls, "the README shows the script's call"
    args, extra = serve.parse(shlex.split(calls[0])[3:])
    command = serve.build(
        args.checkpoint or "/GLM53_CHECKPOINT",
        args.drafter or "/GLM53_DFLASH2_DRAFTER",
        seats=args.seats,
        max_model_len=args.max_model_len,
        port=args.port,
        tool_calls=args.tool_calls,
        trace_region=args.trace_region,
        extra=extra,
        environ=ENV,
    )
    text = command.render()
    for name in ("TT_METAL_HOME", "GLM53_CHECKPOINT", "GLM53_DFLASH2_DRAFTER"):
        text = text.replace(f"/{name}", f"${name}")
    assert shown[0] == text
