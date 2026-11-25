# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The checkpoint's environment variables (`paths.py`): GLM53_CHECKPOINT, and HF_MODEL as its alias (the name other
tt-metal demos use), read only where GLM53_CHECKPOINT is not set and only if it is GLM-5.3-Flash-BF16's checkpoint
directory.  The standard library only: no device, no torch."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from models.demos.blackhole.glm53_flash.paths import (
    checkpoint_dir,
    checkpoint_from_env,
    checkpoint_problem,
    checkpoint_skip_reason,
)

TEXT = {"model_type": "glm5_next_text", "hidden_size": 4096, "num_hidden_layers": 45, "n_routed_experts": 288}
FLASH = {"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next", "text_config": TEXT}


def checkpoint(tmp_path: Path, name: str, config: dict | None = FLASH, index: bool = True) -> Path:
    """A stand-in checkpoint directory: its config.json (none if `config` is None) and an empty index."""
    root = tmp_path / name
    root.mkdir()
    if config is not None:
        (root / "config.json").write_text(json.dumps(config))
    if index:
        (root / "model.safetensors.index.json").write_text("{}")
    return root


def test_glm53_checkpoint_wins(tmp_path):
    """Both set: GLM53_CHECKPOINT, and HF_MODEL is not even looked at (other demos leave it at their model)."""
    root = checkpoint(tmp_path, "glm")
    environ = {"GLM53_CHECKPOINT": str(root), "HF_MODEL": "meta-llama/Llama-3.1-8B-Instruct"}
    assert checkpoint_from_env(environ) == root
    assert checkpoint_skip_reason(environ) is None


def test_hf_model_alias(tmp_path):
    root = checkpoint(tmp_path, "glm")
    assert checkpoint_from_env({"HF_MODEL": str(root)}) == root
    assert checkpoint_skip_reason({"HF_MODEL": str(root)}) is None


def test_neither_set():
    assert checkpoint_from_env({}) is None
    assert "GLM53_CHECKPOINT or HF_MODEL" in checkpoint_skip_reason({})


def other(tmp_path: Path, kind: str) -> str:
    """HF_MODEL values that are not GLM-5.3-Flash-BF16's checkpoint directory."""
    if kind == "hub id":
        return "zai-org/GLM-5.3-Flash-BF16"
    if kind == "missing":
        return str(tmp_path / "nowhere")
    if kind == "no config":
        return str(checkpoint(tmp_path, kind, config=None))
    if kind == "not json":
        root = checkpoint(tmp_path, kind, config=None)
        (root / "config.json").write_text("{not json")
        return str(root)
    configs = {
        "another model": {"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32},
        "the drafter": {"architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "hidden_size": 4096},
        "another shape": {**FLASH, "text_config": {**TEXT, "num_hidden_layers": 78, "n_routed_experts": 256}},
        "quantized": {**FLASH, "quantization_config": {"fmt": "e4m3"}},
    }
    return str(checkpoint(tmp_path, kind, config=configs[kind]))


@pytest.mark.parametrize(
    "kind, reason",
    [
        ("hub id", "not a local directory"),
        ("missing", "not a local directory"),
        ("no config", "no config.json"),
        ("not json", "not readable JSON"),
        ("another model", "model type 'llama'"),
        ("the drafter", "model type 'qwen3'"),
        ("another shape", "another glm5_next model"),
        ("quantized", "quantized checkpoint"),
    ],
)
def test_hf_model_checked(tmp_path, expect_error, kind, reason):
    """A wrong HF_MODEL stops with a message naming both variables (and skips the tests that need a checkpoint)."""
    value = other(tmp_path, kind)
    assert reason in checkpoint_problem(value)
    with expect_error(RuntimeError, "HF_MODEL=.*is not a GLM-5.3-Flash-BF16 checkpoint directory.*GLM53_CHECKPOINT"):
        checkpoint_from_env({"HF_MODEL": value})
    skip = checkpoint_skip_reason({"HF_MODEL": value})
    assert reason in skip and "GLM53_CHECKPOINT" in skip
    assert checkpoint_from_env({"HF_MODEL": value, "GLM53_CHECKPOINT": "/elsewhere"}) == Path("/elsewhere")


def test_checkpoint_without_index(tmp_path):
    root = checkpoint(tmp_path, "glm", index=False)
    assert "no model.safetensors.index.json" in checkpoint_skip_reason({"HF_MODEL": str(root)})


def test_checkpoint_dir(tmp_path, monkeypatch, expect_error):
    """`checkpoint_dir` (what the model, the demo, the tools and the vLLM adapter read): the argument, then the
    environment."""
    root = checkpoint(tmp_path, "glm")
    monkeypatch.delenv("GLM53_CHECKPOINT", raising=False)
    monkeypatch.setenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    assert checkpoint_dir(root) == root  # (an explicit path: the environment is not read)
    with expect_error(RuntimeError, "HF_MODEL="):
        checkpoint_dir()
    monkeypatch.setenv("HF_MODEL", str(root))
    assert checkpoint_dir() == root
    monkeypatch.delenv("HF_MODEL")
    with expect_error(RuntimeError, "GLM53_CHECKPOINT is not set"):
        checkpoint_dir()


def test_real_checkpoint_passes():
    """The real checkpoint's config.json passes the check (where GLM53_CHECKPOINT points at it)."""
    root = os.environ.get("GLM53_CHECKPOINT")
    if not root or not (Path(root) / "config.json").exists():
        pytest.skip("no checkpoint (GLM53_CHECKPOINT)")
    assert checkpoint_problem(root) is None
    assert checkpoint_from_env({"HF_MODEL": root}) == Path(root)
