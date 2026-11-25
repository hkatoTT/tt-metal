"""Where GLM-5.3-Flash's files live, from environment variables (no machine-specific defaults).

    GLM53_CHECKPOINT  the BF16 checkpoint directory (tools/download_checkpoint.py fetches it at the pinned revision)
    HF_MODEL          the same, as other tt-metal demos name it: used when GLM53_CHECKPOINT is not set, and only if it
                      is a local directory holding GLM-5.3-Flash-BF16's config.json (other demos often leave it at
                      another model or at a hub id)
    GLM53_DFLASH2_DRAFTER  the DFlash2 drafter's checkpoint directory (speculative decoding)
    GLM53_DATA_DIR    reference outputs and task files the tests read (default ./glm53_data): the CPU reference's
                      records (tools/run_cpu_reference.py), the task sets (tools/*_tasks.py, tools/*_prompts.py)
    GLM53_OUT_DIR     results the tests and tools write (default ./glm53_out)

The standard library only: tools that run beside a server (no torch) import it too.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

CHECKPOINT_VARIABLES = "GLM53_CHECKPOINT or HF_MODEL"
_DOWNLOAD = "python -m models.demos.blackhole.glm53_flash.tools.download_checkpoint --out DIR fetches it"
# What tells GLM-5.3-Flash's config.json from another model's: the model type and the text model's shape (a model of
# the same type with another shape is not this port's)
_EXPECTED_TEXT = {"hidden_size": 4096, "num_hidden_layers": 45, "n_routed_experts": 288}


def checkpoint_problem(path: Path | str) -> str | None:
    """Why `path` is not a GLM-5.3-Flash-BF16 checkpoint directory, or None: a local directory whose config.json is the
    model's (type glm5_next, the Flash shape) without a quantization config (the FP8 sibling has one)."""
    root = Path(path)
    if not root.is_dir():
        return "not a local directory (a Hugging Face hub id is not read: download the checkpoint first)"
    try:
        config = json.loads((root / "config.json").read_text())
    except FileNotFoundError:
        return "no config.json in it"
    except (OSError, ValueError) as error:
        return f"its config.json is not readable JSON ({error})"
    if not isinstance(config, dict):
        return "its config.json is not a JSON object"
    model_type = config.get("model_type")
    if model_type not in ("glm5_next", "glm5_next_text"):
        return f"its config.json is of model type {model_type!r}, not GLM-5.3-Flash's 'glm5_next'"
    text = config.get("text_config", config)
    wrong = [f"{key} {text.get(key)!r}" for key, value in _EXPECTED_TEXT.items() if text.get(key) != value]
    if wrong:
        expected = ", ".join(f"{key} {value}" for key, value in _EXPECTED_TEXT.items())
        return f"its config.json is another glm5_next model ({', '.join(wrong)}; GLM-5.3-Flash has {expected})"
    if "quantization_config" in config or "quantization_config" in text:
        return "it is a quantized checkpoint (config.json has a quantization_config): this port reads the BF16 one"
    return None


def checkpoint_from_env(environ: Mapping[str, str] | None = None) -> Path | None:
    """``$GLM53_CHECKPOINT``, else ``$HF_MODEL`` if it is GLM-5.3-Flash's checkpoint directory (else a RuntimeError
    naming both variables), else None.  GLM53_CHECKPOINT wins when both are set (HF_MODEL is then not looked at)."""
    environ = os.environ if environ is None else environ
    path = environ.get("GLM53_CHECKPOINT")
    if path:
        return Path(path)
    alias = environ.get("HF_MODEL")
    if not alias:
        return None
    problem = checkpoint_problem(alias)
    if problem:
        raise RuntimeError(
            f"HF_MODEL={alias!r} is not a GLM-5.3-Flash-BF16 checkpoint directory: {problem}. GLM53_CHECKPOINT is not "
            "set, so HF_MODEL was read in its place: set GLM53_CHECKPOINT to the checkpoint directory (it takes "
            f"precedence over HF_MODEL), or point HF_MODEL at it ({_DOWNLOAD})"
        )
    return Path(alias)


def checkpoint_dir(root: Path | str | None = None) -> Path:
    """The checkpoint directory: the argument, else ``$GLM53_CHECKPOINT``, else ``$HF_MODEL``
    (`checkpoint_from_env`)."""
    if root:
        return Path(root)
    path = checkpoint_from_env()
    if path is None:
        raise RuntimeError(
            f"GLM53_CHECKPOINT is not set (nor HF_MODEL): point it at the GLM-5.3-Flash-BF16 checkpoint ({_DOWNLOAD})"
        )
    return path


def checkpoint_skip_reason(environ: Mapping[str, str] | None = None) -> str | None:
    """For tests that need the checkpoint and skip without it: why there is none to use (unset, HF_MODEL at another
    model, no index in the directory), or None."""
    try:
        root = checkpoint_from_env(environ)
    except RuntimeError as error:
        return str(error)
    if root is None:
        return f"no checkpoint ({CHECKPOINT_VARIABLES})"
    if not (root / "model.safetensors.index.json").exists():
        return f"no checkpoint at {root} (no model.safetensors.index.json)"
    return None


def drafter_dir(root: Path | str | None = None) -> Path:
    """The DFlash2 drafter's checkpoint (canada-quant/GLM-5.3-Flash-DFlash2-G): the argument, else
    ``$GLM53_DFLASH2_DRAFTER``."""
    path = root or os.environ.get("GLM53_DFLASH2_DRAFTER")
    if not path:
        raise RuntimeError(
            "GLM53_DFLASH2_DRAFTER is not set: point it at the DFlash2 drafter checkpoint "
            "(huggingface-cli download canada-quant/GLM-5.3-Flash-DFlash2-G --local-dir DIR)"
        )
    return Path(path)


def data_dir(*parts: str) -> Path:
    """A path under ``$GLM53_DATA_DIR`` (default ./glm53_data): reference outputs and task files."""
    return Path(os.environ.get("GLM53_DATA_DIR", "glm53_data"), *parts)


def out_dir(*parts: str) -> Path:
    """A directory under ``$GLM53_OUT_DIR`` (default ./glm53_out) for results, made if missing."""
    path = Path(os.environ.get("GLM53_OUT_DIR", "glm53_out"), *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
