"""Where GLM-5.3-Flash's files live, from environment variables (no machine-specific defaults).

    GLM53_CHECKPOINT  the BF16 checkpoint directory (tools/download_checkpoint.py fetches it at the pinned revision)
    GLM53_DFLASH2_DRAFTER  the DFlash2 drafter's checkpoint directory (speculative decoding)
    GLM53_DATA_DIR    reference outputs and task files the tests read (default ./glm53_data): the CPU reference's
                      records (tools/run_cpu_reference.py), the task sets (tools/*_tasks.py, tools/*_prompts.py)
    GLM53_OUT_DIR     results the tests and tools write (default ./glm53_out)
"""

from __future__ import annotations

import os
from pathlib import Path


def checkpoint_dir(root: Path | str | None = None) -> Path:
    """The checkpoint directory: the argument, else ``$GLM53_CHECKPOINT``."""
    path = root or os.environ.get("GLM53_CHECKPOINT")
    if not path:
        raise RuntimeError(
            "GLM53_CHECKPOINT is not set: point it at the GLM-5.3-Flash-BF16 checkpoint "
            "(python -m models.demos.blackhole.glm53_flash.tools.download_checkpoint --out DIR fetches it)"
        )
    return Path(path)


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
