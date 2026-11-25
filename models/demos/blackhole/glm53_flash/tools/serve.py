# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Start the vLLM OpenAI server for GLM-5.3-Flash with the release's settings (README.md, "vLLM Serving").

    python -m models.demos.blackhole.glm53_flash.tools.serve [--seats 32] [--max-model-len 33792] [--port 8000]
        [--checkpoint DIR] [--drafter DIR] [--tool-calls] [--trace-region BYTES] [--print] [-- more vLLM flags]

Builds the server's command from these few settings and runs it in place of this process (so a SIGINT to this process
stops the server, which shuts its engine down and closes the mesh), from the tt-metal root.  `--print` prints the
command instead.  What it sets:

- the environment: EXTRA_MODELS_DIR (the plugin's model bundle, `tools/vllm_bundle`), MESH_DEVICE (8, 4),
  TT_MESH_GRAPH_DESC_PATH (the Galaxy's 2D torus), GLM53_DFLASH2=1 with GLM53_DFLASH2_DRAFTER, the tt-metal root on
  PYTHONPATH (the scheduler class and the bundle import from it), and TT_METAL_HOME when it is not set;
- DFlash2 speculative decoding with the drafts a step the model uses at this many seats: 7 up to 32 seats, else 3
  (GLM53_DFLASH2_K in the environment overrides both, as in the model);
- chunked prefill in 8,192-token steps with `Glm53Scheduler`, the model's own prefix store (vLLM's off);
- the `tt` block of `--additional-config`: the trace region by `--max-model-len` (768 MiB up to 2,048 tokens, 1 GiB up
  to 33,792, else 2 GiB: as measured), the worker L1 the model's device parameters use (GLM53_KERNEL_CONFIG_EXTRA_KB);
- a default top-k of 32 for sampled requests (`--override-generation-config`), so a request with the model's own
  sampling defaults (temperature 1.0, top-p 0.95, no top-k) is drawn on the device;
- `--tool-calls`: the GLM-4.7 tool-call and reasoning parsers.

Flags after `--` (or any flag this script does not know) go to vLLM after the script's own; a flag the script also sets
replaces the script's (for example `-- --additional-config '{...}'` replaces the whole `tt` block).  The seats and the
context length are this script's own options (`--seats` / `--max-num-seqs`, `--max-model-len`): the drafts a step and
the trace region follow from them.

The standard library only (no torch, no vLLM import): `--print` runs anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[5]  # the tt-metal root this file is in
MODEL_DIR = "models/demos/blackhole/glm53_flash"
MESH_GRAPH = "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto"
SERVED_NAME = "glm-5.3-flash"
SCHEDULER = "models.demos.blackhole.glm53_flash.tools.glm53_vllm_scheduler.Glm53Scheduler"
ARCHITECTURE = "TTGlm5NextForConditionalGeneration"  # (the plugin's name for the bundle's model)
DRAFTER_CLASS = "vllm_tt_plugin.model_owned_drafter"  # (the model drafts on the device itself)
ROWS = 8  # mesh rows: the seats are spread over them
CHUNK_TOKENS = 8192  # a chunked-prefill step (glm53_vllm.CHUNK_GRID: a partway prompt resumes on this grid)
BLOCK_SIZE = 64
L1_SMALL = 32768
DEFAULT_TOP_K = 32  # (the device draws from up to its 32 most likely candidates: ttnn/sampling.py)
TOOL_PARSER = "glm47"
DRAFTER_REPO = "canada-quant/GLM-5.3-Flash-DFlash2-G"
MiB = 1 << 20
# The trace region a chip by max_model_len (a longer cache has more prefill forms to capture): 768 MiB is enough for
# 2,048 and too small for 33,792 (the server stops at start-up), 1 GiB served 4,096 to 33,792, 2 GiB 132,096
TRACE_REGIONS = ((2048, 768 * MiB), (33792, 1024 * MiB))
TRACE_REGION_LONG = 2048 * MiB


def drafts_per_step(seats: int, environ: Mapping[str, str]) -> int:
    """DFlash2's drafts a step at this many seats, as the model takes it (glm53_vllm.dflash2_k): GLM53_DFLASH2_K, else 7
    up to 32 seats, else 3.  vLLM's num_speculative_tokens must be the same: the server refuses another at start-up."""
    given = environ.get("GLM53_DFLASH2_K")
    return int(given) if given else (7 if seats <= 32 else 3)


def trace_region_bytes(max_model_len: int) -> int:
    """The trace region a chip for `max_model_len` (TRACE_REGIONS)."""
    return next((size for limit, size in TRACE_REGIONS if max_model_len <= limit), TRACE_REGION_LONG)


def worker_l1_size(environ: Mapping[str, str]) -> int:
    """The allocator's L1 a core, as `ttnn/common.py` opens the mesh for the tests (DEVICE_PARAMS["worker_l1_size"]):
    1,460,864 bytes less GLM53_KERNEL_CONFIG_EXTRA_KB (default 24) KB given to the kernel config ring."""
    return 1460864 - (int(environ.get("GLM53_KERNEL_CONFIG_EXTRA_KB", "24")) << 10)


def _flag(token: str) -> str | None:
    """A command-line token's flag name, normalized (`--max_num_seqs=8` -> `max-num-seqs`; `--no-x` -> `x`), or None
    for a value."""
    if not token.startswith("--") or token == "--":
        return None
    name = token[2:].split("=", 1)[0].replace("_", "-")
    return name[3:] if name.startswith("no-") else name


@dataclass
class ServeCommand:
    """The server's working directory, the environment it adds, its arguments (after `python -m`), and the flags of
    `extra` that replaced the script's own."""

    cwd: Path
    env: dict
    argv: list
    replaced: list = field(default_factory=list)

    def render(self, python: str = "python") -> str:
        """The command as one shell command line, a flag a line (what `--print` prints)."""
        lines = [f"cd {shlex.quote(str(self.cwd))} && \\"]
        lines += [f"{key}={shlex.quote(str(value))} \\" for key, value in self.env.items()]
        lines.append(f"{shlex.quote(python)} -m {self.argv[0]} \\")
        groups, current = [], []
        for token in self.argv[1:]:
            if token.startswith("--") and current:
                groups.append(current)
                current = []
            current.append(token)
        if current:
            groups.append(current)
        lines += ["  " + " ".join(shlex.quote(t) for t in group) + " \\" for group in groups]
        lines[-1] = lines[-1][: -len(" \\")]
        return "\n".join(lines)


def build(
    checkpoint: str | Path,
    drafter: str | Path,
    seats: int = 32,
    max_model_len: int = 33792,
    port: int = 8000,
    tool_calls: bool = False,
    trace_region: int | None = None,
    extra: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
) -> ServeCommand:
    """The server's command for these settings (no checks of the paths: `main` checks them before it runs)."""
    environ = os.environ if environ is None else environ
    if seats < ROWS or seats % ROWS:
        raise ValueError(f"--seats {seats}: a multiple of {ROWS} (the seats are spread over the {ROWS} mesh rows)")
    if max_model_len < 1:
        raise ValueError(f"--max-model-len {max_model_len}: at least 1")
    root = Path(environ.get("TT_METAL_HOME") or ROOT)
    k = drafts_per_step(seats, environ)
    tt = {
        "l1_small_size": L1_SMALL,
        "trace_region_size": trace_region or trace_region_bytes(max_model_len),
        "worker_l1_size": worker_l1_size(environ),
        "sample_on_device_mode": "decode_only",
        "prefill_chunk_alignment": CHUNK_TOKENS,
        "decode_interleave_prefill_steps": 1,
    }
    speculative = {"method": "custom_class", "model": DRAFTER_CLASS, "num_speculative_tokens": k}
    groups = [
        ["--model", str(checkpoint)],
        ["--served-model-name", SERVED_NAME],
        ["--max_num_seqs", str(seats)],
        ["--block_size", str(BLOCK_SIZE)],
        ["--max-model-len", str(max_model_len)],
        ["--no-enable-prefix-caching"],  # (the model keeps its own per-seat state and prefix store)
        ["--enable-chunked-prefill"],
        ["--max-num-batched-tokens", str(CHUNK_TOKENS)],
        ["--scheduler-cls", SCHEDULER],
        ["--speculative-config", json.dumps(speculative)],
        ["--hf-overrides", json.dumps({"architectures": [ARCHITECTURE]})],
        ["--additional-config", json.dumps({"tt": tt})],
        ["--override-generation-config", json.dumps({"top_k": DEFAULT_TOP_K})],
        ["--port", str(port)],
    ]
    if tool_calls:
        groups += [
            ["--enable-auto-tool-choice"],
            ["--tool-call-parser", TOOL_PARSER],
            ["--reasoning-parser", TOOL_PARSER],
        ]
    given = {_flag(token) for token in extra} - {None}
    for flag in ("max-num-seqs", "max-model-len"):
        if flag in given:
            raise ValueError(f"--{flag} after '--': give it to this script (the drafts and the trace region follow it)")
    kept = [group for group in groups if _flag(group[0]) not in given]
    replaced = [group[0] for group in groups if _flag(group[0]) in given]
    env = {
        "EXTRA_MODELS_DIR": str(root / MODEL_DIR / "tools" / "vllm_bundle"),
        "MESH_DEVICE": "(8, 4)",
        "TT_MESH_GRAPH_DESC_PATH": str(root / MESH_GRAPH),
        "GLM53_DFLASH2": "1",
        "GLM53_DFLASH2_DRAFTER": str(drafter),
    }
    paths = [p for p in environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if str(root) not in paths:
        env["PYTHONPATH"] = os.pathsep.join([str(root)] + paths)
    if not environ.get("TT_METAL_HOME"):
        env["TT_METAL_HOME"] = str(root)
    argv = ["vllm.entrypoints.openai.api_server"] + [t for group in kept for t in group] + list(extra)
    return ServeCommand(cwd=root, env=env, argv=argv, replaced=replaced)


def parse(argv: Sequence[str]) -> tuple[argparse.Namespace, list]:
    """This script's options and the flags for vLLM (after `--`, and any this script does not know)."""
    argv = list(argv)
    split = argv.index("--") if "--" in argv else len(argv)
    ours, after = argv[:split], argv[split + 1 :]
    p = argparse.ArgumentParser(
        prog="python -m models.demos.blackhole.glm53_flash.tools.serve",
        description="Start the vLLM OpenAI server for GLM-5.3-Flash with the release's settings.",
        epilog="Flags after '--' go to vLLM after the script's own (a flag the script also sets replaces its value).",
        allow_abbrev=False,
    )
    p.add_argument("--checkpoint", help="the BF16 checkpoint directory (default: $GLM53_CHECKPOINT, else $HF_MODEL)")
    p.add_argument("--drafter", help="the DFlash2 drafter's directory (default: $GLM53_DFLASH2_DRAFTER)")
    p.add_argument(
        "--seats",
        "--max-num-seqs",
        "--max_num_seqs",
        dest="seats",
        type=int,
        default=32,
        help="concurrent requests, a multiple of 8 (default 32; 7 drafts a step up to 32, else 3)",
    )
    p.add_argument(
        "--max-model-len",
        "--max_model_len",
        dest="max_model_len",
        type=int,
        default=33792,
        help="the longest context a request can reach, prompt and output (default 33792)",
    )
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tool-calls", action="store_true", help="tool calls: the glm47 tool-call and reasoning parsers")
    p.add_argument("--trace-region", type=int, help="the trace region a chip in bytes (default by --max-model-len)")
    p.add_argument("--print", action="store_true", help="print the command and exit without running it")
    args, unknown = p.parse_known_args(ours)
    return args, unknown + after


def main(argv: Sequence[str] | None = None) -> None:
    args, extra = parse(sys.argv[1:] if argv is None else argv)
    sys.path.insert(0, str(ROOT))  # (paths.py's package, when this file runs as a script)
    from models.demos.blackhole.glm53_flash.paths import checkpoint_dir, checkpoint_problem, drafter_dir

    try:
        checkpoint = checkpoint_dir(args.checkpoint)
        drafter = drafter_dir(args.drafter)
        command = build(
            checkpoint,
            drafter,
            seats=args.seats,
            max_model_len=args.max_model_len,
            port=args.port,
            tool_calls=args.tool_calls,
            trace_region=args.trace_region,
            extra=extra,
        )
    except (RuntimeError, ValueError) as error:
        sys.exit(f"serve: {error}")
    if args.print:
        print(command.render())
        return
    problem = checkpoint_problem(checkpoint)
    if problem:
        sys.exit(f"serve: --checkpoint {checkpoint}: {problem}")
    if not (drafter / "config.json").exists():
        sys.exit(f"serve: --drafter {drafter}: no config.json (the DFlash2 drafter: {DRAFTER_REPO})")
    root = os.environ.get("TT_METAL_HOME")
    if root and Path(root).resolve() != ROOT:
        print(f"serve: TT_METAL_HOME is {root}, this script is in {ROOT}: the server runs from {root}", file=sys.stderr)
    print(command.render(python=sys.executable), file=sys.stderr, flush=True)
    for flag in command.replaced:
        print(f"serve: {flag} given after '--' replaces the script's own", file=sys.stderr)
    os.chdir(command.cwd)
    env = {**os.environ, **command.env}
    os.execve(sys.executable, [sys.executable, "-m", *command.argv], env)


if __name__ == "__main__":
    main()
