"""Which requests GLM-5.3-Flash's vLLM serving with DFlash2 takes (tools/glm53_vllm.py), and the start-up checks of its
sampling settings.  No vLLM, plugin or ttnn import at module level (tests/test_vllm_requests_no_device.py runs without
them).

- GLM53_VLLM_SAMPLING (sampled requests): unset ("auto") on with DFlash2 when the installed vllm-tt-plugin certifies
  sampled ids (`spec_ids_are_sampled`, its ca7a182 or later), else off with a warning at start-up; "1" on (the
  start-up stops with the reason on an older plugin); "0" off.  Off as well, with a warning, under the looser
  acceptance (GLM53_ACCEPT_RELAX, the fast presets' delta: greedy ids only).  Off, the launch takes greedy requests
  only: the plugin refuses a request with a temperature (HTTP 400), and a request that sends no temperature takes the
  server's default (the model's generation_config.json: 1.0).
- GLM53_VLLM_HOST_SAMPLING (with sampling on): a request with a control the device does not draw (`host_path`: a
  penalty, logprobs, a grammar or a token filter, min_tokens, a temperature without a top-k of 1 .. 32) puts every step
  it is in into the plugin's "logits" accept mode, each step reading the full logits back to the host (about 1 s a step
  at 32 seats against about 40 ms: the whole batch about 25 times slower while it runs).  "refuse" (default): such a
  request gets HTTP 400 naming its controls, from the API server's per-request check (`install_request_check`: before
  the engine sees it; an error raised in the engine stops the engine for every request); "allow": served on that path.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAMPLING_EPS = 1e-5  # vLLM's: a temperature below it is greedy
VLLM_TEMPERATURE = 1.0  # vLLM's temperature for a request that sends none, without a server default
PLUGIN_KEY = "spec_ids_are_sampled"  # the capability vllm-tt-plugin reads from ca7a182 on
HOST_MODES = ("refuse", "allow")
PENALTIES = (("presence_penalty", 0.0), ("frequency_penalty", 0.0), ("repetition_penalty", 1.0))  # (neutral values)
SLOW = "the whole batch about 25 times slower while it runs"
TOP_K_FLAG = """--override-generation-config '{"top_k": 32}'"""


def plugin_certifies_sampled_ids() -> bool:
    """Whether the installed vllm-tt-plugin reads the `spec_ids_are_sampled` capability (its ca7a182 or later).  The
    plugin's version is 0.1.0 at every commit and an older plugin ignores an unknown capability without a word, so its
    sources are searched for the key; not imported (the adapter is imported while the plugin loads it)."""
    try:
        spec = importlib.util.find_spec("vllm_tt_plugin")
    except (ImportError, ValueError):
        return False
    if spec is None or not spec.submodule_search_locations:
        return False
    for root in spec.submodule_search_locations:
        for path in sorted(Path(root).glob("*.py")):
            try:
                if PLUGIN_KEY in path.read_text(errors="ignore"):
                    return True
            except OSError:
                continue
    return False


@dataclass(frozen=True)
class Sampling:
    """GLM53_VLLM_SAMPLING as resolved when the adapter is imported (its `model_capabilities` follow `on`)."""

    on: bool  # the verify step draws sampled requests (the adapter's SAMPLING)
    asked: str  # GLM53_VLLM_SAMPLING: "auto" (unset), "1", "0" or a wrong value (the start-up stops)
    plugin: bool  # the installed plugin certifies sampled ids (searched only with DFlash2 and "auto" / "1")
    relax: bool = False  # the looser acceptance (GLM53_ACCEPT_RELAX not 0), which the sampled step does not take


def sampling_setting(dflash2: bool, environ: Any = None) -> Sampling:
    """GLM53_VLLM_SAMPLING with DFlash2 (sampled requests need DFlash2's verify step): unset on where the plugin
    certifies sampled ids and the acceptance is exact (`DFlash2Step` takes no sampling with GLM53_ACCEPT_RELAX), "1"
    on (`check_launch` stops on an older plugin or with GLM53_ACCEPT_RELAX), "0" off."""
    env = os.environ if environ is None else environ
    asked = env.get("GLM53_VLLM_SAMPLING", "auto")
    try:
        relax = float(env.get("GLM53_ACCEPT_RELAX", "0") or 0) != 0  # (as ttnn/head.py reads it)
    except ValueError:
        relax = True  # (a wrong value: the head stops on it)
    plugin = dflash2 and asked in ("auto", "1") and plugin_certifies_sampled_ids()
    return Sampling(on=plugin and not relax, asked=asked, plugin=plugin, relax=relax)


def host_mode(environ: Any = None) -> str:
    """GLM53_VLLM_HOST_SAMPLING: "refuse" (default) or "allow"."""
    env = os.environ if environ is None else environ
    mode = env.get("GLM53_VLLM_HOST_SAMPLING", "refuse")
    if mode not in HOST_MODES:
        raise ValueError(f"GLM53_VLLM_HOST_SAMPLING={mode!r}: 'refuse' (default) or 'allow'")
    return mode


def host_path(params: Any, device_top_k: int, device_samples: bool = True, host_seeds: bool = False) -> list:
    """A request's controls (vLLM's SamplingParams) that the device does not draw, as (parameter, text) pairs; [] for a
    request whose every step the device draws (greedy, or a temperature with a top-k of 1 .. device_top_k, any top-p
    and seed).  As vllm-tt-plugin routes it (042a948: `TTModelRunner.check_perform_device_sampling` and
    `_request_is_argmax_certifiable`): a non-zero temperature with no top-k or a larger one is sampled on the host (its
    input batch takes top-k 0 as the whole vocabulary; vLLM keeps a temperature under SAMPLING_EPS, greedy, with top-k
    0); a penalty, logprobs, a grammar, a token filter and min_tokens take the host's acceptance at any temperature.
    `device_samples`: the launch samples on the device (sample_on_device_mode "decode_only"); `host_seeds`: a seeded
    request's step is sampled on the host (seeded_sampling_policy "host")."""
    out = []
    t = float(getattr(params, "temperature", 0.0) or 0.0)
    if t != 0.0:
        k = int(getattr(params, "top_k", 0) or 0)
        if not device_samples:
            out.append(("temperature", f"temperature={t:g} (this launch samples no step on the device)"))
        elif not 1 <= k <= device_top_k:
            shown = k if k >= 1 else "none"
            out.append(
                ("top_k", f"temperature={t:g} with top_k={shown} (send top_k 1 to {device_top_k}, or temperature 0)")
            )
        if float(getattr(params, "min_p", 0.0) or 0.0) > SAMPLING_EPS:
            out.append(("min_p", f"min_p={float(params.min_p):g}"))
    if host_seeds and getattr(params, "seed", None) is not None:
        out.append(("seed", f"seed={params.seed} (seeded_sampling_policy 'host')"))
    for name, neutral in PENALTIES:
        value = getattr(params, name, None)
        if value is not None and float(value) != neutral:
            out.append((name, f"{name}={float(value):g}"))
    if getattr(params, "logprobs", None) is not None:
        out.append(("logprobs", f"logprobs={params.logprobs}"))
    if getattr(params, "structured_outputs", None) is not None:
        out.append(("structured_outputs", "structured outputs (a JSON schema, a grammar, a required or named tool)"))
    for name in ("allowed_token_ids", "bad_words", "logit_bias"):
        if getattr(params, name, None):
            out.append((name, name))
    if int(getattr(params, "min_tokens", 0) or 0) > 0:
        out.append(("min_tokens", f"min_tokens={params.min_tokens}"))
    return out


def refusal(found: list) -> Exception:
    """The error for a request with host-path controls (`host_path`'s pairs): vLLM's VLLMValidationError, which its
    OpenAI server answers with HTTP 400 (a streamed request: an error event in the stream)."""
    message = (
        "This GLM-5.3-Flash server draws tokens on its device and refuses a request with sampling controls the device "
        f"does not draw: {'; '.join(text for _, text in found)}. Such a request would take the host's path, which reads "
        f"the full logits back at every step: {SLOW}. Drop these controls. The operator can admit such requests with "
        "GLM53_VLLM_HOST_SAMPLING=allow."
    )
    try:
        from vllm.exceptions import VLLMValidationError
    except ImportError:  # (tests without vLLM)
        return ValueError(message)
    return VLLMValidationError(message, parameter=found[0][0])


@dataclass(frozen=True)
class RequestCheck:
    """The API server's check of each request (GLM53_VLLM_HOST_SAMPLING=refuse, `install_request_check`): a request
    with host-path controls (`host_path`) refused with HTTP 400."""

    device_top_k: int
    device_samples: bool = True
    host_seeds: bool = False

    def __call__(self, params: Any) -> None:
        if not hasattr(params, "temperature"):  # (not a generation request)
            return
        found = host_path(params, self.device_top_k, self.device_samples, self.host_seeds)
        if found:
            raise refusal(found)


def install_request_check(check: Any, platform: Any = None) -> None:
    """Run `check(params)` after the platform's own per-request check (vLLM's `Platform.validate_request`, which its
    InputProcessor calls in the API server process for each request before the engine sees it: a raise there fails
    that request alone, HTTP 400; a raise in the engine stops the engine for every request).  Installed once on the
    platform's class (`platform`: vLLM's current_platform); a later call replaces the check (None: none).  The plugin's
    own check is not changed and runs first."""
    if platform is None:
        from vllm.platforms import current_platform as platform
    cls = platform if isinstance(platform, type) else type(platform)
    if "_glm53_request_check" not in cls.__dict__:
        original = cls.validate_request  # (bound to cls: a classmethod)

        def validate_request(klass, processed_inputs, params):
            original(processed_inputs, params)
            extra = getattr(klass, "_glm53_request_check", None)
            if extra is not None:
                extra(params)

        cls.validate_request = classmethod(validate_request)
    cls._glm53_request_check = check


def check_launch(sampling: Sampling, tt: dict, defaults: dict, device_top_k: int, environ: Any = None) -> tuple:
    """DFlash2's sampling settings at start-up (the adapter's `spec_plan`, in every process that builds the config, the
    API server's included): raises with the reason for a launch that cannot work; returns the warnings to log and the
    request check to install (None: none).  `tt`: the tt block of --additional-config; `defaults`: the server's default
    sampling parameters (vLLM's `ModelConfig.get_diff_sampling_param`: the model's generation_config.json with
    --override-generation-config on top)."""
    if sampling.asked not in ("auto", "0", "1"):
        raise ValueError(f"GLM53_VLLM_SAMPLING={sampling.asked!r}: unset (on with a plugin that supports it), 1 or 0")
    host = host_mode(environ)
    if sampling.asked == "1" and not sampling.plugin:
        raise ValueError(
            "GLM53_VLLM_SAMPLING=1 needs vllm-tt-plugin ca7a182 or later: this plugin does not read the "
            f"{PLUGIN_KEY} capability, so every step with a sampled request would take the host's path ({SLOW}). "
            "Update the plugin, or unset GLM53_VLLM_SAMPLING for greedy requests only."
        )
    if sampling.asked == "1" and sampling.relax:
        raise ValueError(
            "GLM53_VLLM_SAMPLING=1 with GLM53_ACCEPT_RELAX (the looser acceptance, which compares greedy ids): the "
            "sampled verify step has no looser acceptance. Unset one of them."
        )
    temperature = float(defaults.get("temperature", VLLM_TEMPERATURE))
    top_k = int(defaults.get("top_k", 0) or 0)
    notes = []
    if not sampling.on:
        if sampling.asked == "auto" and sampling.relax:
            notes.append(
                "GLM53_ACCEPT_RELAX (the looser acceptance, greedy ids): sampled requests off (GLM53_VLLM_SAMPLING), "
                "greedy requests only."
            )
        elif sampling.asked == "auto":
            notes.append(
                f"vllm-tt-plugin before ca7a182 (it does not read {PLUGIN_KEY}): sampled requests off "
                "(GLM53_VLLM_SAMPLING), greedy requests only."
            )
        if temperature != 0.0:
            notes.append(
                f"This launch takes greedy requests only, and a request that sends no temperature takes the server's "
                f"default temperature {temperature:g}: the plugin refuses it with HTTP 400. Have clients send "
                "temperature 0, or start the server with --override-generation-config '{\"temperature\": 0}'."
            )
        return notes, None
    mode = tt.get("sample_on_device_mode")
    if mode == "all":
        raise ValueError(
            "Sampled requests (GLM53_VLLM_SAMPLING, on by default with DFlash2) need sample_on_device_mode "
            "'decode_only' in the tt block of --additional-config, got 'all': the prefill's device sampling is the "
            "argmax only, so a sampled request's prefill would stop the engine. Set 'decode_only', or "
            "GLM53_VLLM_SAMPLING=0 for greedy requests only."
        )
    device_samples = mode == "decode_only"
    answer = "refused with HTTP 400" if host == "refuse" else f"served on the host's path, {SLOW}"
    if not device_samples:
        notes.append(
            f"sample_on_device_mode {mode!r}: the device draws no step, so every sampled request is {answer}. Set "
            "'decode_only' in the tt block of --additional-config."
        )
    if temperature != 0.0 and not 1 <= top_k <= device_top_k:
        notes.append(
            f"A request that sends no top_k takes the server's default: temperature {temperature:g} with "
            f"{'no top-k' if top_k < 1 else f'top-k {top_k}'} (the server's defaults: {defaults or 'none'}). The "
            f"device draws from at most the {device_top_k} most likely tokens, so such a request is {answer} "
            f"(GLM53_VLLM_HOST_SAMPLING={host}). To draw it on the device, start the server with {TOP_K_FLAG} (the "
            "default distribution then cut to the 32 most likely tokens), or have clients send top_k."
        )
    penalties = [
        f"{n}={float(defaults[n]):g}" for n, neutral in PENALTIES if float(defaults.get(n, neutral)) != neutral
    ]
    if penalties:
        notes.append(
            f"The server's defaults hold {', '.join(penalties)}: a request that sends none of these is {answer}."
        )
    host_seeds = tt.get("seeded_sampling_policy", "auto") == "host"
    check = RequestCheck(device_top_k, device_samples, host_seeds) if host == "refuse" else None
    return notes, check
