"""The vLLM serving's sampling settings and request check (`tools/glm53_vllm_requests.py`): no device, no vLLM, no
ttnn (stand-ins for vLLM's SamplingParams, its platform class and the plugin's package).

    pytest models/demos/blackhole/glm53_flash/tests/test_vllm_requests_no_device.py \
        --confcutdir=models/demos/blackhole/glm53_flash -p no:cacheprovider
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

from models.demos.blackhole.glm53_flash.tools import glm53_vllm_requests as req

K = 32  # the device's top-k limit (ttnn/sampling.py TOP)


def params(**over):
    """vLLM's SamplingParams as its OpenAI server makes them (the neutral values), with `over` on top."""
    base = dict(
        temperature=0.0, top_k=0, top_p=1.0, min_p=0.0, seed=None, presence_penalty=0.0, frequency_penalty=0.0,
        repetition_penalty=1.0, logprobs=None, structured_outputs=None, allowed_token_ids=None, bad_words=[],
        logit_bias=None, min_tokens=0,
    )  # fmt: skip
    return SimpleNamespace(**{**base, **over})


def names(p, **kw):
    return [name for name, _ in req.host_path(p, K, **kw)]


def test_host_path():
    """The controls the device does not draw, as the plugin routes them."""
    assert names(params()) == []  # greedy
    assert names(params(temperature=1.0, top_k=32, top_p=0.95, seed=7)) == []  # drawn on the device
    assert names(params(temperature=0.7, top_k=1)) == []
    assert names(params(temperature=1.0, top_k=0)) == ["top_k"]  # (no top-k: the whole vocabulary)
    assert names(params(temperature=1.0, top_k=-1)) == ["top_k"]
    assert names(params(temperature=1.0, top_k=50)) == ["top_k"]
    assert names(params(temperature=1e-6, top_k=0)) == ["top_k"]  # (vLLM's greedy, the plugin's sampled row)
    assert names(params(temperature=1.0, top_k=32, min_p=0.1)) == ["min_p"]
    assert names(params(repetition_penalty=1.1)) == ["repetition_penalty"]  # (greedy too)
    assert names(params(temperature=1.0, top_k=32, presence_penalty=0.5, frequency_penalty=-0.5)) == [
        "presence_penalty",
        "frequency_penalty",
    ]
    assert names(params(logprobs=0)) == ["logprobs"]  # (the sampled token's own logprob: the host's walk too)
    assert names(params(structured_outputs=object())) == ["structured_outputs"]
    assert names(params(allowed_token_ids=[1, 2], bad_words=["x"], logit_bias={1: 1.0})) == [
        "allowed_token_ids",
        "bad_words",
        "logit_bias",
    ]
    assert names(params(min_tokens=4)) == ["min_tokens"]
    # a launch that samples no step on the device; seeds sampled on the host
    assert names(params(temperature=1.0, top_k=32), device_samples=False) == ["temperature"]
    assert names(params(), device_samples=False) == []
    assert names(params(temperature=1.0, top_k=32, seed=3), host_seeds=True) == ["seed"]
    assert names(params(temperature=1.0, top_k=32), host_seeds=True) == []


def test_request_check(monkeypatch, expect_error):
    """RequestCheck: a request with host-path controls refused with vLLM's VLLMValidationError (its parameter the first
    control), the others passed; without vLLM a ValueError."""
    check = req.RequestCheck(K)
    check(params())
    check(params(temperature=1.0, top_k=32, top_p=0.95))
    check(SimpleNamespace(task="embed"))  # (not a generation request)
    monkeypatch.setitem(sys.modules, "vllm.exceptions", None)  # (no vLLM: the import fails)
    with expect_error(ValueError, "top_k=none"):
        check(params(temperature=1.0))
    with expect_error(ValueError, r"repetition_penalty=1\.1; logprobs=2.*GLM53_VLLM_HOST_SAMPLING=allow"):
        check(params(repetition_penalty=1.1, logprobs=2))

    class VLLMValidationError(Exception):
        def __init__(self, message, *, parameter=None, value=None):
            super().__init__(message)
            self.parameter = parameter

    fake = types.ModuleType("vllm.exceptions")
    fake.VLLMValidationError = VLLMValidationError
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.exceptions", fake)
    with expect_error(VLLMValidationError, "logprobs=0") as info:
        check(params(logprobs=0, min_tokens=2))
    assert info.value.parameter == "logprobs"


def test_install_request_check(monkeypatch, expect_error):
    """The check runs after the platform's own (which still refuses first), on the platform's class, once; a later
    install replaces the check, None removes it."""
    monkeypatch.setitem(sys.modules, "vllm.exceptions", None)  # (no vLLM: the refusal a ValueError)
    calls = []

    class Platform:
        @classmethod
        def validate_request(cls, processed_inputs, params):
            calls.append(("platform", cls.__name__))
            if getattr(params, "plugin_refuses", False):
                raise ValueError("the plugin's refusal")

    class TTPlatform(Platform):
        pass

    platform = TTPlatform()  # (vLLM's current_platform: an instance)
    req.install_request_check(lambda p: calls.append(("glm", p.n)), platform)
    platform.validate_request({}, SimpleNamespace(n=1))
    assert calls == [("platform", "TTPlatform"), ("glm", 1)]
    req.install_request_check(req.RequestCheck(K), platform)  # (again: replaced, not wrapped twice)
    calls.clear()
    platform.validate_request({}, params(temperature=1.0, top_k=32))
    assert calls == [("platform", "TTPlatform")]
    with expect_error(ValueError, "top_k=50"):
        platform.validate_request({}, params(temperature=1.0, top_k=50))
    with expect_error(ValueError, "the plugin's refusal"):  # (the plugin's check first)
        platform.validate_request({}, params(temperature=1.0, top_k=50, plugin_refuses=True))
    req.install_request_check(None, TTPlatform)
    platform.validate_request({}, params(temperature=1.0, top_k=50))
    Platform.validate_request({}, params(temperature=1.0, top_k=50))  # (the base class untouched)


def test_sampling_setting(monkeypatch):
    """GLM53_VLLM_SAMPLING: unset on where the plugin certifies sampled ids; "1" asks for it; "0" off; DFlash2 only."""
    monkeypatch.setattr(req, "plugin_certifies_sampled_ids", lambda: True)
    assert req.sampling_setting(True, {}) == req.Sampling(True, "auto", True)
    assert req.sampling_setting(True, {"GLM53_VLLM_SAMPLING": "1"}) == req.Sampling(True, "1", True)
    assert req.sampling_setting(True, {"GLM53_VLLM_SAMPLING": "0"}) == req.Sampling(False, "0", False)
    assert req.sampling_setting(False, {}) == req.Sampling(False, "auto", False)
    relax = {"GLM53_ACCEPT_RELAX": "2"}  # (the looser acceptance: the sampled step does not take it)
    assert req.sampling_setting(True, relax) == req.Sampling(False, "auto", True, relax=True)
    assert req.sampling_setting(True, {"GLM53_ACCEPT_RELAX": "0"}) == req.Sampling(True, "auto", True)
    monkeypatch.setattr(req, "plugin_certifies_sampled_ids", lambda: False)
    assert req.sampling_setting(True, {}) == req.Sampling(False, "auto", False)
    assert req.sampling_setting(True, {"GLM53_VLLM_SAMPLING": "1"}) == req.Sampling(False, "1", False)


def test_plugin_certifies_sampled_ids(monkeypatch, tmp_path):
    """The plugin's sources searched for the spec_ids_are_sampled capability (an older plugin: absent)."""
    pkg = tmp_path / "vllm_tt_plugin"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "model_runner.py").write_text('caps.get("max_device_top_k")\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "vllm_tt_plugin", raising=False)
    importlib.invalidate_caches()
    assert not req.plugin_certifies_sampled_ids()
    (pkg / "model_runner.py").write_text('caps.get("spec_ids_are_sampled", False)\n')
    assert req.plugin_certifies_sampled_ids()
    monkeypatch.setattr(req.importlib.util, "find_spec", lambda name: None)  # (no plugin installed)
    assert not req.plugin_certifies_sampled_ids()


GEN = {"temperature": 1.0, "top_p": 0.95}  # (the model's generation_config.json, as vLLM's server defaults)
TT = {"sample_on_device_mode": "decode_only"}
ON, AUTO_OLD, OFF = (
    req.Sampling(True, "auto", True),
    req.Sampling(False, "auto", False),
    req.Sampling(False, "0", False),
)


def test_check_launch_stops(expect_error):
    """The launches that cannot work stop at start-up, with the reason."""
    with expect_error(ValueError, "needs vllm-tt-plugin ca7a182"):
        req.check_launch(req.Sampling(False, "1", False), TT, GEN, K)
    with expect_error(ValueError, "GLM53_VLLM_SAMPLING='yes'"):
        req.check_launch(req.Sampling(False, "yes", False), TT, GEN, K)
    with expect_error(ValueError, "GLM53_VLLM_HOST_SAMPLING='drop'"):
        req.check_launch(ON, TT, GEN, K, {"GLM53_VLLM_HOST_SAMPLING": "drop"})
    with expect_error(ValueError, "with GLM53_ACCEPT_RELAX"):
        req.check_launch(req.Sampling(False, "1", True, relax=True), TT, GEN, K)
    with expect_error(ValueError, "a sampled request's prefill would stop the engine"):
        req.check_launch(ON, {"sample_on_device_mode": "all"}, GEN, K)
    req.check_launch(OFF, {"sample_on_device_mode": "all"}, GEN, K)  # (greedy only: the prefill's argmax is right)


def test_check_launch_greedy_only():
    """Sampled requests off: warned that a request without temperature 0 gets HTTP 400 (the server's default)."""
    notes, check = req.check_launch(AUTO_OLD, TT, GEN, K, {})
    assert check is None and len(notes) == 2
    assert "before ca7a182" in notes[0] and "default temperature 1: the plugin refuses it with HTTP 400" in notes[1]
    notes, check = req.check_launch(req.Sampling(False, "auto", True, relax=True), TT, GEN, K, {})
    assert check is None and len(notes) == 2 and "GLM53_ACCEPT_RELAX" in notes[0]
    notes, check = req.check_launch(OFF, TT, {}, K, {})  # (no server default: vLLM's own temperature 1.0)
    assert check is None and len(notes) == 1 and "HTTP 400" in notes[0]
    assert req.check_launch(OFF, TT, {"temperature": 0.0}, K, {}) == ([], None)


def test_check_launch_sampled():
    """Sampled requests on: the request check (refuse, the default) and the warnings for server defaults the device
    does not draw."""
    notes, check = req.check_launch(ON, TT, GEN, K, {})
    assert check == req.RequestCheck(K, device_samples=True, host_seeds=False)
    assert len(notes) == 1 and "sends no top_k" in notes[0] and "refused with HTTP 400" in notes[0]
    assert req.TOP_K_FLAG in notes[0]
    assert req.check_launch(ON, TT, dict(GEN, top_k=32), K, {}) == ([], check)  # (the README's server)
    assert req.check_launch(ON, TT, dict(GEN, temperature=0.0), K, {}) == ([], check)
    notes, check = req.check_launch(ON, TT, GEN, K, {"GLM53_VLLM_HOST_SAMPLING": "allow"})
    assert check is None and len(notes) == 1 and "25 times slower" in notes[0]
    notes, check = req.check_launch(ON, TT, dict(GEN, top_k=50), K, {})
    assert len(notes) == 1 and "top-k 50" in notes[0]
    notes, check = req.check_launch(ON, TT, dict(GEN, top_k=32, repetition_penalty=1.05), K, {})
    assert len(notes) == 1 and "repetition_penalty=1.05" in notes[0]
    notes, check = req.check_launch(ON, {}, dict(GEN, top_k=32), K, {})  # (no sample_on_device_mode)
    assert len(notes) == 1 and "draws no step" in notes[0] and not check.device_samples
    notes, check = req.check_launch(ON, dict(TT, seeded_sampling_policy="host"), dict(GEN, top_k=32), K, {})
    assert notes == [] and check.host_seeds
