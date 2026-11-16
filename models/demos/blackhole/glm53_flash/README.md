# GLM-5.3-Flash

## Platforms:
    Galaxy (BH): one Blackhole Galaxy, 32 chips as an 8x4 mesh

## Introduction
This is GLM-5.3-Flash (text) on one Blackhole Galaxy, from the BF16 checkpoint
[zai-org/GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16). GLM-5.3-Flash is a mixture-of-experts
model of 320B parameters, 18B of them active per token (as its model card states). Its 45 layers combine linear
attention (Kimi delta attention, KDA, 34 layers) with sparse attention (DeepSeek sparse attention, DSA, 11 layers:
MLA with an indexer that picks the 2,048 most relevant past tokens), 288 routed experts (8 per token) plus a shared
expert, and a residual of 4 streams mixed by manifold hyper-connections (mHC). Its context is up to 1M tokens.

The whole model runs on the 32 chips with BFP8 weights and HiFi4 matmuls. Decoding uses DFlash2 speculative decoding:
a small drafter ([canada-quant/GLM-5.3-Flash-DFlash2-G](https://huggingface.co/canada-quant/GLM-5.3-Flash-DFlash2-G))
runs on the device next to the model.

Measured speed and quality are in [PERF.md](PERF.md); the implementation is described in [docs/DESIGN.md](docs/DESIGN.md).

### Key Features
- **Whole model on one Galaxy**: all 45 layers, BFP8 weights (converted from BF16 at load), HiFi4 matmuls with fp32
  accumulation, bf16 / fp32 activations.
- **Speculative decoding on the device**: the DFlash2 drafter proposes 7 (or 3) tokens a step and the model verifies
  them in one traced step; only drafts equal to the model's greedy token are kept, so the output is the greedy output.
- **Batch decoding**: 8 to 128 users (multiples of 8), up to 7,342 tok/s in all at 128 users.
- **Long contexts**: 1M tokens a user for up to 64 users (576K at 128 users), with the DSA cache in FP8 / BFP4 and split
  over a mesh row's chips.
- **Prefill**: chunked, traced, several requests at once (one per mesh row); a long prompt is split over the 8 mesh
  rows (sequence parallel).
- **Sampling**: temperature, top-p and top-k (up to 32) on the device, with speculative decoding.
- **vLLM-compatible**: an OpenAI-compatible server through the Tenstorrent vLLM TT plugin, with tool calls.
- **Optional fast presets** (lower-precision experts): 5-8 % (A) or 10-14 % (B) faster decode with a small change in
  quality.

## Prerequisites
- This branch of tt-metal, built from source as in [INSTALLING.md](../../../../INSTALLING.md):
  ```
  ./build_metal.sh
  ./create_venv.sh
  source python_env/bin/activate
  ```
- One Blackhole Galaxy, opened as one 8x4 mesh with the 2D torus fabric:
  ```
  export MESH_DEVICE="(8, 4)"
  export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
  ```
- About 650 GB of disk for the checkpoint. The host does not need that much memory: tensors are read through
  memory-mapped views of the safetensors shards.

### Weights
The model reads the BF16 checkpoint at a pinned revision. The download tool fetches it from Hugging Face (643 GB,
resumable) and verifies every file against the revision's listing (SHA-256 for the weights):
```
python -m models.demos.blackhole.glm53_flash.tools.download_checkpoint --out <checkpoint dir>
export GLM53_CHECKPOINT=<checkpoint dir>
```
- `--revision` (default: the pinned one), `--workers` (parallel downloads, 16), `--hash-threads` (16).
- `--verify-only` checks the files without fetching. A file that fails is deleted, so running the tool again fetches it;
  files already verified (same size and modification time) are not hashed again.

The DFlash2 drafter for speculative decoding:
```
huggingface-cli download canada-quant/GLM-5.3-Flash-DFlash2-G --local-dir <drafter dir>
export GLM53_DFLASH2_DRAFTER=<drafter dir>
```

Other paths:
- `GLM53_DATA_DIR` (default `./glm53_data`): reference outputs and task files that some tests read.
- `GLM53_OUT_DIR` (default `./glm53_out`): results that tests and tools write.

Loading the real weights takes about 9-10 minutes (the BF16 matrices are converted to BFP8 as they load).

## How to Run
### Running the Demo
```
pytest models/demos/blackhole/glm53_flash/demo/demo.py
```
The demo loads the model and the drafter, renders the prompt with the checkpoint's chat template, prefills it and
decodes up to 256 new tokens greedily with DFlash2 (K = 7), then prints the text and the decode speed. Without
`GLM53_DFLASH2_DRAFTER` it uses the plain greedy decode (one token a step).

Options (environment variables):
- `GLM53_DEMO_PROMPT`: the prompt (default: a short question).
- `GLM53_DEMO_PROMPTS`: a JSON file with several prompts instead, a list of strings or of `{"prompt": ...}` objects (up
  to 64; prompt i runs on mesh row i % 8).
- `GLM53_DEMO_MAX_NEW`: new tokens per prompt at most (default 256; a prompt stops earlier at an end token).
- `GLM53_DEMO_REASONING`: the chat template's reasoning effort, `low` (default), `high` or `max` (the template's own
  default: a long reasoning part first).
- `GLM53_DEMO_K`: DFlash2's drafts per step, 1 to 7 (default 7).
- `GLM53_SYNTHETIC_WEIGHTS=1`: random weights made on the device; loads in seconds, the text is noise.

For example:
```
GLM53_DEMO_PROMPT="Write a haiku about the sea." GLM53_DEMO_MAX_NEW=128 pytest models/demos/blackhole/glm53_flash/demo/demo.py
```
Every decode step runs all 8 x (users per row) seats of the mesh, so with a single prompt the reported speed is that of
one user in a batch of 8.

## Testing
### CPU tests (no device)
The CPU reference `tt/` follows Hugging Face Transformers' `glm5_next` model operation by operation and matches it bit
for bit. Tests that need no device:
```
pytest models/demos/blackhole/glm53_flash/tests/test_generator_no_device.py \
  models/demos/blackhole/glm53_flash/tests/test_vllm_adapter_no_device.py \
  --confcutdir=models/demos/blackhole/glm53_flash -p no:cacheprovider
```
`tests/test_reference_vs_hf.py` and `tests/test_model_vs_hf.py` compare `tt/` with Transformers on the checkpoint (they
need a Transformers version with `glm5_next`, 5.16 or later).

### Device tests
Device tests run only with `GLM53_DEVICE_TEST=1`:
```
GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/<test>.py -s
```
- `test_device_smoke.py`: opens the mesh and runs the collectives the model uses on both axes.
- `test_dflash2_generator.py`: DFlash2 in the serving generator against the plain greedy decode on the 11 greedy
  reference records (`tools/acceptance/greedy-v1`), with requests arriving into freed seats.
- `test_model_device_accuracy.py`: teacher-forced KLD against the CPU reference's logits and the greedy records (the
  reference logits are made by the tools in `tools/` and read from `GLM53_DATA_DIR`; see the test's docstring).
- `test_model_device_trace.py`, `test_dflash2_speed.py`, `test_prefill_bench.py`: decode and prefill speed.
- `test_model_device_long.py`: long-context passkey retrieval.
- `test_model_device_tasks.py`: task accuracy (GSM8K and others; the task files are made by `tools/*_tasks.py` and
  `tools/*_prompts.py` into `GLM53_DATA_DIR`).

Speed tests can use `GLM53_SYNTHETIC_WEIGHTS=1`: random weights made on the device, of the checkpoint's shapes (the
checkpoint's index and headers are still read), so the model loads in seconds. The timing matches the real weights
where kernels do not depend on values; expert routing is spread evenly (`GLM53_SYNTHETIC_ROUTING=spread`, the default).
Correctness and accuracy need the real weights.

## Details
### Options
Environment variables read when the model is built (set them before starting the process).

| Variable | Default | Effect |
|---|---|---|
| `GLM53_CHECKPOINT` | (required) | The BF16 checkpoint directory |
| `GLM53_DFLASH2_DRAFTER` | (unset) | The DFlash2 drafter's directory |
| `GLM53_DFLASH2` | `0` | vLLM: decode with DFlash2 speculative decoding |
| `GLM53_DFLASH2_K` | by seats | vLLM: DFlash2's drafts per step (default 7 up to 32 seats, else 3) |
| `GLM53_MTP` | `0` | Load the model's own MTP layer; vLLM: speculative decoding with one MTP draft per step instead of DFlash2 |
| `GLM53_ACCEPT_RELAX` | `0` (off) | Relaxed acceptance for greedy requests: a draft is also accepted when its logit is within this many nats of the top. Faster (more drafts accepted) but the text is no longer the greedy output |
| `GLM53_SYNTHETIC_WEIGHTS` | `0` | `1`: random weights made on the device (speed tests and plumbing checks) |
| `GLM53_TRACE_REGION_MB` | `256` | Trace region per chip for the tests and the demo (vLLM takes `trace_region_size` instead) |

**Fast presets** (see [PERF.md](PERF.md#fast-presets)). They change the output relative to the default:

| Preset | Variables |
|---|---|
| A | `GLM53_MOE_PROG_F=0.7 MOE_EXPERTS_FIDELITY=0 GLM53_MOE_PREFILL_FIDELITY=lofi` |
| B | `GLM53_MOE_PROG_F=0.6 MOE_EXPERTS_FIDELITY=0 GLM53_MOE_PREFILL_FIDELITY=lofi` |
| C | `GLM53_MOE_PROG_F=0.5 MOE_EXPERTS_FIDELITY=0 GLM53_MOE_PREFILL_FIDELITY=lofi` |
| All-BFP4 (comparison) | `GLM53_EXPERT_DTYPE=bfp4 MOE_EXPERTS_FIDELITY=0 GLM53_MOE_PREFILL_FIDELITY=lofi` |

- `GLM53_MOE_PROG_F=F`: the routed experts as a BFP4 base plus a BFP4 correction; each decode step reads the
  correction for the experts with the largest routing weight, within F times the bytes of BFP8. The prefill reads every
  correction.
- `MOE_EXPERTS_FIDELITY` (decode experts op: `0` LoFi, `2` / `3` / `4` HiFi2 / HiFi3 / HiFi4, `1` by the weights'
  format) and `GLM53_MOE_PREFILL_FIDELITY` (prefill experts: `lofi`, `hifi2`, `hifi3`, `hifi4`, `auto`): the experts'
  matmul fidelity, HiFi4 by default.

### Long contexts
By default a cache longer than 2,051 tokens keeps the DSA latents in FP8 and the indexer's pooled keys in BFP4, and a
cache of 8,192 tokens or more is split by position over the 4 chips of a mesh row: 1M tokens a user for up to 64 users,
576K at 128 users (see [PERF.md](PERF.md#long-context-and-memory) for memory, speed and quality). Shorter caches use
the dense attention form. Four switches set this back to a bf16 cache, the same on a row's 4 chips (up to 680K tokens
a user at 8 users, 272K at 32, 80K at 128):
```
export GLM53_KV_LATENT_FP8=0          # DSA latent cache in bf16 (default 1: FP8)
export GLM53_DSA_POOLS_DTYPE=bf16     # the indexer's pooled keys in bf16 (default bfp4)
export GLM53_DSA_SHARD_KV=0           # the latent cache not split (default 8192: split from a cache of that many tokens)
export GLM53_DSA_SCORES_SHARD=0       # the indexer's scoring not split (default 8192)
```
Serving with 1M-token contexts has not been measured yet.

### vLLM Serving
The model is served by vLLM 0.31.0 with the [Tenstorrent vLLM TT plugin](https://github.com/tenstorrent/vllm-tt-plugin)
at commit `ca7a182` or later (tested with `042a948`). Install both as the plugin's `docs/install-vllm-tt.sh` does, into
a Python environment that can import this tt-metal build, and add the Transformers version that knows the
`glm5_next` config:
```
pip install transformers==5.16.1
```

Register the model with the plugin and choose the speculative decoding:
```
export EXTRA_MODELS_DIR=$TT_METAL_HOME/models/demos/blackhole/glm53_flash/tools/vllm_bundle
export GLM53_DFLASH2=1
export GLM53_DFLASH2_DRAFTER=<drafter dir>
export MESH_DEVICE="(8, 4)"
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
```

Start the server (32 seats, DFlash2 with 7 drafts):
```
python -m vllm.entrypoints.openai.api_server --model $GLM53_CHECKPOINT --served-model-name glm-5.3-flash \
  --max_num_seqs 32 --block_size 64 --max-model-len 33792 --no-enable-prefix-caching \
  --enable-chunked-prefill --max-num-batched-tokens 8192 \
  --scheduler-cls models.demos.blackhole.glm53_flash.tools.glm53_vllm_scheduler.Glm53Scheduler \
  --speculative-config '{"method":"custom_class","model":"vllm_tt_plugin.model_owned_drafter","num_speculative_tokens":7}' \
  --hf-overrides '{"architectures": ["TTGlm5NextForConditionalGeneration"]}' \
  --additional-config '{"tt": {"l1_small_size": 32768, "trace_region_size": 1073741824, "worker_l1_size": 1436288, "sample_on_device_mode": "decode_only", "prefill_chunk_alignment": 8192, "decode_interleave_prefill_steps": 1}}'
```
- `--max_num_seqs` must be a multiple of 8 (the users are spread over the 8 mesh rows).
- `num_speculative_tokens` must equal the drafts per step the model uses: 7 up to 32 seats and 3 above, or
  `GLM53_DFLASH2_K`. The server refuses another value at start-up.
- The trace region (`trace_region_size`, bytes a chip) holds the captured decode and prefill traces, whose number
  grows with `--max-model-len` (a long cache has more prefill forms): 1 GiB for 33,792 as above; 768 MiB is too small
  there (the server stops at start-up) but enough for 2,048.
- The model keeps its own per-seat state (KDA states, DSA caches), not vLLM's paged KV cache. Prefix caching is not
  supported: every request prefills its whole prompt.
- With DFlash2, the server launches the next decode step while vLLM does its host work (this needs `Glm53Scheduler`).
  `GLM53_VLLM_RUN_AHEAD=0` turns this off.
- `GLM53_MTP=1` (in place of `GLM53_DFLASH2=1`, with `"num_speculative_tokens": 1`) uses the model's own MTP layer as
  the drafter instead (slower).
- `GLM53_PREFILL_CHUNK` sets the prefill chunk lengths (default `128,512,1024`).

Send a request:
```
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "Write a haiku about the sea."}], "max_tokens": 512, "temperature": 0}'
```

**Tool calls**: add `--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47`.

**Sampled requests** (temperature, top-p, top-k): without further settings the server takes greedy requests only
(temperature 0). Set `GLM53_VLLM_SAMPLING=1` and add
`--override-generation-config '{"top_k": 32}'`. The device draws each token from the step's 32 most likely candidates
and accepts the drafts against those draws, so speculative decoding keeps working with sampling. A request without a
top-k (such as the model card's recommended temperature 1.0 / top-p 0.95) needs the whole vocabulary, which the device
does not handle, so the server's default top-k of 32 keeps those requests on the device. Requests with repetition
penalties or logprobs are served too, but while one of them is in the batch, every step reads the full logits back to
the host for the plugin's acceptance, which slows the whole batch down by about 25 times.

**Chunked prefill** (on by default with DFlash2): a long prompt is prefilled in steps of 8,192 tokens with decode
steps between them, so the other users keep generating. With prompts of 30K tokens arriving while 24 requests decoded,
the longest stall of those requests went from 1.65 s (whole prompts) to 0.53 s; the long prompt's own prefill took
0.4 s longer and the throughput was about 2 % lower. It needs the flags of the command above:
`--enable-chunked-prefill --max-num-batched-tokens 8192`, `"prefill_chunk_alignment": 8192` and
`"decode_interleave_prefill_steps": 1` in the `tt` block of `--additional-config`, and `Glm53Scheduler`; the server stops
at start-up with a message when these do not match. `GLM53_VLLM_CHUNKED=0` prefills every prompt whole (then set
`--max-num-batched-tokens` to at least the longest prompt, e.g. 65536).

**Checks** (in the vLLM environment):
- The offline engine on the prompts of the 11 greedy reference records, each output compared with its record (the plain
  greedy decode, without speculative decoding):
  ```
  GLM53_DFLASH2=0 python -m models.demos.blackhole.glm53_flash.tools.vllm_offline_check --model $GLM53_CHECKPOINT
  ```
- Against a running server: `python models/demos/blackhole/glm53_flash/tools/vllm_tool_check.py --port <port>` (tool
  calls), `vllm_sampling_check.py --port <port>` (sampled requests on the device and the host paths) and
  `vllm_chunked_check.py --port <port>` (long prompts arriving while others decode, with and without chunked prefill).

Shut the engine down when a process that uses vLLM in-process ends (as `vllm_offline_check` does): a mesh left open
at exit can keep the boards' Ethernet cores busy for the next process.
