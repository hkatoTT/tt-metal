"""Greedy acceptance through vLLM's offline engine (vllm-tt-plugin + tools/glm53_vllm.py).

The 11 acceptance records' prompts go in as token ids (each `--repeat` times), temperature 0, as many output tokens as
each record holds; the generated ids are compared with the records (first divergence per request) and the wall times
of engine start-up and generation are reported.  Run in the vLLM env with the bundle registered (--model: the
checkpoint, default $GLM53_CHECKPOINT):

    EXTRA_MODELS_DIR=<tree>/models/demos/blackhole/glm53_flash/tools/vllm_bundle MESH_DEVICE="(8, 4)" \
        python -m models.demos.blackhole.glm53_flash.tools.vllm_offline_check --max-num-seqs 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from models.demos.blackhole.glm53_flash.paths import checkpoint_dir

RECORDS = Path(__file__).parent / "acceptance" / "greedy-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--sample-on-device", default="decode_only", help="the plugin's mode, or 'none' (host)")
    parser.add_argument("--out", default=None, help="a JSON file for the results")
    args = parser.parse_args()
    args.model = str(checkpoint_dir(args.model))

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    records = [json.loads(f.read_text()) for f in sorted(RECORDS.glob("*.json"))]
    # (worker_l1_size: the model's kernel config ring takes 24 KB of L1, as ttnn/common.py DEVICE_PARAMS)
    tt = {"l1_small_size": 32768, "trace_region_size": 256 << 20, "worker_l1_size": 1436288}
    if args.sample_on_device != "none":
        tt["sample_on_device_mode"] = args.sample_on_device
    t0 = time.time()
    llm = LLM(
        model=args.model,
        hf_overrides={"architectures": ["TTGlm5NextForConditionalGeneration"]},
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        block_size=64,
        enable_prefix_caching=False,
        additional_config={"tt": tt},
    )
    start_seconds = time.time() - t0
    print(f"engine ready in {start_seconds:.0f} s", flush=True)

    # the engine must be shut down (the TT worker closes the mesh and resets the fabric): a mesh left open at exit
    # wedges the boards' ethernet cores for the next process ("Timed out while waiting for active ethernet core")
    try:
        jobs = [r for r in records for _ in range(args.repeat)]
        prompts = [TokensPrompt(prompt_token_ids=r["prompt_token_ids"]) for r in jobs]
        params = [
            SamplingParams(temperature=0.0, max_tokens=len(r["generated_token_ids"]), ignore_eos=True) for r in jobs
        ]
        t0 = time.time()
        outputs = llm.generate(prompts, params)
        gen_seconds = time.time() - t0
        result = {
            "engine_start_seconds": round(start_seconds),
            "generate_seconds": round(gen_seconds, 1),
            "records": {},
        }
        tokens = 0
        for r, out in zip(jobs, outputs):
            got = list(out.outputs[0].token_ids)
            want = r["generated_token_ids"]
            tokens += len(got)
            div = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            result["records"].setdefault(r["id"], []).append(div)
            if r["id"] not in result.setdefault("texts", {}):
                result["texts"][r["id"]] = out.outputs[0].text
        full = sum(all(d == len(r["generated_token_ids"]) for d in result["records"][r["id"]]) for r in records)
        result["all_match_records"] = full
        print(
            f"{len(jobs)} requests, {tokens} tokens in {gen_seconds:.1f} s; {full}/{len(records)} records match in full",
            flush=True,
        )
        for r in records:
            print(
                f"  {r['id']}: first divergences {result['records'][r['id']]} of {len(r['generated_token_ids'])}",
                flush=True,
            )
        for name, text in result["texts"].items():
            print(f"  [{name}] {text[:160]!r}", flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    finally:
        llm.llm_engine.engine_core.shutdown()
        print("engine shut down", flush=True)


if __name__ == "__main__":
    main()
