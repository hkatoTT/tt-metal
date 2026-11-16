"""The CPU reference (`tt/`, bf16, HF's numerics) of the sequence-parallel test's prompts (tools/passkey.SP_CASES): the
logits after each prompt [V] fp32, and with --tokens N the reference's own greedy continuation (N tokens, the first
the one after the prompt) with the logits at each of those positions [N, V] (what the device test teacher-forces),
for tests/test_model_device_prefill_sp.py's KLD against the reference (which reads them from $GLM53_DATA_DIR/sp-v1/
and sp-v2/).

    python -m models.demos.blackhole.glm53_flash.tools.sp_reference --out $GLM53_DATA_DIR/sp-v1/logits.pt
    python -m models.demos.blackhole.glm53_flash.tools.sp_reference --tokens 24 --out $GLM53_DATA_DIR/sp-v2/teacher.pt

--lengths a,b,..: passkey prompts of about those lengths instead (depths from passkey.DEPTHS, words from passkey.WORDS,
seeded numbers), for the long cache's decode (past 2,048 tokens the indexer picks 512 of the visible pools); the prompt
ids go into the file ("prompts") for the device test (tests/test_model_device_long.py, GLM53_LONG_REF):

    python -m models.demos.blackhole.glm53_flash.tools.sp_reference --lengths 2600 --tokens 24 \
        --out $GLM53_DATA_DIR/long-v1/teacher-2600.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, checkpoint_root
from models.demos.blackhole.glm53_flash.tools.passkey import sp_prompts
from models.demos.blackhole.glm53_flash.tt.model import Glm53TextModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1, help="greedy positions after each prompt (default 1)")
    parser.add_argument(
        "--lengths", default="", help="passkey prompts of these lengths (default: tools/passkey.SP_CASES)"
    )
    args = parser.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(checkpoint_root()) / "tokenizer.json"))
    if args.lengths:
        from models.demos.blackhole.glm53_flash.tools.passkey import DEPTHS, WORDS, prompt_ids

        lengths = [int(x) for x in args.lengths.split(",")]
        g = torch.Generator().manual_seed(13)
        answers, prompts = [], []
        for i, n in enumerate(lengths):
            answer = f"{WORDS[i % len(WORDS)]}-{int(torch.randint(1000, 9999, (1,), generator=g))}"
            answers.append(answer)
            prompts.append(prompt_ids(tok, n, DEPTHS[(i + 3) % len(DEPTHS)], *answer.split("-")))
    else:
        answers, prompts = sp_prompts(tok)
    print(f"prompt lengths {[len(p) for p in prompts]}", flush=True)
    model = Glm53TextModel(Glm53Checkpoint())
    out, tokens = [], []
    for answer, ids in zip(answers, prompts):
        t0 = time.time()
        rows, chosen = [], []
        with torch.no_grad():
            logits, state, _ = model(torch.tensor([ids]), None)
            for step in range(args.tokens):
                rows.append(logits[0, -1].float())
                chosen.append(int(rows[-1].argmax()))
                if step + 1 < args.tokens:
                    logits, state, _ = model(torch.tensor([[chosen[-1]]]), state)
        out.append(torch.stack(rows))
        tokens.append(chosen)
        print(f"{len(ids)} tokens ({answer}): greedy {chosen[:8]}.., {time.time() - t0:.0f} s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    logits = torch.stack(out)  # [prompts, N, V]
    torch.save(
        {
            "lengths": [len(p) for p in prompts],
            "logits": logits[:, 0],
            "tokens": tokens,
            "teacher": logits,
            "prompts": prompts,
            "answers": answers,
        },
        args.out,
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
