"""The CPU reference for a llama.cpp-style KL divergence on wikitext-2 test.

llama.cpp's perplexity tool (--kl-divergence) tokenizes the whole wiki.test.raw, cuts it into chunks of n_ctx tokens,
runs each chunk from an empty context and scores the second half: the logits at positions n_ctx / 2 .. n_ctx - 2
(each predicting the next token); usually 48 chunks of 1,024.  Here the first WT_CHUNKS chunks (default 8) of the same
text through `tt/` (BF16, HF's numerics bit for bit, the LM head in FP32; no BOS: the GLM tokenizer adds none to plain
text), the scored logits saved for the device test (tests/test_model_device_wikitext.py, which reads
data_dir("wikitext", "reference-8x1024.pt") by default).  wiki.test.raw is wikitext-2-raw-v1's test file (the
wikitext-2-raw-v1 zip, as llama.cpp's scripts/get-wikitext-2.sh fetches it).

    python -m models.demos.blackhole.glm53_flash.tools.wikitext_reference <wiki.test.raw> $GLM53_DATA_DIR/wikitext [chunks]
-> <out dir>/reference-<chunks>x<ctx>.pt: tokens [C, ctx] int32, logits [C, ctx / 2 - 1, V] fp32 (positions
   ctx / 2 .. ctx - 2), first (ctx / 2), the text's sha256 and the chunks' count.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import torch

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, checkpoint_root
from models.demos.blackhole.glm53_flash.config import Glm53Config
from models.demos.blackhole.glm53_flash.tools.bfp_study import teacher_forced

CTX = int(os.environ.get("WT_CTX", "1024"))


def chunks_of(path: Path, n: int, ctx: int = CTX) -> torch.Tensor:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(checkpoint_root())
    ids = tok(path.read_text(), add_special_tokens=False)["input_ids"]
    if len(ids) < n * ctx:
        raise ValueError(f"{len(ids)} tokens: fewer than {n} chunks of {ctx}")
    return torch.tensor(ids[: n * ctx], dtype=torch.int32).view(n, ctx)


def main() -> int:
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("WT_CHUNKS", "8"))
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("WT_THREADS", "24")))
    tokens = chunks_of(raw, n)
    first = CTX // 2
    ckpt = Glm53Checkpoint()
    config = Glm53Config.from_dict(ckpt.config)
    t0 = time.time()
    logits = teacher_forced(ckpt, config, [row.tolist() for row in tokens], [slice(first, CTX - 1)] * n)
    path = out / f"reference-{n}x{CTX}.pt"
    torch.save(
        {
            "tokens": tokens,
            "logits": logits.view(n, CTX - 1 - first, -1),
            "first": first,
            "ctx": CTX,
            "text_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "seconds": round(time.time() - t0),
        },
        path,
    )
    print(f"wrote {path}: {n} chunks, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
