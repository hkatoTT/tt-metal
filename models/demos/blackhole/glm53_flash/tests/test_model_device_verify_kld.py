"""The KLD of DFlash2's verify step against the CPU references, teacher-forced: the greedy records (tools/acceptance/
greedy-v1) on the 8 mesh rows (GLM53_VKLD_USERS a row, the records dealt to the slots in turn), every step T =
GLM53_VKLD_T tokens a user (`forward_verify`, eager, its logits read back), the record's own tokens in place of the
drafts (all of them accepted: the next step T positions on), from position 0 (the prompt too: every step is a verify
step).  At the positions that predict the generated tokens, against the CPU bf16 references of
test_model_device_accuracy (the decode path and the prefill path): mean / p99 KLD and the top-1 agreement.

The verify step's own forms that change the values (FIN_PACK, the progressive form GLM53_MOE_PROG_F, ...) measured
in turn on one load, GLM53_VKLD_SETTINGS ";"-separated, each "+"-joined parts
of "env.NAME=value" (an environment variable a program factory reads: the program cache cleared so that the programs
are made again) or "module.NAME=value" (`common.apply_setting`); "" the defaults (the floor).  A setting that changes
the weights (GLM53_EXPERT_DTYPE ...) needs its own load: set it for the run.

    GLM53_VKLD_USERS (1: 8 users, 64 tokens a column at T = 8; a comma list: each on the one load), GLM53_VKLD_T (8:
    K = 7), GLM53_VKLD_SETTINGS ("")
Results: out_dir()/verify-kld-T<T>-<stamp>.json (each setting's per-position KLD in the .pt).
Run (a device test, real weights):
    GLM53_DEVICE_TEST=1 pytest models/demos/blackhole/glm53_flash/tests/test_model_device_verify_kld.py -s
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM53_DEVICE_TEST") != "1", reason="a device test: set GLM53_DEVICE_TEST=1"
)
ttnn = pytest.importorskip("ttnn")

from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, ShapeOnlyCheckpoint  # noqa: E402
from models.demos.blackhole.glm53_flash.paths import data_dir, out_dir  # noqa: E402
from models.demos.blackhole.glm53_flash.ttnn.common import (  # noqa: E402
    DEVICE_PARAMS_TRACE,
    MESH_SHAPE,
    ROWS,
    apply_setting,
)
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model  # noqa: E402

RECORDS = Path(__file__).parents[1] / "tools" / "acceptance" / "greedy-v1"
REFERENCES = {
    # tools/run_cpu_reference.py --teacher-decode, and tools/bfp_study.py e2e's bf16 logits
    "cpu_decode_path": data_dir("teacher-decode-v1", "bf16_decode_logits.pt"),
    "cpu_prefill_path": data_dir("bfp-study-v1", "bf16_logits.pt"),
}
MAX_SEQ = 1024


def apply(mesh, spec: str):
    """One setting: "env.NAME=value" parts set in the environment (the program cache cleared), the others through
    `common.apply_setting` -> its restore."""
    undo_env, module_parts = [], []
    for part in (p for p in spec.split("+") if p):
        if part.startswith("env."):
            name, value = part[4:].split("=", 1)
            undo_env.append((name, os.environ.get(name)))
            os.environ[name] = value
        else:
            module_parts.append(part)
    restore = apply_setting("+".join(module_parts))
    if undo_env:
        mesh.clear_program_cache()

    def undo():
        restore()
        for name, value in reversed(undo_env):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if undo_env:
            mesh.clear_program_cache()

    return undo


def fixed_steps(slots, seqs, users: int, t: int):
    """Every slot from position 0 in steps of t tokens, all of them kept -> (toks, pos, live, scored [(slot, row j, the
    generated token k that row predicts)]) a step."""
    for p0 in range(0, max(len(s) for s in seqs), t):
        toks = torch.zeros(ROWS, t * users, dtype=torch.int64)
        pos = torch.zeros(ROWS, t * users, dtype=torch.int64)
        live = torch.zeros(ROWS, t * users, dtype=torch.bool)
        scored = []
        for s, (rec, seq) in enumerate(zip(slots, seqs)):
            r, u = s % ROWS, s // ROWS
            first = len(rec["prompt_token_ids"]) - 1  # the position predicting the first generated token
            for j in range(t):
                p = p0 + j
                toks[r, j * users + u] = seq[p] if p < len(seq) else 0
                pos[r, j * users + u] = p
                live[r, j * users + u] = p < len(seq)
                if 0 <= p - first < len(rec["generated_token_ids"]):
                    scored.append((s, j, p - first))
        yield toks, pos, live, scored


def run_setting(model, slots, record_of, offsets, refs, users: int, t: int) -> dict:
    """Every slot's record teacher-forced through verify steps of t tokens a user -> the KLD sums by reference."""
    vocab = model.config.vocab_size
    states = model.empty_state(users, spec=t - 1)
    io = model.alloc_spec_io(users, drafts=t - 1, mtp=False)
    seqs = [r["prompt_token_ids"] + r["generated_token_ids"] for r in slots]
    keep = torch.full((ROWS, users), t - 1, dtype=torch.int64)
    sums = {name: {"kld": [], "top1": []} for name in refs}
    where = []
    for toks, pos, live, scored in fixed_steps(slots, seqs, users, t):
        model.write_spec_inputs(io, toks, pos, keep, live=live)
        ids, lg = model.forward_verify(io, states, logits=True)
        logits = model.logits_to_host(lg, t * users).float()  # [ROWS, t U, V]
        for x in (ids, lg):
            ttnn.deallocate(x)
        rows, refrows = [], []
        for s, j, k in scored:
            r, u = s % ROWS, s // ROWS
            rows.append(logits[r, j * users + u, :vocab])
            refrows.append(offsets[record_of[s]] + k)
        if not rows:
            continue
        dev = torch.stack(rows)
        logp_dev = torch.log_softmax(dev, -1)
        top1 = dev.argmax(-1)
        where.append(torch.tensor(refrows))
        for name, ref in refs.items():
            cpu = ref[refrows].float()
            logp_cpu = torch.log_softmax(cpu, -1)
            sums[name]["kld"].append((logp_cpu.exp() * (logp_cpu - logp_dev)).sum(-1))
            sums[name]["top1"].append((top1 == cpu.argmax(-1)).float())
    for st in states:
        free = getattr(st, "deallocate", None)
        if free is not None:
            free()
    out = {"where": torch.cat(where)}
    for name, v in sums.items():
        kld = torch.cat(v["kld"])
        out[name] = {
            "positions": len(kld),
            "mean_kld": kld.mean().item(),
            "p99_kld": kld.quantile(0.99).item(),
            "top1_agreement": torch.cat(v["top1"]).mean().item(),
            "kld": kld,
        }
    return out


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS_TRACE], indirect=True)
def test_model_device_verify_kld(mesh_device):
    t_start = time.time()
    torch.set_num_threads(16)
    users_list = [int(v) for v in os.environ.get("GLM53_VKLD_USERS", "1").split(",")]
    t = int(os.environ.get("GLM53_VKLD_T", "8"))
    settings = os.environ.get("GLM53_VKLD_SETTINGS", "").split(";")
    records = [json.loads(f.read_text()) for f in sorted(RECORDS.glob("*.json"))]
    offsets, n = [], 0  # each record's first row in the reference logits (records in file order)
    for r in records:
        offsets.append(n)
        n += len(r["generated_token_ids"])
    refs = {name: torch.load(path) for name, path in REFERENCES.items() if path.exists()}
    synthetic = os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1"  # (the flow only: the KLD means nothing then)
    model = TtGlm53Model(
        mesh_device, ShapeOnlyCheckpoint() if synthetic else Glm53Checkpoint(), max_seq=MAX_SEQ, mtp=False
    )
    print(
        f"loaded in {time.time() - t_start:.0f} s; users a row {users_list}, T = {t}, settings {settings}", flush=True
    )
    result = {"t": t, "settings": {}}
    per_position = {}
    for users, spec in ((u, sp) for u in users_list for sp in settings):
        nslots = ROWS * users
        record_of = [i % len(records) for i in range(nslots)]  # (one a row: the first 8 records)
        slots = [records[i] for i in record_of]
        t1 = time.time()
        undo = apply(mesh_device, spec)
        try:
            got = run_setting(model, slots, record_of, offsets, refs, users, t)
        finally:
            undo()
        label = f"{nslots} users: " + (spec or "defaults")
        result["settings"][label] = {name: {k: v for k, v in got[name].items() if k != "kld"} for name in refs} | {
            "seconds": round(time.time() - t1)
        }
        per_position[label] = {name: got[name]["kld"] for name in refs} | {"where": got["where"]}
        for name in refs:
            g = got[name]
            print(
                f"verify KLD [{label}] vs {name}: {g['mean_kld']:.5f} (p99 {g['p99_kld']:.4f}), top-1 "
                f"{g['top1_agreement']:.4f}, {g['positions']} positions; {time.time() - t1:.0f} s",
                flush=True,
            )
    path = out_dir() / f"verify-kld-T{t}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    torch.save(per_position, path.with_suffix(".pt"))
    print(f"wrote {path}; {time.time() - t_start:.0f} s", flush=True)
