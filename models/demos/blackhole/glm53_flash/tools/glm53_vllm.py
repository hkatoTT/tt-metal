"""vLLM adapter (vllm-tt-plugin) over the serving generator (`ttnn/generator.py`): text only, BF16, greedy on the device.

vllm-tt-plugin constructs `Glm53ForCausalLM` through `initialize_vllm_model` on the mesh it opened (`MESH_DEVICE="(8,
4)"`, the 2D torus fabric declared below, `l1_small_size` and `trace_region_size` from the `tt` block of
`--additional-config`).  Front-packed execution: vLLM's own data parallelism stays 1 and `--max_num_seqs` is 8 x the
users per mesh row.

- State slots: vLLM's state slot k is a host-side name for one physical slot of the generator (mesh row, user).  A
  prefill into slot k runs on phys[k]; decode row i runs on phys[i]; the plugin's `slot_remap` (row i reads old slot
  remap[i]) permutes the names only, so no state moves on the device.  phys starts interleaved over the mesh rows
  (slot k on row k mod 8, user k // 8): the plugin hands out the lowest free slots, so a batch of new requests lands
  on different rows and prefills together.
- Prefill: each row's prompt (prefix caching is declared unsupported; vLLM's chunked prefill too unless CHUNKED, on by
  default with DFlash2: a long prompt then comes in steps of 8,192 tokens, resumed on that grid, with decode steps
  between them), in the generator's chunks, up to one request per mesh row at once
  (`Glm53Generator.prefill_many`; a wave of one chunk of a captured length replays a trace); returns the last
  position's logits [B, 1, V] fp32.  A partway prompt's next chunk runs on the physical seat its earlier chunks ran on
  (`_resume`: the plugin picks each prefill's state slot afresh, so the prompt may come back in another vLLM slot when a
  request ended between its chunks; without that the chunk would run on that slot's seat, from another row's state
  and the seat's stale DSA rows).
- Prefix caching (on by default with DFlash2's chunked prefill and max_model_len past 8,192; GLM53_PREFIX_ENTRIES=0
  off): a long prompt's state at each multiple of 8,192 kept in a DRAM store, sized at start-up
  from the free DRAM (the log's "prefix store" line); a later request with the same first tokens prefills from there.
- Decode: every physical slot runs each step (one captured trace); rows with `start_pos` -1 run token 0 at position 0
  of their own slot.  Host sampling gets the logits [B, 1, V]; device sampling (`sample_on_device_mode`) is the
  device's argmax, admitted for greedy rows only (`max_device_top_k` 1, no device penalties: the plugin picks host
  sampling for anything else).
- vLLM's page table and KV cache are accepted and ignored: the DSA caches and KDA states are the generator's, per slot.
- GLM53_VLLM_SPREAD_ROWS=1 (default: on with the scheduler's gathering, GLM53_PREFILL_BATCH > 1, which DFlash2's
  serving has by default; else 0): a prefill's new requests onto free physical slots of distinct mesh rows where it
  can (`_place`: a wave takes one request a row, so two on a row take two waves; only free slots' physical slots are
  swapped), and the rows with a free slot published for the scheduler (`glm53_vllm_rows`, read by
  `glm53_vllm_scheduler.Glm53Scheduler` to admit at most one new request a free row).
  The slots in use are those the last decode / verify call ran (start_pos >= 0) and the ones prefilled since.
- MTP self-speculative decode (GLM53_MTP=1; launch with `--speculative-config '{"method": "custom_class", "model":
  "vllm_tt_plugin.model_owned_drafter", "num_speculative_tokens": 1}'`): the plugin's speculative-decode contract for
  a model that drafts itself, with one draft, greedy (`argmax_ids`), every decode step a verify.  `decode_forward` runs the generator's verify step on each row's [committed token, draft] and returns its
  greedy ids; the same trace has already made the next drafts, which `propose_draft_tokens` hands over (A after a
  rejected draft, B after an accepted one; after a prefill the first draft, from the MTP layer's step at the prompt's
  last position).  A row's KDA state goes back to its checkpoint unless the previous step accepted its draft
  (`accepted_counts` 2) or it was just prefilled.
- DFlash2 speculative decode instead (GLM53_DFLASH2=1; `num_speculative_tokens` K, GLM53_DFLASH2_K or by the seats: 7
  up to 32, else 3, the measured best): the drafter on the device (`ttnn/dflash2.py`, GLM53_DFLASH2_DRAFTER its
  checkpoint) drafts K a step; `decode_forward` replays the generator's
  whole DFlash2 step (verify, acceptance, the drafter's context and next block) and returns greedy ids rebuilt from its
  acceptance ([d1 .. dn, bonus, bonus ..]: the plugin's acceptance stops at the first mismatch, so it finds the same n;
  a row with fewer valid drafts than the device accepted takes the plugin's count, its next drafts then miss once);
  `propose_draft_tokens` hands over the step's next drafts (after a prefill the drafter's first drafts).
- DFlash2's serving runs ahead by default: the next inputs made on the device (GLM53_DFLASH2_NEXT_INPUTS), the host's work as whole-tensor ops (GLM53_VLLM_VECTOR), the
  next verify step launched while vLLM does its host work (GLM53_VLLM_AHEAD) and KDA's ping-pong state to throw a
  launched step away (GLM53_KDA_SPEC_PINGPONG: a second set of the verify states, 459 MiB a chip at 32 seats and
  max_model_len 9,408), each "1" unless set (`glm53_vllm_rows.serving_defaults`).  Launch with `--scheduler-cls
  models.demos.blackhole.glm53_flash.tools.glm53_vllm_scheduler.Glm53Scheduler` (each request's end, whether a request
  could be admitted, the hold of a prefill behind a launched step); without it nothing is launched ahead.
  GLM53_VLLM_RUN_AHEAD=0: none of them (each at its own default, off).
- The looser acceptance (GLM53_ACCEPT_RELAX = a gap in nats, default 0: off), with MTP or DFlash2:
  the model's verify step hands back a draft as its greedy id where the draft's logit is within the gap of the top
  (`TtLmHead.greedy_mesh`'s relax), so the plugin's acceptance (MTP) and the device's (DFlash2) take it as an equal
  one; nothing here changes.  The text then differs from the greedy decode's.
- Sampled requests with DFlash2 (GLM53_VLLM_SAMPLING; unset: on where the installed vllm-tt-plugin certifies sampled
  ids, its ca7a182 or later, and GLM53_ACCEPT_RELAX is not set, `glm53_vllm_requests.sampling_setting`; "0" greedy
  requests only): on a device-sampled step the verify step draws each position under its row's temperature / top-k
  (up to 32) / top-p / seed (`slot_sampling`, written to the device when they change), and its ids are those draws
  (`spec_ids_are_sampled`: the plugin's id comparison is then an exact acceptance, the drafts being the drafter's top
  picks); a step with a row the device does not sample (penalties, logprobs, a top-k past 32 or none) is the "logits"
  mode, the target's logits [B, K + 1, V] read back for the plugin's exact acceptance on the host (`_dflash_logits`,
  the whole batch about 25 times slower while such a row runs).  GLM53_VLLM_HOST_SAMPLING ("refuse" by default): such
  a request gets HTTP 400 from the API server's request check instead ("allow": served on that path).  A request
  without a top-k takes the host's path: serve with a default top-k of 32 (`--override-generation-config '{"top_k":
  32}'`, as the README starts the server) to keep it on the device; the start-up warns where the server's defaults
  (the model's generation_config.json: temperature 1.0, top-p 0.95) leave it out (`check_sampling`).

Environment: `MODEL_WEIGHTS_DIR` (default: `--model` as a directory, else `GLM53_CHECKPOINT` or `HF_MODEL`),
`GLM53_PREFILL_CHUNK` (default 128,512,1024: the prefill chunk lengths, chosen per prompt by `Glm53Generator.schedule`;
1,024 gives the one-row form's waves of 1,024 tokens a row and the sequence-parallel waves in pairs),
`GLM53_PREFILL_TRACE` (default: every chunk length; the lengths whose prefill forms are captured as
traces, with GLM53_PREFILL_TRACE_ALL every form their chunks take, the pairs with 1,024; empty for none),
`GLM53_SYNTHETIC_WEIGHTS=1` (random weights made on the device: seconds to load,
for plumbing checks only; the text is noise), `GLM53_VLLM_TIMELINE` (a file: one JSON line per prefill and decode
call, its wall-clock start and end, rows, tokens and active rows, to see where serving time goes).  vLLM imports this
module in several processes: nothing here opens a device.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.glm53_flash.checkpoint import Glm53Checkpoint, ShapeOnlyCheckpoint, checkpoint_root
from models.demos.blackhole.glm53_flash.tools import glm53_vllm_requests, glm53_vllm_rows
from models.demos.blackhole.glm53_flash.ttnn import prefix_keys as prefix_keys_module
from models.demos.blackhole.glm53_flash.ttnn.common import COLS, MESH_SHAPE, ROWS, fabric_router_config
from models.demos.blackhole.glm53_flash.ttnn.generator import Glm53Generator
from models.demos.blackhole.glm53_flash.ttnn.model import TtGlm53Model
from models.demos.blackhole.glm53_flash.ttnn.sampling import TOP as DEVICE_TOP_K

SAMPLING_EPS = 1e-5  # vLLM's: a temperature below it is greedy
TIMELINE = os.environ.get("GLM53_VLLM_TIMELINE")
MTP = os.environ.get("GLM53_MTP", "0") == "1"  # self-speculative decode (the model's own MTP layer drafts)
DFLASH2 = os.environ.get("GLM53_DFLASH2", "0") == "1"  # DFlash2's drafter on the device drafts K a step
SPEC = MTP or DFLASH2
DEFAULT_CHUNKS = "128,512,1024"  # GLM53_PREFILL_CHUNK's default
# a prefill's requests on distinct rows (`_place`).  Unset: on with the scheduler's gathering (GLM53_PREFILL_BATCH > 1,
# its default in DFlash2's serving: `glm53_vllm_rows.prefill_batch`; two requests of one call on a mesh row take two
# waves, 265 against 220 ms at 1K, 8 x 1K 550-680 against 360 ms), else off
SPREAD_ROWS = os.environ.get("GLM53_VLLM_SPREAD_ROWS", "1" if glm53_vllm_rows.prefill_batch()[0] > 1 else "0") == "1"
# vLLM's chunked prefill: a long prompt in chunks of up to the step's token budget
# (--max-num-batched-tokens 8192: one SP wave pair) with decode steps between them (the plugin's decode interleave), so
# that a long prompt no longer stops every running request for its whole prefill.  The scheduler
# (`glm53_vllm_scheduler.Glm53Scheduler`) admits no new request while a chunked prompt is partway (the generator resumes
# it from its rows' wave state: `Glm53Generator.prefill_many`'s `begin`), and the chunks end on the 8,192 grid
# (additional_config tt prefill_chunk_alignment 8192 at the server's launch).  On by default with DFlash2 (with
# prompts of 30K tokens arriving while 24 requests decode, the longest stall of those went from 1.65 to 0.53 s, for
# +0.4 s on the long prompt's own prefill and about 2 % less throughput); 0: whole prompts only (the default with MTP
# or the plain decode)
CHUNKED = os.environ.get("GLM53_VLLM_CHUNKED", "1" if DFLASH2 else "0") == "1"
CHUNK_GRID = 8192  # (the grid a partway prompt resumes on: `Glm53Generator.prefill_many`'s `begin`, one SP wave pair)
SCHEDULER_PATH = "models.demos.blackhole.glm53_flash.tools.glm53_vllm_scheduler.Glm53Scheduler"
# DFlash2's verify and propose calls' host work as torch's whole-tensor ops (`_dflash_verify_vec`, `_dflash_propose_vec`)
# in place of the loops over the rows (measured with the loops: the call's prep 0.24 and the rest after the step
# 1.33 ms at 32 seats); the same inputs and outputs.  Read at each call (tests switch it).  0: the loops (the
# default but in DFlash2's serving: `glm53_vllm_rows.serving_defaults`)
VECTOR = "GLM53_VLLM_VECTOR"
# DFlash2's next verify step launched at the end of this one (with VECTOR and the generator's NEXT_INPUTS; KDA's ping-pong
# state, GLM53_KDA_SPEC_PINGPONG, to throw it away), so that it runs while vLLM does its host work: "1" launch, "dry"
# decide and tell the scheduler without launching (a test's reference: the same steps), "0" (default but in DFlash2's
# serving, `glm53_vllm_rows.serving_defaults`) never.  Launched only when every seat's next inputs are those the device
# made (no proposal cut, the cache's end not near) and the scheduler (`glm53_vllm_scheduler.Glm53Scheduler`) said no
# request could be admitted (`glm53_vllm_rows.admission`: not published, no launch); a prefill after it throws it away
# (`prefill_forward`).  Read at each call (tests switch it)
AHEAD = "GLM53_VLLM_AHEAD"
# How near the end a step may be launched ahead (AHEAD; read at each call, tests switch it).  "vllm" in DFlash2's
# serving unless set (`glm53_vllm_rows.RUN_AHEAD_VALUES`), else "cache".  "cache": the next step's first position
# + 2 (K + 1) + the drafter's block below the cache's rows, which stops every seat's launch in the last ~25 positions of
# any request (a 1K + 1K request at max_model_len 2,048: about 5 steps each).  "vllm": only what the launched step
# needs to be the one vLLM asks for: vLLM (0.31 scheduler: a step's new tokens <= max_model_len - computed - 1) cuts a
# request's drafts once its next position p + K + 2 > max_model_len, and a cut step's inputs differ from the ones the
# device made, so p + K + 2 <= max_model_len for every running seat; the device side needs no more (the generator
# clamps positions past the cache's last row, the drafter's cache holds max_seq + its block).  "cut" (with
# AHEAD_SEATS): a seat past that edge does not stop the launch; the step is launched on inputs the host makes as vLLM
# will send them (`_given_inputs`: that seat's valid drafts min(K, max_model_len - p - 2), the others zero), so the
# last steps of a request that runs into max_model_len (1K + 1K at 2,048: ~2 steps each, every seat's launch stopped
# there) run ahead too.  The same steps and values either way; "vllm" launches more of them than "cache", "cut" more
# than "vllm"
AHEAD_EDGE = "GLM53_VLLM_AHEAD_EDGE"
# GLM53_VLLM_AHEAD_SEATS=1 (read at each call; on in DFlash2's serving unless set, `glm53_vllm_rows.RUN_AHEAD_VALUES`;
# else 0): with AHEAD, a step is also launched ahead where the device's own next inputs are not the next step's for
# some seats, on inputs the host makes from them (`Glm53Generator.dflash_launch_given`): (a) at a verify where
# requests end (the scheduler's budgets), their seats sitting out as an empty seat does (`_dflash_inputs`: token 0, the
# seat's idle position, keep K, not live); (b) at a prefill's end (the scheduler says the next step is a decode,
# `glm53_vllm_scheduler`'s admission after a prefill step), each seat just prefilled taking its first verify as the
# plugin sends it (its first id at its prompt's end, no drafts: the plugin's post-prefill default, keep K), the other
# seats as the last verify left them (its next inputs, the seats that ended there sitting out).  The same steps and
# values (the next verify takes the launched one only when its inputs are these, else throws it away and runs its own);
# the host's work before those two verifies then overlaps the device
AHEAD_SEATS = "GLM53_VLLM_AHEAD_SEATS"
# GLM53_VLLM_AHEAD_HOST_FIRST=1 (read at each call; with AHEAD_SEATS; default 0): (b) also when the prefill is sampled
# on the host (sample_on_device_mode "decode_only", as served with SAMPLING: the call returns the logits), for a call
# whose every request draws its first token as the logits' argmax, as the scheduler published them
# (`glm53_vllm_rows.first_argmax`): the launch on those argmax ids (vLLM's greedy pick, the first index of the
# maximum).  Without it, (b) never runs in such serving (no prefill launched ahead)
AHEAD_HOST_FIRST = "GLM53_VLLM_AHEAD_HOST_FIRST"
# GLM53_VLLM_IDLE_ZERO (read at each call; with AHEAD; 1 by default; =0: off): before a verify the scheduler said no
# request could be admitted for (`glm53_vllm_rows.admission()` False, which holds no partway prompt: a pending chunk
# makes it True, `Glm53Scheduler._admission`), each seat without a request in it whose idle position is not 0 (a
# request that ended at its prefill's first token: the prefill left the seat at its prompt's end,
# `Glm53Generator.prefill_many`, to keep a partway prompt's first rows) back to 0, as a seat whose request ran a verify
# is (`Glm53Generator._dflash_inputs`) and where the device's next inputs put a seat without a request
# (next_pack.cpp): a step launched ahead is then not thrown away for it.  Not the seats a launched step ran as sitting
# out at their next position (`_idle_moved`: `_idle_back` after the verify).  The same values (the seat is free: no
# request reads its rows).  Counted in the generator's dflash_writes["idle_zeroed"]
IDLE_ZERO = "GLM53_VLLM_IDLE_ZERO"
# Sampled requests with DFlash2: the verify step draws each position from the target's distribution under its row's
# temperature / top-k / top-p / seed (`DFlash2Step`'s GLM53_DFLASH2_SAMPLING), declared to the plugin as
# `spec_ids_are_sampled` (its "argmax_ids" then the device's draws, which the id comparison accepts exactly against the
# drafter's top picks), with the "logits" accept mode for the steps the device does not sample (penalties, logprobs, a
# top-k past the device's 32 or none: the plugin's exact acceptance on the host from the full-vocabulary logits,
# GLM53_DFLASH2_LOGITS).  Needs vllm-tt-plugin ca7a182 or later (vLLM 0.31): unset, on where the installed plugin has
# it, else off (a warning at start-up); "1" on (the start-up stops on an older plugin); "0" greedy requests only (the
# plugin refuses a sampled one with HTTP 400); off under the looser acceptance (GLM53_ACCEPT_RELAX), which the sampled
# step does not take.  `glm53_vllm_requests.sampling_setting`; the start-up checks, the warnings and the request check
# (GLM53_VLLM_HOST_SAMPLING) in `check_sampling`
SAMPLING_SETTING = glm53_vllm_requests.sampling_setting(DFLASH2)
SAMPLING = SAMPLING_SETTING.on
# The prefix store (with DFlash2's chunked prefill and a cache past CHUNK_GRID): the generator keeps entries of a
# prefill's state at a multiple of CHUNK_GRID.  Each prompt's continuation call first snapshots what its earlier calls
# left (one entry a request, extended call by call); a request that the scheduler
# (`glm53_vllm_scheduler.Glm53Scheduler`) found in the store's index comes from start_pos = the prefix's length, alone:
# its state is restored into a free seat of the entry's mesh row (`_prefix_seat`) and the prompt goes on from there;
# when the entry is gone or no seat of its row is free, the whole prompt so far from 0 (the same values, the time of a
# whole prefill).  The store's index and each mesh row's free seats are published for the scheduler after every call.
# GLM53_PREFIX_ENTRIES: unset or "auto" (the default): as many entries as the free DRAM holds at start-up
# (`prefix_store.auto_entries`: none when not one fits, the reason in the log); a number: that many; 0: none of it
PREFIX_SETTING = prefix_keys_module.entries_setting("auto")
PREFIX = DFLASH2 and CHUNKED and PREFIX_SETTING != 0


def prefill_chunks() -> tuple:
    """The prefill chunk lengths (GLM53_PREFILL_CHUNK)."""
    return tuple(int(c) for c in os.environ.get("GLM53_PREFILL_CHUNK", DEFAULT_CHUNKS).split(","))


def dflash2_k(seats: int) -> int:
    """DFlash2's drafts a step: GLM53_DFLASH2_K, else by the seats (the measured best: K = 7 up to 32 users, K = 3
    from 64; K = 4 .. 6 a row's tokens not a multiple of 32, slower everywhere)."""
    given = os.environ.get("GLM53_DFLASH2_K")
    return int(given) if given else (7 if seats <= 32 else 3)


def w1_shape(seats: int) -> tuple | None:
    """The DFlash2 verify step's (users a mesh row, tokens a user) for TtGlm53Model's `dflash_shape` (the L1 landing of
    the next MoE layer's expert weights by default, GLM53_W1, w1_landing.landing_c), None without DFlash2 (the landing
    only if GLM53_W1 asks)."""
    return (seats // ROWS, dflash2_k(seats) + 1) if DFLASH2 else None


def _record(kind: str, start: float, **fields: Any) -> None:
    """One line of the serving timeline (GLM53_VLLM_TIMELINE): the call's kind, wall-clock start and end, `fields`."""
    if TIMELINE:
        with open(TIMELINE, "a") as f:
            f.write(json.dumps(dict(kind=kind, start=start, end=time.time(), **fields)) + "\n")


PROTOCOL_SHIM = "Glm53ForCausalLM runs on the device through prefill_forward and decode_forward"


def _rows(values: Any) -> list:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        return values.tolist()
    return list(values)


def _greedy_only(params: Any, rows: int, call: str) -> None:
    """A device-sampled call that takes the argmax (a prefill, the plain decode; SAMPLING's draws are the verify
    step's): refused for a sampled row rather than answered greedily.  A backstop, not reached as served: a raise here
    is in the engine and stops it for every request (vLLM's EngineCore), so the plugin keeps a sampled row off these
    calls (`max_device_top_k` 1 without SAMPLING; with it, a prefill is never device-sampled, sample_on_device_mode
    "decode_only" checked at start-up by `check_sampling`, and DFlash2 sends every decode step as a verify)."""
    temperatures = _rows(getattr(params, "temperature", None))
    top_ks = _rows(getattr(params, "top_k", None))
    for i in range(rows):
        t = float(temperatures[i]) if i < len(temperatures) else 0.0
        k = int(top_ks[i]) if i < len(top_ks) else 1
        if t >= SAMPLING_EPS and k != 1:
            raise ValueError(f"row {i}: the {call}'s device sampling is greedy only (temperature {t}, top_k {k})")


def slot_sampling(params: Any, phys: list, slots: int, seed_base: list) -> tuple:
    """The plugin's device-sampling parameters (`TTSamplingParams`: per row lists, None for a greedy step) -> each
    physical slot's (top-k, top-p, temperature, seed) [slots] for `Glm53Generator.dflash_sampling`: row i on slot
    phys[i]; a slot without a row greedy (k 1); a row without a seed its slot's base (drawn at its prefill)."""
    k, p = torch.ones(slots, dtype=torch.int64), torch.ones(slots)
    t, seed = torch.zeros(slots), torch.tensor(seed_base, dtype=torch.int64)
    if params is None:
        return k, p, t, seed
    temps, top_ks, top_ps = (_rows(getattr(params, f, None)) for f in ("temperature", "top_k", "top_p"))
    seeds = _rows(getattr(params, "seed", None))
    for i, slot in enumerate(phys):
        if i < len(temps) and float(temps[i]) >= SAMPLING_EPS:
            t[slot] = float(temps[i])
            k[slot] = int(top_ks[i]) if i < len(top_ks) else 0
            p[slot] = float(top_ps[i]) if i < len(top_ps) else 1.0
            if i < len(seeds) and seeds[i] is not None:
                seed[slot] = int(seeds[i])
    return k, p, t, seed


def check_chunked(vllm_config: Any) -> None:
    """CHUNKED with vLLM's chunked prefill on: the settings the generator's resumption needs, else a clear error at
    start-up (a misplaced chunk would fail its request, `prefill_many`'s `begin`).  A partway prompt resumes from its
    rows' wave state as if it had run alone: its chunks must end on the CHUNK_GRID (additional_config tt
    prefill_chunk_alignment, a multiple of it), a step's prompt tokens must be exactly that alignment (then a prompt cut
    short never shares its step: any other prompt in the step leaves less than the alignment, and the plugin defers a
    chunk short of the grid), and no other prompt may run between its chunks (Glm53Scheduler).  vLLM 0.31 turns
    chunked prefill on by default, so a server started without these flags stops here (GLM53_VLLM_CHUNKED=0 or
    --no-enable-chunked-prefill: whole prompts)."""
    sched = getattr(vllm_config, "scheduler_config", None)
    if not CHUNKED or sched is None or not getattr(sched, "enable_chunked_prefill", False):
        return
    from vllm_tt_plugin.config import get_tt_prefill_chunk_alignment

    align = get_tt_prefill_chunk_alignment(vllm_config)
    limit = int(sched.max_num_batched_tokens)
    if int(getattr(sched, "long_prefill_token_threshold", 0) or 0) > 0:
        limit = min(limit, int(sched.long_prefill_token_threshold))
    cls = sched.scheduler_cls
    names = [cls] if isinstance(cls, str) else [c.__name__ for c in getattr(cls, "__mro__", ())]
    wrong = []
    if align <= 0 or align % CHUNK_GRID:
        wrong.append(f"additional_config tt prefill_chunk_alignment {align} (a multiple of {CHUNK_GRID})")
    if limit != align:
        wrong.append(f"a step's prompt tokens {limit} (--max-num-batched-tokens equal to the alignment)")
    if not any(str(n).endswith("Glm53Scheduler") for n in names):
        wrong.append(f"scheduler {cls} (--scheduler-cls {SCHEDULER_PATH})")
    if wrong:
        raise ValueError(
            "GLM-5.3-Flash's chunked prefill (GLM53_VLLM_CHUNKED=1) needs: "
            + "; ".join(wrong)
            + ". Or GLM53_VLLM_CHUNKED=0 / --no-enable-chunked-prefill for whole prompts."
        )


def check_sampling(vllm_config: Any) -> None:
    """DFlash2's sampling settings at start-up (`glm53_vllm_requests.check_launch`, in each process that builds the
    config: the API server's too): stops with the reason for a launch that cannot work (GLM53_VLLM_SAMPLING=1 on a
    plugin before ca7a182, sample_on_device_mode "all"), logs the warnings (a server default the device does not draw,
    e.g. no top-k), and with SAMPLING installs the API server's request check (GLM53_VLLM_HOST_SAMPLING=refuse, the
    default: a request the device does not draw gets HTTP 400 there instead of slowing the whole batch down)."""
    from vllm_tt_plugin.config import get_tt_config

    model_config = getattr(vllm_config, "model_config", None)
    defaults = model_config.get_diff_sampling_param() if model_config is not None else {}
    setting = dataclasses.replace(SAMPLING_SETTING, on=SAMPLING)  # (a test may switch SAMPLING)
    notes, check = glm53_vllm_requests.check_launch(setting, get_tt_config(vllm_config), defaults, DEVICE_TOP_K)
    for note in notes:
        logger.warning(note)
    if SAMPLING:
        glm53_vllm_requests.install_request_check(check)


class Glm53ForCausalLM:
    model_capabilities = {
        "supports_chunked_prefill": CHUNKED,
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,
        # (SAMPLING: the device draws a top-k of up to its 32 candidates; else greedy only)
        "max_device_top_k": DEVICE_TOP_K if SAMPLING else 1,
        "spec_ids_are_sampled": SAMPLING,
        "supports_device_penalties": False,
        # (the plugin passes these to ttnn.set_fabric_config: the routers' payload as the device tests open the mesh)
        "fabric_config": {"config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "router_config": fabric_router_config()},
        "supports_spec_decode": SPEC,
        "spec_requirements": ["device_propose"] if SPEC else [],
    }

    def __init__(self, generator: Glm53Generator, max_seq_len: int, **kwargs: Any) -> None:
        """`**kwargs` takes vLLM's `vllm_config` / `prefix`, whose presence its registry checks (never passed here)."""
        self.gen = generator
        self.max_seq_len = max_seq_len
        users = generator.slots // ROWS
        self.phys = [(k % ROWS) * users + k // ROWS for k in range(generator.slots)]  # vLLM state slot -> physical
        self.prefill_warm = False
        self.fresh: set = set()  # MTP: physical slots prefilled since their last verify (their state is current)
        self.drafts = None  # MTP: the last verify step's next drafts [slots, 2] (A, B); DFlash2: [slots, K]
        self.active: set = set()  # vLLM state slots in use (the last decode's rows at start_pos >= 0, then prefills)
        traced = os.environ.get("GLM53_PREFILL_TRACE", ",".join(str(c) for c in generator.chunks))
        self.prefill_traced = tuple(int(c) for c in traced.split(",") if c and int(c) in generator.chunks)
        self._no_scheduler_told = False  # (AHEAD without Glm53Scheduler's admission: warned once)
        # AHEAD_SEATS: what the last verify left for the next one (None: not foreseen: a seat whose next inputs are not
        # the device's, the cache's end, an end without a budget), as the physical seats whose requests end there; and
        # each seat prefilled since its last verify: (its first id, its prompt's length)
        self._seat_plan: set | None = None
        self._cut_plan: dict = (
            {}
        )  # AHEAD_EDGE "cut": the seats (physical) whose next drafts vLLM cuts -> their valid ones
        self._fresh_first: dict = {}
        # AHEAD_SEATS: the seats a launched step ran as sitting out at their next position (not their idle one, 0):
        # set back to 0 after the next verify (`_idle_back`)
        self._idle_moved: set = set()
        # SAMPLING: each physical slot's base seed for a request without its own (drawn when the slot is prefilled), and
        # the sampling parameters last written to the device (`_write_sampling`)
        self.seed_base = [0] * generator.slots
        self._seeds = random.SystemRandom()
        self._sampling_written = None
        # PREFIX: the store in use (a cache past CHUNK_GRID), each physical slot's entry of its own request's snapshots,
        # and the counts of hits / prompts from 0 in place of a hit / snapshots
        self.prefix_on = PREFIX and getattr(generator, "prefix", None) is not None
        self.prefix_own: dict = {}
        self.prefix_counts = {"hits": 0, "misses": 0, "snapshots": 0}
        self.seat_moves = 0  # (partway prompts brought back to their seat, `_resume`)
        self._note: dict = {}  # (the current prefill call's prefix / seat events, for the timeline)
        if self.prefix_on:
            self._publish()

    # -- vLLM's text-generation protocol: inspected by the registry, never called on the TT path -------------------
    def embed_input_ids(self, input_ids: Any) -> Any:
        raise NotImplementedError(PROTOCOL_SHIM)

    def forward(self, input_ids: Any, positions: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(PROTOCOL_SHIM)

    def compute_logits(self, hidden_states: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(PROTOCOL_SHIM)

    def modules(self):
        """No torch modules (vLLM's in-process engine walks them at exit to clear caches)."""
        return iter(())

    # -- construction ----------------------------------------------------------------------------------------------
    @classmethod
    def spec_plan(cls, vllm_config: Any, max_num_seqs: int, requested_k: int) -> Any:
        """The plugin's speculative admission (GLM53_MTP=1): one draft a step from the model's MTP layer, verified by
        greedy ids, every decode step a verify (no narrow decode).  The extra device bytes a sequence: the KDA
        checkpoints (34 layers x 16 heads x 128 x 128 fp32 a chip, and the conv history's oldest row); a token: the
        MTP layer's latent cache."""
        from vllm_tt_plugin.spec_decode import ACCEPT_MODE_ARGMAX_IDS, DRAFTER_STATE_INTERNAL, SpecPlan, SpecReject

        check_chunked(vllm_config)  # (the speculative serving's start-up: its config in hand)
        if DFLASH2:
            check_sampling(vllm_config)
            k = dflash2_k(max_num_seqs)
            if requested_k != k:
                return SpecReject(reason=f"DFlash2 at {max_num_seqs} seats drafts {k} a step", supported_k=(k,))
            modes = (ACCEPT_MODE_ARGMAX_IDS,)
            if SAMPLING:  # (the "logits" mode: the steps the device does not sample, the host's exact acceptance)
                from vllm_tt_plugin.spec_decode import ACCEPT_MODE_LOGITS

                modes += (ACCEPT_MODE_LOGITS,)
            return SpecPlan(
                effective_k=k,
                lanes_per_request=1,
                # the KDA checkpoints of the K + 1 tokens (34 layers, fp32 at most) and the drafter's K / V a token
                # (8 layers x K and V x 2 KV heads a chip x 128, bf16)
                extra_bytes_per_seq=34 * (16 * 128 * 128 + 6144) * 4 * k,
                extra_bytes_per_token=8 * 2 * 2 * 128 * 2,
                accept_modes=modes,
                drafter_state=DRAFTER_STATE_INTERNAL,
                supports_narrow_decode=False,
            )
        if not MTP or requested_k < 1:
            return SpecReject(reason="GLM-5.3-Flash drafts with its MTP layer only with GLM53_MTP=1", supported_k=(1,))
        return SpecPlan(
            effective_k=1,
            lanes_per_request=1,
            extra_bytes_per_seq=34 * (16 * 128 * 128 + 6144) * 4,
            extra_bytes_per_token=512 * 2,
            accept_modes=(ACCEPT_MODE_ARGMAX_IDS,),
            drafter_state=DRAFTER_STATE_INTERNAL,
            supports_narrow_decode=False,
        )

    @classmethod
    def get_max_tokens_all_users(
        cls, *, max_model_len: int | None = None, max_num_seqs: int | None = None, **kwargs: Any
    ) -> int:
        """The KV token pool the plugin sizes its block table from: one full context per sequence and an eighth more
        (no preemption: at contexts near max_model_len the scheduler's block rounding and free-block watermark left
        two of 32 seats unfilled and preempted and re-prefilled requests over and over).  The device keeps its
        own per-slot state, so this pool is bookkeeping only."""
        return int(max_model_len) * int(max_num_seqs) * 9 // 8

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config: Any,
        mesh_device: Any,
        max_batch_size: int,
        max_seq_len: int,
        tt_data_parallel: int = 1,
        optimizations: str | None = None,
        **kwargs: Any,
    ) -> "Glm53ForCausalLM":
        # DFlash2's serving: the run-ahead's switches on where unset (before the model and its states are made)
        if glm53_vllm_rows.serving_defaults():
            logger.info(
                "DFlash2 serving runs ahead (GLM53_VLLM_RUN_AHEAD=0: off): "
                + ", ".join(f"{v}={os.environ[v]}" for v in glm53_vllm_rows.RUN_AHEAD_SWITCHES)
            )
        if SAMPLING:  # the verify step's draws and its logits step (read when the step is made)
            os.environ["GLM53_DFLASH2_SAMPLING"] = os.environ["GLM53_DFLASH2_LOGITS"] = "1"
            logger.info(
                f"DFlash2 serves sampled requests (GLM53_VLLM_SAMPLING {SAMPLING_SETTING.asked}: spec_ids_are_sampled, "
                f"logits mode); requests the device does not draw: {glm53_vllm_requests.host_mode()} "
                "(GLM53_VLLM_HOST_SAMPLING)"
            )
        if tt_data_parallel != 1:
            raise ValueError(
                f"GLM-5.3-Flash serves front-packed on one 8x4 mesh: data parallel 1, got {tt_data_parallel}"
            )
        shape = tuple(int(x) for x in mesh_device.shape)
        if shape != MESH_SHAPE:
            raise ValueError(f"GLM-5.3-Flash needs the {MESH_SHAPE} mesh (MESH_DEVICE='(8, 4)'), got {shape}")
        if max_batch_size % ROWS:
            raise ValueError(f"max_num_seqs ({max_batch_size}) must be a multiple of the {ROWS} mesh rows")
        chunks = prefill_chunks()
        chunk = max(chunks)
        if os.environ.get("GLM53_SYNTHETIC_WEIGHTS") == "1":
            checkpoint = ShapeOnlyCheckpoint()
            logger.warning("GLM53_SYNTHETIC_WEIGHTS=1: random weights, the text is noise (plumbing checks only)")
        else:
            root = os.environ.get("MODEL_WEIGHTS_DIR")
            name = getattr(hf_config, "_name_or_path", None)
            if not root and name and Path(name).is_dir():
                root = name
            checkpoint = Glm53Checkpoint(checkpoint_root(root))
        t0 = time.time()
        max_seq = -(-max_seq_len // chunk) * chunk
        model = TtGlm53Model(mesh_device, checkpoint, max_seq=max_seq, mtp=MTP, dflash_shape=w1_shape(max_batch_size))
        dflash = None
        if DFLASH2:  # the drafter's cache: the model's and one block more (the warm-up prefills the whole cache)
            from models.demos.blackhole.glm53_flash.ttnn.dflash2 import TtDFlash2, load_weights

            k = dflash2_k(max_batch_size)
            cfg, w, mask = load_weights(os.environ.get("GLM53_DFLASH2_DRAFTER"))
            drafter = TtDFlash2(
                mesh_device, cfg, w, mask, users=max_batch_size // ROWS, max_ctx=max_seq + 8, block=4 if k <= 3 else 8,
                head_weight=model.lm_head.mesh_weight, embedding=model.embed,
            )  # fmt: skip
            dflash = (drafter, k)
            logger.info(f"DFlash2: K = {k}, the drafter's block {drafter.block}")
        entries = PREFIX_SETTING if PREFIX and max_seq > CHUNK_GRID else 0
        if PREFIX and not entries:
            logger.info(f"prefix store off: a cache of {max_seq} positions (none past {CHUNK_GRID} to keep)")
        gen = Glm53Generator(
            model, max_batch_size // ROWS, chunk=chunks, spec=MTP, dflash=dflash, prefix_entries=entries
        )
        if PREFIX and entries:  # (the store's size: "auto" from the free DRAM, else GLM53_PREFIX_ENTRIES)
            store = getattr(gen, "prefix", None)
            note = getattr(gen, "prefix_note", None) or (
                f"prefix store: {len(store.entries)} entries, {store.bytes_a_chip() / 2**30:.2f} GiB a chip "
                f"(GLM53_PREFIX_ENTRIES={PREFIX_SETTING})"
                if store is not None
                else "prefix store off"
            )
            (logger.info if store is not None else logger.warning)(note + " (GLM53_PREFIX_ENTRIES=0: off)")
        logger.info(
            f"GLM-5.3-Flash on {ROWS}x{COLS}: {max_batch_size // ROWS} users per row, cache {gen.max_seq} rows, "
            f"prefill chunks of {chunks}, loaded in {time.time() - t0:.0f} s"
        )
        return cls(gen, max_seq_len)

    def allocate_kv_cache(self, kv_cache_shape: Any, dtype: Any, num_layers: int) -> None:
        """vLLM's paged KV is not used: the generator holds each slot's DSA caches and KDA states."""
        logger.info(f"vLLM KV cache {tuple(kv_cache_shape)} {dtype} x {num_layers} not allocated (per-slot state)")

    # -- warm-up: the prefill first, then the traces (ttnn/generator.py, `warmup`) ------------------------------------
    def warmup_model_prefill(self, kv_cache: Any = None, enable_trace: bool = False, **kwargs: Any) -> None:
        if not self.prefill_warm:
            seconds = self.gen.warmup(greedy=False, logits=False)
            self.prefill_warm = True
            logger.info(f"prefill warm-up {seconds}")

    def warmup_model_decode(self, kv_cache: Any = None, enable_trace: bool = False, **kwargs: Any) -> None:
        if not self.prefill_warm:
            self.warmup_model_prefill()
        if not self.gen.traces:
            seconds = self.gen.warmup(greedy=True, logits=True, tokens=self.gen.chunk, traced=self.prefill_traced)
            logger.info(f"prefill traces {self.prefill_traced} and decode traces captured {seconds}")
            try:  # (a chip's DRAM once everything is made: what the serving's temporaries have, past the prefix store)
                view = ttnn.get_memory_view(self.gen.mesh, ttnn.BufferType.DRAM)
                banks, gib = int(view.num_banks), 2**30
                store = getattr(self.gen, "prefix", None)
                logger.info(
                    f"DRAM a chip after the warm-up: {int(view.total_bytes_allocated_per_bank) * banks / gib:.2f} GiB "
                    f"allocated, {int(view.total_bytes_free_per_bank) * banks / gib:.2f} GiB free (largest block "
                    f"{int(view.largest_contiguous_bytes_free_per_bank) * banks / gib:.2f}); the prefix store "
                    + (f"{len(store.entries)} entries, {store.bytes_a_chip() / gib:.2f} GiB" if store else "off")
                )
            except Exception as err:  # (a stand-in generator in tests)
                logger.debug(f"no DRAM view: {err!r}")

    # -- the two forward calls -------------------------------------------------------------------------------------
    def prefill_forward(
        self,
        tokens: torch.Tensor,
        start_pos: Any,
        prompt_lens: Any,
        page_table: Any = None,
        kv_cache: Any = None,
        enable_trace: bool = False,
        empty_slots: Any = None,
        sampling_params: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        t0 = time.time()
        if self.gen.dflash_ahead == "live":  # a verify step launched ahead: a prefill now throws it away (AHEAD)
            self.gen.dflash_void()
        glm53_vllm_rows.set_ahead(False)
        starts, ends, slots = _rows(start_pos), _rows(prompt_lens), _rows(empty_slots)
        rows = tokens.shape[0]
        requests, begin = [], 0
        for i in range(rows):
            start, end = int(starts[i]), int(ends[i])
            if start != 0 and not (CHUNKED and rows == 1):  # (CHUNKED: a partway prompt comes alone, the scheduler's)
                raise ValueError(f"row {i}: prefill from position {start}; only whole prompts (from 0) are served")
            if end > self.max_seq_len:
                raise ValueError(f"row {i}: a {end}-token prompt exceeds max_model_len {self.max_seq_len}")
            slot = int(slots[i]) if slots else i
            requests.append((slot, tokens[i, :end].tolist()))
            begin = start
        self._note = {}  # (this call's prefix / seat events for the timeline)
        if self.prefix_on:  # (a hit restored, a continuation snapshotted; 0: from position 0)
            begin = self._prefix_call(requests, begin, page_table)
        elif begin:  # (a partway prompt's next chunk on its earlier chunks' seat; 0: from position 0)
            begin = self._resume(requests[0][0], begin)
        if SPREAD_ROWS and not begin:  # (a partway prompt keeps its seat: its earlier chunks' state is there)
            self._place([slot for slot, _ in requests])
            if self.prefix_on:  # (each new request's own snapshots start afresh on the seat it now has)
                for slot, _ in requests:
                    self.prefix_own.pop(self.phys[slot], None)
        self.active.update(slot for slot, _ in requests)
        requests = [(self.phys[slot], prompt) for slot, prompt in requests]
        # (AHEAD_SEATS: a seat a launched step ran as sitting out at its next position, prefilled now, keeps the idle
        # position this prefill gives it (its prompt so far, `prefill_many`): the next verify's `_idle_back` would set
        # it to 0, and a verify after that would write a partway prompt's first rows)
        self._idle_moved.difference_update(p for p, _ in requests)
        self._note["seats"] = [p for p, _ in requests]  # (the timeline's: each request's physical seat, its mesh row)
        for slot, _ in requests:  # (a new request's draws from its own base, unless it brings a seed)
            self.seed_base[slot] = self._seeds.getrandbits(31)
        waves = getattr(self.gen, "counts", {}).get("waves", 0)
        tokens_in = sum(int(e) - int(s) for s, e in zip(starts[:rows], ends[:rows]))
        more = {"begin": begin} if begin else {}
        if sampling_params is not None:  # device sampling (greedy): the ids (model.PREFILL_IDS: argmax on the device)
            _greedy_only(sampling_params, rows, "prefill")
            ids = torch.tensor(self.gen.prefill_many(requests, greedy_ids=True, **more), dtype=torch.int64)
            self.fresh.update(slot for slot, _ in requests)  # (MTP: their first drafts in self.gen.first)
            self._publish()
            ahead = self._prefill_launch(requests, ids, begin)
            _record("prefill", t0, rows=rows, tokens=tokens_in, begin=begin, waves=self._waves(waves), ahead=ahead,
                    **self._note)  # fmt: skip
            return ids[:, None]
        # (a call with a partway chunk comes here: the plugin samples on the host then, and drops that chunk's token)
        logits = torch.stack(self.gen.prefill_many(requests, **more))[:, None, :]  # [B, 1, V] fp32
        self.fresh.update(slot for slot, _ in requests)  # (MTP: their first drafts in self.gen.first)
        for slot, _ in requests:  # (AHEAD_SEATS: a first token drawn on the host, not known here)
            self._fresh_first.pop(slot, None)
        self._publish()
        ahead = False
        if not begin and self._host_first_known(page_table, slots, rows):  # (AHEAD_SEATS: the plugin's argmax)
            ids = logits[:, 0, :].argmax(-1)  # (the first index of the maximum, as vLLM's greedy pick)
            ahead = self._prefill_launch(requests, ids, begin)
        _record("prefill", t0, rows=rows, tokens=tokens_in, begin=begin, waves=self._waves(waves), ahead=ahead,
                **self._note)  # fmt: skip
        return logits

    def _host_first_known(self, page_table: Any, slots: list, rows: int) -> bool:
        """AHEAD_SEATS with the prefill sampled on the host (sample_on_device_mode "decode_only", as served with
        SAMPLING): whether every request of this call draws its first token as the logits' argmax, as the scheduler
        published them (`glm53_vllm_rows.first_argmax`, by first KV block: the page table's first entry, or the vLLM
        slot without page tables), so that `_prefill_launch` may launch the next verify on those first ids."""
        if os.environ.get(AHEAD_SEATS, "0") != "1" or os.environ.get(AHEAD_HOST_FIRST, "0") != "1" or not self.gen.dk:
            return False
        known = glm53_vllm_rows.first_argmax()
        if not known:
            return False
        table = page_table if isinstance(page_table, torch.Tensor) and page_table.numel() else None
        keys = (
            table.reshape(table.shape[0], -1)[:rows, 0].tolist() if table is not None else (slots or list(range(rows)))
        )
        return len(keys) == rows and all(int(k) in known for k in keys)

    def _prefill_launch(self, requests: list, ids: torch.Tensor, begin: int) -> bool:
        """AHEAD_SEATS at a prefill's end (its requests (physical seat, prompt) and their first ids): the next verify
        launched now when the scheduler says it comes next (its admission after a prefill step) and the last verify
        foresaw the other seats (`_seat_plan`); the seats prefilled since their last verify as the plugin sends them
        (`_given_inputs`).  Whether it was launched (in "dry": whether it would be)."""
        if os.environ.get(AHEAD_SEATS, "0") != "1" or not self.gen.dk:
            return False
        for (p, prompt), first in zip(requests, ids.view(-1).tolist()):
            self._fresh_first[p] = (int(first), len(prompt))  # (its first token's position: the prompt's end)
        mode = os.environ.get(AHEAD, "0")
        if mode not in ("1", "dry") or begin or self._seat_plan is None or glm53_vllm_rows.admission() is not False:
            return False
        if any(p not in self._fresh_first for p in self.fresh):
            return False
        inputs = self._given_inputs(self._seat_plan, {p: self._fresh_first[p] for p in self.fresh}, self._cut_plan)
        if inputs is None:
            return False
        launched = self.gen.dflash_launch_given(inputs) if mode == "1" else True
        glm53_vllm_rows.set_ahead(launched)
        return bool(launched)

    def _waves(self, before: int) -> int:
        """The prefill waves the generator ran since `before` (its counts["waves"]; the rounds over the rows apart)."""
        return int(getattr(self.gen, "counts", {}).get("waves", 0)) - int(before)

    def _place(self, new: list) -> None:
        """SPREAD_ROWS, before a prefill: its requests' vLLM slots `new` onto free physical slots with as few on one
        mesh row as there can be (distinct rows while there are enough rows with a free slot: one wave), by swapping
        physical slots between free vLLM slots only: no request in use moves, and the generator's state is per
        physical slot (the KDA / DSA / drafter's state, `fresh`, `first`), as before.  A request already on a chosen
        row keeps its slot."""
        users = self.gen.slots // ROWS
        fresh_new = set(new)
        busy = self.active - fresh_new
        by_row: dict = {}  # mesh row -> its free vLLM slots (the new ones' own included)
        for j in range(self.gen.slots):
            if j not in busy:
                by_row.setdefault(self.phys[j] // users, []).append(j)
        # the rows to fill, in rounds: every row with a free slot, then those with two, ..: at most ceil(m / rows) a row
        caps = {r: len(v) for r, v in by_row.items()}
        want: list = []
        level = 0
        while len(want) < len(new):
            level += 1
            for r in sorted(caps, key=lambda r: (-caps[r], r)):
                if caps[r] >= level and len(want) < len(new):
                    want.append(r)
        move = []
        for s in new:  # a request already on a wanted row keeps its slot
            r = self.phys[s] // users
            if r in want:
                want.remove(r)
            else:
                move.append(s)
        for s, r in zip(move, want):
            j = next(j for j in by_row[r] if j not in fresh_new)  # (a free slot of row r: there is one by its cap)
            old = self.phys[s] // users
            self.phys[s], self.phys[j] = self.phys[j], self.phys[s]
            by_row[r].remove(j)
            by_row[old].append(j)

    def _prefix_call(self, requests: list, begin: int, page_table: Any) -> int:
        """PREFIX, before a prefill call's requests (vLLM slot, the prompt so far) go to the generator -> the position
        the call goes on from.  From 0: each request's seat starts its own snapshots afresh.  From `begin` (one request,
        the scheduler's): a hit (`glm53_vllm_rows.take_hit` by the page table's first block) is restored into a free seat
        of its entry's row; else the next chunk of the seat's prompt, whose state at `begin` (a multiple of CHUNK_GRID) is
        snapshotted first.  0 when neither holds (the entry gone, no free seat on its row, the seat's state not the
        prompt's): the prompt so far from position 0 instead."""
        from models.demos.blackhole.glm53_flash.ttnn.prefix_keys import prefix_keys

        gen = self.gen
        if not begin:
            for slot, _ in requests:
                self.prefix_own.pop(self.phys[slot], None)
            return 0
        ((slot, prompt),) = requests
        table = page_table if isinstance(page_table, torch.Tensor) else None
        first = int(table.reshape(table.shape[0], -1)[0, 0]) if table is not None and table.numel() else None
        hit = glm53_vllm_rows.take_hit(first)
        if hit is not None and prefix_keys(prompt[: hit[0]])[-1:] != [hit[1]]:  # (published for another prompt)
            logger.warning(f"slot {slot}: the prefix hit at {hit[0]} is not this prompt's; ignored")
            hit = None
        if hit is None:  # the next chunk of this seat's prompt (on its earlier chunks' seat)
            if not self._resume(slot, begin):
                self.prefix_counts["misses"] += 1
                return 0
            p = self.phys[slot]
            if begin % CHUNK_GRID == 0:
                keys = prefix_keys(prompt[:begin])
                t0 = time.time()
                try:
                    self.prefix_own[p] = gen.prefix_snapshot(p, begin, keys, self.prefix_own.get(p))
                    self.prefix_counts["snapshots"] += 1
                    self._note["snapshot_ms"] = round((time.time() - t0) * 1e3, 2)
                except RuntimeError as err:  # (every entry held)
                    logger.warning(f"no prefix snapshot at {begin}: {err}")
            return begin
        length, key = hit
        entries = gen.prefix.entries
        entry = next((i for i, e in enumerate(entries) if e.key == key and e.length == length), None)
        if length != begin or entry is None or not self._prefix_seat(slot, entries[entry].row):
            why = "gone" if entry is None else ("no free seat on its row" if length == begin else f"not at {begin}")
            logger.warning(f"slot {slot}: the prefix hit at {length}: {why}; its prompt from 0")
            self.prefix_counts["misses"] += 1
            self.prefix_own.pop(self.phys[slot], None)
            return 0
        p = self.phys[slot]
        t0 = time.time()
        gen.prefix_restore(entry, p)
        self._note.update(hit=length, restore_ms=round((time.time() - t0) * 1e3, 2))
        self.prefix_own.pop(p, None)  # (its own snapshots go to a new entry: this one stays for others)
        self.prefix_counts["hits"] += 1
        return begin

    def _resume(self, slot: int, begin: int) -> int:
        """CHUNKED, a partway prompt's next chunk from `begin` in vLLM slot `slot` (no hit) -> `begin` when it goes on
        from its earlier chunks' state, else 0 (the prompt so far from position 0: the same values, a whole prefill's
        time).  That state is on the physical seat its last chunk ran on (`gen.prefill_end`: no other prefill since,
        Glm53Scheduler admitting none while a prompt is partway): the rows' wave states with their roles (the seat's row
        against the others), the seat's DSA rows and drafter context.  The plugin picks each prefill's state slot afresh
        (its `_alloc_prefill_state_slots`: for a call of one request the lowest slot no other request holds), so the
        prompt comes back in another vLLM slot when a request with a lower slot ended between its chunks (e.g. slot 6,
        then 5).  Its earlier vLLM slot is then free (the plugin moved the request off it), and so is `slot`'s seat:
        the two vLLM slots' physical seats are swapped (`_prefix_seat`'s way), and the chunk runs on its own seat."""
        end = getattr(self.gen, "prefill_end", None)
        if end is None or end[1] != begin:
            logger.warning(f"slot {slot}: a chunk from {begin}, the last prefill ended at {end}: from 0")
            self._note["from0"] = begin
            return 0
        seat = end[0]
        if self.phys[slot] != seat:
            j = self.phys.index(seat)
            self.phys[slot], self.phys[j] = self.phys[j], self.phys[slot]
            self.active.discard(j)  # (the request's earlier vLLM slot: free now)
            self.seat_moves += 1
            self._note["kept_seat"] = seat
            logger.info(f"slot {slot}: the chunk from {begin} on its seat {seat} (its earlier vLLM slot {j})")
        return begin

    def _prefix_seat(self, slot: int, row: int) -> bool:
        """PREFIX: vLLM slot `slot` (free before this call) onto a physical seat of mesh row `row` (a swap with
        another free vLLM slot's, as `_place`) -> False when the row has none free."""
        users = self.gen.slots // ROWS
        if self.phys[slot] // users == row:
            return True
        busy = self.active - {slot}
        j = next(
            (j for j in range(self.gen.slots) if j != slot and j not in busy and self.phys[j] // users == row), None
        )
        if j is None:
            return False
        self.phys[slot], self.phys[j] = self.phys[j], self.phys[slot]
        return True

    def _seen(self, starts: list, rows: int) -> None:
        """A decode / verify call's rows: the vLLM slots in use are those it ran (start_pos >= 0)."""
        self.active = {i for i in range(rows) if int(starts[i]) >= 0}
        self._publish()

    def _publish(self) -> None:
        """SPREAD_ROWS: the mesh rows with a free physical slot, for the scheduler (`glm53_vllm_rows`); PREFIX: the
        prefix store's index and each mesh row's free seats."""
        if self.prefix_on:
            users = self.gen.slots // ROWS
            counts = [0] * ROWS
            for j in self.active:
                counts[self.phys[j] // users] += 1
            index = {e.key: (e.length, e.row) for e in self.gen.prefix.entries if e.length and e.key}
            glm53_vllm_rows.publish_prefix(index, [users - c for c in counts])
        if not SPREAD_ROWS:
            return
        users = self.gen.slots // ROWS
        counts = [0] * ROWS
        for j in self.active:
            counts[self.phys[j] // users] += 1
        glm53_vllm_rows.publish(sum(1 for r in range(ROWS) if counts[r] < users))

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: Any = None,
        kv_cache: Any = None,
        enable_trace: bool = True,
        read_from_device: bool = True,
        slot_remap: Any = None,
        sampling_params: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        t0 = time.time()
        if slot_remap is not None:
            remap = _rows(slot_remap)
            self.phys = [self.phys[int(r)] for r in remap] + self.phys[len(remap) :]
        self._page_table = page_table  # (GLM53_VLLM_AHEAD: each row's request, by its first KV block)
        if kwargs.get("spec_mode") is not None:
            return self._verify(
                tokens,
                start_pos,
                kwargs["spec_mode"],
                kwargs.get("num_valid_drafts"),
                kwargs.get("accepted_counts"),
                t0,
                sampling_params,
            )
        if self.gen.dflash_ahead == "live":  # (a plain decode after a verify launched ahead: thrown away)
            self.gen.dflash_void()
        rows = tokens.shape[0]
        greedy = sampling_params is not None
        if greedy:
            _greedy_only(sampling_params, rows, "decode")
        n = self.gen.slots
        where = self.phys[:rows]
        # (host lists, not torch scalar indexing: ~1 ms of the step at 32 rows)
        tok, starts = tokens[:, 0].tolist(), _rows(start_pos.view(rows, -1)[:, 0])
        self._seen(starts, rows)
        ids, positions = [0] * n, [-1] * n
        for i, p in enumerate(where):
            if starts[i] >= 0:
                ids[p], positions[p] = tok[i], starts[i]
        ids, positions = torch.tensor(ids, dtype=torch.int64), torch.tensor(positions, dtype=torch.int64)
        t1 = time.time()
        out = self.gen.decode(ids, positions, greedy=greedy)
        if TIMELINE:
            write, run = self.gen.verify_seconds
            _record("decode", t0, rows=rows, active=int((positions >= 0).sum()), prep=t1 - t0, write=write, run=run)
        index = torch.tensor(where)
        if greedy:
            return out[index].view(rows, 1)
        return out[index].view(rows, 1, -1)

    # -- MTP self-speculative decode -------------------------------------------------------------------------------
    def _verify(self, tokens, start_pos, spec_mode, num_valid, accepted, t0: float, sampling_params=None) -> Any:
        """The plugin's verify call: tokens, start_pos [B, 2] (each row's committed token and its draft) -> its
        VerifyOutput, the greedy ids after both [B, 2]; the next drafts kept for `propose_draft_tokens`."""
        from vllm_tt_plugin.spec_decode import VerifyOutput

        if self.gen.dk:
            self._seat_plan = None  # (AHEAD_SEATS: set again by a verify that foresees the next one)
            self._cut_plan = {}
            if SAMPLING:
                from vllm_tt_plugin.spec_decode import ACCEPT_MODE_LOGITS

                if spec_mode == ACCEPT_MODE_LOGITS:
                    logits = self._dflash_logits(tokens, start_pos, num_valid, accepted, t0)
                    return VerifyOutput(spec_mode=spec_mode, logits=logits)
                self._write_sampling(sampling_params, tokens.shape[0])
            return self._dflash_verify(tokens, start_pos, spec_mode, num_valid, accepted, t0)
        rows = tokens.shape[0]
        vocab = self.gen.model.config.vocab_size
        valid, counts = _rows(num_valid), _rows(accepted)
        # (host lists, not torch scalar indexing: 1.9 ms of the step at 32 rows)
        tok, starts = tokens.tolist(), _rows(start_pos[:, 0])
        self._seen(starts, rows)
        n = self.gen.slots
        ids, positions, keep = [[0, 0] for _ in range(n)], [[-1, -1] for _ in range(n)], [1] * n
        for i in range(rows):
            start = starts[i]
            if start < 0:
                continue
            p = self.phys[i]
            t, draft = tok[i]
            if not (i < len(valid) and valid[i] > 0) or not 0 <= draft < vocab:
                draft = 0  # (a padded column's placeholder)
            ids[p], positions[p] = [t, draft], [start, start + 1]
            # the state after the last draft holds where it was accepted, or after a prefill; else the checkpoint
            keep[p] = 1 if p in self.fresh or (i < len(counts) and counts[i] >= 2) else 0
            self.fresh.discard(p)
        ids, positions = torch.tensor(ids, dtype=torch.int64), torch.tensor(positions, dtype=torch.int64)
        t1 = time.time()
        out, self.drafts = self.gen.verify(ids, positions, torch.tensor(keep, dtype=torch.int64))
        if TIMELINE:
            write, run = self.gen.verify_seconds
            _record(
                "verify",
                t0,
                rows=rows,
                active=int((positions[:, 0] >= 0).sum()),
                prep=t1 - t0,  # the adapter's own host work before the generator
                write=write,  # the generator's host-to-device writes
                run=run,  # the trace's replay and the ids read back
            )
        index = torch.tensor(self.phys[:rows])
        return VerifyOutput(spec_mode=spec_mode, argmax_ids=out[index].to(torch.int32))

    def _dflash_verify(self, tokens, start_pos, spec_mode, num_valid, accepted, t0: float) -> Any:
        """The plugin's verify call with DFlash2 (K drafts): tokens, start_pos [B, K + 1] (each row's committed token
        and its drafts) -> its VerifyOutput, greedy ids [B, K + 1] rebuilt from the device's acceptance (the drafts it
        accepted, the bonus, the bonus again: the plugin's comparison stops at the first mismatch); the step's next
        drafts kept for `propose_draft_tokens`."""
        from vllm_tt_plugin.spec_decode import VerifyOutput

        if os.environ.get(VECTOR, "0") == "1":
            return VerifyOutput(
                spec_mode=spec_mode, argmax_ids=self._dflash_verify_vec(tokens, start_pos, num_valid, accepted, t0)
            )
        k = self.gen.dk
        rows = tokens.shape[0]
        vocab = self.gen.model.config.vocab_size
        valid, counts = _rows(num_valid), _rows(accepted)
        tok, starts = tokens.tolist(), _rows(start_pos[:, 0])
        self._seen(starts, rows)
        n = self.gen.slots
        ids, positions, keep = [[0] * (k + 1) for _ in range(n)], [[-1] * (k + 1) for _ in range(n)], [k] * n
        usable = [0] * rows  # each row's valid drafts
        for i in range(rows):
            start = starts[i]
            if start < 0:
                continue
            p = self.phys[i]
            v = min(int(valid[i]), k) if i < len(valid) else 0
            drafts = [d if j < v and 0 <= d < vocab else 0 for j, d in enumerate(tok[i][1 : k + 1])]  # (padding: 0)
            ids[p], positions[p], usable[i] = [tok[i][0]] + drafts, [start + j for j in range(k + 1)], v
            # the state after the drafts holds where they were accepted (keep: the last step's accepted drafts), or
            # after a prefill (all of it)
            keep[p] = k if p in self.fresh else (max(int(counts[i]) - 1, 0) if i < len(counts) else 0)
            self.fresh.discard(p)
        t1 = time.time()
        self._idle_zero({self.phys[i] for i in range(rows) if starts[i] >= 0})
        got, bonus, self.drafts = self.gen.dflash_verify(
            torch.tensor(ids, dtype=torch.int64), torch.tensor(positions, dtype=torch.int64), torch.tensor(keep)
        )
        self._idle_back()
        out = torch.zeros(rows, k + 1, dtype=torch.int32)
        for i in range(rows):
            if starts[i] < 0:
                continue
            p = self.phys[i]
            a = min(int(got[p]), usable[i])  # (the plugin counts the valid drafts only)
            nxt = int(bonus[p]) if a == int(got[p]) else ids[p][a + 1]  # (cut short: the greedy id there is the draft)
            out[i, :a] = torch.tensor(ids[p][1 : a + 1], dtype=torch.int32)
            out[i, a:] = nxt
        if TIMELINE:
            write, run = self.gen.verify_seconds
            _record(
                "verify", t0, rows=rows, active=sum(s >= 0 for s in starts[:rows]), prep=t1 - t0, write=write, run=run
            )
        return VerifyOutput(spec_mode=spec_mode, argmax_ids=out)

    def _dflash_host_inputs(self, tokens, start_pos, num_valid, accepted) -> tuple:
        """A DFlash2 verify call's rows -> the generator's inputs by physical slot (ids, positions [slots, K + 1], keep
        [slots]), each row's valid drafts [B] and its start [B] (-1: no request)."""
        k, n = self.gen.dk, self.gen.slots
        rows = tokens.shape[0]
        vocab = self.gen.model.config.vocab_size
        valid, counts = _rows(num_valid), _rows(accepted)
        tok, start_l = tokens.tolist(), _rows(start_pos[:, 0])
        self._seen(start_l, rows)
        ids_l, pos_l, keep_l = [[0] * (k + 1) for _ in range(n)], [[-1] * (k + 1) for _ in range(n)], [k] * n
        usable = [0] * rows  # each row's valid drafts
        for i in range(rows):
            start = start_l[i]
            if start < 0:
                continue
            p = self.phys[i]
            v = min(int(valid[i]), k) if i < len(valid) else 0
            drafts = [d if j < v and 0 <= d < vocab else 0 for j, d in enumerate(tok[i][1 : k + 1])]  # (padding: 0)
            ids_l[p], pos_l[p], usable[i] = [tok[i][0]] + drafts, [start + j for j in range(k + 1)], v
            keep_l[p] = k if p in self.fresh else (max(int(counts[i]) - 1, 0) if i < len(counts) else 0)
            self.fresh.discard(p)
        return (
            torch.tensor(ids_l, dtype=torch.int64),
            torch.tensor(pos_l, dtype=torch.int64),
            torch.tensor(keep_l, dtype=torch.int64),
            torch.tensor(usable, dtype=torch.int64),
            torch.tensor(start_l[:rows], dtype=torch.int64),
        )

    def _dflash_logits(self, tokens, start_pos, num_valid, accepted, t0: float) -> torch.Tensor:
        """SAMPLING: the plugin's verify call in the "logits" mode (a step with a row the device does not sample: the
        plugin accepts on the host) -> the target's logits [B, K + 1, V] fp32, column j the distribution after the
        row's token j (its token, then its drafts); the step's next drafts kept for `propose_draft_tokens` (from the
        device's own acceptance: where the host's differs, the next step's drafts miss once).  Never launched ahead."""
        rows = tokens.shape[0]
        ids, positions, keep, _, starts = self._dflash_host_inputs(tokens, start_pos, num_valid, accepted)
        t1 = time.time()
        self._idle_zero({self.phys[i] for i in range(rows) if int(starts[i]) >= 0})
        _, _, self.drafts = self.gen.dflash_verify(ids, positions, keep, logits=True)
        self._idle_back()
        glm53_vllm_rows.set_ahead(False)
        out = self.gen.dflash_logits[torch.tensor(self.phys[:rows], dtype=torch.int64)]
        if TIMELINE:
            write, run = self.gen.verify_seconds
            _record("verify_logits", t0, rows=rows, active=int((starts >= 0).sum()), prep=t1 - t0, write=write, run=run)
        return out

    def _write_sampling(self, params: Any, rows: int) -> None:
        """SAMPLING: each row's sampling parameters (the plugin's, on a device-sampled step; None: every row greedy)
        onto its physical slot, written to the device when they changed (`Glm53Generator.dflash_sampling`)."""
        values = slot_sampling(params, self.phys[:rows], self.gen.slots, self.seed_base)
        key = tuple(tuple(v.tolist()) for v in values)
        if key != self._sampling_written:
            self.gen.dflash_sampling(*values)
            self._sampling_written = key

    def _dflash_verify_vec(self, tokens, start_pos, num_valid, accepted, t0: float) -> torch.Tensor:
        """`_dflash_verify` with its greedy ids made by whole-tensor ops (VECTOR; the generator's inputs by the loop over
        the rows, the faster for 32 seats): the same generator inputs, the same greedy ids [B, K + 1] int32."""
        k = self.gen.dk
        rows = tokens.shape[0]
        ids, positions, keep, v, starts = self._dflash_host_inputs(tokens, start_pos, num_valid, accepted)
        act = starts >= 0
        phys = torch.tensor(self.phys[:rows], dtype=torch.int64)
        t1 = time.time()
        self._idle_zero(set(phys[act].tolist()))
        got, bonus, self.drafts = self.gen.dflash_verify(ids, positions, keep)
        self._idle_back()
        g, b = got[phys], bonus[phys]
        a = torch.minimum(g, v)  # (the plugin counts the valid drafts only; 0 on the rows without a request)
        mine = ids[phys]  # [rows, K + 1]
        cut = mine.gather(1, (a + 1).clamp(max=k)[:, None]).squeeze(1)  # (cut short: the greedy id there is the draft)
        nxt = torch.where(a == g, b, cut)
        shifted = torch.cat([mine[:, 1:], torch.zeros(rows, 1, dtype=torch.int64)], dim=1)
        at = torch.arange(k + 1, dtype=torch.int64)[None, :]
        out = torch.where(at < a[:, None], shifted, nxt[:, None])
        out = torch.where(act[:, None], out, 0).to(torch.int32)
        t2 = time.time()
        mode = os.environ.get(AHEAD, "0")
        launched = False
        if mode in ("1", "dry"):
            want = self._ahead_ok(act, a, g, starts, k, out)
            given = None
            if not want and os.environ.get(AHEAD_SEATS, "0") == "1":  # (a request ends here: its seat sits out)
                given = self._ends_launch(act)
            if want and mode == "1":
                launched = self.gen.dflash_launch_ahead()
            elif given is not None and mode == "1":
                launched = self.gen.dflash_launch_given(given)
            glm53_vllm_rows.set_ahead((want or given is not None) if mode == "dry" else launched)
        if TIMELINE:
            write, run = self.gen.verify_seconds
            _record(
                "verify", t0, rows=rows, active=int(act.sum()), prep=t1 - t0, write=write, run=run,
                ahead=self.gen.dflash_writes.get("ahead", 0), launched=launched, launch=time.time() - t2,
            )  # fmt: skip
        return out

    def _ahead_ok(self, act, a, g, starts, k: int, out) -> bool:
        """Whether the next verify step may be launched now (AHEAD): some seat runs, every running seat takes the
        device's acceptance (a == n: its next inputs are those the device made), the cache's end is not near (the
        next step's positions and its drafter block within both caches), the scheduler said no request could be
        admitted (a prefill would throw the step away), and no request ends at this step (`_ends`: its seat would be
        empty in the next one)."""
        admission = glm53_vllm_rows.admission()
        if admission is None and not self._no_scheduler_told:  # (vLLM's scheduler not Glm53Scheduler: no launch)
            self._no_scheduler_told = True
            logger.warning(
                f"{AHEAD}: no admission published (--scheduler-cls models.demos.blackhole.glm53_flash.tools."
                "glm53_vllm_scheduler.Glm53Scheduler): no verify step is launched ahead"
            )
        cut = os.environ.get(AHEAD_SEATS, "0") == "1" and os.environ.get(AHEAD_EDGE, "cache") == "cut"
        if os.environ.get(AHEAD_SEATS, "0") == "1":  # (what the next verify gets, for a launch after a prefill)
            if cut:  # (the seats ending here sit out next: their acceptance and positions do not matter, e.g. a
                # request cut at max_model_len, whose acceptance is the plugin's count of its valid drafts)
                rows = self._ending_rows(act, a, out) if bool(act.any()) else None
                if rows is not None and not self._ahead_core(act, a, g, starts, k, skip=rows):
                    rows = None
            else:
                core = bool(act.any()) and self._ahead_core(act, a, g, starts, k)
                rows = self._ending_rows(act, a, out) if core else None
            self._seat_plan = None if rows is None else {self.phys[i] for i in rows}
            self._cut_plan = self._edge_cuts(act, a, starts, k, rows) if cut and rows is not None else {}
        if not bool(act.any()) or admission is not False:
            return False
        if not self._ahead_core(act, a, g, starts, k):
            return False
        if self._cut_plan:  # (the device's own next inputs are not vLLM's: launched on the host's, `_ends_launch`)
            return False
        return not self._ends(act, a, out)

    def _ahead_core(self, act, a, g, starts, k: int, skip=()) -> bool:
        """`_ahead_ok`'s checks of the seats themselves: each running seat takes the device's acceptance and the cache's
        end is not near (AHEAD_EDGE); the rows in `skip` (those ending here, AHEAD_EDGE "cut") left out."""
        if len(skip):
            act = act.clone()
            act[list(skip)] = False
            if not bool(act.any()):
                return True
        if bool((a[act] != g[act]).any()):
            return False
        nxt = int((starts[act] + a[act]).max()) + 1  # (the next step's first position, the furthest seat's)
        edge = os.environ.get(AHEAD_EDGE, "cache")
        if edge == "cut" and os.environ.get(AHEAD_SEATS, "0") == "1":  # (past vLLM's edge: `_edge_cuts`)
            return True
        if edge in ("vllm", "cut"):  # (vLLM's draft cut at max_model_len, AHEAD_EDGE)
            return nxt + k + 2 <= self.max_seq_len and nxt + k < (self.gen.max_seq or nxt + k + 1)
        top = nxt + 2 * (k + 1) + (self.gen.drafter.block if self.gen.drafter else 0)
        limit = min(self.gen.max_seq or top + 1, self.gen.drafter.max_ctx if self.gen.drafter else top + 1)
        return top < limit

    def _ends_launch(self, act) -> tuple | None:
        """AHEAD_SEATS at a verify: the inputs to launch the next step on when requests end here and nothing else stops
        it (the scheduler said no request could be admitted, every other check passed, some seat runs on): the
        device's next inputs with the ending seats sitting out; else None."""
        ending = self._seat_plan
        if ending is None or not (ending or self._cut_plan) or glm53_vllm_rows.admission() is not False:
            return None
        if len({self.phys[i] for i in range(act.shape[0]) if bool(act[i])} - ending) == 0:
            return None
        return self._given_inputs(ending, {}, self._cut_plan)

    def _edge_cuts(self, act, a, starts, k: int, ending_rows: list) -> dict:
        """AHEAD_EDGE "cut": the running seats (physical) that go on past this step and whose next drafts vLLM cuts at
        max_model_len -> their valid drafts min(K, max_model_len - p - 2), p the seat's next position (vLLM 0.31's
        scheduler: a step's new tokens <= max_model_len - computed - 1); the seats ending here sit out instead."""
        nxt = (starts + a + 1).tolist()
        on, end = act.tolist(), set(ending_rows)
        cuts = {}
        for i in range(act.shape[0]):
            if on[i] and i not in end and nxt[i] + k + 2 > self.max_seq_len:
                cuts[self.phys[i]] = max(0, self.max_seq_len - nxt[i] - 2)
        return cuts

    def _given_inputs(self, ending: set, fresh: dict, cut: dict | None = None) -> tuple | None:
        """The next verify's inputs as `_dflash_inputs` makes them (tokens, positions, keep, live by mesh row), from the
        device's own next inputs (`Glm53Generator.dflash_next`, the last verify's): each seat in `ending` (physical)
        sitting out, each in `fresh` (physical -> (its first id, its prompt's length)) taking its first verify (no
        drafts, keep K), each in `cut` (physical -> its valid drafts, AHEAD_EDGE "cut") with its drafts past those
        zero (as the adapter fills a row's invalid drafts).  None when there is nothing to start from."""
        base = self.gen.dflash_next
        if base is None:
            return None
        tokens, positions, keep, live = (t.clone() for t in base)
        u, n, dk = self.gen.users, self.gen.dk + 1, self.gen.dk
        moved: dict = {}
        top = (self.gen.max_seq - 1) if self.gen.max_seq is not None else None
        for p, v in (cut or {}).items():
            if p in ending or p in fresh:
                continue
            r, j = divmod(p, u)
            for t in range(1 + v, n):  # (the drafts vLLM does not send)
                tokens[r, t * u + j] = 0
        if top is not None and cut:  # (as `_dflash_inputs`: a request's last positions past the cache on its last row)
            positions = positions.clamp(max=top)
        for p in set(ending) | set(fresh):
            r, j = divmod(p, u)
            cols = [t * u + j for t in range(n)]
            if p in fresh:
                first, start = fresh[p]
                tokens[r, cols] = 0
                tokens[r, j] = int(first)
                pos = torch.arange(start, start + n, dtype=positions.dtype)
                positions[r, cols] = pos.clamp(max=top) if top is not None else pos
                live[r, cols] = True
            else:  # (sitting out at its next position: rows past its tokens, which a rerun writes if it goes on)
                tokens[r, cols] = 0
                idle = int(positions[r, j])
                positions[r, cols] = min(idle, top) if top is not None else idle
                live[r, cols] = False
                moved[p] = int(positions[r, j])
            keep[r, j] = dk
        if (
            self.gen.drafter is not None
            and int(positions.max()) + 1 + self.gen.drafter.block > self.gen.drafter.max_ctx
        ):
            return None
        for p, pos in moved.items():  # (the next call's `_dflash_inputs` then makes the same inputs: `_idle_back`)
            self.gen.idle_pos[p] = pos
        self._idle_moved |= set(moved)
        return tokens, positions, keep, live

    def _idle_zero(self, running: set) -> None:
        """IDLE_ZERO, before a verify whose running seats (physical) are `running`: the other seats' idle positions to 0
        where no request could be admitted (no partway prompt), except the seats a launched step moved (`_idle_moved`).
        """
        if os.environ.get(IDLE_ZERO, "1") != "1" or glm53_vllm_rows.admission() is not False:
            return
        idle = self.gen.idle_pos
        keep = set(running) | self._idle_moved
        zero = [p for p in range(self.gen.slots) if p not in keep and int(idle[p]) != 0]
        if zero:
            idle[zero] = 0
            writes = self.gen.dflash_writes
            writes["idle_zeroed"] = writes.get("idle_zeroed", 0) + len(zero)

    def _idle_back(self) -> None:
        """AHEAD_SEATS, after a verify: the seats a launched step ran as sitting out at their next position back to
        their idle position 0 (`Glm53Generator.idle_pos`; a seat running again was set to 0 by the verify itself), so
        that the steps after it sit out where the device's next inputs put them."""
        for p in self._idle_moved:  # (a seat running again is at 0 already)
            self.gen.idle_pos[p] = 0
        self._idle_moved = set()

    def _ends(self, act, a, out) -> bool:
        """Whether a running request ends at this step as vLLM's `check_stop` will see it, from the scheduler's budgets
        (`glm53_vllm_rows.publish_budgets`: its tokens left, its stop ids) and the a + 1 tokens the step gives it
        (`out`); a row whose request has no budget counts as ending.  No budgets published: none foreseen."""
        rows = self._ending_rows(act, a, out)
        return rows is None or bool(rows)

    def _ending_rows(self, act, a, out) -> list | None:
        """The rows whose request ends at this step (`_ends`), or None when a running row's request has no budget (its
        end not foreseen)."""
        budgets = glm53_vllm_rows.budgets()
        if budgets is None:
            return []
        table = getattr(self, "_page_table", None)
        rows = act.shape[0]
        keys = table[:rows, 0].tolist() if isinstance(table, torch.Tensor) else list(range(rows))
        on, gave = act.tolist(), (a + 1).tolist()
        ending = []
        for i in range(rows):
            if not on[i]:
                continue
            entry = budgets.get(int(keys[i]))
            if entry is None:
                return None
            left, stops = entry
            if gave[i] >= left or (stops and not stops.isdisjoint(out[i, : gave[i]].tolist())):
                ending.append(i)
        return ending

    def _dflash_propose_vec(self, num_drafts: int, rows: int) -> torch.Tensor:
        """`propose_draft_tokens`'s DFlash2 rows as whole-tensor ops (VECTOR): the same [B, num_drafts] int32."""
        out = torch.zeros(rows, num_drafts, dtype=torch.int32)
        phys = torch.tensor(self.phys[:rows], dtype=torch.int64)
        if self.drafts is not None:
            w = min(num_drafts, self.drafts.shape[1])
            out[:, :w] = self.drafts[phys][:, :w].to(torch.int32)
        for i, p in enumerate(self.phys[:rows]):  # (the rows just prefilled: the drafter's first drafts)
            if p in self.fresh and p in self.gen.first:
                row = self.gen.first[p][:num_drafts]
                out[i] = 0
                out[i, : len(row)] = torch.tensor(row, dtype=torch.int32)
        return out

    def propose_draft_tokens(
        self, num_drafts: int, committed_tokens: Any, committed_positions: Any, accepted_counts: Any, hidden: Any = None
    ) -> Any:
        """The plugin's propose call after a commit: each row's next draft [B, num_drafts] int32 (one real): after a
        prefill the first draft (`Glm53Generator.first`), else the last verify's (B where its draft was accepted, A
        where not)."""
        from vllm_tt_plugin.spec_decode import DraftOutput

        t0 = time.time()
        rows = committed_tokens.shape[0]
        if self.gen.dk and os.environ.get(VECTOR, "0") == "1":
            out = self._dflash_propose_vec(num_drafts, rows)
            if TIMELINE:
                _record("propose", t0, rows=rows)
            return DraftOutput(draft_token_ids=out)
        if self.gen.dk:  # DFlash2: the step's next drafts, or after a prefill the drafter's first
            out = torch.zeros(rows, num_drafts, dtype=torch.int32)
            drafts = self.drafts.tolist() if self.drafts is not None else None
            for i in range(rows):
                p = self.phys[i]
                if p in self.fresh and p in self.gen.first:
                    row = self.gen.first[p]
                elif drafts is not None:
                    row = drafts[p]
                else:
                    continue
                out[i, : min(num_drafts, len(row))] = torch.tensor(row[:num_drafts], dtype=torch.int32)
            if TIMELINE:
                _record("propose", t0, rows=rows)
            return DraftOutput(draft_token_ids=out)
        counts = _rows(accepted_counts)
        drafts = self.drafts.tolist() if self.drafts is not None else None
        first = [0] * rows
        for i in range(rows):
            p = self.phys[i]
            if p in self.fresh and p in self.gen.first:
                first[i] = self.gen.first[p]
            elif drafts is not None:
                first[i] = drafts[p][1 if i < len(counts) and counts[i] >= 2 else 0]
        out = torch.zeros(rows, num_drafts, dtype=torch.int32)
        out[:, 0] = torch.tensor(first, dtype=torch.int32)
        if TIMELINE:
            _record("propose", t0, rows=rows)
        return DraftOutput(draft_token_ids=out)

    # -- lifecycle -------------------------------------------------------------------------------------------------
    def release_persistent_capture(self) -> None:
        self.gen.release()
