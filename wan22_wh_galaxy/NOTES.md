# Running large video models on a Wormhole Galaxy — notes

> The detailed record behind [README.md](README.md). Commit hashes in this document refer to the history before the 2026-10-13 rebase onto main; the branch `hkato/wan-wh-opt` contains the same changes under new hashes.

A sequel to earlier work on a P150 (one Blackhole card), where Wan2.1-1.3B was optimized. The setting moves to
**a Wormhole Galaxy (32 chips)**, with larger models (Wan 2.2 A14B, MiniMax-H3).
This document grows by adding chapters.

- Chapter 1: Survey of prior work — read the PR that brought MiniMax-H3 up on the WH Galaxy (#56886, merged) and
  its follow-up optimization PR (#57941, not merged), and sort out the Wan 2.2 support status and the performance gap between Blackhole and Wormhole.
  Section 7 adds a model comparison of Wan 2.2 and MiniMax-H3 (release date, strengths, official benchmarks).
- Chapter 2 (section 8 onward): Our own measurements — run Wan 2.2 on a WH Galaxy and compare it with the public benchmarks. Section 9 covers the module structure and the breakdown of one block, section 10 the optimization history, section 11 notes on questions and answers, and section 12 operational lessons.

Survey date: 2026-09-29. All numbers are quoted from the documents and code in the PRs (not measured by us).
Values we computed ourselves are marked "(computed)".

---

<!-- toc -->
## Contents

- [0. Summary in 3 lines](#0-summary-in-3-lines)
- [1. Glossary (terms used in this chapter)](#1-glossary-terms-used-in-this-chapter)
- [2. Does Wan 2.2 already support the WH Galaxy?](#2-does-wan-22-already-support-the-wh-galaxy)
- [3. Details of PR #56886 "Enable MiniMax-H3 on Wormhole Galaxy"](#3-details-of-pr-56886-enable-minimax-h3-on-wormhole-galaxy)
  - [3.1 Mesh configuration per arch (① does not start)](#31-mesh-configuration-per-arch--does-not-start)
  - [3.2 Memory: FSDP and the "not coresident" setting (② does not fit in memory)](#32-memory-fsdp-and-the-not-coresident-setting--does-not-fit-in-memory)
  - [3.3 Matching the AGMM core placement to the link count (③ crashes midway, part 1)](#33-matching-the-agmm-core-placement-to-the-link-count--crashes-midway-part-1)
  - [3.4 Keeping the Ring K-block condition (③ crashes midway, part 2)](#34-keeping-the-ring-k-block-condition--crashes-midway-part-2)
  - [3.5 Making the fused MM+RS check include the core grid (③ hang + ④ silent slowdown) — the core of this PR](#35-making-the-fused-mmrs-check-include-the-core-grid--hang---silent-slowdown--the-core-of-this-pr)
  - [3.6 Keying the tuning tables on "the row count that actually flows" (④ silent slowdown)](#36-keying-the-tuning-tables-on-the-row-count-that-actually-flows--silent-slowdown)
  - [3.7 Blocking search for WH and its numerical check (④)](#37-blocking-search-for-wh-and-its-numerical-check-)
  - [3.8 Getting the weights (⑤)](#38-getting-the-weights-)
  - [3.9 Read-back verification of the weight cache (⑤)](#39-read-back-verification-of-the-weight-cache-)
  - [3.10 Bring-up lessons (taken from this PR)](#310-bring-up-lessons-taken-from-this-pr)
- [4. Performance: Blackhole vs Wormhole](#4-performance-blackhole-vs-wormhole)
  - [4.1 Hardware differences (1 Galaxy = 32 chips)](#41-hardware-differences-1-galaxy--32-chips)
  - [4.2 End-to-end comparison (H3, 768P 16:9, 50 steps = 49 forwards, warm)](#42-end-to-end-comparison-h3-768p-169-50-steps--49-forwards-warm)
  - [4.3 Per-block breakdown (15 s, per chip, Tracy)](#43-per-block-breakdown-15-s-per-chip-tracy)
  - [4.4 The WH roofline — how fast can it get?](#44-the-wh-roofline--how-fast-can-it-get)
  - [4.5 All WH conditions (#57941 README, with FSDP, 2026-09-17)](#45-all-wh-conditions-57941-readme-with-fsdp-2026-09-17)
  - [4.6 Progress in the unmerged PR #57941 "MiniMax-H3: Wormhole Galaxy optimizations"](#46-progress-in-the-unmerged-pr-57941-minimax-h3-wormhole-galaxy-optimizations)
  - [4.7 Comparing parallel layouts (WH, 15 s, 10 steps)](#47-comparing-parallel-layouts-wh-15-s-10-steps)
  - [4.8 Quality](#48-quality)
- [5. Implications for our work](#5-implications-for-our-work)
- [6. Operational notes (hangs and board recovery)](#6-operational-notes-hangs-and-board-recovery)
- [7. Model comparison: Wan 2.2 and MiniMax-H3](#7-model-comparison-wan-22-and-minimax-h3)
  - [7.1 Release timing and positioning](#71-release-timing-and-positioning)
  - [7.2 Strengths of each](#72-strengths-of-each)
  - [7.3 Official Wan 2.2 benchmarks](#73-official-wan-22-benchmarks)
- [8. Measured on our setup: Wan 2.2 T2V 480p on WH Galaxy 4×8](#8-measured-on-our-setup-wan-22-t2v-480p-on-wh-galaxy-48)
  - [8.1 Measurement conditions](#81-measurement-conditions)
  - [8.2 Results](#82-results)
  - [8.3 Comparison with benchmarks (480p, T2V-A14B, total time per video)](#83-comparison-with-benchmarks-480p-t2v-a14b-total-time-per-video)
  - [8.4 Problems hit before the first run](#84-problems-hit-before-the-first-run)
- [9. Inside Wan 2.2: module structure and per-block breakdown](#9-inside-wan-22-module-structure-and-per-block-breakdown)
  - [9.1 Module structure (T2V-A14B)](#91-module-structure-t2v-a14b)
  - [9.2 Per-block breakdown (Tracy measurement, 480p, per chip, average over 32 chips, 2026-09-30)](#92-per-block-breakdown-tracy-measurement-480p-per-chip-average-over-32-chips-2026-09-30)
  - [9.3 Takeaways (improvement candidates)](#93-takeaways-improvement-candidates)
  - [9.4 What the fused path is (terminology)](#94-what-the-fused-path-is-terminology)
  - [9.5 Current breakdown and theoretical values (as of #15d, 2026-10-02)](#95-current-breakdown-and-theoretical-values-as-of-15d-2026-10-02)
  - [9.6 Current breakdown and theoretical values (as of #30, 2026-10-05)](#96-current-breakdown-and-theoretical-values-as-of-30-2026-10-05)
  - [9.7 Current (#77) vs. baseline (2026-10-07)](#97-current-77-vs-baseline-2026-10-07)
  - [9.8 Current (#107) breakdown and theoretical values (2026-10-08)](#98-current-107-breakdown-and-theoretical-values-2026-10-08)
- [10. Optimization history (Wan 2.2 T2V 480p, WH Galaxy 4×8)](#10-optimization-history-wan-22-t2v-480p-wh-galaxy-48)
  - [#1 details: FFN2 unfused](#1-details-ffn2-unfused)
  - [#2 details: Matmul blocking for WH, and retuning the fused MM+RS](#2-details-matmul-blocking-for-wh-and-retuning-the-fused-mmrs)
  - [#3 details: ring SDPA chunk size](#3-details-ring-sdpa-chunk-size)
  - [#4 details: Automatically deciding whether to use the fused MM+RS](#4-details-automatically-deciding-whether-to-use-the-fused-mmrs)
  - [#5a details: FSDP weight prefetch (no effect)](#5a-details-fsdp-weight-prefetch-no-effect)
  - [#5b preparation: Free device memory, and the steps each of the two models handles (2026-10-01)](#5b-preparation-free-device-memory-and-the-steps-each-of-the-two-models-handles-2026-10-01)
  - [#5b details: Keeping the FFN weights resident (`transformer` side)](#5b-details-keeping-the-ffn-weights-resident-transformer-side)
  - [#6 details: Dropping fp32 accumulation in the FFN (C)](#6-details-dropping-fp32-accumulation-in-the-ffn-c)
  - [#6b details: No fp32 accumulation for the attention projections either (optional, default off)](#6b-details-no-fp32-accumulation-for-the-attention-projections-either-optional-default-off)
  - [#21 details: LoFi for the ring SDPA matmuls only (2026-10-05)](#21-details-lofi-for-the-ring-sdpa-matmuls-only-2026-10-05)
  - [#23 details: LoFi matmuls (rejected, 2026-10-05)](#23-details-lofi-matmuls-rejected-2026-10-05)
  - [#7 details: Building the softmax row sums inside DEST (failed)](#7-details-building-the-softmax-row-sums-inside-dest-failed)
  - [#10 details: Caching the cross-attention K/V](#10-details-caching-the-cross-attention-kv)
  - [#11 details: trace (no effect)](#11-details-trace-no-effect)
  - [#13 details: Time per section of one forward](#13-details-time-per-section-of-one-forward)
  - [Speeding up experiments: comparing with fewer steps (2026-10-01)](#speeding-up-experiments-comparing-with-fewer-steps-2026-10-01)
  - [#15 details: CFG batching (merging the conditional and unconditional forwards)](#15-details-cfg-batching-merging-the-conditional-and-unconditional-forwards)
  - [#16 (in progress, 2026-10-01 18:00 onward): SDPA precision settings, and folding the CFG-batched self-attention into heads](#16-in-progress-2026-10-01-1800-onward-sdpa-precision-settings-and-folding-the-cfg-batched-self-attention-into-heads)
  - [Check at 720p (D, 2026-10-01)](#check-at-720p-d-2026-10-01)
  - [Time per zone inside the SDPA kernel (2026-10-01)](#time-per-zone-inside-the-sdpa-kernel-2026-10-01)
  - [Experiment and measurement methods (summary, 2026-10-02)](#experiment-and-measurement-methods-summary-2026-10-02)
  - [Zones inside the SDPA kernel (after LoFi, 2026-10-05, #26)](#zones-inside-the-sdpa-kernel-after-lofi-2026-10-05-26)
  - [Confirming the Phase 1 bottleneck, and the outlook for a kernel rewrite (2026-10-05, #27, #28)](#confirming-the-phase-1-bottleneck-and-the-outlook-for-a-kernel-rewrite-2026-10-05-27-28)
  - [Considering fundamental speedups (2026-10-05, after #29)](#considering-fundamental-speedups-2026-10-05-after-29)
  - [#30 details: Faster GELU_TANH in FFN1 (2026-10-05)](#30-details-faster-gelu_tanh-in-ffn1-2026-10-05)
  - [SDPA rewrite design: Freezing the max at the first chunk (2026-10-05, started)](#sdpa-rewrite-design-freezing-the-max-at-the-first-chunk-2026-10-05-started)
  - [#32 details: Prototype that removes the row max (2026-10-05)](#32-details-prototype-that-removes-the-row-max-2026-10-05)
  - [#33 details: Re-analyzing why ring SDPA is slow (2026-10-05)](#33-details-re-analyzing-why-ring-sdpa-is-slow-2026-10-05)
  - [#34 details: Re-measuring the no-max upper bound with LoFi, and per-core analysis (2026-10-06)](#34-details-re-measuring-the-no-max-upper-bound-with-lofi-and-per-core-analysis-2026-10-06)
  - [#35 details: "The matmul gets slower" was a measurement artifact (2026-10-06)](#35-details-the-matmul-gets-slower-was-a-measurement-artifact-2026-10-06)
  - [#36 details: Folding the two CFG passes into heads, and dealing SDPA evenly over 64 cores (2026-10-06)](#36-details-folding-the-two-cfg-passes-into-heads-and-dealing-sdpa-evenly-over-64-cores-2026-10-06)
  - [Explainer: the K/V relay (chain) and how uneven work makes cores wait (2026-10-06)](#explainer-the-kv-relay-chain-and-how-uneven-work-makes-cores-wait-2026-10-06)
  - [#37 details: the "no max" upper bound in the real ring SDPA (2026-10-06)](#37-details-the-no-max-upper-bound-in-the-real-ring-sdpa-2026-10-06)
  - [#38 details: isolating why K/V arrive late (2026-10-06)](#38-details-isolating-why-kv-arrive-late-2026-10-06)
  - [#39 details: overlapping the K and V reads on the chain-head core (2026-10-06, rejected)](#39-details-overlapping-the-k-and-v-reads-on-the-chain-head-core-2026-10-06-rejected)
  - [#40 details: all cores read either K or V from DRAM (2026-10-06, rejected)](#40-details-all-cores-read-either-k-or-v-from-dram-2026-10-06-rejected)
  - [#41 details: trying the experimental ring SDPA op on WH 4×8 (2026-10-06, did not work)](#41-details-trying-the-experimental-ring-sdpa-op-on-wh-48-2026-10-06-did-not-work)
  - [#42-#45 details: Q interleave to reduce K/V DRAM reads (2026-10-06)](#42-45-details-q-interleave-to-reduce-kv-dram-reads-2026-10-06)
  - [#46 details: bounded exp mode, subtracting a per-row upper bound (2026-10-06, rejected)](#46-details-bounded-exp-mode-subtracting-a-per-row-upper-bound-2026-10-06-rejected)
  - [#47 details: the matmul (AGMM) is bound by data supply, not compute (2026-10-06)](#47-details-the-matmul-agmm-is-bound-by-data-supply-not-compute-2026-10-06)
  - [#48-#49 details: BFP8 weights and activations (2026-10-06)](#48-49-details-bfp8-weights-and-activations-2026-10-06)
  - [#50-#52 details: fully resident weights and splitting the text encoder (2026-10-06)](#50-52-details-fully-resident-weights-and-splitting-the-text-encoder-2026-10-06)
  - [#53-#58 details: analyzing what bounds AGMM (fused matmul and all-gather) (2026-10-06)](#53-58-details-analyzing-what-bounds-agmm-fused-matmul-and-all-gather-2026-10-06)
  - [#59-#61 details: BFP8 activations, and fixing bugs in BFP8 SDPA (2026-10-06)](#59-61-details-bfp8-activations-and-fixing-bugs-in-bfp8-sdpa-2026-10-06)
  - [#66-#68 details: fixing how the experimental switches are applied (2026-10-06)](#66-68-details-fixing-how-the-experimental-switches-are-applied-2026-10-06)
  - [#83-#85 details: the GELU epilogue of FFN1 (2026-10-07)](#83-85-details-the-gelu-epilogue-of-ffn1-2026-10-07)
  - [#86-#89 details: confirming what bounds FFN2, ring SDPA and the matmuls (2026-10-07)](#86-89-details-confirming-what-bounds-ffn2-ring-sdpa-and-the-matmuls-2026-10-07)
  - [#90 details: assessing a ring SDPA rewrite in advance (2026-10-08)](#90-details-assessing-a-ring-sdpa-rewrite-in-advance-2026-10-08)
  - [#92 details: improving the VAE decode (2026-10-08)](#92-details-improving-the-vae-decode-2026-10-08)
  - [#150 details: relaying ring SDPA K/V between chips (2026-10-10, parked)](#150-details-relaying-ring-sdpa-kv-between-chips-2026-10-10-parked)
  - [#154 details: an 8×10 compute grid with Ethernet dispatch (2026-10-10, optional)](#154-details-an-810-compute-grid-with-ethernet-dispatch-2026-10-10-optional)
  - [How the experiments were run (making the experiments themselves fast and reliable, 2026-10-07)](#how-the-experiments-were-run-making-the-experiments-themselves-fast-and-reliable-2026-10-07)
- [11. Q&A notes (2026-09-30)](#11-qa-notes-2026-09-30)
  - [11.1 Is video generation bound by compute or communication?](#111-is-video-generation-bound-by-compute-or-communication)
  - [11.2 The K/V ring needs data from distant chips; is that a problem?](#112-the-kv-ring-needs-data-from-distant-chips-is-that-a-problem)
  - [11.3 Does splitting the sequence (SP) break the dependencies between frames?](#113-does-splitting-the-sequence-sp-break-the-dependencies-between-frames)
  - [11.4 Where are SDPA's intermediate results kept?](#114-where-are-sdpas-intermediate-results-kept)
  - [11.5 What is the fused path?](#115-what-is-the-fused-path)
  - [11.6 Idle cores in SDPA, and other bottlenecks](#116-idle-cores-in-sdpa-and-other-bottlenecks)
  - [11.7 How complex can processing on a Tensix core be?](#117-how-complex-can-processing-on-a-tensix-core-be)
  - [11.8 What limits experiment turnaround](#118-what-limits-experiment-turnaround)
- [12. Operational lessons](#12-operational-lessons)
- [13. MiniMax-H3](#13-minimax-h3)
- [Sources](#sources)

<!-- /toc -->

## 0. Summary in 3 lines

1. **Wan 2.2 already supports the WH Galaxy** (there is `_PRESETS_WH` and a CI performance target). It does not need bring-up work like the H3 PR,
   so it is the safer first model to run.
2. **The H3 WH PR (#56886) is a record of fixing, one by one, the places that "silently fall onto a slow path".** Problems appeared in this order: does not start → does not fit in memory →
   crashes midway → runs but slowly. Most causes were "tables and checks that assume Blackhole silently missed on WH".
3. **Performance: with the same 32 chips and the same parallel layout, WH is about 2 to 2.4x slower than BH** (15-second 768P video, per forward:
   BH 5.1 s / WH 12.4 s → 11.9 s after optimization). On WH, attention (Ring SDPA) takes 70% of the time.
   Config tuning can no longer shrink it; it is kernel-rewrite territory.
4. **Wan 2.2 T2V-A14B 480p (81 frames, 40 steps) on a WH Galaxy 4×8 went from 110.39 s → 45.00 s (2.45x faster)**
   (as of 2026-10-10, #143. Details in section 10). Enabling fast exp (#138), which is off by default because it affects quality, gives 43.88 s (2.52x faster).
   As a further optional configuration, moving dispatch from Tensix row 10 to the Ethernet cores widens the compute grid from 8×9 → 8×10 (#154) and gives
   **43.01 s** (2.57x faster), and **42.14 s** (2.62x faster) with fast exp. Taking off only one link and keeping 3 SP links (#155) gives
   **42.68 s** (2.59x faster), and **41.85 s** (2.64x faster) with fast exp. Fixing a write race in the matmul op so that QKV and FFN1 also use row 10 (#156) gives
   41.87 s (41.02 s with fast exp). However, with this setup two 40-step runs did not match exactly, so
   this configuration was changed to not use #109, which enlarges the kernel config region (#157): 42.00 s (41.19 s with fast exp).
   The send scheme that avoids the race in the ring all-gather with 3 SP links (#155) had never actually been enabled because of a wrong env var check; fixing that (#162) gives
   41.81 s (2.64x faster), and 41.01 s (2.69x faster) with fast exp. An exact transform that folds the cross-attn text padding (zero padding up to 512 tokens)
   into a weighted key (#164) gives 41.64 s (2.65x faster), and 40.80 s (2.71x faster) with fast exp.
   The rare result divergence turned out to be a race that already exists in the default configuration, not caused by the enlarged config region, so the enlargement was restored at +64 KB,
   and a defect where the ring SDPA K/V buffers did not alternate between calls was also fixed (#165): 41.49 s (2.66x faster),
   and 40.70 s (2.71x faster) with fast exp. Computing the timestep embeddings for all steps at once (#166) gives **41.43 s** (2.66x faster).
   Note that the e2e script synchronizes after every step while measuring; without that, the true time is 41.26 s (2.68x faster), and 40.43 s (2.73x faster) with fast exp. Computing the softmax row sum as a 5th column of the P·V matmul (a tile with 1 only in column 0), and making the TRISC side wait for the pack thread's SFPU (#167), gives **41.00 s** (2.69x faster), **40.81 s** (2.70x faster) without sync, and **40.07 s** (2.75x faster) with fast exp. Keeping the residual stream of the VAE decoder in row-major (#174, bit-identical) brings the VAE from 0.51 → 0.49 s, and **40.78 s** without sync. Keeping the causal conv's temporal cache with its halo, and letting neighbor pad copy it as-is as the first frame (#175, bit-identical), brings the VAE from 0.49 → 0.46 s, and **40.76 s** without sync. In the mid-block attention, letting chips that hold the same frame split the queries among themselves (#176, bit-identical) brings the VAE from 0.46 → 0.445 s. Having the RMSNorm that builds the causal conv's input write directly into the conv's halo buffer (#177), and packing the sticks that neighbor pad sends to its neighbor 4 at a time into fabric scatter packets (#178, both bit-identical), brings the VAE from 0.447 → 0.410 s. Allocating both assembly buffers for the host readback on the first call, and converting the VAE input to bf16 in torch before sending it (#180, same values), brings the VAE to 0.39-0.41 s. Distributing the frames of the mid-block attention with all-to-all (#181) and splitting the input into contiguous blocks (#182, both same values) brings the VAE to 0.37 s. Searching the conv3d blocking again on the 8×10 grid of Ethernet dispatch (#185, same values) brings the VAE to 0.36 s. Fusing the final add of the residual block into the output side of the second conv (#189, #190, bit-identical) brings the VAE to 0.346 s. Fusing untilize into the last pass of the row-major RMSNorm, and removing the padding tiles and the per-row re-initialization (#191, bit-identical), brings the VAE to 0.341 s. Letting the spatial conv after the 2x upsample read the low-resolution tensor directly by remapping coordinates (#192, bit-identical) brings the VAE to 0.333 s. Also keeping the residual stream of the latent-resolution layers in row-major (#195, bit-identical) brings the VAE to 0.330 s. A hang bisection showed that the VAE hang happens only when conv3d runs on the 8×10 grid of Ethernet dispatch (row 9), so on the WH Galaxy conv3d is now limited to 9 rows by default (#197, bit-identical; on 8×9 the blocking goes back to the one from before #185). The VAE is 0.340-0.343 s. **Final status (2026-10-13):** the default configuration is **40.66 / 40.72 s** without sync (#197), and **39.902 / 39.909 s** with the optional fast exp (#138) (#197); 39.888 / 39.982 s with #192, 39.968 / 39.975 s with #195, 39.982 / 39.982 s with #191, 39.924 / 39.998 s with #190, 39.915 / 40.005 s with #189, and before that 39.922 / 39.938 / 39.995 s with #185 (21 of the 27 runs since #180 that did not hang are under 40 s, 2.76x faster). However, in the 2026-10-13 e2e runs there were ring SDPA hangs in denoising: 3 of 12 runs with fast exp, and 2 of 7 runs even with the default configuration (unrelated to the VAE changes; since it also happens with the default configuration, it is not specific to fast exp). The 39.67 s from stacking the optional Phase 2 merge (#172) does not count as an improvement, because 5 of 6 runs hung. With a configuration that does not add hangs, we are now just under 40 s (denoise varies by about ±50 ms, so not on every run yet). Two kinds of hangs remain, in the denoising ring SDPA and in the VAE conv3d (one core on row 9 of one chip stops); the cause is still under investigation. For the VAE one, we found that it happens only when conv3d runs on the 8×10 grid of Ethernet dispatch (the bisection row in section 10). What helped: folding the two CFG passes into heads, BFP8 weights with all of them resident,
   LoFi and BFP8 for the projections and SDPA, the ring SDPA work assignment, multicast, and fewer writes from the compute pack thread,
   removing small ops, fusing normalization, and so on. Of the remaining time, 60% is ring SDPA (all 3 threads are 80-90% busy with work),
   and a little over 30% is matmul (bound by the per-tile hand-off between unpack and math).

---

[↑ Back to contents](#contents)

## 1. Glossary (terms used in this chapter)

| Term | Meaning |
|---|---|
| **Galaxy** | A Tenstorrent server with 32 chips in one chassis. The chips are connected to each other over Ethernet in a mesh. There are WH and BH versions. |
| **mesh 4×8** | A logical arrangement that treats the 32 chips as a grid of 4 rows × 8 columns. "axis 0" is the row direction (4), "axis 1" the column direction (8). |
| **TP (Tensor Parallel)** | Parallelism where one matrix (weight) is split across several chips, and the compute is split too. H3 uses TP=4 on the 4 chips of axis 0. |
| **SP (Sequence Parallel)** | Parallelism that splits the input sequence (the video tokens) across several chips. H3 uses SP=8 on the 8 chips of axis 1. |
| **DiT** | Diffusion Transformer. The core of video generation. H3 has 50 layers; Wan 2.2 has 40 layers × 2 models. |
| **forward / step** | One pass through the DiT is one forward. For 50 diffusion steps, H3 runs 49 forwards (the last step is not needed). H3 is guidance-distilled, so 1 step is 1 forward; Wan 2.2 uses CFG, so 1 step is 2 forwards. |
| **warm / cold** | cold = the first run (includes kernel compilation and building the weight cache). warm = the second and later runs. Compare performance on warm runs. |
| **Tensix core / core grid** | The compute units inside a chip. A WH Galaxy can use 8×9=72 per chip, a BH Galaxy 12×10=120. |
| **DRAM / bank** | The main memory attached to a chip. WH has 12 GB (12 banks ≈ 1 GB/bank), BH has 32 GB. Whether an allocation fits is decided by the free space per bank. |
| **L1** | A small SRAM next to each core (WH about 1.43 MiB, BH 1.5 MiB). If a matrix block does not fit in L1, the kernel cannot be built. |
| **fabric / link / payload** | The fabric is the chip-to-chip communication system. A link is a physical Ethernet connection between chips (WH uses 4 × 12.5 GB/s, BH 2 × 25 GB/s). The payload is the most data one packet can carry (WH opens with 4 KB, BH with 8 KB). |
| **Ring / Linear topology** | The shape of the communication path. Ring is a loop with the ends connected; Linear is a straight line. Ring is faster but has algorithmic constraints (such as the K_block condition described later). |
| **FSDP** | Fully Sharded Data Parallel. Instead of copying the full weights to every chip, the weights are split and gathered just before use (all-gather). Memory goes down, communication goes up. |
| **all-gather / reduce-scatter** | Basic collective communication (CCL) operations. all-gather = collect each chip's fragment so that everyone has the whole. reduce-scatter = add up everyone's partial sums, then split the result and hand out the pieces. |
| **AGMM** | An op that fuses All-Gather + Minimal MatMul into one kernel. It starts the matmul while gathering the TP-split input, so communication and compute overlap. Used for to_qkv / to_out / ff1. |
| **MM+RS (fused MMRS)** | A fused matmul + reduce-scatter op. Used for ff2 (the second matmul of the FFN). |
| **Blocking (M, K, N, (sub_h, sub_w))** | How a matmul is split up when it is assigned to cores. Numbers are in tiles (32×32 elements). Bigger blocks are more efficient but use more L1. The best values depend on the chip's core count and shape, so each arch needs its own search (sweep). |
| **M_per_core** | The number of tiles in the M direction that one core handles. This decides how good a blocking is. |
| **Fallback** | A generic config used when no dedicated config is found. Often slow, and easy to miss because it only prints one warning line in the log. |
| **SDPA / Ring Joint SDPA** | Scaled Dot-Product Attention (the core of attention). The Ring version computes while passing the SP-split K/V around the ring. The chunk size (q, k) drives performance. |
| **packed sequence** | H3 packs text, audio, and video into one sequence for the DiT; this is its row count. It is padded to be divisible by SP. |
| **HiFi2 / LoFi** | Precision modes of the matrix unit. HiFi2 is about bf16 precision; LoFi is fast but low precision. The theoretical performance limit (roofline) is computed based on HiFi2. |
| **fp32 dest** | A setting that does the matmul accumulation in fp32. More accurate but slower. |
| **roofline / utilization (util)** | The theoretical minimum time, based on the hardware's theoretical performance (here, the peak FLOPS of the compute units). The inverse of measured ÷ theoretical is the utilization. "headroom 2x" means it could still get 2x faster in theory. |
| **PCC** | Pearson correlation coefficient. A numerical check that compares the output with the torch reference implementation (1.0 is a perfect match). |
| **CLIP / VBench** | Quality metrics for generated video. CLIP measures agreement with the prompt (the H3 pass line is 33.0); VBench covers many items such as consistency and smoothness. |
| **Tracy** | A profiler that records the run time of each op on the device. |
| **adaLN** | A mechanism that modulates each layer's normalization according to the diffusion timestep. In H3 it is implemented as a table lookup (embedding). |
| **SwiGLU** | The FFN activation function. It is fused as post-processing (epilogue) of the ff1 matmul. |
| **trace** | A feature that records the op issue sequence once and replays it. It removes the host-side issue cost. |

---

[↑ Back to contents](#contents)

## 2. Does Wan 2.2 already support the WH Galaxy?

**Yes.** It does not need extra work like the H3 PR.

`models/tt_dit/pipelines/wan/pipeline_wan.py` (main) has a preset table per arch, and it switches automatically by chip type:

| Mesh | Arch | SP axis | TP axis | links | Topology | FSDP |
|---|---|---|---|---|---|---|
| 4×8 | **WH** | 1 | 0 | **4** | Ring | **enabled** |
| 2×4 | WH | 0 | 1 | 1 | Linear | enabled (with dynamic_load) |
| 4×8 | BH | 1 | 0 | 2 | Ring | disabled |
| 2×4 / 1×4 / 4×32 etc. | BH | … | … | 2 | … | … |

Note that this has **the same shape** as the H3 WH support: on WH, links go up to 4, and FSDP is enabled to fit in 12 GB of memory.
It reads as if Wan 2.2 took this path first, and H3 later applied the same approach.

CI performance targets (`tests/models/wan2_2/test_performance_wan.py`, 81 frames, 40 steps. According to the comments, the 4×8 values are measured values plus a 30% margin):

| Condition | BH target (total) | WH target (total) | WH/BH (computed) |
|---|---|---|---|
| 4×8, 480p | 76 s (denoise 75) | 144 s (denoise 142) | 1.9x |
| 4×8, 720p | 206 s (denoise 205) | 452 s (denoise 449) | 2.2x |
| 2×4, 480p | 255 s | 850 s | 3.3x |

Here too, **on 4×8, WH takes about 2x the time of BH**. This is almost the same as the H3 gap described later (2.0-2.4x).

---

[↑ Back to contents](#contents)

## 3. Details of PR #56886 "Enable MiniMax-H3 on Wormhole Galaxy"

- Author jamesleeTT, merged 2026-09-25, 34 commits, 23 files, +1082 / −120 lines
- The PR description said the WH performance details would go in the follow-up PR #57029, but #57029 was closed with no description.
  The actual WH performance docs are in #57941 (not merged, see below).

The problems surfaced in the following order. **Fixing one stage makes the problems of the next stage visible**: this order itself is typical of bring-up.

| Stage | Symptom | Section |
|---|---|---|
| ① Does not start | The tests skip WH 4×8 | 3.1 |
| ② Does not fit in memory | OOM on 10-second and 15-second videos | 3.2 |
| ③ Crashes midway | assert after a 20-minute weight load, intermittent hangs | 3.3, 3.4, 3.5 |
| ④ Runs but slowly | Dedicated configs do not match and it falls back to generic ones (only one warning line) | 3.5, 3.6, 3.7 |
| ⑤ Reliability around it | A broken cache looks like a model bug; getting the weights | 3.8, 3.9 |

### 3.1 Mesh configuration per arch (① does not start)

**What was happening:** The H3 pipeline had only the Blackhole config table `_PRESETS_BH`. Its 4×8 row requested an 8 KB fabric
payload, which exceeds the WH (wormhole_b0) limit, so on main the WH 4×8 tests were skipped before the mesh was opened.

**Fix (`pipelines/minimax_h3/pipeline_minimax_h3.py`):**

```python
_PRESETS_WH = {
    (4, 8): {
        "tp_axis": 0, "sp_axis": 1,       # same axis assignment as BH (TP=4 is set by the model structure)
        "num_links": 4,                   # match the WH mesh with a 4 KB payload
        "topology": ttnn.Topology.Ring,
        "coresident": False,              # see 3.2
        "dit_fsdp": True,                 # see 3.2
    },
}

def _presets_for_this_arch():
    if is_blackhole():     return "Blackhole", _PRESETS_BH
    if is_wormhole_b0():   return "Wormhole", _PRESETS_WH
    raise NotImplementedError(...)        # unknown archs are rejected explicitly, not dropped into either table
```

- The config lookup key changed from "mesh shape" to "arch + mesh shape".
- Unknown archs (future chips, etc.) **are not silently given an existing table; they raise an error**. The table values are tuned to that chip's
  DRAM capacity and fabric.
- WH has no (4, 32) row (the quad, 4 systems linked). The quad is a BH-only configuration.
- The tests also got a `4x8_WH` row and parameters for the 4 KB payload (`ring_params_4k_req_exact_devices`).

### 3.2 Memory: FSDP and the "not coresident" setting (② does not fit in memory)

**What was happening:** A WH chip has 12 GB of DRAM (BH has 32 GB). Holding the DiT weights with TP=4 alone uses
799.5 MiB per bank (about 9.4 GiB over 12 banks, computed), which leaves almost no room for activations (the intermediate results of the video token sequence).

**Fix 1 — hold the DiT with FSDP:** The weights are split further across the 8 chips of the SP axis and gathered with all-gather just before each layer's compute.
It is enabled by default in the WH preset. The env var `MINIMAX_H3_DIT_FSDP=0/1` overrides it on either arch.

| | DiT use per bank | Free per bank | Largest contiguous free region |
|---|---|---|---|
| Without FSDP | 799.5 MiB | 221.7 MiB | 221.7 MiB |
| With FSDP | **101.8 MiB** | **919.4 MiB** | **917.7 MiB** |

- Use drops to 1/7.85 (divided by SP=8). **The output is exactly the same** (CLIP matches to 2 decimal places in 6 cases of 5-second video).
- The cost is +5-11% denoise time (more communication and layout conversion).
- Without FSDP, only 5-second videos run. For 10 seconds, 3 of 6 cases OOM in warm-up, and the rest OOM on the second DiT reload.
  For 15 seconds, every case OOMs in warm-up.

**Fix 2 — `coresident: False`:** On BH the text encoder (Qwen3-VL, about 1.6 GB/chip) stays resident together with the DiT, but
on WH it overflows by **a few MB**. So each stage (text encoder → DiT → VAE) is unloaded from memory once it is done.
The cost is reloading the encoder for every request. The code has a TODO: "think about keeping it resident once generation gets down to about 1 minute".

> **Note for beginners:** "Largest contiguous free region" is listed next to "free per bank" because when memory is fragmented,
> a large tensor cannot be allocated even if the total free space is enough. Look at both when you investigate an OOM.

### 3.3 Matching the AGMM core placement to the link count (③ crashes midway, part 1)

**Background:** AGMM places the "senders" (in0 sender / mux), which distribute the gathered input to each core, on one row (or one column) of the core grid.
The senders must also split into exactly `num_links` groups, one per link; an assert inside the op checks this.

**What was happening:** The BH grid sides (12, 10) divide by 2 links, so there was no problem. On the WH 8×9 grid with
4 links and a non-transposed layout, 9 rows have to be split into 4 groups, and with `ceil(9/4)=3` rows each there are only 3 groups.

**Fix (`agmm_worker_grid` in `utils/matmul.py`):** Shrink the sender axis to a length that splits into exactly `num_links` groups.
On WH 8×9, 8 of the 9 rows are used (2 rows × 4 groups). H3 runs with the transposed layout (`force_transpose=True`, the last row reserved for the mux),
so the actual AGMM grid is **8×8** (12×9 on BH).

### 3.4 Keeping the Ring K-block condition (③ crashes midway, part 2)

**Background:** AGMM on a Ring requires the tile count in the K direction (the direction of the inner product) to be divisible by K_block
(`K_tiles_per_device % K_block == 0`. It sends half in each direction and cannot handle a partial block. Linear can.)

**What was happening:** Shapes without a dedicated config fall back to the generic config (8, 8, 8). With TP=4 the dimensions met the condition by chance, but
in a TP=8 experiment the 5376-dim input became 168/8 = **21 tiles** per chip, which is not divisible by K_block=8, so the assert fired.
Worse, this happened **at the first matmul, after a 20-minute weight load**.

**Fix (`_ring_safe_k_block`):** Only when the generic config is used, K_block is lowered to the largest divisor of 21 (7), and a warning is printed once
per shape. Table configs set by search and configs passed explicitly by the caller stay as they are (= the TP=4 production config is bit-identical).

### 3.5 Making the fused MM+RS check include the core grid (③ hang + ④ silent slowdown) — the core of this PR

**Background:** ff2 is the flow "matmul → reduce-scatter → gated residual (addcmul)", and `has_mmrs_config(m, k, n)` decided
whether to use the fused op that does all of it in one. The fused op's config came from two sources:

1. A table set by search (`_SWEPT_BLOCKINGS`) — **for the BH 12×8 matmul grid**
2. The rule engine (v2.3) — **BH only**

**What was happening:** The check looked only at the shape, so on WH too it answered "can fuse" for any M aligned to tile boundaries.
But on WH neither 1 nor 2 matched, and **the fused op ran with the generic default config (fallback)**. That is 50 times per denoise step (once in each of the 50 layers).

On the WH 8×9 grid, that default config gave the reduce-scatter only **1 core per link**,
and this was the real cause of the **intermittent hangs in the middle of denoising** (per the performance README: the full-condition sweep that had stopped at 13/18 cases completed 18/18 after the fix).

**Fix:**

```python
def has_mmrs_config(m, k, n, core_grid):          # core_grid was added
    if not (k == _K and n == _N and m % _TILE == 0):
        return False
    if m in _SWEPT_BLOCKINGS and core_grid == _DEVICE_GRID:
        return True                                # table configs are valid only on the grid they were searched on
    return resolves_fused_mmrs_config(m, _K, _N, core_grid)   # query with the same priority order as the actual resolver
```

- `resolves_fused_mmrs_config()` is a side-effect-free query function that **shares as-is** the production config resolver (`_resolve_fused_mmrs_config`).
  If the check and the real processing are separate copies, they will drift apart sooner or later.
- The caller (`transformer_block_minimax_h3.py`) passes `mesh_device.compute_with_storage_grid_size()`.
- Table configs are also registered only when they match the grid they were searched on (a config that assumes 12×8 goes outside the grid on an 8×9 chip).
- **Guard against deadlock:** The formula for the number of reduce-scatter cores can give 0 or less (e.g. on an 8×9 chip, using 8×7 for the matmul gives
  1 core/link with 4 links, and 0 with 8 links). A collective with 0 workers does not fail; it **hangs**, so it now raises `ValueError` at that point.
  The comment says "a deadlock is the most expensive way to find out the cause".

**Effect:** WH stopped fusing and now runs ff2 as a regular matmul + a separate reduce-scatter. The hangs went away, and every case got **2.0-4.0% faster (3.0% on average)**.
(The follow-up #57941 put back a fused config properly searched for WH on an 8×7 grid, and shortened it further. See section 4.5.)

### 3.6 Keying the tuning tables on "the row count that actually flows" (④ silent slowdown)

**Background:** The matmul blocking table and the SDPA chunk size table are looked up by the row count per chip (M).

**What was happening:** The benchmark code that built the tables in the BH days (a) counted the audio latent as 1 latent = 1 row (it is actually 2 rows, since it is 2ch), and
(b) assumed a 512-token prompt. As a result, the table keys were **row counts that the pipeline never actually produces**.

| Item (768P, 15 s) | Actual pipeline | Old benchmark (computed) |
|---|---|---|
| Video | 107 latent frames × 24×42 = 107,856 rows | same |
| Audio | 603 latents × 2ch = **1,206 rows** | 603 rows |
| Text | 39 tokens | 512 tokens |
| Total → padding (multiple of 256) | 109,101 → 109,312 | 108,971 → 109,056 |
| Per chip (SP=8) | **13,664** | 13,632 |

Table keys for 5 / 10 / 15 seconds: old 4768 / 9216 / 13632 → new **4736 / 9184 / 13664**. 13632 is a value that never appears for 15 seconds, whatever the prompt length.

**Fix:**
- Added `packed_sequence_length()` and `padded_sequence_length()` to `packing.py`; both the pipeline and the tests use them
  (the same calculation is not written in two places, so they cannot drift).
- Re-keyed the SDPA chunk size table (`attention_minimax_h3.py`) to the new keys. The comment states that "the 5-second values were measured on BH's 110 SDPA cores and
  are not verified on WH's 63 cores".
- **Fallback to the same M_per_core** (`_same_m_per_core_match`): when there is no exact match, use the entry with the same (K, N) and the same
  "M tiles per core". How good a blocking is depends on M_per_core, not on M itself. Example: 13632 and 13664 both give
  M_per_core = 54 on an 8-column grid. The table keeps matching even when the prompt length changes within the padding range. It logs once when used.

### 3.7 Blocking search for WH and its numerical check (④)

The WH core grid (8×9) differs from BH (12×10), so **all the blockings H3 had were searched on grids that do not exist on WH**.
A search on the WH 4×8 Ring configuration was run with `sweep_mm_block_sizes.py`, with these results for 15 seconds (M=13664) (1419 configs, all finished normally):

| Shape | op / grid | Configs searched | Previous config | Adopted config | Rank | Gain |
|---|---|---|---|---|---|---|
| ff1 | AGMM 8×8 | 320 | (8,3,14) 16,718 µs | **(8,7,10)** 15,710 µs | 1st | 6.0% |
| ff2 | matmul 8×9 | 322 | (8,8,8) 7,013 µs | **(8,7,10)** 6,771 µs | 2nd | 3.5% |
| to_qkv | AGMM 8×8 | 425 | (8,7,12) | unchanged | 2nd | 0.5% behind 1st |
| to_out | AGMM 8×8 | 352 | (8,8,6) | unchanged | 3rd | 0.5% behind 1st |

In total, 1,250.5 µs per block = 62.5 ms per step = 0.5% of one forward. A same-host A/B of the whole pipeline gave **−69.9 ms/forward (−0.58%)**, almost matching the prediction.

Decisions in this section worth copying:
- **ff2's 1st place (12,7,8) was 1.5% faster but was not adopted.** The search tool only measures time and does not check PCC.
  The adopted (8,7,10) was verified afterward at PCC 1.0000000 vs torch, and ff1's (8,7,10) at PCC 0.9999843 on a 4-chip Ring (it matches the old config to 6 decimal places).
- **to_qkv / to_out were not added, since the gap was 0.5%.** That is about the same as the run-to-run variation (about 0.3%), so they cannot be told apart.
- **The tables were split by M.** On BH, the key was only (K, N), on the assumption that "the block shape does not depend on M", but on WH this does not hold.
  Example: the best config for ff2 at M=4768 ranks 71st at M=9216, 14.6% slower than the best, and worse than the generic (8,8,8).
- **Fixes to the search tool:**
  - An attempt to lower the L1 budget cap for WH was a mistake (WH L1 is 1,499,136 B, BH 1,572,864 B). Lowering it to 1328 KB
    excluded both ff1's best (estimated 1380 KB) and qkv's existing value (1352 KB), so even the baseline could not be measured; it was reverted.
  - The profiler buffer dump interval `PROFILER_DUMP_EVERY` can now be changed with the env var `MM_SWEEP_PROFILER_DUMP_EVERY`.
    **On a WH Galaxy, the default of 10 always hangs at the first dump (the 10th config)** (see section 6).

### 3.8 Getting the weights (⑤)

`resolve_weights_dir()` in the new file `weights_minimax_h3.py` looks for the weights in this order:

1. The env var `MINIMAX_H3_MODEL_PATH`
2. The HuggingFace cache
3. Download (only when `TT_DIT_ALLOW_HF_DOWNLOAD=1`. If it is unset, the tests skip instead of downloading 144 GB on their own)

It downloads only the parts it needs:

| Part | Size | Use |
|---|---|---|
| `text_encoder/` | 66.7 GB | all tasks (Qwen3-VL) |
| `transformer/` | 66.3 GB | t2va, fl2va |
| `transformer_ref/` | 66.3 GB | ref2va only |
| `vae/` | 10.4 GB | all tasks |
| `audio_vae/` | 0.6 GB | all tasks |
| tokenizer etc. | about 20 MB | always fetched |

For t2va, about 144 GB (the whole repository is 498 GB. `FL2VA/` and `Ref2VA/` are duplicates in another format and are not fetched).
**In our environment, check the free disk space first.**

### 3.9 Read-back verification of the weight cache (⑤)

**What was happening:** tt_dit saves the converted and split weights in a cache (`TT_DIT_CACHE_DIR`) and uses it from the next run on.
A cache built during a run with a broken device setting, `l1_small_size=0`, kept being reused in later runs.
The result was "text embedding max abs value 2.5e30" and "**a video that is not broken, but shows a different subject**" (CLIP 13.12, normally 37.36).
It looked like a model bug and caused a long detour.

**Fix (`utils/cache.py`):** When `TT_DIT_CACHE_VERIFY=1`, right after the cache is written, all tensors are read back and compared with the in-memory weights, shard by shard
(`verify_saved_model`). If they do not match, the completion mark is not written, so the cache is rebuilt next time. Reading the whole cache back once has a cost, so this is optional.
**Unresolved:** The cache key does not include device settings (such as `l1_small_size`).

> **Note for beginners:** The tricky part is that the output is "a plausible but different video", not "a broken video". It is the same kind of trap as
> missing gelu_fast on the P150, and you need **both a numerical metric (CLIP) and visual inspection**.

### 3.10 Bring-up lessons (taken from this PR)

1. **Keep arch-dependent values in a table keyed by arch, and reject unknown archs.** Do not allow "BH values happen to work on WH too".
2. **The check and the real processing share the same code.** The biggest incident (hang + slowdown) came from a shape-only check that disagreed with whether a config could actually be resolved.
3. **Table keys are the values the pipeline actually sends.** Keep the row count calculation in one place, and have the tests use it too. Print `rows/device` in the log to confirm.
4. **Count fallbacks in the log.** Behind "one warning line", 50 layers were going down the slow path. Always check the fallback warnings in the run log (same as the `[fallback]` check on the P150).
5. **Searching for speed and checking correctness are separate.** The search tool only measures time, so verify the adopted config with PCC afterward. Do not adopt an unverified 1st place.
6. **If a single difference is within the variation, do not change.** Do not put in a config that differs by 0.5%.
7. **Fail early and clearly.** An assert after 20 minutes, a hanging collective: both were turned into "errors you can read on the spot".
8. **Cache and environment bugs look like model bugs.** Have a way to verify.

---

[↑ Back to contents](#contents)

## 4. Performance: Blackhole vs Wormhole

### 4.1 Hardware differences (1 Galaxy = 32 chips)

| Item | Wormhole Galaxy | Blackhole Galaxy | Source |
|---|---|---|---|
| Compute cores / chip | 8×9 = 72 | 12×10 = 120 | sdpa_perf_utils.py |
| Clock | 1.0 GHz | 1.35 GHz | transformer_roofline.py |
| HiFi2 theoretical / chip | 147.5 TFLOPS | 331.8 TFLOPS (computed) | 72 (120) cores × 2048 FLOP/cycle × clock |
| 32-chip total | 4.72 PFLOPS | 10.6 PFLOPS (computed) | |
| DRAM capacity / chip | 12 GB | 32 GB | |
| DRAM bandwidth / chip | 288 GB/s | 512 GB/s | |
| L1 / core | 1,499,136 B | 1,572,864 B | sweep_mm_block_sizes.py |
| Chip-to-chip links (H3 setting) | 4 × 12.5 GB/s | 2 × 25 GB/s | 50 GB/s total for both |
| fabric payload | 4 KB (limit 7,616 B) | 8 KB (limit 15,232 B) | |
| Cores used for SDPA | 63 (7×9) | 110 (11×10) | attention_minimax_h3.py |
| AGMM worker grid | 8×8 | 12×9 | |
| H3 DiT FSDP | required (enabled) | not needed (disabled) | |

The ratio of theoretical compute per chip is BH/WH ≈ **2.25x** (computed). The measured gap is larger than this (section 4.3).

### 4.2 End-to-end comparison (H3, 768P 16:9, 50 steps = 49 forwards, warm)

| Condition | BH 4×8 | WH 4×8 (#57941 baseline) | WH/BH (computed) |
|---|---|---|---|
| 5 s (124 frames) denoise | 67.0 s (1,366.5 ms/forward) | 135.3 s (2,761 ms/forward) | 2.0x |
| 5 s total | 72.7 s | 147.6 s | 2.0x |
| 15 s (362 frames) denoise | 252.3 s (5,149 ms/forward) | 609.3 s (12,435 ms/forward) | 2.4x |
| 15 s VAE decode | 11.5 s | 15.8 s | 1.4x |
| 15 s total | 268.5 s | 633.1 s | 2.4x |
| Reference: BH 4×32 (quad, 128 chips, trace) 15 s | total 125.0 s (1,896 ms/forward) | — | |

Caveats when reading (all of them are stated in the sources):
- **The measurement date, commit, and host differ.** The BH values are from the merged `MiniMaxH3.md`; the WH values from the #57941 README (2026-09-17).
- The BH 5-second value does not include the text encoder time (prompt embeddings were cached in advance). The BH audio decode value is from before the default precision setting changed.
- **Variation:** ±8% on BH (denoise 56.6-71.3 s under the same conditions). On WH too, about 3% difference between hosts with the same code, and about 2% across a board reset, are reported.
- Even so, a gap of "2-2.4x" is well above the variation.

### 4.3 Per-block breakdown (15 s, per chip, Tracy)

One H3 DiT layer (one of the 50), measured with the same shape (M=13664 rows/chip) on both archs.

| op | BH (ms) | BH util | WH baseline (ms) | WH util | WH/BH (computed) |
|---|---|---|---|---|---|
| Ring Joint SDPA | 51.79 | 68% | **174.56** | 48% | 3.4x |
| ff1 AGMM (SwiGLU fused) | 5.49 | 64% | 15.70 | 51% | 2.9x |
| to_qkv AGMM | 3.65 | 73% | 10.26 | 59% | 2.8x |
| to_out AGMM | 2.92 | 50% | 5.3-6.3 | 32-46% | about 2x |
| ff2 (BH: fused MM+RS / WH: matmul + RS + residual) | 2.50 | 79% | 10.27 | 52% | 4.1x |
| adaLN lookup (Embeddings ×6) | 5.13 | DRAM 8% | 10.41 | DRAM 7% | 2.0x |
| **Block total (device time)** | **74.8** | | **246.9** | 36% | **3.3x** |

- BH: 2026-09-22/23, without FSDP. WH: 2026-09-17, with FSDP (233.4 ms without FSDP).
- **What the gap is made of:** On top of the 2.25x theoretical gap, each op's utilization on WH is 10-20% lower than on BH (SDPA 48% vs 68%, matmul 51-59% vs 64-79%). Multiplied together, this gives about 3x.
- BH 74.8 ms × 50 layers = 3.74 s/forward (computed), which is shorter than the end-to-end 5,149 ms/forward. The end-to-end value is older and also includes gaps between ops and host-side work. Do not compare the block ratio (3.3x) directly with the end-to-end ratio (2.4x).

### 4.4 The WH roofline — how fast can it get?

The theoretical minimum time per forward, from the HiFi2 theoretical performance of 32 WH chips (4.72 PFLOPS), versus measured:

| Video length | Sequence length after padding | Attention share of compute | Theoretical min ms/forward | Measured ms/forward | Compute utilization | Headroom |
|---|---|---|---|---|---|---|
| 5 s | 37,888 | 58% | 746 | 2,761 | 27% | 3.7x |
| 10 s | 73,472 | 73% | 2,240 | 6,422 | 35% | 2.9x |
| 15 s | 109,312 | 80% | 4,523 | 12,435 | 36% | 2.75x |

- The longer the video, the more attention (which grows with the square of the sequence length) dominates. At 15 seconds, 80% of the compute is attention.
- A realistic upper limit is about 70% utilization (well-tuned WH matmuls reach 80-93% of the HiFi2 theoretical). In that case, 15 seconds would take about 6.5 s/forward (denoise 317 s).
- **Communication is not the bottleneck.** For 15 seconds, per layer, passing 686 MB of KV around the ring takes about 14 ms at 50 GB/s, hidden behind the 175 ms of SDPA.
  The README's conclusion is that "only the compute units are the bottleneck".
- 5 seconds is especially low because of occupancy (with the SDPA chunk assignment, 16.7% of the 63 slots sit idle, and M per chip is small).
- The BH block is 74.8 ms against a theoretical sum of 46.3 ms (headroom 1.62x). WH is 246.9 ms against 109.1 ms (2.26x). **WH is farther from its theoretical value.**

### 4.5 All WH conditions (#57941 README, with FSDP, 2026-09-17)

An excerpt of 16:9 and 1:1 (all 6 aspect ratios × 3 lengths = 18 cases completed):

| Length | Aspect | Canvas | warm total | denoise | ms/forward | Real-time ratio | CLIP |
|---|---|---|---|---|---|---|---|
| 5 s | 16:9 | 1344×768 | 147.6 s | 135.3 s | 2,761 | 28.6x | 37.42 |
| 5 s | 1:1 | 768×768 | 79.3 s | 69.4 s | 1,417 | 15.3x | 36.32 |
| 10 s | 16:9 | 1344×768 | 333.4 s | 314.7 s | 6,422 | 32.9x | 36.92 |
| 10 s | 1:1 | 768×768 | 159.2 s | 146.4 s | 2,988 | 15.7x | 37.54 |
| 15 s | 16:9 | 1344×768 | 633.1 s | 609.3 s | 12,435 | 42.0x | 36.31 |
| 15 s | 1:1 | 768×768 | 266.6 s | 251.2 s | 5,126 | 17.7x | 38.26 |

- "Real-time ratio" = seconds taken to generate ÷ seconds of video (smaller is better).
- denoise is 87-96% of the warm total. The time is decided by the **pixel count** and barely depends on the aspect ratio (the three 1.03 MPix cases match within 1 second).
- It grows **faster than linearly** with length (2.33x from 5→10 seconds, 1.94x from 10→15 seconds), because attention grows with the square of the sequence length.
- The text encoder takes about 3.5 s in every case.

### 4.6 Progress in the unmerged PR #57941 "MiniMax-H3: Wormhole Galaxy optimizations"

- Status: **Open, Draft** (updated 2026-09-28), 27 commits, 59 files, +11,684 / −168 lines, no PR description.
- Most of the content is documentation (per-op analyses in `models/tt_dit/models/minimax_h3_wormhole/*.md`, BH comparisons in `minimax_h3_blackhole/`) and
  analysis tools (roofline, Tensix performance counters, single-op benchmarks). It also has **changes to the kernels themselves**:
  the WH matmul LLK (`tt_llk_wormhole_b0/llk_lib/llk_math_matmul.h` and others), SDPA's `compute_streaming.hpp`, the AGMM and minimal_matmul kernels, and the SwiGLU LUT.

Optimizations and their effects (15 s 16:9, 1 block, per chip):

| Optimization | Status | Effect |
|---|---|---|
| Pack in the SDPA inner loop made 4 wide | included | SDPA 174.56 → 172.16 ms (−2.4 ms) |
| silu in the ff1 SwiGLU switched to the bf16-precision version | included | ff1 kernel −3.6%, no PCC change |
| Instruction reordering in the matmul K loop (LLK rewrite) | included | ff1 −0.35 ms per call, to_qkv 11.25 → 11.03 ms |
| ff2 fused MM+RS searched again for WH (8×7 grid, (6,7,8)) | included | ff2 as a whole 10.27 → 8.24 ms (−2.0 ms) |
| adaLN table replicated 16 times to avoid concentration on one bank | included | lookup 10.45 → 2.39 ms, block 238.5 → 230.8 ms (−3.2%), no PCC change |
| fp32 dest off for ff1 / to_qkv | **switch only** (`MINIMAX_H3_MM_FP32_DEST=0`) | ff1 15.63 → 14.30 ms (single-op benchmark) |
| SwiGLU sigmoid approximated with a LUT | **switch only** (`TT_MM_SWIGLU_LUT_SILU=1`) | ff1 15.7 → 12.1 ms with both switches (utilization 51% → 66%) |

- Everything on (including the 2 switches): block 246.92 → 238.27 ms (−3.5%), then **230.8 ms** after the adaLN improvement.
  1 step 12.26 → **11.93 s** (49-step denoise about 584 s). End-to-end CLIP 35.90 (pass line 33.0).
- **The 2 precision-related switches are disabled by default.** Adopting them is left to "the owner's decision", and VBench with the LUT enabled has not been run.
- **SDPA barely shrank** (utilization still 48%). It is now 72% of the block, and 89 ms of the remaining headroom (129 ms) is SDPA.
  Chunk size tuning is exhausted (several q, k combinations were tried, and the existing (256, 512) is the best); from here on it is kernel work.
- Same-host A/B is the rule. The PR records a case where "compared with a baseline from another host it looked like a 3.6% improvement, but it was actually the host difference".

### 4.7 Comparing parallel layouts (WH, 15 s, 10 steps)

The only meshes that can be opened with 32 chips are 8×4 / 4×8 / 1×32 / 32×1 (2×16 and 16×2 are rejected). 4×8 and 8×4 are physically the same ring.
With 56 heads, TP=16 is not possible. In practice there are only 3 options:

| Layout | TP / SP | ms/forward | vs the production layout |
|---|---|---|---|
| 4×8 tp0 sp1 (production) | 4 / 8 | 12,382 | — |
| 4×8 tp1 sp0 | 8 / 4 | 12,890 | +4.1% (slower). The output is the same video, with small jitter in the details from the bf16 rounding order |
| 1×32 | 1 / 32 | — | **hangs every time** (cause unknown, needs `tt-smi -r all`) |

**Stay with the production TP4/SP8.**

### 4.8 Quality

- The WH VBench (16:9, 5 seconds) is above the pass line on every item (e.g. subject_consistency 0.9793 / pass line 0.95). CLIP 37.36 (BH 37.37).
- VBench was checked only for 16:9, 5 seconds. The sweep ran with `RUN_VBENCH=0` (CLIP only).

---

[↑ Back to contents](#contents)

## 5. Implications for our work

- **Direction (decided 2026-09-29): the optimization target is Wan 2.2.**
- **First, run Wan 2.2 on WH 4×8.** There are WH presets and a CI target (480p within 144 s in total), so it is easy to judge "whether it runs correctly".
- H3 has 144 GB of weights, and building the cache reads files on the order of 62 GB (1054 s the first time). Estimate disk space and time first.
- The main battlefield for performance is **SDPA on WH** (70% of the block, utilization 48%). The SDPA chunk tuning we did for Wan2.1 on the P150 is already exhausted for H3.
- When comparing, use the same host and the same commit, and measure the variation (a few % on WH) first. What we learned on the P150 is written there as-is.

[↑ Back to contents](#contents)

## 6. Operational notes (hangs and board recovery)

The README records several cases that **do not recover without resetting the board with `tt-smi -r`**. Our policy is "avoid resets where possible", so we avoid these cases up front:

1. **Running `sweep_mm_block_sizes.py` with the defaults on a WH Galaxy hangs.** It stops making progress at the profiler dump on the 10th config, and the board needs a reset.
   Setting the interval large, like `MM_SWEEP_PROFILER_DUMP_EVERY=100000`, is required (it also reproduces with the existing Wan2.2 shapes; not specific to H3). The root cause is not known.
2. **TP1/SP32 (1×32) always hangs.** Do not use it.
3. **What a hang looks like:** the log stops, there is no exception, the CPU keeps spinning at 170-370% (the host keeps waiting for the device),
   **the pytest timeout does not work**, and after the run the board shows "FW initialization failed" and "ETH heartbeat timeout". `kill -TERM` does not bring it back.
4. **Recommended settings:** With `TT_METAL_OPERATION_TIMEOUT_SECONDS=300`, `TT_METAL_INSPECTOR=1` and so on, a hang turns into an exception plus a record of the device state (the README has example settings).
   Run each configuration in a separate process, wrapped in the shell's `timeout`.
5. **Do not wait on `pgrep -f <name>`.** It also matches the waiting shell's own command and waits forever (we made the same mistake on the P150).
6. The 300 s timeout in `pytest.ini` is too short for 15-second videos. Pass something like `--timeout 3600` explicitly.
7. The README has a step to check "whether any process has the device open" before the recovery procedure (look for tenstorrent in `/proc/*/fd`).

---

[↑ Back to contents](#contents)

## 7. Model comparison: Wan 2.2 and MiniMax-H3

### 7.1 Release timing and positioning

**They were released exactly one year apart** (Hugging Face repository creation dates). Wan 2.2 on 2025-07-28, MiniMax-H3 on 2026-07-28.

| | Wan 2.2 (T2V-A14B) | MiniMax-H3 |
|---|---|---|
| Developer | Alibaba (Wan team) | MiniMax (the company behind the "Hailuo" video service) |
| License | Apache 2.0 (commercial use also free) | Custom Community License (application form for the US, EU, UK, and South Korea) |
| Output | Video only (no audio) | **Generates video + 32 kHz stereo audio together** |
| Length, resolution | About 5 seconds (81 frames), 480p / 720p | 4-15 seconds, 24 fps, short side 768 px (2K is a non-public module) |
| DiT structure | 14B × 2 MoE (switches between an early-stage model for heavy noise and a late-stage model that finishes the details) | 33B dense single Transformer (about 13B of it is the adaLN part, which can be precomputed) |
| Compute per step | 2 forwards because of CFG | 1 forward, CFG-distilled |
| Text encoder | umT5 | Qwen3-VL-32B (uses the output of layer 50) |
| Input conditions | Text (image-to-video is a separate model, I2V) | Text, first and last frames, reference material (up to 9 images, 3 videos, 3 audio clips) |

- **MoE (Mixture of Experts):** A structure with several "expert" models that are used depending on the situation. Wan 2.2 uses separate 14B models for the early diffusion steps (which decide the composition) and the late steps (which draw the details). 27B in total, but only 14B runs at each step.
- **CFG (Classifier-Free Guidance):** A method that computes 2 forwards, with and without the prompt, and amplifies the difference to follow the prompt more closely. "CFG distillation" retrains the model to get the effect of those 2 passes in 1.

### 7.2 Strengths of each

**Wan 2.2**
- Open and easy to work with: Apache 2.0. A year's worth of ecosystem has built up: diffusers and ComfyUI support, LoRAs, distilled versions, and so on.
- Light: the weights are less than half of H3's. In tt-metal too, it is a "reference model" with both WH and BH configs and CI performance targets in place.
- Video quality: with MoE, separate expert models handle the early and late stages. A selling point is that you can direct cinematic expression such as camera work and lighting.

**MiniMax-H3**
- Generates video with audio in one model (dialogue, sound effects, and BGM in sync with the video). Dialogue is stable in 11 languages (including Japanese).
- Up to 15 seconds, aspect ratios from 21:9 to 9:16, first and last frames can be specified, and the input can combine several reference materials.
- CFG-distilled, so 1 step is 1 forward.
- Caveats: only the main model, H3-Base, is public. H3-Context-IR, which refines prompts, and H3-Regenerate-2K, which upscales to 2K, are API-only. **The public version has full attention only**; sparse attention is "to be released later".

> **Note from the optimization side:** On WH, 72% of H3's time is SDPA (section 4.6). If sparse attention is released, the attention compute itself could drop a lot, so it is better to check this outlook before investing in the H3 SDPA kernel.

### 7.3 Official Wan 2.2 benchmarks

**Speed — tt-metal's `models/tt_dit/models/Wan2_2.md`** (81 frames, 40 steps, total seconds per video. The document was last updated on 2026-03-19 and may differ from the current main)

| Resolution | System | Arch | SP / TP | Total |
|---|---|---|---|---|
| 720p | Galaxy 4×8 | **WH** | 8 / 4 | **354 s** |
| 720p | Galaxy 4×8 | BH | 8 / 4 | 168 s |
| 480p | Loud Box 2×4 | WH | 2 / 4 | 735 s |
| 480p | Loud Box 2×4 | BH | 4 / 2 | 207 s |
| 480p | Quiet Box 2×2 | BH | 2 / 2 | 466 s |

- At 720p, WH/BH ≈ 2.1x (computed). Almost the same as H3's 2.0-2.4x.
- The CI target of 452 s is consistent with 354 s × 1.3. WH 4×8 at 480p is not in the table; its CI target is 144 s.
- Future improvements the document lists: higher matmul and SDPA utilization, overlapping AllGather-Matmul / Matmul-ReduceScatter, fusing binary ops, and hiding the FSDP weight gather behind compute. These are the same issues we saw with H3.

**Speed — GPU measurements from the official model card (T2V-A14B, seconds / peak memory GB)**

| GPU | Resolution | 1 GPU | 4 GPUs | 8 GPUs |
|---|---|---|---|---|
| H100/H800 | 480P | 326.9 / 41.3 | 91.7 / 40.9 | 51.5 / 26.3 |
| H100/H800 | 720P | 1041.5 / 59.8 | 288.7 / 51.7 | 155.1 / 37.1 |
| A100/A800 | 480P | 785.7 / 41.3 | 215.2 / 40.9 | 119.9 / 26.3 |
| A100/A800 | 720P | 2735.7 / 59.8 | 725.3 / 51.7 | 386.7 / 37.0 |
| H20 | 480P | 1133.9 / 41.3 | 306.6 / 40.9 | 170.5 / 26.2 |
| H20 | 720P | 4048.7 / 59.8 | 1067.8 / 51.7 | 564.7 / 37.0 |

- The GPU side uses FSDP + Ulysses (sequence parallel) + FlashAttention3 (Hopper generation). The 5B model (TI2V-5B) does 720P on one RTX 4090 in 534.7 s.
- **Rough ordering (720p):** H100 × 8 = 155 s, BH Galaxy = 168 s, WH Galaxy = 354 s, A100 × 8 = 387 s. A WH Galaxy (32 chips) is about the same as 8 A100s. However, the implementations and measurement dates differ, and the card does not state the GPU step count, so this is not a strict comparison.

**Quality — Wan-Bench 2.0 from the official model card** (a benchmark made by the Wan team itself)

| Item | Wan2.2-A14B | Seedance 1.0 | Hailuo 02 | KLING 2.0 | Sora | Wan2.1-14B |
|---|---|---|---|---|---|---|
| Aesthetics Quality (visual beauty) | **85.3** | 84.3 | 77.2 | 80.1 | 82.3 | 75.9 |
| Dynamic Degree (amount of motion) | **52.0** | 38.6 | 38.5 | 48.7 | 47.6 | 50.5 |
| Text Rendering (text inside the video) | **66.5** | 61.1 | 47.0 | 19.9 | 50.4 | 65.1 |
| Camera Control (camera work) | **39.6** | 38.0 | 39.4 | 37.1 | 20.8 | 27.9 |
| Video Fidelity (faithfulness of the video) | 73.7 | **81.8** | 63.6 | 64.8 | 70.9 | 63.6 |
| Object Accuracy (accuracy of objects) | 78.2 | 73.8 | 80.1 | **82.1** | 69.0 | 75.4 |

- 1st in 4 of the 6 items (Seedance is ahead on fidelity, KLING on object accuracy). It is **self-reported on the team's own benchmark**, so take it with a grain of salt.
- Hailuo 02 in the comparison is MiniMax's previous-generation commercial model. The H3 model card does not seem to have comparisons like this with other companies (the only H3 quality metrics we have are the CLIP / VBench pass checks on the tt-metal side).

[↑ Back to contents](#contents)

## 8. Measured on our setup: Wan 2.2 T2V 480p on WH Galaxy 4×8

Results of running the performance test that ships with tt-metal, as is, on another WH Galaxy on 2026-09-29.
These are the "starting point" numbers: **no code changes and no configuration changes**.

### 8.1 Measurement conditions

| Item | Value |
|---|---|
| Hardware | One Wormhole Galaxy (32 chips, 6U, every chip directly attached over PCIe) |
| tt-metal | `a1146e45eef` (Release build), ttnn 0.75.0rc10.dev2278, diffusers 0.38.0 |
| Model | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (revision `5be7df96`) |
| Test | `test_performance_wan.py::test_pipeline_performance[wormhole_b0-t2v-resolution_480p-wh_4x8_sp1tp0]` |
| Generation | 832×480, 81 frames (about 5 s at 16 fps), 40 steps, one prompt (a butterfly and a flower) |
| Parallelism | mesh 4×8, SP=8 (axis 1), TP=4 (axis 0), Ring, 4 links, FSDP enabled (as in the WH preset) |
| Environment variables | `TT_DIT_CACHE_DIR` (cache of converted weights), `TT_DIT_CACHE_VERIFY=1` (only matters on the first run), `TT_METAL_OPERATION_TIMEOUT_SECONDS=600` |

Note: on WH 4×8 this test does no warm-up, so the single measured run also includes compilation and similar work. However, denoise differed by only 0.01 s between the two measurements, so compile time had almost no visible effect here (probably because the kernel cache already existed).

### 8.2 Results

| Stage | 1st run (creates the weight cache) | 2nd run (cache present) | CI target |
|---|---|---|---|
| Text encode | 0.09 s | 0.09 s | 0.23 s |
| denoise (40 steps) | 108.92 s | 108.93 s | 142 s |
| VAE decode | 1.25 s | 1.36 s | 1.55 s |
| **Total** | **110.26 s** | **110.39 s** | **144 s** |
| Total test time | 18 min (about 14 min of it for weight conversion and cache creation) | about 3 min | — |

- 2.72 s per step. Wan 2.2 runs 2 forwards per step because of CFG, so one DiT forward takes about 1.36 s (computed).
- The cache of converted weights is 66 GB. The downloaded weights are 126 GB (about 5.5 min, about 380 MB/s).
- The performance test passed. The generated video looks fine by eye (figure below).
- The log has 33 `[fallback]` warnings, all from the VAE conv3d. No DiT matmul switched to a generic config (there is one notice that "the config for the same M_per_core is reused").

![Frames of the generated video (top left 0 s, top right 1.5 s, bottom left 3 s, bottom right 4.9 s). Prompt: A close-up of a beautiful butterfly landing on a flower, wings gently moving in the breeze.](img/wan22_wh4x8_480p_frames.jpg)

[Generated video (832×480, 81 frames, 16 fps, mp4)](img/wan22_wh4x8_480p.mp4)

### 8.3 Comparison with benchmarks (480p, T2V-A14B, total time per video)

| Setup | Chips / GPUs | Total | Ratio to this run (computed) | Source |
|---|---|---|---|---|
| **This work: WH Galaxy 4×8 (optimized, default)** | **32** | **40.66 s** | 0.37 (2.7x faster than the start; about 1.27x faster than the 51.5 s of 8 × H100) | Measured (#197, no sync) |
| **This work: WH Galaxy 4×8 (optimized, optional fast exp)** | **32** | **39.90 s** | 0.36 (2.76x faster than the start) | Measured (#197, no sync; rare denoise hangs remain, section 10) |
| **This work: WH Galaxy 4×8 (start)** | **32** | **110.3 s** | 1.00 | Measured |
| WH Galaxy 4×8 (back-computed from the CI target) | 32 | about 111 s (144 ÷ 1.3) | 1.0 | test_performance_wan.py |
| BH Galaxy 4×8 (back-computed from the CI target) | 32 | about 58 s (76 ÷ 1.3) | 0.53 (BH is about 1.9x faster) | test_performance_wan.py |
| WH Loud Box 2×4 | 8 | 735 s | 6.7 (this run is 6.7x faster) | Wan2_2.md |
| BH Loud Box 2×4 | 8 | 207 s | 1.9 | Wan2_2.md |
| H100/H800 × 8 | 8 | 51.5 s | 0.47 (8 × H100 is about 2.1x faster) | Official model card |
| H100/H800 × 4 | 4 | 91.7 s | 0.83 | Official model card |
| A100/A800 × 8 | 8 | 119.9 s | 1.09 (this run is slightly faster) | Official model card |
| H20 × 8 | 8 | 170.5 s | 1.55 | Official model card |

How to read this:
- **The result matches what tt-metal expects.** The CI target is the measured value plus a 30% margin, so the back-computed ~111 s and this run's 110.3 s almost match. The environment setup is fine.
- **After optimization (section 10): 40.66 s by default, 39.90 s with the optional fast exp.** That is 2.7–2.8x faster than the initial 110.3 s, and on paper about 1.3x faster than the official 8 × H100 number (51.5 s) (the conditions are not matched; see the caveat below). However, rare hangs in the denoise ring SDPA and in the VAE conv3d still remain.
- **Against GPUs, the initial WH Galaxy (32 chips) is on par with 8 × A100 and about half the speed of 8 × H100.** The conditions are not matched, though. The GPU numbers come from the official implementation with FSDP + Ulysses + FlashAttention3, and the step count could not be confirmed on the model card. The measurements were also taken at different times.
- The "back-computed from the CI target" values are estimates that simply divide out the "30% margin" stated in the code comment. They are not actual measurement logs.
- 720p has not been run yet (for reference: Wan2_2.md lists WH Galaxy 354 s and BH Galaxy 168 s; the official card lists 155.1 s for 8 × H100).

### 8.4 Problems hit before the first run

| Symptom | Cause | Fix |
|---|---|---|
| The test fails after about 20 s (`OfflineModeIsEnabled`) | `HF_HUB_OFFLINE=1` was set to load the weights offline, but diffusers' `from_pretrained(trust_remote_code=True)` queries the Hub API | Unset it. Watched the HF cache size to confirm that no re-download happened |
| The video is not saved (`imageio_ffmpeg not available`) | `imageio-ffmpeg` is not installed in the venv | `uv pip install imageio-ffmpeg` (0.6.0) |
| `hf download` exits with code 1 | After a successful download, the CLI raises `click.exceptions.Exit: 0` as an exception | Checked the sizes of all 49 files against the Hub values to confirm success |

None of these problems involve the device.
As staged checks beforehand, an add on the 32-chip mesh (no communication) and an all_gather on the Ring fabric (4 links, correct values on 32/32 chips) also passed.

[↑ Back to contents](#contents)

## 9. Inside Wan 2.2: module structure and per-block breakdown

### 9.1 Module structure (T2V-A14B)

```
prompt ─→ [UMT5 text encoder] ─→ text embedding (512 tokens × 4096)
noise (latent 16ch × 21 × 60 × 104) ─→ patchify (1×2×2) ─→ 32,760 tokens × 5120
   40 steps (1 step = 2 forwards because of CFG)
     early (high noise): transformer / late (low noise): transformer_2   ← MoE, boundary boundary_ratio = 0.875
       both are 40 layers of DiT blocks (same structure)
─→ [VAE decoder (3D conv, causal in time)] ─→ 832×480 × 81 frames
```

One DiT block (hidden 5120, 40 heads × 128, FFN 13824; checked in the config):
1. **self-attention**: norm (modulated by adaLN) → QKV projection → RMSNorm on q/k → RoPE → Ring SDPA → output projection (residual + gate fused)
2. **cross-attention**: norm → Q projection (video side), K/V projection (text side, 512 tokens) → SDPA → output projection → residual add
3. **FFN**: norm (adaLN) → FFN1 (5120→13824, GELU) → FFN2 (13824→5120) → residual + gate

In the full pipeline, the DiT takes 98.8% of the time (text encoder 0.09 s, DiT 108.9 s, VAE 1.3 s; section 8).

### 9.2 Per-block breakdown (Tracy measurement, 480p, per chip, average over 32 chips, 2026-09-30)

Measurement: one run of `test_wan_transformer_block[wormhole_b0-14b-480p-4x8sp1tp0nl4_ring_is_fsdp1]` with `python -m tracy -r -p -v -m pytest` (PCC 99.9919%).
These are on-device kernel times, so they exclude compilation and host-side time.

**Assumptions:** theoretical compute is 147.5 TFLOPS per chip (72 cores × 2048 FLOP/cycle × 1.0 GHz, HiFi2; the same constants as in the MiniMax-H3 work).
Communication is estimated at 50 GB/s total over 4 links, DRAM at 288 GB/s. Per chip: M = 4,096 rows (32,760 ÷ SP8, rounded up), 10 heads (40 ÷ TP4), full sequence S = 32,768.
Matmul FLOPs are 2·M·K·N; attention FLOPs are 4·M·S·128·(number of heads). All values are computed.

**Compute-bound ops**

| Part | op | Shape (M×K×N) | FLOPs | Theoretical minimum | Measured | Utilization |
|---|---|---|---|---|---|---|
| self-attn | QKV projection (AGMM) | 4096×5120×3840 | 161.1 GFLOP | 1.09 ms | 2.34 ms | 47% |
| | **SDPA (Ring Joint)** | Q 4096 × KV 32768, 10 heads | **687.2 GFLOP** | **4.66 ms** | **14.69 ms** | **32%** |
| | Output projection (AGMM) | 4096×5120×1280 | 53.7 GFLOP | 0.36 ms | 1.12 ms | 33% |
| cross-attn | Q projection (AGMM) | 4096×5120×1280 | 53.7 GFLOP | 0.36 ms | 1.05 ms | 35% |
| | K/V projection (text) | 512×5120×2560 | 13.4 GFLOP | 0.09 ms | 0.40 ms | 23% |
| | SDPA | Q 4096 × KV 512 | 10.7 GFLOP | 0.07 ms | 0.28 ms | 26% |
| | Output projection (AGMM) | 4096×5120×1280 | 53.7 GFLOP | 0.36 ms | 1.05 ms | 35% |
| FFN | FFN1 (AGMM + GELU) | 4096×5120×3456 | 145.0 GFLOP | 0.98 ms | 2.71 ms | 36% |
| | **FFN2 (fused MM+RS, default config)** | 4096×3456×5120 | 145.0 GFLOP | 0.98 ms | **5.18 ms** | **19%** |
| **Subtotal** | | | **1,323 GFLOP** | **8.97 ms** | **28.79 ms** | **31%** |

**Communication- and memory-bound ops**

| op | Bytes moved | Theoretical minimum | Measured | Fraction of bandwidth |
|---|---|---|---|---|
| FSDP weight gather ×7 (QKV 34.4 MB, FFN1/FFN2 31.0 MB each, etc.) | 154 MB total | 3.07 ms (50 GB/s) | 2.89 ms | about 100% (bandwidth limit) |
| Norm ×7 (reads and writes 10.5 MB per call) | 21 MB per call | 0.07 ms per call (288 GB/s) | 0.18–0.26 ms per call | about 30–40% |
| Residual add | 31 MB | 0.11 ms | 0.15 ms | about 75% |
| Head split/merge ×4, etc. | — | — | 0.9 ms | — |

**Whole block:** measured 33.67 ms (the union of all op execution intervals is 33.79 ms) × 40 layers = **1.347 s/forward** (matches the ~1.36 s measured in the pipeline).
Utilization against the compute-only theoretical minimum of 8.97 ms is 27%.

| Category | Time | Share |
|---|---|---|
| self-attention SDPA | 14.7 ms | 43.6% |
| 7 matmuls | 13.8 ms | 41.1% |
| FSDP weight gather | 2.9 ms | 8.6% |
| Norms | 1.4 ms | 4.2% |
| Other | 0.9 ms | 2.5% |

### 9.3 Takeaways (improvement candidates)

1. **FFN2 takes a slow path.** It takes about 2x as long as FFN1, which has the same FLOPs (19% utilization). The log also shows the warning "no fused MM+RS config for the WH 8×9 grid, using defaults" 3,360 times in one run. This is the same kind of problem that PR #56886 fixed for H3. → Optimization #1 in section 10.
2. **The FSDP weight gather runs at the bandwidth limit and is effectively not hidden.** An earlier version of this note said "union ≈ plain sum, so nothing overlaps", but that was how it looked in a single-run measurement (with host waits between ops). **Correction**: in the warm state, running the block 3 times in a row, the weight gather runs in parallel with the matmul that uses it. But the matmul waits until the weights are complete, so the section length barely changes (QKV part: 2.95 ms without overlap → 3.00 ms with overlap; whole block 31.71 → 31.54 ms). To hide it, **weights used later must be gathered ahead of time (prefetch)**, for example during SDPA (14.7 ms). The expected gain is up to 2–2.9 ms per layer (estimate).
3. **SDPA utilization is low at 32%** (48% for the 15 s H3 case). One likely reason is the small number of Q rows per chip (about 4,096). Candidates: tuning the chunk sizes or reworking the kernel.
4. **All main matmuls run with the generic blocking (8,8,8)** (the `No known best blocking` warning in the log). Utilization is 33–47%. A search for WH-specific settings should give a few percent.
5. Small ops such as norms reach 30–40% of the bandwidth limit, but they add up to only 4%, so they have low priority.

### 9.4 What the fused path is (terminology)

With TP=4, the K dimension of FFN2 is split across 4 chips, so it needs the following 3 stages.

| Stage | What it does |
|---|---|
| ① Matmul | Each chip computes with its own 1/4 of K (the results are partial sums) |
| ② reduce-scatter | Add up the partial sums of the 4 chips, split the result into 4 equal parts and distribute them (communication) |
| ③ addcmul | Residual connection (original input + result × gate value) |

- **Unfused**: run the 3 ops in order. Each op writes to DRAM and the next one reads it back. The communication in ② starts only after ① is completely done.
- **Fused** (MM+RS): one kernel sends each matmul tile to the other chips as soon as it is ready and adds it there. Compute and communication overlap, and there are fewer DRAM round trips and op launches.
- **Pitfall**: cores given to communication are taken away from the matmul. The config ("how many cores for the matmul, how many for communication, how many blocks") is found by a search for each grid shape. Without a config, the defaults are used, and they are slow (for H3 on WH, the defaults even caused hangs).
- The same idea on the input side is AGMM (all-gather + matmul).

### 9.5 Current breakdown and theoretical values (as of #15d, 2026-10-02)

Re-measured the current best configuration (#15d, CFG batching) with the same method as section 9.2 (before optimization).

**Assumptions**
- On-device time of one block (Tracy, warm, average over 32 chips). With CFG batching, one block computes both the conditional and the unconditional pass (8,192 rows per chip).
- Block test settings: FFN weights resident (same as the high-noise `transformer`), cross-attention K/V already cached, FFN without fp32 accumulation.
- Theoretical values use the same assumptions as section 9.2 (147.5 TFLOPS per chip (HiFi2), communication 50 GB/s, DRAM 288 GB/s).

**Compute-bound ops**

| Part | op | Shape (both passes) | FLOPs | Theoretical minimum | Measured | Utilization |
|---|---|---|---|---|---|---|
| self-attn | QKV projection (AGMM) | 8192×5120×3840 | 322 GFLOP | 2.18 ms | 4.44 ms | 49% |
| | **ring SDPA (one pass at a time ×2)** | Q 4096 × KV 32768, 10 heads ×2 | **1,374 GFLOP** | **9.32 ms** | **26.12 ms** | **36%** |
| | Output projection + residual (AGMM) | 8192×5120×1280 | 107 GFLOP | 0.73 ms | 2.07 ms | 35% |
| cross-attn | Q projection | 8192×5120×1280 | 107 GFLOP | 0.73 ms | 1.90 ms | 38% |
| | K/V projection | (cached) | — | — | 0 | — |
| | SDPA (B=2) | Q 8192 × KV 512 | 21 GFLOP | 0.15 ms | 0.56 ms | 26% |
| | Output projection | 8192×5120×1280 | 107 GFLOP | 0.73 ms | 1.88 ms | 39% |
| FFN | FFN1 (AGMM + GELU) | 8192×5120×3456 | 290 GFLOP | 1.97 ms | 4.62 ms | 43% |
| | FFN2 (fused MM+RS + residual) | 8192×3456×5120 | 290 GFLOP | 1.97 ms | 4.16 ms | 47% |
| **Subtotal** | | | **2,620 GFLOP** | **17.76 ms** | **45.75 ms** | **39%** |

**Communication, memory and glue ops**

| Category | Contents | Measured | Theoretical estimate |
|---|---|---|---|
| FSDP weight gather | 4 calls (QKV, output projection, cross Q and output). None for FFN because it is resident | 1.56 ms | about 92 MB ÷ 50 GB/s ≈ 1.8 ms (close to the bandwidth limit) |
| Norm (RMSNorm) | 6 calls | 1.82 ms | 42 MB read and written per call ÷ 288 GB/s ≈ 0.15 ms (measured: about 2x, 0.3 ms) |
| CFG batching glue | transpose ×3 (0.54), q/k/v slice ×6 (0.62), concat (0.21) | 1.37 ms | work that should not be needed |
| Head split/merge | 3 calls | 0.61 ms | — |
| Residual etc. | adds and others | 0.37 ms | — |
| **Whole block** | | **51.82 ms** (25.9 ms per pass) | 17.76 ms for compute alone (34% utilization) |

**Share by category**

| Category | Time | Share | Gap to theoretical |
|---|---|---|---|
| ring SDPA | 26.1 ms | 50% | **−16.8 ms** (theoretical 9.3 ms) |
| 6 matmuls | 19.1 ms | 37% | **−10.8 ms** (theoretical 8.3 ms) |
| Norms, glue, head ops, etc. | 4.2 ms | 8% | most of it can be removed |
| FSDP weight gather | 1.6 ms | 3% | close to the bandwidth limit |
| cross SDPA | 0.6 ms | 1% | small |

**Breakdown of one video (480p, 40 steps): 86.90 s**

| Part | Time | Contents |
|---|---|---|
| denoise | 85.59 s | block 51.82 ms × 40 layers × 40 steps ≈ **82.9 s**. The remaining ~2.7 s is the FFN weight gathers of `transformer_2` (14 steps), the timestep embedding and output-side processing, the scheduler, etc. (estimate) |
| VAE decode | 1.21 s | |
| Text encoder | 0.09 s | |

**Theoretical floor (compute only):** at 100% utilization, 17.76 ms × 40 layers × 40 steps ≈ **28 s**. Well-tuned WH matmuls reach 80–93% of theoretical, so a realistic best case (70% utilization) is estimated at about **41 s**. Compared with the current 86.9 s, there is still about 2x of headroom.

**Starting points for discussion**
1. **The largest headroom is in ring SDPA:** it takes half of a block at 36% utilization. Measurements inside the kernel showed that in Phase 1 (QKᵀ and softmax) the pack thread is the bottleneck and the FPU is busy only 49% of the time (section 10, "Per-zone time inside the SDPA kernel").
2. **Matmul utilization is 35–49%:** the blocking search is nearly exhausted, so going further probably needs changes to how matmul and communication overlap (inside AGMM) or to precision (handling of fp32 accumulation).
3. **About 1.4 ms of CFG batching glue:** folding into heads (#16b, implemented but not yet measured) should remove most of it.
4. **Norms:** reach only about half the bandwidth; 1.8 ms in total.
5. **FSDP weight gather:** close to the bandwidth limit, so the only ways to reduce it are "do not gather (keep resident)" or "hide it behind SDPA". However, prefetching before SDPA (#5a) did not help, because SDPA then waits for the communication resources.

### 9.6 Current breakdown and theoretical values (as of #30, 2026-10-05)

Re-measured the current best (#30: CFG batching + LoFi for the SDPA matmuls + polynomial tanh for GELU) with the same method as section 9.5 (the test Galaxy, warm, average over 32 chips).

**Assumptions:** peak per chip (72 cores, 1 GHz) is 147.5 TFLOPS with HiFi2 and 295 TFLOPS with LoFi. The SDPA matmuls use LoFi since #21, so the SDPA theoretical value is computed for LoFi (the HiFi2 value is listed too). With CFG batching, one block computes both the conditional and the unconditional pass.

**Compute-bound ops**

| Part | op | FLOPs | Theoretical minimum | Measured | Utilization |
|---|---|---|---|---|---|
| self-attn | QKV projection (AGMM, HiFi2) | 322 GFLOP | 2.18 ms | 4.43 ms | 49% |
| | **ring SDPA ×2 (LoFi matmuls)** | **1,374 GFLOP** | **4.66 ms** (9.32 with HiFi2) | **24.08 ms** | **19%** (39% in HiFi2 terms) |
| | Output projection + residual (AGMM) | 107 GFLOP | 0.73 ms | 2.00 ms | 36% |
| cross-attn | Q projection (AGMM) | 107 GFLOP | 0.73 ms | 1.89 ms | 39% |
| | SDPA (B=2, HiFi2) | 21 GFLOP | 0.15 ms | 0.56 ms | 26% |
| | Output projection (AGMM) | 107 GFLOP | 0.73 ms | 1.88 ms | 39% |
| FFN | FFN1 (AGMM + GELU) | 290 GFLOP | 1.97 ms | 4.12 ms | 48% |
| | FFN2 (matmul + reduce-scatter) | 290 GFLOP | 1.97 ms | 4.20 ms | 47% |
| **Subtotal** | | **2,620 GFLOP** | **13.12 ms** | **43.16 ms** | **30%** |

**Breakdown of the SDPA bottleneck (reference)**
- exp (SFPU, pack thread): the approximate exp takes about 68 cycles per tile. 1.31 million tiles per pass ÷ 72 cores ≈ **1.24 ms** (assuming it can overlap with the matmuls).
- Changing the matmul fidelity in single-chip SDPA (same shape, one pass): LoFi 13.24 ms, HiFi2 13.14 ms, HiFi4 17.93 ms. LoFi and HiFi2 take the same time, so **the current bottleneck is not the FPU** (it is pack, unpack and the hand-off between threads). The FPU becomes the bottleneck only at HiFi4.
  - Note: in the single-chip tests so far, the matmul fidelity setting had no effect (the single-chip SDPA op did not read `matmul_math_fidelity`). This was confirmed on the machine by adding the same wiring as in ring. The single-chip values in #27 and #32 are HiFi2 values.

**Communication, memory and glue ops**

| Category | Contents | Measured | Theoretical estimate |
|---|---|---|---|
| FSDP weight gather | 4 AllGathers | 1.46 ms | close to the bandwidth limit |
| Norm (RMSNorm) | 6 calls | 1.82 ms | about 0.9 ms (about half the bandwidth) |
| CFG batching glue | transpose ×3, slice ×6, concat | 1.35 ms | should not be needed |
| Head split/merge | 3 calls | 0.58 ms | — |
| Residual etc. | adds and others | 0.36 ms | — |
| **Whole block** | | **49.13 ms** | 13.12 ms for compute alone (27% utilization) |

**Share by category**

| Category | Time | Share | Theoretical | Gap to theoretical |
|---|---|---|---|---|
| ring SDPA | 24.1 ms | 49% | 4.7 ms | **−19.4 ms** |
| 6 matmuls | 18.5 ms | 38% | 8.3 ms | **−10.2 ms** |
| Norms, glue, head ops, etc. | 4.1 ms | 8% | about 1 ms | about −3 ms |
| FSDP weight gather | 1.5 ms | 3% | close to the limit | — |
| cross SDPA | 0.6 ms | 1% | 0.15 ms | −0.4 ms |

**One video (480p, 40 steps): 81.84 s** = denoise 80.56 s (block 49.13 ms × 1,600 ≈ 78.6 s + about 2 s of other work) + VAE 1.19 s + text encoder about 0.1 s.

**Theoretical floor:** compute alone is 13.12 ms × 1,600 ≈ **21 s** (about 30 s at 70% utilization). About 2/3 of the gap to the current 81.8 s is in ring SDPA.

### 9.7 Current (#77) vs. baseline (2026-10-07)

**Current (#77) breakdown and theoretical values:** taken from the raw logs of the block test with the #77 configuration (one block for one step, excluding the FSDP gather that exists only in the block test). Theoretical values assume 8×8 cores per chip at 1 GHz, LoFi matmuls (16 cycles per tile product), and DRAM at 288 GB/s for memory-bound ops.

| Module | Measured (ms) | Theoretical (ms) | Efficiency | Likely bottleneck |
|---|---|---|---|---|
| ring SDPA (self-attn) | 18.03 | 5.24 (matmuls only) / 8.0 (if exp is also serial) | 29–44% | relaying the K/V chain and the non-matmul parts of softmax (breakdown below) |
| QKV projection (AGMM) | 2.02 | 1.23 | 61% | compute + in0 hand-off |
| FFN1 (AGMM + GELU) | 2.38 | 1.11 | 47% | compute + DM |
| FFN2 (matmul + reduce-scatter + residual) | 2.62 | 1.26 | 48% | compute + reduce-scatter transfers |
| self-attn output projection (+ residual) | 1.30 | 0.41 | 32% | in0 (activation) supply |
| cross-attn Q projection | 1.19 | 0.41 | 34% | same |
| cross-attn output projection (+ residual) | 1.28 | 0.41 | 32% | same |
| LayerNorm ×3 | 0.91 | 0.33 | 36% | DRAM bandwidth |
| RMSNorm (Q, K, cross Q) | 0.59 | 0.24 | 41% | same |
| cross-attn SDPA | 0.40 | 0.08 | 20% | fixed cost of a small op |
| Head creation, concat, transpose, etc. | about 0.8 | about 0.3 | — | memory copies |
| **Total** | **about 31.6** | **about 13.8** | **44%** | |

40 steps: denoise 52.4 s (theoretical estimate about 22.6 s), VAE 0.95 s, text encode 0.08 s, total 53.47 s.

The comparison is for one block of one step (both the conditional and the unconditional CFG pass). The baseline predates CFG batching (one forward pass at a time), so the section 9.2 measurements are doubled. Current values come from the raw logs of the block test with the #77 configuration (excluding the FSDP gather that exists only in the block test). Theoretical values assume HiFi2 and BF16 for the baseline and LoFi and BFP8 for the current version (8×8 cores per chip, 1 GHz, 16 cycles per matmul tile, DRAM 288 GB/s).

| Module | Baseline (ms) | Current (ms) | Reduction | Theoretical baseline → current (ms) |
|---|---|---|---|---|
| ring SDPA (self-attn) | 29.38 | **18.03** | −39% | 9.32 → 5.24 |
| QKV projection | 4.68 | 2.02 | −57% | 2.18 → 1.23 |
| self-attn output projection (+ residual) | 2.24 | 1.30 | −42% | 0.73 → 0.41 |
| cross-attn Q projection | 2.10 | 1.19 | −43% | 0.73 → 0.41 |
| cross-attn K/V projection (text) | 0.80 | 0 (cached) | −100% | 0.18 → 0 |
| cross-attn SDPA | 0.56 | 0.40 | −29% | 0.15 → 0.08 |
| cross-attn output projection (+ residual add) | 2.40 | 1.28 | −47% | 0.73 → 0.41 |
| FFN1 (+ GELU) | 5.42 | 2.38 | −56% | 1.97 → 1.11 |
| FFN2 (matmul + reduce-scatter + residual) | 10.36 | 2.62 | −75% | 1.97 → 1.26 |
| FSDP weight gather | 5.78 | 0 (all resident) | −100% | — |
| Norms | 2.80 | 1.50 | −46% | 0.5 → 0.57 |
| Glue such as head split/merge | 1.80 | about 0.8 | −56% | — |
| **Block total** | **67.34** | **about 31.6** | **−53%** | 17.9 → 13.8 |

| One generation | Baseline | Current | Reduction |
|---|---|---|---|
| Denoise | 108.9 s | 52.4 s | −52% |
| VAE decode | 1.25–1.36 s | 0.95 s | −27% |
| Text encode | 0.09 s | 0.08 s | — |
| **Total** | **110.39 s** | **53.47 s** | **−51.6% (2.06x faster)** |

**ring SDPA breakdown (timing-only variant, #77 configuration):** skipping the matmuls (QKᵀ, P·V) gives 16.37 ms, skipping exp gives 15.85 ms, and skipping both gives 14.87 ms (normal: 18.03 ms). The matmuls and exp together take about 3 ms; the remaining ~15 ms is K/V supply and the rest of softmax. Reader wait times (median of one core) in the normal run: waiting for compute to free space K 3.5, V 1.9 ms; waiting to receive from the chain K 4.0, V 3.1 ms; forwarding to the chain K 2.2, V 2.1 ms. With the matmuls and exp skipped, the wait for compute drops to almost 0 and forwarding grows to K 3.8, V 3.8 ms. With chunk (128, 1024), each core handles 10 Q chunks for different heads, so one op receives the K/V of 10 heads (about 89 MB) and forwards it to the next core. **When compute gets lighter, relaying the K/V chain (store-and-forward) becomes the bottleneck** (the DM-only floor is about 14.8 ms).

### 9.8 Current (#107) breakdown and theoretical values (2026-10-08)

Recorded a full 40-step run with the device profiler (`WAN_PROF_FLUSH_EVERY=5`) and took the per-op median over the 40 blocks of the 4th forward (a steady-state step) on chip 0. Units are M cycles (same as ms at 1 GHz). Theoretical values use the same assumptions as section 9.7 (LoFi, BFP8, 8×8 cores, 1 GHz, DRAM 288 GB/s).

| Module | #77 (section 9.7) | Current #107 | Theoretical | Efficiency |
|---|---|---|---|---|
| ring SDPA (self-attn) | 18.03 | **17.68** | 5.24 (matmuls only) / 8.0 (if exp is also serial) | 30–45% |
| QKV projection (AGMM) | 2.02 | 2.05 | 1.23 | 60% |
| FFN1 (AGMM + GELU) | 2.38 | 2.01 | 1.11 | 55% |
| FFN2 (matmul + reduce-scatter + residual) | 2.62 | 2.59 | 1.26 | 49% |
| self-attn output projection (+ residual) | 1.30 | 1.23 | 0.41 | 33% |
| cross-attn Q projection | 1.19 | 1.09 | 0.41 | 38% |
| cross-attn output projection (+ residual) | 1.28 | 1.22 | 0.41 | 34% |
| Norms ×6 (LayerNorm ×3, RMSNorm ×3) | 1.50 | 1.41 | 0.57 | 40% |
| cross-attn SDPA | 0.40 | 0.37 | 0.08 | 22% |
| Head ops, type casts, adds, etc. | about 0.8 | 0.20 | about 0 | — |
| Gaps between ops | — | 0.12 | 0 | — |
| **Total** | **about 31.6** | **30.0** | **10.7–13.5** | **36–45%** |

- The 6 matmuls add up to 10.2 (theoretical 4.8). ring SDPA is 59% of the block.
- The clock under load is about 0.98 GHz (some chips drop to 925 MHz; see the row below #108 in section 10), so in wall time one block takes about 30.6 ms and 40 blocks take 1,224 ms. About 14 ms is spent outside the blocks, so one step is about 1,236 ms.

| One generation | Measured | Theoretical (100% efficiency) | Realistic best case (70% efficiency) |
|---|---|---|---|
| Denoise | 49.64 s | 17.2–21.6 s | about 25–31 s |
| VAE, text encode, etc. | 0.6 s | — | — |
| **Total** | **50.25 s** (2.20x faster than the baseline) | **about 18–23 s** | **about 26–32 s** |

**As of #117 (2026-10-09):** 29.68 M cycles per block (the larger kernel config region in #109 removed the gaps between ops, #114 cut the small modulation ops from 10 to 6, and #117 removed the type-cast op after ring SDPA). About 1,223 ms per step, denoise 49.11 s, total **49.73 s** (2.22x faster than the baseline). ring SDPA (17.63) and the 6 matmuls (about 10.2) take 94% of a block, and the bottleneck of both is confirmed (for ring SDPA, per-thread cycle measurements show all 3 threads busy 80–90% of the time; the matmuls are bound by unpack; #115, #116).

**As of #126 (2026-10-09):** cutting the pack thread's writes and the SFPU waits in ring SDPA (#118–#120) brought ring SDPA to about 16.4 ms, and the fused LayerNorm (#123, #125, #126) went from about 300k to about 220k cycles per call. About 30.9 ms per block (block test), about 1,162 ms per steady-state step, denoise 46.62 s, total **47.22 s** (2.34x faster than the baseline).

**As of #128 (2026-10-09):** the device profile of a steady-state step (#126 configuration) shows 28.24 M cycles per block: ring SDPA 16.33 (58%), 6 matmuls 10.17 (36%), 6 norms 1.26, cross SDPA 0.37. #128 cut ring SDPA by another ~0.24 ms: about 1,154 ms per steady-state step, denoise 46.35 s, total **46.96 s** (2.35x faster than the baseline). Units are M cycles (one block; ring SDPA is the #126 profile value minus the 0.24 saved by #128). Theoretical values use the same assumptions as the table above.

| Module | #107 | Current #128 | Theoretical | Efficiency | Bottleneck (confirmed) |
|---|---|---|---|---|---|
| ring SDPA (self-attn) | 17.68 | **about 16.09** | **8.0** (matmuls 5.24 + exp about 2.8; on WH the SFPU exp and the FPU matmul barely overlap, #35) | 50% | number of tiles unpacked (all 3 threads busy 80–90% of the time) |
| QKV projection (AGMM) | 2.05 | 2.01 | 1.23 | 61% | data movement (floor 1.92) |
| FFN1 (AGMM + GELU) | 2.01 | 2.01 | 1.11 | 55% | compute (floor 1.95) |
| FFN2 (matmul + reduce-scatter + residual) | 2.59 | 2.61 | 1.26 | 48% | reduce-scatter and sync between chips |
| self-attn output projection (+ residual) | 1.23 | 1.23 | 0.41 | 33% | data movement (floor 1.01–1.13) |
| cross-attn Q projection | 1.09 | 1.09 | 0.41 | 38% | same |
| cross-attn output projection (+ residual) | 1.22 | 1.22 | 0.41 | 34% | same |
| Norms ×6 (LayerNorm ×3, RMSNorm ×3) | 1.41 | 1.26 | 0.57 | 45% | wait in the all-gather of the statistics (start-time skew between chips) |
| cross-attn SDPA | 0.37 | 0.37 | 0.08 | 22% | — |
| Small modulation ops, type casts, etc. | 0.20 | 0.03 | about 0 | — | — |
| Gaps between ops | 0.12 | about 0.01 | 0 | — | — |
| **Total** | **30.0** | **about 27.9** | **13.5** | **48%** | |

| One generation | Measured (#128) | Theoretical (100% efficiency) | Realistic best case (70% efficiency) |
|---|---|---|---|
| Denoise | 46.35 s (about 1,154 ms per step) | about 21.6 s | about 31 s |
| VAE, text encode, etc. | 0.6 s | — | — |
| **Total** | **46.96 s** (2.35x faster than the baseline) | **about 22 s** | **about 32 s** |

**Second look at the parts below 40% efficiency (2026-10-09):** the theoretical values count compute only, not communication or DRAM reads.
- The 3 output-projection-style AGMMs (1.09–1.23 ms, theoretical 0.41 ms) are set by the floor of the data movement inside the op (1.0–1.15 ms, #80, #89). The #132 measurements showed that the bottleneck is not the fabric but the serial processing in each row's injector; moving the output writes to the in1 side saved about 21 ms per step. Breakdown: the fabric all-gather alone is 0.59 ms (about 33 MB in BF8, saturating at about 58 GB/s, measured on its own), and distributing the data to the cores inside the chip adds 0.4–0.55 ms on top. The compute floor (about 23 cycles per matmul tile + about 23% for post-processing) is 0.86–0.96 ms, so the realistic floor is about 0.9 ms (current efficiency against it: 75–85%). Tightening the on-chip distribution could save 0.15–0.3 ms per op, but that means reworking the CCL inside AGMM.
- cross-attn SDPA (0.37 ms, theoretical 0.08 ms): skipping the matmuls and exp saves only about 0.06 ms per block, and K/V is already shared between cores through the non-causal chain (about 14 MB of DRAM reads per call). Reusing K/V inside a core across the same head does not make it faster (#131). The 320 work units (20 heads × 16 Q chunks) are split 4–5 per core over 72 cores, so the cores with 5 units set the total (median 323 µs, max 398 µs). One unit (128 S tiles) takes about 80 µs, i.e. about 500 cycles per S tile, which is the same per-tile hand-off overhead as in ring SDPA. There is no easy improvement left.

**Optional fast exp (#138, 2026-10-09):** the softmax exp in ring SDPA normally takes 2 instructions per row (LOADMACRO and SHFT2), about 2.1 cycles. #138 places the multiply-add result at the top bits of FP32 so that out-of-range inputs overflow to ∞. This makes it 1 instruction per row (about 1 cycle) and also removes the limit on the input range. ring SDPA goes from 16.12 → 15.40 ms (one block), about 1,100 ms per steady-state step, total **44.76 s** (2.47x faster). The approximation is the same as the standard one (Schraudolph); only the rounding position differs. Because it affects quality, it is off by default (enable it with `TT_RING_SDPA_EXP_OVF=1`).

**ring SDPA breakdown (on top of #138, 2026-10-09):** of 15.98 M cycles per core, unpack spends 8.4 on the QKᵀ and P·V matmuls, 1.8 on sub+exp, 0.65 on the max, and 0.73 on the correction. Pack spends 7.06 on sub+exp (writing out P and the row sums) and 2.32 on the correction factor (time the SFPU waits for the FPU to be free). Upper bounds measured by skipping work (measurement only): −1.05 ms for the max of the 2nd and later chunks, −0.70 ms for the correction factor and the correction, −1.86 ms (−12%) for both. However, in real data, layers 39 and 40 have heads whose row max jumps by more than 100 nat across chunks, so the max must be computed for every chunk (#139). What remains is skipping the correction only for chunks where the max barely grows (upper bound −0.70 ms).

**FFN2 on the matmul cores only (#142, 2026-10-10):** the fused FFN2 op (2.55 ms) was bound by its matmul, which ran on 56 cores because the RS workers and the mux took 16 cores. #142 runs the matmul on all 72 cores: in each ring iteration, each chip computes its own slice and adds the partial sums from the neighbor chip, and the cores in row 9 send the per-column partial sums directly to the same cores on the neighbor chip. 2.30 ms per call. Then #143 splits the in0 reads across the 4 cores of a row group and does the DRAM reads on NOC0 (reads on NOC1 are about 3x slower per tile), reaching 1.97 ms, almost the same as the matmul alone (transposed grid) at 1.92 ms.

**Removing the dedicated all-gather cores of ring SDPA (#150, 2026-10-10, parked):** when the writers of the SDPA cores relay K/V between chips one hop at a time and SDPA runs on 72 cores, ring SDPA goes from 15.39 → 13.89 ms (about 1.4 s saved per video). However, long tests rarely changed the results and the cause could not be found, so it was not adopted (#150 in section 10).

**8×10 grid with Ethernet dispatch (#154, 2026-10-10, optional):** WH Galaxy reserves Tensix row 10 (8 cores) for dispatch, so ops can use only 8×9. Running dispatch on Ethernet cores would give 8×10, but on a Galaxy all 16 Ethernet links per chip are connected (torus) and used by the fabric, so no core is free. So on the SP axis (the ring of 8 chips), 2 of the 4 Ethernet links of each connection are taken out of the fabric, and dispatch runs on those Ethernet cores. ring SDPA runs on 72 cores (Q chunks per core 10 → 9) plus the all-gather on row 10, and the QKV and FFN1 matmuls split N across 9 rows. One block (block test without FSDP) 26.47 → 25.19 ms; 40 steps 45.00 → 43.01 s (42.14 s with the fast exp). To keep 3 links on the SP axis: WH's ring SDPA all-gather (which splits each head's contiguous page range across the links) changes the results with 3 links, so this was fixed by using the Blackhole send scheme ("split the work by DRAM bank") on WH too (#155, 42.68 s, 41.85 s with the fast exp). Next, a race was fixed in which the function that writes the matmul (AGMM) output one row at a time freed the CB without waiting for its writes to complete (#156); with QKV and FFN1 also using row 10, 41.87 s (41.02 s with the fast exp). In this form, two 40-step runs diverged from step 32, so the kernel config region is not enlarged when dispatch_s is absent (#157, 42.00 s, 41.19 s with the fast exp). Finally, a wrong env var check had kept the bank-owned send scheme of the 3-link SP all-gather (#155) from taking effect; this was fixed (#162, **41.81 s**, **41.01 s** with the fast exp; chained block test 5/5 matching).

**Compressing the text for cross-attn (#164, 2026-10-11):** the text encoder pads the prompt with zeros up to 512 tokens, and cross-attn attends to all of them without a mask. The K/V at the padding positions are all identical, so K/V is packed into 256 entries, and the padding in columns 16–31 of each tile counts with weight 3 (the weight is applied to the `col_identity` column of the SDPA row sum and to the V rows; mathematically identical). cross-attn SDPA goes from 0.31 → 0.18 ms per block; 40 steps 41.81 → **41.64 s** (41.01 → **40.80 s** with the fast exp).

**Larger config region and K/V alternation (#165, 2026-10-11):** the reason the config region enlargement was removed in #157 (rare result divergence) turned out to be a race that already exists in the default configuration, with a frequency that depends on the device state. Alternating with the baseline, all 6 runs with +64 KB matched (the baseline diverged in 1 of 6 runs and hung once), so the enlargement was restored at +64 KB. In the same change, a defect was fixed in the default: the K and V all-gather buffers of ring SDPA have the same shape, so they competed for the same ping-pong pair and did not alternate across calls. 40 steps 41.64 → **41.49 s** (40.80 → **40.70 s** with the fast exp); the latents are identical.

**Batched timestep embedding (#166, 2026-10-11):** the per-step fp32 timestep MLP (M = 1; mostly reading the 5120×30720 weights, about 3.1 ms per step) is computed once, with the timesteps of all steps stacked as rows (bit-identical results). With the run script's per-step sync, the start of each step is bound by the host and this cost was hidden behind it (no effect in #159). Without sync it goes 41.32–41.36 → 41.25–41.27 s, and with sync 41.49 → 41.43 s.

**Folding the row sum into P·V (#167, 2026-10-11):** the ring SDPA pack thread built the softmax row sum by packing each P tile with accumulation into L1 (128 times per set), which cost as much as the exp and P packs. If the P·V matmul multiplies by a "tile with 1 only in column 0" as a 5th column after the 4 V tiles, it reuses the P already in srcB and produces the row sum in column 0 of the 5th DEST tile (one pack per row). Unpack and matmul grow by about 1/4 of the P·V part. In the same change, the pack thread's SFPU work (exp and correction factor) is now issued only after the pack TRISC waits on a dedicated semaphore that math raises after its commit (with the row-sum change alone, a hang in which the math commit and the pack thread's SFPU preamble waited for each other appeared in 3 of 9 runs). With sync 41.43 → **41.00 s**; without sync 41.26–41.31 → **40.79–40.85 s**. The result divergence did not appear in any of 13 runs in one time window, but came back in a later window (e.g. 2 of 3 runs), so it is not gone.

**Row-major VAE residuals (#174, 2026-10-12):** the VAE decoder takes about 112 ms (device time) per 5-frame chunk: conv3d 41%, neighbor pad 18%, and RMSNorm, the concat with the cache, and tilize / untilize 6–8% each. Inside conv3d, the heavy parts are the reads that rearrange the 27 sticks of the receptive field in L1 for each output position (vol2col), and the compute-side tilize and matmul unpack (see the measurement row). In residual blocks without a 1×1 shortcut, keeping the inputs and outputs row-major (RMSNorm and the add read row-major directly; bit-identical) removes the untilize after norm1, the tilize before the add, and the tilize after the upsampler. Chunks with a cache 118.4 → 111.9 ms; e2e VAE 0.51 → 0.49 s.

**VAE cache with halo (#175, 2026-10-12):** the causal conv (3×3×3) kept the last 2 frames of the previous chunk's input as a cache, concatenated them in front of the next chunk's input, and then added the halo with neighbor pad (the concat copies the whole input once and neighbor pad copies it again). The last 2 frames of the neighbor pad output (with halo) are exactly the first frames of the next chunk, so those are kept as the cache, and a new `front_tensor` input of neighbor pad copies them page by page in place of the leading zero frames (the W-direction exchange is not needed for the leading frames either). The concat disappears; chunks with a cache 111.9 → 103.5 ms; e2e VAE 0.49 → 0.46 s.

**Splitting the queries of the VAE attention (#176, 2026-10-12):** the mid-block attention (per frame, 1 head, 6,240 tokens) distributed the frames over the 8 chips of the W axis, and the 4 chips of the H axis in the same column all computed all 6,240 queries of the same frame (for a 1-frame chunk, all 32 chips did the same computation). If each chip computes only its own H rows (1,560 queries), or only its own H×W share (195 queries) for a 1-frame chunk, the output is already in its own share, and the H-direction re-split and the W-direction re-gather also get smaller. VAE test 180.2 → 173.3 ms (SDPA 7.87 → 2.31 ms); e2e VAE 0.460 → 0.445 s.

**Norm output into the conv's halo buffer (#177) and neighbor pad scatter packets (#178), 2026-10-12:** the RMSNorm (+ SiLU) that produces the causal conv input writes its row-major output directly into the interior of the neighbor pad's halo buffer, and neighbor pad fills in only the halo in place (the full copy of the conv input disappears). The W-direction sends that bound neighbor pad used one packet per stick (192 B), and the send rate was set by the number of buffer slots in the sender's EDM (batching the reader's reads did not change it), so the sticks sent to a neighbor are now grouped 4 at a time into fabric scatter packets. On the chips at the W edge, the reader writes the zero columns at the edge directly (the edge chips alone were slow because of the per-row hand-off of zero sticks). 5-frame chunk 99.7 → 88.1 ms (neighbor pad 19.7 → 8.1 ms); e2e VAE 0.447 → 0.410 s.

**VAE input and output (#180, 2026-10-12):** the e2e VAE (with sync) breaks down into input 26 ms, decoder 371 ms, and readback to the host 31 ms (DMA 10 ms, assembly 17 ms). The assembly (97 MB) alternates between 2 output buffers, and the 2nd buffer was first created and zero-filled on the 2nd call, i.e. in the real run after the allocation run (in a standalone benchmark the assembly takes 5 ms). Creating both on the 1st call gives 17.4 → 8.6 ms. On the input side, the float32 → bf16 conversion in `ttnn.from_torch` is slow (7.7 ms on the host vs. 0.1 ms in torch, same rounding); converting to a contiguous bf16 tensor in torch before passing it in gives 21.1 → 13.1 ms.

**All-to-all for the VAE attention (#181) and block-wise input upload (#182), 2026-10-12:** before handing one frame to each of the 8 chips of the W axis, the mid-block attention had every chip gather the chunk's 5 frames in both the H and W directions (the W-direction all-gather alone took 2.6 ms, about 21 MB per chip). Now each frame is sent to its owner chip with an all-to-all on the W axis, only the H direction is gathered, and a row-major permute restores raster order (the output uses the reverse all-to-all). 5-frame chunk 88.2 → 83.9 ms. For the input, instead of splitting on the host with ShardTensor2dMesh (about 9 ms), torch lays out each chip's piece as a contiguous block and the tensor is distributed along dim 0 (1.6 ms).

**Fusing the VAE residual add into conv (#189), 2026-10-13:** of a 5-frame chunk (about 81 ms on device), conv3d takes 46 ms, and inside it the bottleneck is compute (tilize and the HiFi2 matmul). Skipping the reader's vol2col copy gains only −3 to −6% (#187), so a rewrite to implicit GEMM is not worth it. The final add of each residual block (BinaryNg, reading and writing DRAM in row-major) takes 4.3 ms per chunk, of which the 3 at the top level (96 channels, 20×120×104 per chip) take 2.7 ms. A `residual_tensor` input was added to conv3d: the reducer's writer reads the residual rows of each output block into a CB in the same order as the output, and compute tilizes them and adds them after the bias and before the untilize. conv2 goes 2018 → 2074 µs, the 3 adds (about 0.9 ms each) disappear, and VAE goes 357.8 → 348.3 ms. The remaining adds (the 192- and 384-channel layers) have multiple C_in blocks with fp32 partial sums, so the result is first rounded into a bf16 CB with the same read path as untilize (copy to srcA) and then added (#190, bit-identical). The 192-channel conv2 costs +90 µs; VAE 348.6 → 344.9 ms. Reworking the host-side input and output (#188) made no difference without sync.

**Row-major RMSNorm (#191), 2026-10-13:** the RMSNorm (+ SiLU) before the causal conv runs 7 passes per tile row (3 tiles for 96 channels): tilize → square → row sum → eps and rsqrt → multiply by rsqrt → multiply by gamma and SiLU → untilize. Lowering the FPU fidelity changes nothing (#186), and removing SiLU gives −21%, so the SFPU and the fixed cost per pass dominate. The gamma (+ SiLU) pass now writes directly to the row-major output with `pack_untilize` (the tiled output CB is bf16, the same format as the output, so the values are the same), the block width is made a divisor of Wt to remove padding tiles, and the per-tile-row init touches only DST and the packer (the last two had been skipped in #179 as under 2%). Norm at T = 6: 11.69 → 10.78 ms (−8%).

**Spatial conv after upsample (#192), 2026-10-13:** after the 2x spatial upsample (nearest neighbor) in the up blocks, the high-resolution tensor was given a halo with neighbor pad before the (1,3,3) conv (the upsample itself takes 1.03 ms per chunk, and the high-resolution neighbor pad, which also does an internal copy, takes 1.71 ms). An `input_upsample=2` option was added to conv3d: the input stays the low-resolution tensor with halo, and when the reader gathers input pieces, it maps row/column Y of the high-resolution tensor with halo to (Y+1)/2 at low resolution (the same formula also works for the halo rows and columns; the vol2col assembly is unchanged). The upsample op disappears and neighbor pad runs at low resolution in 0.69 ms. VAE 342.2 → 334.2 ms.

**Residual stream at latent resolution (#195) and neighbor pad breakdown, 2026-10-13:** the decoder converted to tiles before the mid block, so only in the layers at 15×13 per chip with 384 channels (the 5 residual blocks of the mid block and the first up block) was the residual stream tiled, and the fusions of #177, #189 and #190 did not apply there. Passing it in row-major and having the a2a attention also add the residual in row-major cut the ops per 5-frame chunk from 212 → 193 and the device time from 73.15 → 72.68 ms. Looking at the zones of the top-level in-place neighbor pad (498 µs): the W-direction sends (set by the number of packets) take about 485 µs, and the H direction takes about 305–343 µs in parallel. Grouping the W sends in the receiver's L1 would make H the floor and save only about 150 µs per call, so it was not pursued.

**Gaps between ops (0.12 ms per block):** most are 0.6 µs, but the gap from cross-attn SDPA → the output projection AGMM is 32 µs, from the q/k norm → ring SDPA 22 µs, and the 3 gaps before norms are 5–7 µs. In the 32 µs gap, the 72 cores of the AGMM all start at the same time (within 0.4 µs), so it is a dispatch wait, not a kernel wait. The Tensix kernel config region (a ring buffer that holds kernel binaries and runtime args) is 69 KB by default on WH, and the kernel binaries are about 60 KB for ring SDPA, about 50 KB for cross-attn SDPA, about 29 KB for norms and about 18 KB for AGMM. When an op and the previous op together exceed 69 KB, the next op's binaries are written only after the previous op finishes. → #109 in section 10.

[↑ Back to contents](#contents)

## 10. Optimization history (Wan 2.2 T2V 480p, WH Galaxy 4×8)

> **Current status (updated 2026-10-08)**
> - **Latest (2026-10-13):** Rebased the branch `hkato/wan-wh-opt` onto upstream main (`e7c94bbed8b`, 480 commits past the fork point) (the commit hashes in this section are from before the rebase). Resolved conflicts in 7 of the 80 commits (new arguments on the main side in `linear.py`, `feedforward.py` and `manager.py`, the main-side rewrite of the ring SDPA compute, the argument indices of the conv3d writer, and the mid-kernel initialization in LayerNorm, which main had already removed). Two fix commits after the rebase (a renamed SDPA window check, and the `#endif` nesting in `compute_streaming.hpp`). The VAE output is bit-identical to before the rebase, the fast exp e2e video also has a 0.0 difference from the reference, and e2e is 39.973 / 39.890 s with fast exp and 40.712 s with the default (VAE 0.336 s). One more fast exp run hung at denoising step 8 (triage could not be captured, because the debugger version that main requires (tt-exalens 0.4.1) was not installed yet). Classified the Wan optimization items by whether they apply to MiniMax-H3 (MiniMax-H3 work is tracked separately). Before that, a hang bisection (the investigation row below) showed that the VAE hang only occurs when conv3d runs on the 8×10 grid of Ethernet dispatch (row 9), so on WH Galaxy conv3d is now limited to 9 rows by default (#197; `WAN_VAE_C3D_GRID_Y=0` goes back to 8×10). On 8×9 the blocking from before #185 is faster (VAE alone 410 → 353 ms), so grids with fewer than 10 rows use it. The VAE output is bit-identical, VAE 0.330 → 0.340–0.343 s, with fast exp 39.902 / 39.909 s, default 40.720 / 40.658 s (commit `60736588c4c`; 1 of 5 runs hung in the denoising ring SDPA). The night before (after 21:00 machine time), the VAE hang (re-reading triage: one core in row 9 of one chip stops in the middle of conv3d or RMSNorm, and the other chips wait in the next neighbor pad) occurred once every few to 100+ decodes of 81 frames, even in the #185 state without today's changes (the 2026-10-12 neighbor pad changes #175, #177, #178 still included), and the denoising ring SDPA hangs continued. The cause is not identified (recounting the logs, that day's VAE hangs were clustered in two time windows). The optional on-device latent hand-off #196 (bit-identical, `WAN_LAT_DEVICE=1`) stays disabled by default. Before that, #195, which also keeps the residual stream of the latent-resolution layers (the mid block and the first up block) in row-major, gave VAE 0.330 s (bit-identical) and 39.968 / 39.975 s with fast exp (commit `375f45d2e63`). Per-chunk reads (#193) and reads with larger pages (#194) had no effect and were rejected. Before that, #192, where the spatial conv after upsample reads the low-resolution tensor directly (conv3d `input_upsample=2`), gave VAE 0.333 s (bit-identical), **39.888** / 39.982 s with fast exp, default 40.632 s (commit `05dcc613f76`, no hangs). Before that, #191, which fuses the untilize of the row-major RMSNorm into the last pass and removes the padding tiles and the per-row re-initialization, gave VAE 0.341 s (bit-identical) and 39.982 / 39.982 s with fast exp (commit `c088a418610`). One run of the default config hung in the denoising ring SDPA (it happens with the default too). Before that, #190, which also for conv2 with multiple C_in blocks (192 and 384 channels) rounds to bf16 before adding the residual, gave VAE 0.346 s, 39.924 / 39.998 s with fast exp, default 40.656 / 40.665 s (commit `1860a57323d`, no hangs). Before that, #189, which fuses the final add of the residual block into the output side of the second conv (conv3d `residual_tensor`), gave VAE 0.348 s (bit-identical, 120 of 120 stress test runs matched), 39.915 / 40.005 s with fast exp, default 40.662 / 40.677 s (commit `1d717855da4`). The host-side I/O rework #188 showed no difference in unsynchronized timing and was rejected; a timing run that skips the conv3d vol2col (#187) confirmed that compute is the bottleneck. On 2026-10-13, 17:42–18:01, 3 of 5 e2e runs hung in the denoising ring SDPA (fast exp, unrelated to the VAE). Before that, on 2026-10-12: #185, which re-searches the conv3d blocking on the 8×10 grid, gave VAE 0.36 s, 39.922 / 39.938 / 39.995 s with fast exp (commit `2d210b9eb8a`). Before that, #181 (distribute the attention frames with all-to-all) and #182 (block-wise input distribution) gave VAE 0.37 s (commit `c18627970be`), 39.934 / 39.943 / 40.021 s with fast exp. The kT = 1 conv for the first chunk (#183) reduces device time but has no effect on e2e, so it was rejected. The VAE stress test divergence rate was the same before and after today's changes (3 of 120 each, no hangs). Before that, the host-side VAE I/O rework #180 (allocate the read assembly buffer up front, convert the input to bf16 in torch) gave VAE 0.39–0.41 s, 39.951–40.048 s with fast exp (3 of 4 runs under 40 s). Before that, #177 (RMSNorm writes its output directly into the conv's padded buffer) and #178 (bundle the neighbor pad fabric sends into scatter packets of 4) gave VAE 0.447 → 0.410 s (both bit-identical, commit `1fa87c13974`). Unsynchronized e2e is 40.72–40.74 s with the default and 39.999 / 40.006 / 40.064 s with the optional fast exp (no hangs), right around 40 s. Before that, adopted #174 (keep the VAE residual stream in row-major; VAE 0.51 → 0.49 s), #175 (keep the causal conv cache with padding and remove the concat; VAE 0.49 → 0.46 s, unsynchronized e2e 40.76 s) and #176 (chips that hold the same frame split the attention queries; VAE 0.46 → 0.445 s) (all bit-identical). Today's e2e hangs: 6 in the denoising ring SDPA, 3 in the VAE neighbor pad, and 2 in the first all-gather of the text encoder in the first run right after glx_reset (right after "Pipeline allocation run"). The default config is 40.75 s unsynchronized, 40.04–40.10 s with the optional fast exp (#138), and 39.67 s with the optional #172 on top, but 5 of 6 runs hang, so it does not count as an improvement. The fastest config that does not add hangs is fast exp at 40.01–40.11 s, just short of 40 s. The hangs concentrated on a few specific chips, so a hardware-specific cause is possible; the investigation was closed (the investigation row below). Also tried the VAE conv3d halo mode (the row before #174, parked because of hangs) and DEST full-sync (#173, rejected).
> - **2026-10-11:** The current fastest is **41.00 s** (#167, **2.69x** faster than the 110.39 s baseline, with the script's per-step sync). The real time without the sync is **40.81 s** (2.70x), and **40.07 s** (2.75x) with fast exp. Up to #166 it was 41.43 s (41.26 s without sync, 40.43 s with fast exp (#138), which is off by default because it affects quality). Adopted #167, which computes the softmax row sums with the P·V matmul and makes the SFPU on the pack thread wait on the TRISC side. Adopted on 2026-10-10 and 2026-10-11: an 8×10 grid with Ethernet dispatch (#154–#157), the fix of the bank-owned check for the 3-link SP all-gather (#162), exact compaction of the cross-attn text (#164), +64 KB for the kernel config region and the fix of the K/V alternation in ring SDPA (#165), and batched computation of the timestep embedding (#166). A rare race remains even in the default config: about once in several tens of runs, the latent diverges or a hang occurs (in the denoising ring SDPA only one device cannot finish, or in the VAE) (the investigation row in section 10).
> - **Current best:** **86.90 s** at 480p (#15d, CFG batching; −21.3% from the 110.39 s baseline; the re-measurement #20 on the test Galaxy gave 86.97 s). The previous best was 89.79 s (#10). **Further, #21** (LoFi only for the SDPA matmuls, enabled by default, not committed) gives 82.76 s, **#30** (the tanh in GELU_TANH as a polynomial for bf16, not committed) 81.84 s, **#36** (fold the two CFG passes into heads, spread ring SDPA evenly over 64 cores, committed) 77.93 s, and **#45** (ring SDPA Q interleave + reader on NoC1, not committed, enabled by environment variables) **75.49 s** (−31.6% from the baseline). #21 changes the picture content a lot, but the visual quality is kept.
> - **Current breakdown and theoretical values:** The latest is in section 9.8 (as of #107: of 30.0 M cycles per block, ring SDPA takes 59%; the theoretical lower bound is about 18–23 s per run). Older values are in section 9.6 (as of #30: of 49.13 ms per block, ring SDPA takes 49% (19% of theoretical) and matmuls 38% (45% of theoretical); the compute-only theoretical lower bound is 21 s per run).
> - **In progress (2026-10-05):** As fundamental speedups, tried (1) overlapping the GELU post-processing with the matmul (#31, effect −0.1%, parked) and (2) a rewrite of SDPA (#32, a prototype that removes the row max and the correction; the speed limit is −8% in SDPA, but making it correct needs changes to the op; stopped). Next: study the efficiency of the SDPA QKᵀ matmul itself (the math–unpack hand-off) on one chip. For the history of the hangs on another Galaxy, see the "#16" item below (on the latest main, "LoFi only for the matmuls" does not hang, #21).
> - **BFP8 and resident weights (added 2026-10-06):** Weights in BFP8 (#48): 73.86 s. Using the freed DRAM, all high-noise weights resident (#50): 71.61 s; the text encoder split 8 ways and the low-noise FFN also resident (#51): 71.34 s; ring SDPA inputs also in BFP8 (#52, the composition changes): 68.05 s. After the AGMM analysis (#53–#56), the BF16 Dst for the attention projections (#57) combined with LoFi for the projections: 66.88 s (#58). With the AGMM input activations in BFP8 (#59, #60) and everything combined: 62.28 s (#61). Fixed the problem that the experiment switches were applied after the perf test's warm-up (#66), made the output projection weights BFP8 too (#67), and re-searched the blocks: 58.14 s (#68). Ring SDPA chunk (128, 1024): 57.08 s (#69); cross-attn SDPA also in BFP8: 56.41 s (#70). Split the encoder embedding table and used the freed DRAM to make the low-noise side fully resident too (#71, #72), QKV output in BFP8: 54.95 s (#73, 2.01x); VAE decoding 3 frames at a time: 54.48 s (#75); FFN2 partial sums sent in BFP8 (#76) and the fused cross-attn add: **53.47 s** (#77, 2.06x). All of these are optional experiment switches.
> - **Commits (added 2026-10-07):** Added the #45, #21, #30, #48–#70 changes to the branch `hkato/wan-wh-opt` as 4 commits (`abe8475bfd3` ring SDPA, `77362cc7978` GELU, `c0dd743b676` AGMM switches, `4a1fe297dde` Wan experiment switches, `4f33f89562a` #71–#73) (temporary measurements for investigation were left out).
> - **Commits (added 2026-10-06):** Added #36 to the branch `hkato/wan-wh-opt` as the 9th commit `414d32c85d8`. #21 and #30 are still not committed.
- **Commits (2026-10-02):** Organized the changes into 8 commits, one per optimization item, rebased them onto the latest `tenstorrent/tt-metal` main (`d1489f53`), and published them on the fork **`hkatoTT/tt-metal`, branch `hkato/wan-wh-opt`** (https://github.com/hkatoTT/tt-metal/tree/hkato/wan-wh-opt , head `27e6010c`). The 7 changed files had no upstream changes from the base `a1146e45eef` to main, and there were no conflicts. The performance numbers were measured on the `a1146e45eef` base; the rebased code is not yet verified on hardware. Unverified or rejected experiments, such as head folding and the SDPA precision switches, are not included.
> - **Machine (2026-10-05):** Work moved to **the test Galaxy**. Setup took about 15 minutes (118 GB from Hugging Face in 5.5 minutes, 8 minutes to build the organized branch). The rebased branch reproduced the earlier block values (B=1: 26.63 ms, CFG batching: 51.72 ms, same PCC).
> - **Approach:** Iterate quickly at 480p (screening with 6 steps, adoption decisions with 40 steps). 720p comes later.
> - "#number" is the row number in the table below (the running number of the optimization). The details of each number are in the subsection "Details of #N" within this section 10.

**Adopted optimizations (as of 2026-10-06, 480p, 40 steps, WH Galaxy 4×8)**

The total time is the cumulative value when the rows are stacked from top to bottom. "Composition" is the difference of the video from the previous version (repeat runs with the same settings have a mean pixel difference of about 0.11).

| # | Change | Total (s) | Delta | How to enable | Composition | Commit |
|---|---|---|---|---|---|---|
| 0 | Baseline | 110.39 | — | — | — | — |
| 1, 4 | Unfused FFN2 path, and automatic selection when a config for the fused version exists | 104.43 | −5.96 | Default | No change | 0001, 0002 |
| 2b | Matmul blocking for WH, and the fused MM+RS config | 102.72 | −1.71 | Default | No change | 0003 |
| 3 | Ring SDPA chunk (256, 256) → (256, 512) | 97.35 | −5.37 | Default | Changes slightly (divergence is visible) | 0004 |
| 5b | High-noise transformer FFN weights resident (not gathered by FSDP) | 95.24 | −2.14 | `WAN_FFN_RESIDENT=transformer` | No change | 0005 |
| 6 | FFN matmuls accumulate in BF16 (+ blocking re-search) | 92.35 | −2.89 | `WAN_MM_FP32_DEST=ffn` | No change | 0006 |
| 10 | Cache the cross-attention K/V across steps | 89.79 | −2.56 | `WAN_CROSS_KV_CACHE=1` | No change | 0007 |
| 15d | Combine the conditional and unconditional CFG passes into one forward | 86.90 | −2.89 | Default (`WAN_CFG_BATCH=0` disables) | Changes slightly (mean pixel difference about 12) | 0008 |
| 21 | LoFi only for the ring SDPA matmuls | 82.76 | −4.21 | Default (`WAN_SDPA_MATMUL_FIDELITY=none` disables) | **Changes a lot** (butterfly species, color, size; mean pixel difference 29.7). Picture quality is kept | **Not committed** |
| 30 | The tanh in GELU_TANH as a polynomial for BF16 | 81.84 | −0.92 | Always on (LLK header change) | Changes slightly (mean pixel difference 17.4) | **Not committed** |
| 36 | Fold the two CFG passes into heads, spread ring SDPA evenly over 64 cores | **77.93** | −3.91 | Default (`WAN_CFG_HEADFOLD=0` disables) | No change (mean pixel difference 0.03) | 0009 |
| 45 | Ring SDPA Q interleave + reader on NoC1, chain to the left | 75.49 | −2.44 | `TT_RING_SDPA_Q_INTERLEAVE=1 TT_RING_SDPA_READER_NOC1=1 TT_RING_SDPA_Q_INTERLEAVE_ORDER=rev` | Changes very slightly (mean pixel difference 4.15) | **Not committed** |
| 48 | Weights only in BFP8 (QKV, cross Q and KV, FFN1, FFN2) (optional) | (75.49 → 73.86) | −1.63 | `WAN_WEIGHTS_BF8=qkv,xq,xkv,ff1,ff2` | Changes slightly (same butterfly position, different details; mean pixel difference 18.9) | **Not committed (experiment)** |
| 49 | + Ring SDPA inputs Q, K, V also in BFP8 (optional) | (73.86 → 70.74) | −3.12 | + `WAN_SDPA_BF8=1` | **Changes a lot** (butterfly position and orientation change; mean pixel difference 28.4). Picture quality is kept | **Not committed (experiment)** |
| 50 | Convert the weights to BFP8 at load time and make the high-noise transformer fully resident (removes the FSDP weight gather) | (73.86 → 71.61) | −2.25 | + `WAN_WEIGHTS_BF8_LOAD=1 WAN_FULL_RESIDENT=transformer` | The composition changes (pixel difference 30.5 vs #48, because the weight rounding moves to the host) | **Not committed (experiment)** |
| 51 | Split the text encoder 8 ways and use the freed DRAM to make the low-noise FFN resident too | (71.61 → 71.34) | −0.27 | + `WAN_ENC_ON_SP_AXIS=1 WAN_FFN_RESIDENT=transformer,transformer_2` | Almost the same (pixel difference 20.2, same composition) | **Not committed (experiment)** |
| 52 | #51 + ring SDPA inputs in BFP8 (the #49 switch) | (71.34 → **68.05**) | −3.29 | + `WAN_SDPA_BF8=1` | **Changes a lot** (same kind as #49, pixel difference 15.7). Picture quality is kept | **Not committed (experiment)** |
| 57 | Attention projections also accumulate in the BF16 Dst, QKV subblock 4×2 (extension of #6b) | (71.34 → 70.05) | −1.29 | + `WAN_MM_FP32_DEST=all WAN_AGMM_OVR=8192x5120x3840=8,10,10,4,2` | The composition changes (29.7 vs #51). Picture quality is kept | **Not committed (experiment)** |
| 58 | #57 + LoFi for the projections + ring SDPA inputs in BFP8 | (70.05 → **66.88**) | −3.17 | + `WAN_PROJ_LOFI=attn,ffn WAN_SDPA_BF8=1` | **Changes a lot** (pixel difference 26.8–32.9). Picture quality is kept | **Not committed (experiment)** |
| 59 | AGMM inputs (activations) in BFP8 (LayerNorm writes BFP8 directly; type conversion only before the output projection) | (71.34 → 67.64) | −3.70 | + `WAN_AGMM_IN_BF8=1` | The composition changes (31.5 vs #51). Picture quality is kept | **Not committed (experiment)** |
| 60 | FFN1 output (FFN2 input) also in BFP8 | (67.64 → 66.89) | −0.75 | + `WAN_FFN2_IN_BF8=1` | Changes slightly (13.2 vs #59) | **Not committed (experiment)** |
| 61 | #60 + #57, #56 (BF16 Dst, LoFi) + ring SDPA inputs in BFP8 (with a fix that keeps the residual in BF16 and a program cache fix) | (66.89 → **62.28**) | −4.61 | + `WAN_SDPA_BF8=1 WAN_MM_FP32_DEST=all WAN_AGMM_OVR=… WAN_PROJ_LOFI=attn,ffn` | **Changes a lot** (even the flower species changes). Picture quality is kept | **Not committed (experiment)** |
| 66 | (Fix) Apply the experiment switches (LoFi, SDPA inputs in BFP8) before the pipeline warm-up (when the model is built) | (62.28 → 59.33) | −2.95 | Same switches | The composition changes (LoFi now applies to all matmuls) | **Not committed (experiment)** |
| 67 | Output projection (self, cross) weights also in BFP8 (removes the AGMM restriction so that the residual and the weights can have different formats) | (59.33 → 58.88) | −0.45 | + `WAN_WEIGHTS_BF8=qkv,out,xq,xkv,xout,ff1,ff2 TT_AGMM_TERNARY_ANY_DF=1` | Changes slightly | **Not committed (experiment)** |
| 68 | AGMM blocks set to the #62 values (re-measured with LoFi applied correctly) | (58.88 → **58.14**) | −0.74 | + `WAN_AGMM_OVR=8192x5120x3840=16,10,10,4,2;8192x5120x3456=16,8,14,4,2;8192x5120x1280=16,8,5,8,1` | Changes slightly | **Not committed (experiment)** |
| 69 | Ring SDPA chunk (256, 512) → (128, 1024) (uses the L1 freed by K and V in BFP8) | (58.14 → **57.08**) | −1.06 | + `WAN_SDPA_CHUNK=128,1024` | Changes slightly | **Not committed (experiment)** |
| 70 | Cross-attn SDPA also in BFP8 (Q is the normalization output; K and V are converted once, when they are cached) | (57.08 → **56.41**) | −0.67 | + `WAN_XATTN_BF8=1` | Changes slightly | **Not committed (experiment)** |
| 71 | Shard the text encoder token embedding table (256k × 4,096; 2.1 GB per chip when replicated) on the TP axis, and all-gather after the lookup | (56.41 → 56.62) | ±0 | + `WAN_ENC_EMB_SHARD=1` | No change (pixel difference 0.0) | **Not committed (experiment)** |
| 72 | Low-noise transformer also fully resident (uses the DRAM freed by #71) | (56.62 → 55.89) | −0.73 | + `WAN_FULL_RESIDENT=transformer,transformer_2` | No change (pixel difference 0.0) | **Not committed (experiment)** |
| 73 | QKV AGMM output in BFP8 (BFP8 input to the Q and K RMSNorm; the V type conversion also goes away) | (55.89 → **54.95**) | −0.94 | + `WAN_QKV_OUT_BF8=1 TT_DIT_RMSNORM_ALLOW_BF8=1` | Changes slightly | **Not committed (experiment)** |
| 75 | VAE decoding 1 latent frame at a time → 3 at a time | (54.95 → **54.48**) | −0.47 | + `WAN_VAE_T_CHUNK=3` | No change (pixel difference 0.0) | **Not committed (experiment)** |
| 76 | Send the FFN2 (matmul + reduce-scatter) partial sums in BFP8, and write only the final residual add in BF16 | (54.48 → 53.73) | −0.75 | + `WAN_FFN2_RS_BF8=1 TT_MMRS_SPLIT_FINAL_DF=1` | Changes slightly | **Not committed (experiment)** |
| 77 | Fuse the residual add after cross-attn into the output projection AGMM (re-evaluation of #65) | (53.73 → **53.47**) | −0.26 | + `WAN_XATTN_FUSED_ADD=1` | Changes slightly | **Not committed (experiment)** |
| 78 | Multicast the ring SDPA K/V to the Q interleave group (4 rows, 32 cores) (removes the chain relay) | (53.47 → **53.01**) | −0.46 | + `TT_RING_SDPA_Q_INTERLEAVE_MCAST=1` | No change (pixel difference 0.0) | Committed |
| 80 | Distribute AGMM in0 (activations) to each column by multicast instead of a chain (helps matmuls with a small N, such as the output projection) | (53.01 → **52.51**) | −0.50 | + `TT_AGMM_IN0_MCAST=1` | No change (pixel difference 0.0) | Committed |
| 84 | Compute the FFN1 GELU_TANH two rows at a time (fills the SFPMAD waits), and compute it on the pack thread in the AGMM post-processing | (52.51 → **51.88**) | −0.63 | + `TT_AGMM_ACT_ON_PACK=1` (the two-row computation is always on) | No change (pixel difference 0.0) | Committed |
| 92 | Search the VAE 3D conv blocking for WH 4×8, 480p, 3 frames at a time (12 layers; the 2 layers in the time direction got 41x and 16x faster), and split the VAE mid-block attention across only the 8 chips of the W axis | (51.88 → **51.70**) | −0.18 | + `WAN_VAE_SDPA_W_ONLY=1` (the blocking is added to the table, always on) | Only the VAE rounding changes (pixel difference 2.5, same composition) | Committed |
| 93 | In the neighbor pad (halo exchange) before the VAE convs, place the W-direction worker cores in the second row so the links go from 1 to 4 (on the WH 8-column grid, the H and W worker cores competed for the first row and W was cut down to 1 link), and batch the copy of the chip's own part 16 rows at a time | (51.70 → **51.49**) | −0.21 | + `TT_NP_W_ROW1=1 TT_NP_LC_BATCH=16` | No change (pixel difference 0.0 vs #92) | Committed |
| 94 | VAE decoding 3 latent frames at a time → 5 at a time (chunks: 8 passes of 1 + 3×6 + 2 → 5 passes of 1 + 5×4). Also searched the conv blocking for 5 frames and added it to the table | (decoder body 662 → **593 ms**) | −0.07 | `WAN_VAE_T_CHUNK=5` (changed from 3) | Only the VAE rounding changes (pixel difference 2.5, no peak of differences at the chunk boundary frames) | Committed |
| 95 | Concat with the cache before the VAE convs (row-major concat): from reading and writing one page at a time and waiting at a barrier, to batching 16 pages at a time | (→ **51.37**, −0.12 together with #94 from 51.49 at #93) | −0.12 | + `TT_CONCAT_RM_BATCH=16` | No change (pixel difference 0.0 vs #94) | Committed |
| 96 | The writer of the row-major conversion (untilize) waited at a barrier for every row; now it waits once per tile row (32 rows) | (→ **51.34**) | −0.03 | + `TT_UNTILIZE_BATCH_BARRIER=1` | No change (pixel difference 0.0 vs #95) | Committed |
| 98 | In the W-direction sends of the neighbor pad, drop the per-row "ready to receive" semaphore packet (which waits until the send finishes) and send one at the end for all rows, and also drop the per-row write completion wait | (VAE −8 ms, total within noise) | −0.01 | + `TT_NP_W_FAST=1` | No change (pixel difference 0.0 vs #96, latents also match) | Committed |
| 99 | Change the SiLU fused into the VAE normalization from fp32-precision exp and reciprocal (2 Newton iterations) to a tanh polynomial for BF16 (sigmoid(x)=0.5+0.5·tanh(x/2), two rows at a time). The normalization output is BF16, so the precision is enough | (VAE −32 ms, total within noise) | −0.03 | + `TT_SILU_FAST=2` | Only the VAE rounding (pixel difference 2.5 vs #98, same composition) | Committed |
| 100 | Stitching the VAE output together on the host (about 97 MB) allocated the output buffer every time (handling of first-touched pages made it vary from 9 to 275 ms); now two buffers are allocated and reused in turn, and the 32 pieces are written directly by 16 threads | (stitching 9–275 → **17–18 ms**) | −0.05 (average; the variance also goes away) | + `WAN_D2H_FAST=1` | No change (pixel difference 0.0 vs #99) | Committed |
| 101 | Fold the two cross-attn CFG passes into the head dimension, like self-attn (Q is only reshaped; the text K/V alternate conditional and unconditional per head when the cache is built). The per-block Q transpose (0.10 ms) goes away | (51.29 → **51.17**) | −0.12 | + `WAN_XATTN_FOLD=1` | No change (pixel difference 0.0 vs #100) | Committed |
| 102 | Do the BF8 type conversion of the self-attn output before the head concat instead of after it (the concat moves half the data; the concat is only a copy, so the result is the same) | (51.17 → **50.98**) | −0.19 | + `WAN_SDPA_OUT_CAST_FIRST=1` | No change (pixel difference 0.0 vs #101) | Committed |
| 103 | Intermediate buffers of the fused distributed normalizations (q/k RMSNorm + RoPE and the adaLN LayerNorm) from fp32 → BF16 (DEST accumulation stays fp32; pack and unpack move half the bytes) | (50.98 → **50.84**) | −0.14 | + `TT_DRMS_INTER_BF16=1 TT_DLN_INTER_BF16=1` | The composition changes, picture quality is the same (pixel difference 37.4 vs #102, block PCC 99.9647 → 99.9636%) | Committed |
| 104 | Write the SDPA output directly in head-concatenated order (the self-attn ring SDPA also unfolds the CFG pair; the cross-attn SDPA adds the pair unfolding to its existing concatenated output). The two head concat ops (0.106 + 0.100 ms/block) go away, and only a reshape remains | (50.84 → **50.57**) | −0.27 | + `TT_RING_SDPA_OUT_CONCAT=2 TT_SDPA_OUT_PAIRFOLD=1` | No change (pixel difference 0.0 vs #103, latents also match) | Committed |
| 105 | In the writer of the fused QKV matmul (AGMM, output split 3 ways), write only the V part in per-head order ([1, H, 2n, E]). The V head split op (0.098 ms/block) goes away, and only a reshape remains | (50.57 → **50.39**) | −0.18 | + `TT_AGMM_V_HEADS=4` | No change (pixel difference 0.0 vs #104, latents also match) | Committed |
| 106 | Cross-attn SDPA chunk (256, 256) → (256, 512) (the K chunk covers the whole text length, so the softmax correction goes away) | (50.39 → **50.34**) | −0.04 | + `WAN_XSDPA_CHUNK=256,512` | Same composition, details differ very slightly (pixel difference 8.0 vs #105) | Committed |
| 107 | In the adaLN modulation of each block (split the coefficient table + timestep embedding into 6), a chain of row-major conversions and slices ran 5 times per block. Now the timestep embedding is split once per step (shared by the 40 blocks) and the table once per block, so the block itself does only 6 adds | (50.34 → **50.25**) | −0.09 | + `WAN_TEMB_PRESLICE=1` | Looks the same (pixel difference 3.1 vs #106; the rounding position of the slicing changes) | Committed |
| 109 | Grow the Tensix kernel config region (the ring buffer that holds kernel binaries and runtime arguments, default 69 KB) by +128 KB (reduce `worker_l1_size`). The binaries of the ops after ring SDPA (binary about 60 KB) and the cross-attn SDPA (about 50 KB) can now be written while the previous op runs, so the dispatch waits between ops (32 µs, 22 µs, etc.) go away | (50.23 → **50.14**) | −0.09 | + `WAN_KCFG_EXTRA_KB=128` | No change (pixel difference 0.0 vs #197) | Committed |
| 111 | Before the final projection (`proj_out`, 5120 → 64), the fp32 `norm_out` output (split 4 ways by TP) was all-gathered (8192 × 5120 × 4 B per chip). Now each chip computes partial sums from only its own 1280 columns and the matching rows of the weight, and the small partial sums (8192 × 64, fp32) are summed across the 4 TP chips (row-wise reduce-scatter + all-gather). The bias is added once, after the sum | (50.14 → **49.99**) | −0.15 | + `WAN_PROJ_OUT_TP=1` | Same composition, the butterfly wing angle differs very slightly (pixel difference 12.2 vs #109; the fp32 summation order changes) | Committed |
| 112 | Every step wrote the timestep (1 element) and CFG coefficient (constant) tensors to the device. This write made the host wait until the 40 blocks of the previous step finished, so the device was idle while the first op of the next step was issued. Now the timesteps for all steps are created before the loop, and the coefficient only once | (49.99 → **49.94**) | −0.05 | + `WAN_PREUPLOAD=1` | No change (pixel difference 0.0 vs #111) | Committed |
| 114 | Reduce the 10 small modulation ops per block (6 adds of table + timestep embedding, 2 "1 + scale", 2 gate type conversions) to 6. The "1 +" is added once to the per-block constant table, and the gate uses the add output directly as BF16 | (49.94 → **49.93**) | −0.01 to −0.03 | + `WAN_TEMB_FOLD=1` | Same composition, the butterfly wing angle differs very slightly (pixel difference 12.6 vs #112; the fp32 summation order changes) | Committed |
| 117 | (Re-adoption of #108) Only in the last ring iteration of ring SDPA, write the normalized output to the joint output in BF8, removing the BF8 type conversion before the output projection (0.14 ms/block). Re-measured on the current config, the e2e regression of #108 and #206 did not reproduce | (49.93 → **49.73**) | −0.20 | + `TT_RING_SDPA_FINAL_BF8=1` | Same composition (pixel difference 7.6 vs #114) | Committed |
| 118 | In the ring SDPA compute, change the condition for blocked pack from "8 tiles or more" to "4 tiles or more" (the WH default is 8; Wan's shape uses a 2×4 subblock, 4 tiles at a time, so it packed one tile at a time). Now that ring SDPA is bound by the pack thread, the gain is large (in #28 the difference was within noise) | (49.73 → **48.92**) | −0.81 | + `TT_SDPA_BLOCKED_PACK_MIN=4` | No change (pixel difference 0.0 vs #218) | Committed |
| 119 | In the ring SDPA softmax, compute the row sums of P with one pack per row instead of accumulating into L1 for each P tile (one tile pack at a time). Keep the width-4 blocked pack setting from packing P, set the destination stride (`OUTPUT_ADDR_OFFSET`) to 0, and accumulate the 4 tiles into the same sum tile in order | (48.92 → **48.30**) | −0.62 | + `TT_RING_SDPA_SUM_STRIDE0=1` | No change (pixel difference 0.0 vs #219; the accumulation order is the same) | Committed |
| 120 | Compute the ring SDPA softmax exp for the 8 tiles of a subblock in one SFPU call, instead of one SFPU call per P tile (each call sets the DEST address and waits for the SFPU to finish) | (48.31 → **47.54**) | −0.77 | + `TT_RING_SDPA_EXP_BATCH=1` | No change (pixel difference 0.0 vs #222) | Committed |
| 123 | In the first half of the fused distributed LayerNorm (per-chip mean and variance), replace Welford's method, which transposes each tile and updates sequentially on the SFPU, with an FPU row-wise reduce (sum) and an x² accumulation (sum of squares); the variance is computed as E[x²] − mean² | (47.54 → **47.33**) | −0.21 | + `TT_DLN_FPU_PRE=1` | Same composition, the butterfly wing angle differs very slightly (pixel difference 12.4 vs #223) | Committed |
| 125 | The second half of the fused distributed LayerNorm (x − mean, × 1/σ, × weight, + bias): from 4 passes that each round-trip through an L1 intermediate buffer, to computing them back to back within one DEST acquire per block and packing once (stages 2–4 combine "DEST reuse", which moves the DEST value to srcA, with broadcast) | (47.32 → **47.28**) | −0.04 | + `TT_DLN_POST_FUSE=1` | The composition changes, picture quality is the same (pixel difference 36.6 vs #225) | Committed |
| 126 | The two multiplies in the second half of #125 (× 1/σ and × weight) from HiFi4 → HiFi2 (LoFi failed the block PCC check) | (47.28 → **47.22**) | −0.06 | + `TT_DLN_POST_FID=2` | The composition changes, picture quality is the same (pixel difference 49.4 vs #225) | Committed |
| 128 | Ring SDPA QKᵀ matmul output subblock 2×4 → 1×8. Unpack keeps the Q tile in srcB and loads the K tile into srcA for every matmul, so the loads per K step drop from 2 + 8 = 10 tiles to 1 + 8 = 9 tiles (ring SDPA is bound by the number of tiles unpacked) | (47.22 → **46.96**) | −0.26 | + `TT_RING_SDPA_QK_SB=1,8` | Same composition (pixel difference 16.0 vs #227; the accumulation order changes) | Committed |
| 132 | Move the matmul (AGMM) output writes from the in0 (activation) data movement kernel to the in1 (weight) side. On the in0 side, the first core of each row (injector) did the DRAM reads, the multicast to the row and the write of its own output serially on one RISC. The in1 side spends much of its time waiting for free space and has headroom | (46.96 → **46.17**) | −0.79 | + `TT_AGMM_OUT_IN1=1` | No change (pixel difference 0.0 vs #231, latents also identical) | Committed |
| 133 | Forwarding AGMM in0 to the fabric is done by 2 cores in each row (forwarders). A forwarder told the injector it was ready for the next block only after it finished sending the received block to the fabric, so the send time (about 11 µs per block) became wait time for the injector. Now it signals ready first and sends the previous block afterwards | (46.17 → **45.93**) | −0.24 | + `TT_AGMM_FAB_DEFER=1` | No change (pixel difference 0.0 vs #235) | Committed |
| 138 | (Optional, off by default) Ring SDPA softmax exp in one instruction per row. The multiply-add result is placed in the top bits of FP32, and inputs below the range that can be computed correctly overflow to ∞ and become 0, so the haze seen in #136 and #137 does not appear | (45.93 → **44.76**) | −1.17 | + `TT_RING_SDPA_EXP_OVF=1` | Same composition, details differ very slightly (pixel difference 14.6 vs #236) | Committed |
| 142 | Run FFN2 (the fused matmul + reduce-scatter + residual op) as a new program on the matmul cores only. Before, 56 cores (8×7) did the matmul, with the RS workers and muxes in the top 2 rows. The new program runs the matmul on all 72 cores; in each ring iteration it computes its own slice and adds the partial sums of the same tiles that arrived from the neighbor chip. The core in the 9th row of each column writes the partial sums directly to the same core on the neighbor chip over its own fabric connection | (45.93 → **45.43**) | −0.50 | + `TT_MMRS_RING=1` | Same composition (pixel difference 14.6 vs #236) | Committed |
| 143 | Change how #142 distributes in0 (activations). Each of the 4 cores in a row group reads only its own rows from DRAM and writes them directly to the same CB location on the other 3 cores (before, one core read all rows and forwarded them in a chain). Also put the data movement kernels in dynamic NOC mode, and do the in0 DRAM reads on NOC0, which is meant for reads (reads on NOC1, which is meant for writes, were about 3x slower per tile) | (45.43 → **45.00**) | −0.43 | (default with `TT_MMRS_RING=1`) | Same composition (pixel difference 3.6 vs #247) | Committed |
| 148 | (Fix) In the #143 in0 split, the "written" flag could arrive before the rows written to the other cores, and that core could then run the matmul on stale rows (rare, timing dependent). Add a flush after the row writes and before the flag | (45.00 → 45.00) | 0 | (default with `TT_MMRS_RING=1`) | Same result as before (all step hashes match in a 40-step e2e) | Committed |
| 149 | (Fix) The #142 in1 chain forwarding (sending in1 down the column) and the in0 chain forwarding when the split is not used had the same kind of race: the "valid" flag was written right after the block without a flush (minimal_matmul has the flush). Add the flush | (45.00 → 45.00) | 0 | (default with `TT_MMRS_RING=1`) | Same result as before | Committed |
| 154 | (Optional config) Move dispatch from the 10th Tensix row to Ethernet cores and grow the compute grid from 8×9 → 8×10. There are no free Ethernet cores, so 2 of the 4 links of each SP-axis connection are taken out of the fabric and used. Ring SDPA uses 72 cores + all-gather on the 10th row; the QKV and FFN1 matmuls use up to the 10th row | (45.00 → **43.01**, with fast exp 43.88 → **42.14**) | −1.99 | `source models/tt_dit/tests/models/wan2_2/eth_dispatch/env.sh` | Same composition (pixel difference 12.2 vs #249) | Committed |
| 155 | Reduce the links taken out by #154 from 2 to 1 per side (3 on the SP axis; the 2 freed Ethernet cores run prefetch and dispatch without dispatch_s). With 3 links the WH ring SDPA all-gather changes the result, so WH also uses the Blackhole send scheme "split by DRAM bank, 4 pages per packet". QKV and FFN1 go back to the 8×9 blocking (using up to the 10th row changes the result with 3 links) | (43.01 → **42.68**, with fast exp 42.14 → **41.85**) | −0.33 | `source models/tt_dit/tests/models/wan2_2/eth_dispatch/env.sh` (default) | Almost the same (pixel difference 3.6 vs #249) | Committed |
| 156 | (Fix) The function that writes the matmul (AGMM) output one row at a time freed the output CB right after issuing the NOC write of the row (it waited for completion only at the end of the block). When compute wrote the next block into the freed space, rows not yet read out were overwritten. Now it waits for completion per row before freeing (the same function in minimal_matmul was already fixed). This also makes QKV and FFN1 deterministic when they use up to the 10th row | (42.68 → **41.87**, with fast exp 41.85 → **41.02**) | −0.81 | `source models/tt_dit/tests/models/wan2_2/eth_dispatch/env.sh` (default) | Same composition (pixel difference 12.2 vs #249) | Committed |
| 157 | (Fix) With the #156 setup, running the 40-step e2e twice made the latents diverge from step 32 (the chained block test matched). It happens with the combination of Ethernet dispatch without dispatch_s and #109, which grows the kernel config region (active only in e2e) (without growing the config region, all steps matched in all 3 runs; the standard config with dispatch_s matched even with it grown). Without dispatch_s, #109 is not used (+3.6 ms per step) | (41.87 → **42.00**, with fast exp 41.02 → **41.19**) | +0.13 | `source models/tt_dit/tests/models/wan2_2/eth_dispatch/env.sh` (default) | Same composition (pixel difference 12.2 vs #249) | Committed |
| 162 | (Fix) The bank-owned send scheme that avoids the race in the 3-link SP ring all-gather (#155, `TT_RING_AG_BANK_OWNED_WH=1`) was never active, because of a bug in the environment variable check (`e[0] == 1` compared with the number 1, not the character '1'). The 6/6 matches of #155 were luck; on 2026-10-11 the chained block test of the current default matched only about 40% of the time. With the fix, 5/5 match, and e2e is also a bit faster with 4-page packets | (42.00 → **41.81**, with fast exp 41.19 → **41.01**) | −0.19 | `source models/tt_dit/tests/models/wan2_2/eth_dispatch/env.sh` (default) | The default latents are bit-identical to #157 (pixel difference 12.2 vs #249). Fast exp: pixel difference 8.4 vs #250 | Committed |
| 164 | Compact the cross-attn text padding exactly. The prompt is zero-padded to 512 tokens and all padding K/V are identical, so they are compacted into 256 keys, and the padding in columns 16–31 of each tile is counted with weight 3 (weights on the `col_identity` of the SDPA row sums and on the V rows). Cross-attn SDPA 0.31 → 0.18 ms | (41.81 → **41.64**, with fast exp 41.01 → **40.80**) | −0.17 | `WAN_XATTN_COMPACT=1` | Same composition (pixel difference 8.9 vs #162), latents matched 10/10 times | Committed (`627c534df9a`) |
| 165 | Bring back the kernel config region growth (#109) with +64 KB, also for Ethernet dispatch without dispatch_s (the divergence that made #157 remove it was a race already present in the default config). Use separate ping-pong buffers for K and V in the ring SDPA K/V all-gather (they have the same shape, so they competed for the same 2 buffers every time) | (41.64 → **41.49**, with fast exp 40.80 → **40.70**) | −0.15 | Perf test default (`WAN_KCFG_NODS_MAX_KB=64`), `WAN_AG_PP_FIX` (on by default) | Latents bit-identical to #164 | Committed (`c737cad24ea`) |
| 166 | Compute the timestep embedding (fp32 MLP, M = 1, about 3.1 ms per step) for all steps at once, and each step uses its row (re-evaluation of #159; bit-identical) | (41.49 → **41.43**; without sync 41.32–41.36 → **41.26**, with fast exp **40.43**) | −0.06 | `WAN_TEMB_BATCH=1` | Latents bit-identical to #165 | Committed (`a85a84b6be5`) |
| 167 | Compute the ring SDPA softmax row sums with a 5th column of the P·V matmul (a tile with 1s only in column 0) instead of packing into L1 for each P tile (relieves the pack thread bottleneck). The SFPU work on the pack thread is issued after the pack TRISC waits on a dedicated semaphore raised after the math commit (stalling in hardware caused a hang where it and the math commit waited on each other) | (41.43 → **41.00**; without sync 41.26–41.31 → **40.81**, with fast exp 40.43 → **40.07**) | −0.43 | `TT_RING_SDPA_PV_SUM=1 TT_RING_SDPA_SFPU_GO=1` | Same composition (pixel difference 9.4 vs #166). In one time window the latents matched 13 times, but in a later window they diverged again | Committed (`910ca50e9e6`) |
| 174 | In the VAE decoder, keep the inputs and outputs of the residual blocks without a 1×1 shortcut in row-major (RMSNorm and the add read row-major directly). The untilize after norm1, the tilize before the add, and the tilize after the upsampler go away | (without sync 40.88–40.91 → **40.78**; VAE 0.509–0.516 → 0.490) | −0.02 (VAE) | On by default (`WAN_VAE_RM_RESID=0` reverts) | Bit-identical (VAE output) | Committed (`c17778dcac0`) |
| 175 | Keep the VAE causal conv cache with padding: the last 2 frames of the neighbor pad output become the leading frames of the next chunk, and the new neighbor pad `front_tensor` copies them in instead of zeros. The concat with the cache (a copy of each chunk's input) goes away | (without sync 40.87 → **40.76**; VAE 0.489 → 0.458) | −0.03 (VAE) | On by default (`WAN_VAE_NP_CACHE=0` reverts) | Bit-identical (VAE output) | Committed (`32533e8f0be`) |
| 176 | In the VAE mid-block attention, chips that hold the same frame split the queries (when frames are distributed on the W axis, the 4 chips on the H axis take only their own rows; for a 1-frame chunk, each chip takes only its own H×W part). The output is already the chip's own part, so the re-split and re-gather get smaller | (VAE 0.460 → 0.445 / 0.446; e2e 40.78 → 40.77–40.84, within the denoising variance) | −0.015 (VAE) | On by default (`WAN_VAE_ATTN_QSPLIT=0` reverts) | Bit-identical (VAE output) | Committed (`bc07569a9a8`) |
| 177 | When the input of a VAE causal conv is the RMSNorm (+ SiLU) output, the norm writes its row-major output directly into the interior of the neighbor pad's padded buffer (ping-pong), and the neighbor pad fills only the borders in place (removes the interior copy) | (5-frame chunk 99.7 → 97.1 ms in the T = 6 VAE test) | −0.01 (VAE) | On by default (`WAN_VAE_NP_NORM=0` reverts) | Bit-identical (VAE output) | Committed (`1fa87c13974`, together with #178) |
| 178 | Neighbor pad fabric sends: bundle the sticks sent to the neighbor 4 at a time into one scatter packet (the number of EDM buffer slots set the send speed), the reader reads 8 at a time, and the W reader writes the zeros at the tensor edges directly | (VAE 0.444–0.449 → 0.409–0.413; without sync 40.77–40.82 → **40.72–40.74**, with fast exp 39.999 / 40.006 / 40.064) | −0.037 (VAE, including #177) | On by default (`TT_NP_SCATTER=0`, `TT_NP_RD_BATCH=1` revert) | Bit-identical (VAE output) | Committed (`1fa87c13974`) |
| 180 | VAE host-side I/O: allocate the two 97 MB read assembly buffers on the first call (the second one was first created during the real run), and make the input latent contiguous bf16 in torch before passing it to `from_torch` | (with fast exp, without sync 39.985 / 39.986 / 40.048 / 39.951; VAE 0.391–0.407) | −0.01 (VAE) | On by default (`WAN_VAE_IN_BF16=0` reverts the input) | Same values | Committed (`f3722c34766`) |
| 181 | VAE mid-block attention: instead of every chip gathering all frames of the chunk over H and W and then keeping one frame, send each frame to its owner chip with an all-to-all on the W axis and gather only over H (the output uses the reverse all-to-all) | (5-frame chunk 88.2 → 83.9 ms; VAE 0.391–0.403 → 0.368–0.373, with fast exp 39.934 / 39.943 / 40.021) | −0.02 (VAE, including #182) | On by default (`WAN_VAE_ATTN_A2A=0` reverts) | Bit-identical (VAE output) | Committed (`c18627970be`) |
| 182 | Arrange each chip's piece of the VAE input into contiguous blocks in torch and distribute it over the whole mesh on dim 0 (host-side split of ShardTensor2dMesh 9 ms → 1.6 ms) | (measured together with #181) | (included in #181) | On by default (`WAN_VAE_IN_SHARDS=0` reverts) | Same values (match confirmed on device) | Committed (`c18627970be`) |
| 184 | Hand-off of the final denoising latent: instead of reading 4 MB from one chip after the SP-axis all-gather (9–10 ms on that chip's command queue), read the pieces from all chips in one async read and join them on the host | (hand-off 11 → 7.8 ms; with fast exp 39.940 / 39.952 / 39.943) | −0.003 | On by default (`WAN_LAT_SHARDS=0` reverts) | Same values (match confirmed) | Committed (`849a100d9ae`) |
| 185 | Re-search the VAE conv3d blocking on the 8×10 grid of Ethernet dispatch (the 120×104 96→96 3×3×3 conv and the 192→96 spatial conv were missing from the table and ran with default values). Updated conv_out, up2_res, up1_res, up3_res and up2_spatial; up1_spatial is excluded | (5-frame chunk 83.9 → 81.0 ms (including up1_spatial); VAE 0.366–0.371 → 0.359–0.361, with fast exp 39.922 / 39.938 / 39.995) | −0.008 (VAE) | On by default (`WAN_C3D_RESWEEP=0` reverts) | Bit-identical (VAE output) | Committed (`2d210b9eb8a`) |
| 189 | Fuse the final add of the VAE residual block into the output side of the second conv (conv3d `residual_tensor`: the writer reads the residual rows, compute tilizes them and adds them before the untilize; for layers with one C_in block of 96 channels) | (removes 3 adds, 2.7 ms per chunk, at +56 µs per conv2; VAE alone 357.8 → 348.3 ms, e2e VAE 0.359–0.361 → 0.348 s, with fast exp 39.915 / 40.005) | −0.012 (VAE) | On by default (`WAN_VAE_C3D_RESID=0` reverts) | Bit-identical (VAE output) | Committed (`1d717855da4`) |
| 190 | Extend the #189 fusion to conv2 with multiple C_in blocks (fp32 partial sums) (round to bf16 with the same read pattern as untilize, then add) | (VAE alone 348.6 → 344.9 ms, e2e VAE 0.348 → 0.346 s, with fast exp 39.924 / 39.998) | −0.002 (VAE) | On by default (`WAN_VAE_C3D_RESID_ALL=0` goes back to the #189 scope) | Bit-identical (VAE output) | Committed (`1860a57323d`) |
| 191 | Row-major RMSNorm (with gamma): write row-major directly with `pack_untilize` from the gamma (+ SiLU) pass, make the block width a divisor of Wt, and limit the per-tile-row initialization to DST and the packer | (norm 11.69 → 10.78 ms (T = 6), e2e VAE 0.346 → 0.341 s, with fast exp 39.982 / 39.982) | −0.005 (VAE) | On by default (`TT_LN_FUSED_UNTILIZE=0`, `TT_LN_BLK_DIV=0`, `TT_LN_NO_RESTART=0` revert) | Bit-identical (VAE output) | Committed (`c088a418610`) |
| 192 | The spatial conv after upsample reads the padded low-resolution tensor directly by remapping the reader coordinates (Y → (Y+1)/2) (conv3d `input_upsample=2`). The upsample op and the high-resolution neighbor pad go away | (2.74 → 0.69 ms per chunk, VAE alone 342.2 → 334.2 ms, e2e VAE 0.341 → 0.333 s, with fast exp 39.888 / 39.982) | −0.008 (VAE) | On by default (`WAN_VAE_C3D_UP=0` reverts) | Bit-identical (VAE output) | Committed (`05dcc613f76`) |
| 195 | Also keep the residual stream of the latent-resolution layers (the mid block and the first up block, 15×13, 384 channels) in row-major, and use the fused norm, neighbor pad and residual add path | (ops per chunk 212 → 193, 73.15 → 72.68 ms; VAE alone 332.3 → 330.8 ms, e2e VAE 0.333 → 0.330 s, with fast exp 39.968 / 39.975) | −0.003 (VAE) | On by default (`WAN_VAE_RM_MID=0` reverts) | Bit-identical (VAE output) | Committed (`375f45d2e63`) |
| 197 | On WH Galaxy, limit the VAE conv3d to 9 rows (8×9) by default. This avoids the hang where conv3d stalls on the row-9 cores of the 8×10 Ethernet dispatch grid (8 of 630 runs on 8×10, 0 of 1,876 on 8×9). Grids with fewer than 10 rows use the blocking from before #185 | (VAE alone 332 → 353 ms, e2e VAE 0.330 → 0.340–0.343 s, with fast exp 39.902 / 39.909, default 40.720 / 40.658) | +0.010 (VAE) | On by default (`WAN_VAE_C3D_GRID_Y=0` goes back to 8×10) | Bit-identical (VAE output) | Committed (`60736588c4c`) |
| 6b | Attention projections also accumulate in BF16 (optional) | (92.35 → 91.70) | −0.65 | `WAN_MM_FP32_DEST=all` | Changes clearly | 0006 (disabled by default) |

- Total: 110.39 → **75.49 s (−31.6%, up to #45)**. With all of #48–#52 (BFP8 and resident weights) it is 68.05 s, and with #57–#68 on top (BF16 Dst, LoFi, BFP8 activations and output projection, the fix to how experiment switches are applied, block re-search, SDPA chunk, BFP8 cross-attn, sharded encoder embedding, fully resident weights, BFP8 QKV output, VAE decoding 3 frames at a time, BFP8 FFN2 partial sums, fused cross-attn add, K/V multicast, AGMM in0 multicast, GELU computed 2 rows at a time and on the pack thread, VAE conv blocking and attention split, parallel neighbor pad (halo exchange), decoding 5 frames at a time, batched writes in concat and untilize, SiLU polynomial, host-side stitching, folding cross-attn into heads, order of type cast and head concat, BF16 norm intermediates, concatenated SDPA output, V output in head order, cross-attn SDPA chunk, slicing the modulation once per step, larger kernel config region, TP split of the final projection, moving small per-step writes earlier, fewer small modulation ops, BF8 final SDPA output, SDPA width-4 batched writes, P row sums in one write, exp in one SFPU call, LayerNorm statistics on the FPU) it is **47.33 s (−57.1%, 2.33x the baseline)**. Within the changes that keep the composition (#48, #50, #51; the rounding difference of #50 does change the composition slightly) it is 71.34 s. (Earlier entry: 110.39 → 77.93 s, up to #36.) Of these, only #21 (−4.2 s) changes the composition a lot. Without #21 and #30 the estimate is about 83 s.
- The three non-default settings (`WAN_FFN_RESIDENT=transformer`, `WAN_MM_FP32_DEST=ffn`, `WAN_CROSS_KV_CACHE=1`) are needed to get the current best numbers.
- Commits are on the branch `hkato/wan-wh-opt`.

Add one row per optimization. Compare on **the same machine with the same measurement method**.
"Block" is the same Tracy measurement as section 9.2 (union of the execution spans of all ops, per chip),
"Total" is the same performance test as section 8 (80 forwards + text encoder + VAE).
Changes are stacked on the tt-metal branch `hkato/wan-wh-opt` (base `a1146e45eef`).

**How the block is measured (since 2026-09-30):** Run the block 3 times in a row (`WAN_BLOCK_TEST_ITERS=3`, a switch added to the test) and record "start of the first op → end of the last op" of the last run ("Block (warm)" in the table below). The "Block" column of #0 and #1 is the union of all ops from a single run.

| # | Date | Change | Switch | Block (ms) | vs prev. | Total (s) | vs prev. | Quality | Decision |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2026-09-29/30 | Baseline (no changes) | — | 33.79 | — | 110.39 | — | PCC 99.9919%, video OK | — |
| 1 | 2026-09-30 | FFN2 unfused (AGMM → matmul → reduce-scatter → addcmul) | `WAN_FFN2_UNFUSED=1` | 31.71 | −6.2% | 104.43 | −5.4% | PCC 99.9922%, video OK | adopted (enabled in later experiments) |
| 1' | 2026-09-30 | (#1 re-measured with the warm method; reference value) | `WAN_FFN2_UNFUSED=1` | warm 31.54 | — | — | — | PCC 99.9922% | — |
| 2a | 2026-09-30 | #1 + matmul blocking for WH (3 AGMM shapes, FFN2 matmul) | `WAN_FFN2_UNFUSED=1` + config table | warm 31.15 | −1.2% | 103.07 | −1.3% | PCC 99.9922%, video OK | rejected (2b is faster) |
| **2b** | 2026-09-30 | Blocking + FFN2 back to a fused MM+RS tuned for WH (8×7 grid) | config table (no switch) | warm **30.97** | −1.8% | **102.72** | −1.6% | PCC 99.9919%, video OK | **adopted** |
| **3** | 2026-09-30 | ring SDPA chunk (256, 256) → **(256, 512)** | default in `attention_wan.py` (override with `WAN_SDPA_CHUNK`) | warm **29.46** | −4.9% (vs 31.04 from re-measuring 2b the same day) | **97.35** (97.18 with the env var set) | −5.2% | PCC 99.9938%, video OK (fine details differ more than in 2b) | **adopted** |
| **4** | 2026-10-01 | Choose fused or unfused FFN2 automatically by "whether the config resolves" (`resolves_fused_mmrs_config`) | code (`WAN_FFN2_UNFUSED=1` forces unfused) | — (480p takes the same path as #3) | — | **97.38** (480p) / **311.22** (720p) | 480p ±0, 720p −3.8% | 0 warnings, video OK | **adopted** |
| 5a | 2026-10-01 | FSDP weight prefetch (minimal version: issue the output projection weights before SDPA) | `WAN_FSDP_PREFETCH=1` | warm 29.55 | +0.3% (no effect) | — | — | PCC 99.9938%, no hang | rejected |
| **5b** | 2026-10-01 | Keep the FFN weights of the high-noise `transformer` (26/40 steps) resident instead of FSDP | `WAN_FFN_RESIDENT=transformer` (cache is `…-ffnresident`) | — | — | **95.24** (2nd run; 1st run 95.41) | −2.1% | video nearly identical to #3, free memory 1.93 GiB | **adopted** (480p; 720p not checked) |
| **6** | 2026-10-01 | Drop fp32 accumulation in the FFN matmuls and use blocking re-searched with 4×2 subblocks | `WAN_MM_FP32_DEST=ffn` (+ #5b) | block PCC 99.9910% | — | **92.35** | −3.0% | video OK (slight differences in details) | **adopted** |
| 6b | 2026-10-01 | No fp32 accumulation for the attention projections (QKV, output projection, cross) either + re-searched blocking | `WAN_MM_FP32_DEST=all` (+ #5b) | — | — | 91.70 | −0.7% | not broken, but the picture visibly changes (butterfly color and size, stem placement) | **optional (disabled by default)** |
| 7 | 2026-10-01 | SDPA softmax: build the per-row sum with SFPU adds on DEST and write it once per row, instead of a second write of P (L1 accumulation) | kernel (`compute_streaming.hpp`) changed directly | ring SDPA 13.06 → **16.79 ms** | **+28% (worse)** | — | — | PCC 99.9938 → 99.9779%, no hang | **rejected** (reverted) |
| **10** | 2026-10-01 | Cache the cross-attention K/V (text side) across steps | `WAN_CROSS_KV_CACHE=1` (+ #5b, #6) | warm 27.59 → **26.71** | −3.2% | **89.79** | −2.8% | the video difference is about the same as between repeated runs with the same settings, free memory 1.54 GiB | **adopted** |
| 11 | 2026-10-01 | Trace one denoise step (record the command stream in advance and replay it) | `WAN_TRACED=1` (K/V cache disabled automatically) + #5b, #6 | — | — | 92.71 | same as without trace (92.35) | video OK | rejected (host op dispatch is not the bottleneck) |
| 13 | 2026-10-01 | (Measurement) Time each section of one forward with syncs | `WAN_STEP_TIMING=1`, `WAN_PERF_STEPS=6` | — | — | — | — | — | — (measurement only) |
| 15a | 2026-10-01 | CFG batching: merge the conditional and unconditional forwards into one (stack each chip's rows vertically to M=8192, one SDPA with batch 2) | `WAN_CFG_BATCH=1 WAN_CFG_SDPA_SPLIT=0` (+ #10) | warm 77.11 (vs 53.42 for two runs) | +44% | — (3,157 ms/step over 6 steps, reference 2,197) | worse | PCC 99.9939%, picture not broken | rejected (ring SDPA with batch 2 takes 4x as long) |
| 15b | 2026-10-01 | CFG batching + call only the ring SDPA twice, one half each | `WAN_CFG_BATCH=1` (+ #10) | warm **51.96** (vs 53.42 for two runs) | −2.7% | **87.50** (2,131 ms/step over 6 steps, reference 2,199) | −2.6% | PCC 99.9939%, video OK (the composition shifts slightly) | **candidate** (optional, enabled by default; disable with `WAN_CFG_BATCH=0`) |
| **15d** | 2026-10-01 | #15b + register blocking for the M=8192 attention projections; enable CFG batching by default | default (disable with `WAN_CFG_BATCH=0`) + #10 | warm **51.82** | −0.3% (vs 15b) | **86.90** | −3.2% (vs #10) | PCC 99.9939%, video same as 15b | **candidate** (optional, enabled by default) |
| 20 | 2026-10-05 | (Re-measurement) Rebased the cleaned-up branch onto the latest main (`d1489f53`) and re-measured on the test Galaxy | same as #15d | warm 26.63 (B=1) / 51.72 (CFG batching) | ±0 | 86.97 | ±0 | PCC identical, video nearly the same as #15d (mean difference 3.3) | — (reproduction check) |
| 21 | 2026-10-05 | LoFi only for the ring SDPA matmuls (QKᵀ and P·V) (`matmul_math_fidelity` from PR #58223; the compute config stays HiFi2) | default (`WAN_SDPA_MATMUL_FIDELITY`, uncommitted) | warm 26.63 → **25.57** (B=1, ring SDPA 13.00 → 12.04) | −4.0% | **82.76** | −4.8% | no hang. PCC 99.9772%. Picture quality is good, but the content changes a lot (butterfly color, species, size; mean difference vs #20 29.7) | **candidate** (optional, **enabled by default**; disable with `WAN_SDPA_MATMUL_FIDELITY=none`; uncommitted) |
| 22 | 2026-10-05 | Fold the CFG-batched self-attention into "20 heads, B=1" and run the ring SDPA once (no transpose, slice or concat needed) | `WAN_CFG_HEADFOLD=1` (+ #21) | warm 49.59 → 50.48 (CFG batching; ring SDPA 12.04 × 2 → 26.06) | **+1.8% (worse)** | — | — | PCC 99.9833% (same as without folding) | **rejected** (removed from the code) |
| 23 | 2026-10-05 | LoFi for the attention projection and FFN matmuls too (weights stay bf16) | `WAN_MM_FIDELITY=LoFi` (+ #21) | warm 49.59 → 49.10 (CFG batching; FFN2 4.17 → 3.87, QKV and FFN1 about the same) | −1.0% | — | — | PCC 99.9612%, **relative RMSE 5.1%** (above the 3.2% limit, so the test fails), CCC 99.86% (one-sided bias) | **rejected** (removed from the code) |
| 24 | 2026-10-05 | Also keep the FFN weights of the low-noise `transformer_2` resident (both experts) | `WAN_FFN_RESIDENT=transformer,transformer_2` | — | — | failed | — | free memory after init is 0.753 GiB per chip (11.21 of 12 GiB used); `TT_FATAL: Out of Memory` on the first 20 MB buffer allocation during generation | **rejected** (CFG batching doubles the activations, so memory is short) |
| 25 | 2026-10-05 | Re-search the ring SDPA chunk now that the SDPA matmuls are LoFi (B=1 block) | `WAN_SDPA_CHUNK` (+ #21) | (256, 512) 25.67 ms (ring SDPA 12.04) is best. (352, 320) 25.97, (256, 384) 26.07, (192, 640) 27.27, (288, 448) 27.92, (160, 768) 29.39 | ±0 | — | — | PCC 99.976–99.978% | no change (keep the current (256, 512)) |
| 26 | 2026-10-05 | (Measurement) Zones inside the SDPA kernel after the SDPA matmuls went LoFi | kernel profiling switch enabled temporarily (reverted after measuring) | — | — | — | — | PCC 99.9772% (unchanged) | — (measurement only; details in "Zones inside the SDPA kernel (after LoFi)" below) |
| 27 | 2026-10-05 | (Measurement) Measure the net work without waits in single-chip SDPA (zones added temporarily) | measurement zones added to the kernel (reverted after measuring) | — | — | — | — | — | — (measurement only; see "Checking the Phase 1 bottleneck" below) |
| 28 | 2026-10-05 | Lower the Wormhole blocked-pack threshold from 8 to 4 tiles (write the QKᵀ output in batches) | kernel constant (reverted after testing) | single-chip SDPA 12.995 → 12.897 ms | −0.75% | — | — | PCC same | **rejected** (gain within noise) |
| 29 | 2026-10-05 | Change the buffer for QKᵀ and P (`cb_qk_im`) from bf16 to bf8 to halve the data written | env var `TT_SDPA_QK_IM_BFP8` (op C++ and kernels changed temporarily, reverted after testing) | single-chip SDPA 13.01 → 13.13 ms (version with added format switches for max and sum) | +0.9% (worse) | — | — | **PCC nan** (computation breaks) | **rejected** |
| **30** | 2026-10-05 | Compute the tanh inside GELU_TANH with a bf16 polynomial when DEST is bf16 (same GELU_TANH formula; the result is rounded to bf16 anyway) | LLK headers (`ckernel_sfpu_gelu.h`, `gelu.h`) changed (uncommitted) | warm 49.59 → **49.13** (CFG batching; FFN1 4.62 → 4.12) | −0.9% | **81.84** | −1.1% | PCC 99.9834% (unchanged). The video has the same butterfly and flowers as #21 with equal quality; the composition shifts slightly (mean difference 17.4) | **candidate** (enabled by default, uncommitted) |
| 31 | 2026-10-05 | In the AGMM epilogue, compute GELU_TANH on the pack thread instead of the math thread (`gelu_tanh_tile_pack`) | test define in the AGMM compute kernel (`AGMM_ACT_ON_PACK`, GELU_TANH hard-coded, uncommitted) | warm 49.13 → 49.07 (CFG batching; FFN1 4.12 → 3.94) | −0.1% | — | — | PCC 99.9834% (same) | parked (small gain; adopting it needs a switch per activation type) |
| 32 | 2026-10-05 | (Prototype) Remove the row max and the correction (SALAD) from SDPA; take QKᵀ through exp inside DEST and write only P | prototype kernel path (`SDPA_RAW_EXP`, single-chip SDPA only; reverted after testing) | single-chip SDPA 12.995 → 12.24 ms (with a copy of the previous result), 11.96 ms (no copy = upper bound) | upper bound −8.0% | — | — | **PCC 79% (incorrect)**: the SDPA exp is an approximate version that assumes inputs ≤ 0 | **abandoned** (a correct version needs op changes to compute a per-row score bound in advance; small payoff for an upper bound of −8%) |
| 33 | 2026-10-05 | (Measurement) Re-analysis of why ring SDPA is slow. Fixed single-chip SDPA so the matmul fidelity setting takes effect, and re-measured the net zones with LoFi | single-chip SDPA op wiring (same as ring), measurement zones (reverted after measuring) | single-chip SDPA: LoFi 13.24 / HiFi2 13.14 / HiFi4 17.93 ms | — | — | — | PCC 99.927% (LoFi) | — (measurement only; see "Details of #33" below. The #26 and #27 numbers included the first run, so they were corrected) |
| 34 | 2026-10-06 | (Measurement) Re-measured the no-max upper bound of #32 with LoFi, plus the per-core step time and K/V waits | #32 timing-only variant, measurement zones (both reverted after measuring) | single-chip SDPA: 12.75 → 11.84 ms (−7%). One step 59.6 → 42.8 µs (−28%) | — | — | — | — (the timing-only variant has PCC 51%, not correct) | — (measurement only; see "Details of #34" below. The op does not shrink because some cores wait on the K/V relay and the work is unevenly split) |
| 35 | 2026-10-06 | (Measurement) With the #34 timing-only variant, removed exp, replaced it with NOPs, etc. to isolate why the matmul slows down | timing-only variant + measurement zones (reverted after measuring) | single-chip SDPA: 11.84 → 11.52 ms. Matmul subblock about 1,600 → about 950 cycles | — | — | — | — (incorrect variant) | — (measurement only; see "Details of #35" below. The matmul only looked slow because the math zone includes Dst waits) |
| 36 | 2026-10-06 | Fold the two CFG halves into heads (one ring SDPA with 20 heads) and run ring SDPA on 8×8 = 64 cores to balance the work (5 per core) | `WAN_CFG_HEADFOLD=1` + `WAN_SDPA_GRID=8,8` (uncommitted) | warm 49.13 → 47.46 (CFG batching; ring SDPA 24.08 → 22.39) | −3.4% | **77.96** | −4.7% (from 81.82) | PCC 99.9834% (same), video nearly the same as #30 (mean pixel difference 0.03) | **adopted** (enabled by default, disable with `WAN_CFG_HEADFOLD=0`; commit `414d32c85d8`) |
| 37 | 2026-10-06 | (Measurement) Upper bound of the no-max timing-only variant in the production ring SDPA (#36 configuration) | timing-only variant adapted to ring (reverted after measuring) | ring SDPA 22.29 → 21.50 ms (−3.6%). The K/V arrival wait grows from 1.6 to 13.7 µs per step | — | — | — | — (incorrect variant) | — (measurement only; see "Details of #37" below. Reworking softmax gains little with the current way K/V are delivered) |
| 38 | 2026-10-06 | (Measurement) Isolate the cause of late K/V arrival on the reader side (ring signal wait / chain relay wait / DRAM read) | wait-time measurement added to the reader temporarily (reverted after measuring) | ring signal wait is 0.1 ms or less per core. DRAM reads of the chain head core take about 14 ms per call | — | — | — | — | — (measurement only; see "Details of #38" below. The bottleneck is the DRAM reads of the chain head core, not ring communication) |
| 39 | 2026-10-06 | On the chain head core, overlap the K and V DRAM reads (issue the V requests before K completes) | temporary reader change (reverted after measuring) | ring SDPA 22.31 → 24.20 ms (timing-only variant also 21.49 → 24.27 ms) | +8.5% | — | — | PCC 99.9834% (same) | rejected (DRAM/NoC congestion is the bottleneck, and overlapping makes it slower; see "Details of #39" below) |
| 40 | 2026-10-06 | Stop the chain relay for either K or V and have each core read from DRAM (split the chain head core's work) | temporary reader change (reverted after measuring) | ring SDPA 22.31 → 34.29 ms (V) / 34.94 ms (K) | +54% | — | — | PCC 99.9834% (same) | rejected (total DRAM read bandwidth is the wall; see "Details of #40" below) |
| 41 | 2026-10-06 | Try the experimental ring SDPA op (streams K/V only once per row) with Wan on WH 4×8 | experimental switches such as `WAN_EXP_RING_SDPA=1` (reverted after trying) | does not run (after working around the constraints on link count, column multiples, even rows and worker count, it does not fit in L1: needs 1.54–2.02 MB, 1.39 MB available) | — | — | — | — | rejected (needs op changes; see "Details of #41" below) |
| 42 | 2026-10-06 | (Measurement) Upper bound when each core reads K/V from DRAM only once per head (timing only) | temporary reader change (reverted) | ring SDPA 22.31 → 20.23 ms, 15.57 ms combined with the timing-only variant | — | — | — | — (incorrect variant) | — (measurement only; see "Details of #42–#45" below) |
| 43 | 2026-10-06 | Q interleave (core c handles chunks c, c+64, …; a chain over each 16-core group reads K/V only once per pass) | `TT_RING_SDPA_Q_INTERLEAVE=1` (host op and kernel changes, uncommitted) | chips 0–15: 22.31 → 19.96 ms, chips 16–31: 22.42 → 23.15 ms | — | — | — | PCC 99.9834% (same) | — (the left-half chips are slow; see #45) |
| 44 | 2026-10-06 | (Measurement) Isolate why the left-half chips are slow; compare chain length and direction | measurement zones (reverted) | DRAM reads are about 2.5x slower only in the iterations where an all-gather is still running. Shorter or reversed chains are worse | — | — | — | — | — (measurement only) |
| 45 | 2026-10-06 | #43 + reader on NoC1 and the chain toward the left (align the NoC direction with the relay direction) | `TT_RING_SDPA_Q_INTERLEAVE=1 TT_RING_SDPA_READER_NOC1=1 TT_RING_SDPA_Q_INTERLEAVE_ORDER=rev` (uncommitted) | ring SDPA max 22.76 → 20.84 ms (even across all chips) | −8.4% | **75.49** | −3.1% (from 77.93) | PCC 99.9834% (same), positions in the video shift slightly (mean pixel difference 4.15) | candidate |
| 46 | 2026-10-06 | Bounded exp mode: subtract a guaranteed per-row upper bound inside the matmul and drop the max and SALAD | `TT_RING_SDPA_BOUNDED=1 WAN_SDPA_BOUNDED=1` (host op, kernels, model; reverted after trying) | ring SDPA 21.4 ms (#45 is 20.8) | +3% | — | — | **PCC 95.8% (fails)** | rejected (BF16 Dst is not precise enough; see "Details of #46" below) |
| 47 | 2026-10-06 | (Measurement) Timing-only AGMM variant with the matmul compute skipped, to see whether compute or data feed is the bottleneck | temporary compute kernel change (reverted) | QKV 4.45 → 4.17, FFN1 4.12 → 3.94, output projection 2.03 → 1.86 ms | — | — | — | — (incorrect variant) | — (measurement only; data feed is the bottleneck; see "Details of #47" below) |
| 48 | 2026-10-06 | BFP8 for the weights only (QKV, cross Q and KV, FFN1, FFN2; the output projection stays BF16 because of a fusion constraint). Compute config unchanged | `WAN_WEIGHTS_BF8=qkv,xq,xkv,ff1,ff2` (experimental, uncommitted) | 45.61 → 43.73 ms | −4.1% | **73.86** | −2.2% | PCC passes, video details change (mean pixel difference 18.9, same composition) | optional (see "Details of #48–#49" below) |
| 49 | 2026-10-06 | #48 + BFP8 for the ring SDPA inputs (Q, K, V) | + `WAN_SDPA_BF8=1` (experimental, uncommitted) | (Tracy post-processing failed; steady-state step 1,769 → 1,686 ms) | — | **70.74** | −4.2% | block PCC 99.9882%, the composition changes (mean pixel difference 28.4). Picture quality is kept | optional (changes the composition) |
| 50 | 2026-10-06 | Convert the #48 weights to BFP8 at load time (host side) and keep the whole high-noise transformer resident without FSDP. Fully resident in BF16 runs out of DRAM | `WAN_WEIGHTS_BF8_LOAD=1 WAN_FULL_RESIDENT=transformer` (experimental, uncommitted) | (steady-state step 1,769 → 1,732 ms) | — | **71.61** | −3.0% | the composition changes slightly (30.5 vs #48). Picture quality is kept | optional (see "Details of #50–#52" below) |
| 51 | 2026-10-06 | Split the text encoder 8 ways (SP axis) instead of TP=4 to free DRAM, and keep the low-noise FFN resident too | `WAN_ENC_ON_SP_AXIS=1 WAN_FFN_RESIDENT=transformer,transformer_2` (experimental, uncommitted) | (low-noise steady-state step 1,797 → 1,769 ms) | — | **71.34** | −0.4% | nearly the same (20.2 vs #50, same composition) | optional |
| 52 | 2026-10-06 | #51 + BFP8 ring SDPA inputs | + `WAN_SDPA_BF8=1` | (steady-state step 1,647 / 1,685 ms) | — | **68.05** | −4.6% | the composition changes (same kind as #49). Picture quality is kept | optional (changes the composition) |
| 53 | 2026-10-06 | (Measurement) Breakdown of the AGMM DM kernel waits (waiting for in0/in1 chain receives, waiting for CB space, DRAM reads, sends) and the compute-only and DM-only lower bounds | temporary DM kernel instrumentation (reverted) | QKV (BFP8 weights): measured 4.02, compute only 3.73, DM only 3.22 ms. FFN1: 3.74 / 3.51 / 2.90 ms | — | — | — | — (measurement only) | — (see "Details of #53–#58" below) |
| 54 | 2026-10-06 | AGMM blocks shaped to reduce DRAM re-reads (QKV N in one block, FFN1 M at 16) | `WAN_AGMM_OVR` (experimental) | QKV 4.02 → 4.45, FFN1 3.74 → 3.91 ms | worse | — | — | PCC passes | rejected (re-reads were not the bottleneck; small K blocks mean more accumulation passes, and the subblocks become 4×1) |
| 55 | 2026-10-06 | Triple-buffer the AGMM in0 and in1 CBs | `TT_AGMM_CB_DEPTH=3` (host op, experimental) | FFN1 3.74 → 3.58, output projection 1.95 → 2.05 ms | about 0 | — | — | same | rejected (the ops that gain and the ops that lose cancel out) |
| 56 | 2026-10-06 | LoFi for the projection (attention, FFN) matmuls | `WAN_PROJ_LOFI=attn,ffn` (experimental) | (steady-state step 1,732 → 1,711 ms) | −1.2% | 72.15 | — | block PCC 99.9549% (RMSE/σ 4.0%), the composition changes | small gain on its own (once compute is lighter, DM becomes the bottleneck) |
| 57 | 2026-10-06 | Accumulate the attention projections in BF16 Dst too (#6b), QKV subblock 2×2 → 4×2 | `WAN_MM_FP32_DEST=all WAN_AGMM_OVR=8192x5120x3840=8,10,10,4,2` (experimental) | (steady-state step 1,732 / 1,769 → 1,702 / 1,740 ms) | — | **70.05** | −1.8% (from #51) | the composition changes (29.7 vs #51). Picture quality is kept | optional (changes the composition) |
| 58 | 2026-10-06 | #57 + #56 + BFP8 ring SDPA inputs | + `WAN_PROJ_LOFI=attn,ffn WAN_SDPA_BF8=1` | (steady-state step 1,594 / 1,637 ms) | — | **66.88** | −4.5% (from #57) | the composition changes. Picture quality is kept | optional (changes the composition) |
| 59 | 2026-10-06 | BFP8 for the AGMM input (in0, activations). The LayerNorm (norm1–3) outputs are BFP8 directly; only the input of the output projection (the SDPA output) is type-cast. The AGMM output stays BF16 | `WAN_AGMM_IN_BF8=1` (experimental, uncommitted) | QKV 4.02 → 3.25, output projection 1.95 → 1.47, cross Q 1.84 → 1.23, cross output 1.88 → 1.28, FFN1 3.74 → 3.55 ms (with BFP8 weights) | — | **67.64** | −5.2% (from #51) | block PCC 99.9824%, the composition changes. Picture quality is kept | optional (see "Details of #59–#61" below) |
| 60 | 2026-10-06 | BFP8 for the FFN1 output (the input of the FFN2 MM+RS) too (the FFN2 output is fixed to the residual's dtype) | + `WAN_FFN2_IN_BF8=1` | FFN2 3.71 → 3.36, FFN1 3.55 → 3.41 ms | — | **66.89** | −1.1% | block PCC 99.9777% | optional |
| 61 | 2026-10-06 | #60 + #57 + #56 + BFP8 ring SDPA inputs. Also fixed 2 bugs: (1) with BFP8 SDPA inputs, the output of the output projection + residual add (the residual) also became BFP8; (2) the empty joint input was type-cast on every call, so the ring SDPA program was rebuilt on every call | + `WAN_SDPA_BF8=1 WAN_MM_FP32_DEST=all WAN_AGMM_OVR=8192x5120x3840=8,10,10,4,2 WAN_PROJ_LOFI=attn,ffn` | (steady-state step 1,498 / 1,538 ms) | — | **62.28** | −6.9% | the composition changes a lot (the flower species also changes). Picture quality is kept | optional (changes the composition) |
| 62 | 2026-10-06 | Re-search the AGMM blocks in the #61 configuration (using the L1 freed by BFP8 in0). QKV (16,10,10), FFN1 (16,8,14), output projection (16,8,5) | `WAN_AGMM_OVR` (experimental) | block test: QKV 2.72 → 2.52, FFN1 2.74 → 2.38 ms | — | 62.41 | ±0 | — | rejected (in e2e the steady-state step does not change, 1,498 → 1,503 ms). For FFN2 (MM+RS), larger blocks do not fit because they collide with the L1 window |
| 63 | 2026-10-06 | Approximate mode for the ring SDPA exp | `WAN_SDPA_EXP_APPROX=1` (experimental) | no change (PCC identical too) | 0 | — | — | same | rejected (the streaming kernel already uses the approximate version for the main exp; the setting only affects the first column) |
| 64 | 2026-10-06 | (Measurement) Breakdown of one step (synced, timed per section) and a re-evaluation of trace | `WAN_FWD_TIMING=1`, `WAN_TRACED=1` | before the blocks 4.4, 40 blocks 1,491, after the blocks 7, scheduler 2–3 ms. Trace in the #51 configuration: 1,732 → 1,763 ms | — | — | — | — | — (almost no time outside the blocks. Trace is slower because the K/V cache is off. The #61 configuration stops during trace capture with "program not in cache") |
| 65 | 2026-10-06 | Fuse the residual add after cross-attn into the addcmul of the output projection AGMM (gate = 1) | `WAN_XATTN_FUSED_ADD=1` (experimental) | add 0.26 → 0.02 ms, cross output projection 1.21 → 1.42 ms | about 0 | — | — | PCC 99.9600% | rejected |
| 66 | 2026-10-06 | (Fix) The performance test's warm-up runs while the pipeline is built, and the experiment switches (LoFi and BFP8 SDPA inputs, applied via `set_weights_only_cast`) were applied after it. So the first measured step built new programs (about 0.4–1.6 s), and LoFi only took effect on some matmuls. Fixed to apply them when the model is built | (set in the transformer build) | #61 configuration: steady-state step 1,498 → 1,434 ms, first step 1,848 → 1,461 ms | — | **59.33** | −4.7% (from #61) | the composition changes (51.2 vs #61). Picture quality is kept | fix (see "Details of #66–#68" below) |
| 67 | 2026-10-06 | BFP8 weights for the output projections (self and cross) too. Removed the AGMM constraint that "the residual and weight tiles must have the same size and format" (the DM kernel takes the residual tile size from its own CB, and the compute side re-configures the format before the add, so the constraint is not needed) | + `TT_AGMM_TERNARY_ANY_DF=1` (host op, experimental), add out and xout to `WAN_WEIGHTS_BF8` | self output projection 1.39 → 1.30, cross output projection 1.22 → 1.14 ms | — | **58.88** | −0.8% | block PCC 99.9660% | optional |
| 68 | 2026-10-06 | Set the AGMM blocks to the #62 values (re-measured in e2e after the #66 fix) | `WAN_AGMM_OVR=8192x5120x3840=16,10,10,4,2;8192x5120x3456=16,8,14,4,2;8192x5120x1280=16,8,5,8,1` | (steady-state step 1,424 / 1,454 → 1,409 / 1,439 ms) | — | **58.14** | −1.3% | changes slightly (12.2 vs #67) | optional (fastest so far) |
| 69 | 2026-10-06 | ring SDPA chunk (256, 512) → (128, 1024). BFP8 K/V freed L1. The compute per chunk pair is the same, while the target of the per-K-chunk max update and output correction (the number of Q rows) is halved. (256, 1024) does not fit in L1 (1.68 MB); (128, 512) is slow (19.60 ms). For the Q interleave chain, 32 (default) and 16 are equal and 8 is slower. FFN2 through DRAM and larger blocks were also tried but are slower | + `WAN_SDPA_CHUNK=128,1024` | ring SDPA 18.75 → 18.12 ms (block test), steady-state step 1,409 / 1,439 → 1,381 / 1,412 ms | — | **57.08** | −1.8% | changes slightly (11.7 vs #68) | optional (fastest so far) |
| 70 | 2026-10-06 | BFP8 for the cross-attn SDPA too. Q: the q norm outputs BFP8 directly; K/V: cast when cached (once per call). The SDPA output becomes BFP8, and the type cast before the output projection also goes away | + `WAN_XATTN_BF8=1` (experimental) | block 35.14 → 34.51 ms (cross SDPA 0.55 → under 0.4, no type cast), steady-state step 1,381 / 1,412 → 1,361 / 1,391 ms | — | **56.41** | −1.2% | block PCC 99.9721%, changes slightly (15.8 vs #69). Picture quality is kept | optional (fastest so far, `img/wan22_70_fastest_cmp.png`) |
| 71 | 2026-10-07 | The token embedding table of the text encoder (UMT5) was replicated on all chips and used 2.1 GB per chip. Shard the embedding dimension over the TP axis (8 devices) and all-gather after the lookup (same computation) | `WAN_ENC_EMB_SHARD=1` (experimental; the cache uses a different name) | free DRAM (during steps) 1.76 → 3.47 GiB | — | 56.62 | ±0 | unchanged (pixel difference 0.0 vs #70) | optional |
| 72 | 2026-10-07 | Keep the low-noise transformer fully resident without FSDP too (runs out of DRAM without #71) | + `WAN_FULL_RESIDENT=transformer,transformer_2` | low-noise steady-state step 1,391 → 1,360 ms (same as high-noise) | — | **55.89** | −1.3% | unchanged (pixel difference 0.0) | optional |
| 73 | 2026-10-07 | BFP8 for the QKV AGMM output. Removed the input dtype constraint of the distributed RMSNorm (the factory and kernels take the tile size from the input format, so BFP8 works as is) | + `WAN_QKV_OUT_BF8=1 TT_DIT_RMSNORM_ALLOW_BF8=1` (host op, experimental) | QKV 2.54 → 2.03 ms, block 34.67 → 33.83 ms, steady-state step 1,360 → 1,332 ms | — | **54.95** | −1.7% | block PCC 99.9731%, changes slightly (22.3 vs #72). Picture quality is kept | optional (fastest so far, **2.01x**, `img/wan22_73_2x_cmp.png`) |
| 75 | 2026-10-07 | Decode the VAE 3 latent frames (out of 21) at a time instead of 1. The WH 4×8 config had no `vae_t_chunk_size`, so it used the default of 1. 7 frames and all frames run out of DRAM (2.5 GiB is not enough even after freeing the DiT communication buffers), 4 frames is slow (1.15 s), 5 frames is 0.99 s, 6 frames runs out of DRAM | + `WAN_VAE_T_CHUNK=3` (pipeline config, experimental) | VAE 1.41 → 0.95 s | — | **54.48** | −0.9% | unchanged (pixel difference 0.0 vs #73) | optional (fastest so far, 2.03x) |
| 76 | 2026-10-07 | In the fused FFN2 op (matmul + strided reduce-scatter + residual addcmul), make the partial sums (exchanged among the 4 TP devices) BFP8, and keep only the last step's addcmul and its output in the residual format (BF16). The reduce-scatter used one format for all CBs, so an output CB for the last step was added, and the addcmul CBs (temp, residual, gate) each got their own format and page size (changed the factory, the compute, reader and writer kernels, and the op's output spec) | + `WAN_FFN2_RS_BF8=1 TT_MMRS_SPLIT_FINAL_DF=1` (experimental) | FFN2 3.10 → 2.58 ms, steady-state step 1,333 → 1,313 ms | — | **53.73** | −1.4% | block PCC 99.9641% (99.9731% before), changes slightly (18.5 vs #75) | optional |
| 77 | 2026-10-07 | Fuse the residual add after cross-attn into the addcmul of the output projection AGMM (gate = 1). It had no effect in #65, but it helps now that the output projection weights are BFP8 too | + `WAN_XATTN_FUSED_ADD=1` | block 33.88 → 33.78 ms (the 0.27 ms add disappears, cross output projection 1.19 → 1.28 ms), steady-state step 1,313 → 1,307 ms | — | **53.47** | −0.5% | changes slightly (26.8 vs #76). Picture quality is kept | optional (fastest so far, **2.06x**, `img/wan22_77_cmp.png`) |
| 78 | 2026-10-07 | Multicast the ring SDPA K/V to each Q interleave group. The group's head core reads each K/V chunk only once and delivers it to the whole group with one write (before: relayed core by core along a chain). It hung at first: the reader runs on NoC1, but the multicast rectangle was given in the NoC0 direction (top-left → bottom-right), so data only reached the rightmost core of each row (in the triage call stacks, the head core was waiting for the ready signal, and only the rightmost cores were stuck waiting to receive the next V). Fixed by using the NoC1 direction (bottom-right → top-left). Group size: 8 cores (1 row) 18.26 ms, 16 cores 21.29 ms, 32 cores (4 rows) 17.70 ms | + `TT_RING_SDPA_Q_INTERLEAVE_MCAST=1` (host op, committed) | ring SDPA 18.03 → 17.70 ms, steady-state step 1,307 → 1,289 ms | — | **53.01** | −0.9% | unchanged (pixel difference 0.0 vs #77) | candidate (fastest so far, 2.08x). The receivers now wait for the head core's read + delivery (receive wait about 6 ms) |
| 79 | 2026-10-07 | (Follow-up to #78) Tried to parallelize the multicast delivery. From the design of exp_ring_joint_sdpa (a separate op for BH, used by MiniMax H3), took only the part where "the receivers send the ready signal (credit) for the next chunk early, and the head core waits on the cumulative credits". Tried: (1) move the wait for write completion to just before the next delivery, (2) deliver V with a separate semaphore from a separate head core (the last core of the group), (3) the credit scheme, (4) on the V head core, read V before receiving K, (5) triple-buffer the K/V CBs. Also re-swept the chunk (128×512 19.2, 256×512 18.9, 64×1024 22.5 ms; 128×2048 does not fit in L1) | experimental switches (`TT_RING_SDPA_MCAST_SPLIT_KV` etc., reverted) | ring SDPA 17.66–18.05 ms (no change); the lower bound with matmul and exp skipped is also unchanged at 12.4–12.6 ms | 0 | — | — | same | rejected. No change on the delivery side moves the lower bound, so the bottleneck now is the compute side (the non-matmul work: softmax max, subtract, sum and correction, writing P, etc.), not delivery |
| 80 | 2026-10-07 | Stop relaying the AGMM in0 (activations) along a chain from the head core of each column; multicast it to the whole column at once (the receivers only signal the head core and do not relay; the rectangle follows the in0 NoC direction). For the 3 output projections (8192×5120×1280, N per core 5 tiles), the "data feed only" time was nearly the same as the measured time, so the in0 relay was the bottleneck (feed only 1.31, 1.15, 1.26 ms; compute only 0.96, 0.86, 0.96 ms) | + `TT_AGMM_IN0_MCAST=1` (host op and kernels, committed) | self output projection 1.33 → 1.22, cross Q 1.19 → 1.09, cross output 1.28 → 1.23 ms (the feed-only lower bound is now 1.15, 1.00, 1.12 ms), QKV 2.02 → 2.07 ms. Steady-state step 1,289 → 1,277 ms | — | **52.51** | −0.9% | unchanged (pixel difference 0.0 vs #78) | adopted |
| 81 | 2026-10-07 | (Isolating the ring SDPA compute side, and changing how the row sums are made) Timing-only variants: K/V not moved 17.51 ms (normal 17.70, so compute-bound), + no matmul 15.30, + no exp 14.45, + neither 12.02, + also no row-sum write (the pack that accumulates in L1 for each tile of S) 10.64 ms. The sum write took about 1.5 ms, so two replacements were tried. (a) Accumulate a matmul of P × a tile column of ones per row: 18.84 ms because of the fixed cost of one-tile calls, and PCC also broke. (b) Append one column of tiles to V that is "1 only in column 0" and get the sum from the P·V matmul (the reader writes V shifted by one tile and prepares the ones column in a CB; the sum format is unchanged): it worked correctly with PCC 99.9630%, but the width-1 matmul calls are heavy, 19.14 ms | experimental switches (`TT_RING_SDPA_V_AUG` etc., reverted) | slower (17.70 → 18.84 / 19.14 ms) | worse | — | — | (b) PCC 99.9630% | rejected. The fixed cost of an extra matmul call is larger than the sum write (1.5 ms). Dst (8 tiles) is too small to add a 5th column to the existing P·V call |
| 82 | 2026-10-07 | Update the ring SDPA row max less often. In the timing-only variant the max reduce is about 1.7 ms (the early-signal mechanism saves 0.4 ms). Online softmax is mathematically correct with any shift, not only the max, so take the true max only for the first K chunk (or once every N chunks) and use that value (+ a margin) afterwards. The approximate exp assumes inputs ≤ 0, so a small margin causes errors on positive inputs and a large margin causes errors on large negative inputs. Margin sweep (in score units, scale 1/√128): 0 gives PCC 99.475, 45 99.64, 90 99.74, 150 99.83, 200 99.90, 250 99.94, 300 99.954, 400 99.961, 450 99.961, 500 99.962, 600 99.91%. Once every N chunks (N=2, 4) gives 99.50–99.68% | `TT_RING_SDPA_STALE_MAX=1 TT_RING_SDPA_STALE_MAX_MARGIN=450` (experimental, reverted) | ring SDPA 17.70 → 16.49 ms, steady-state step 1,277 → 1,238 ms | −7% | 50.79 | −3.3% | block PCC 99.9614%, but **the video looks hazy** (contrast [std of pixel values] 57 → 49.7, edges 8.8 → 8.0, `img/wan22_82_stalemax_zoom.png`, left #45, middle #80, right #82) | rejected (picture quality degrades). Switching to the exact exp nearly doubles the exp time, which outweighs the 1.3 ms gained |
| 83 | 2026-10-07 | (Isolating the cost of GELU in FFN1, and fusing the epilogue) Removing GELU in a timing-only run takes FFN1 from 2.36 to 1.81 ms (GELU is about 0.55 ms per block, about 1.3 µs of SFPU work per tile). Fused the AGMM epilogue (a separate loop that reads the partial sums accumulated in L1 back one tile at a time, applies bias + GELU and writes them out) into the last K block: load "partial sum + bias" into Dst, accumulate the last K block's matmul on top, compute GELU on the pack thread and write directly to the output CB | `TT_AGMM_FUSE_EPI=1` (experimental, reverted) | FFN1 2.36 → 2.33 ms; combined with #84 it is slower instead, 2.10 → 2.14 ms, and the output projection also goes 1.19 → 1.35 ms | ±0 | — | — | PCC unchanged | rejected. GELU overlaps only with the matmul of the last K block (1 of 20), so it is barely hidden. The fixed cost of switching formats on each read-back grows |
| **84** | 2026-10-07 | Run WH GELU_TANH (when DEST is BF16) through the tanh polynomial 2 rows at a time (`_sfpu_tanh_polynomial_x2_`; the tanh op already used it, but GELU_TANH did 1 row at a time and waited a lot on SFPMAD results). Folded √(2/π) into the cubic coefficient to save one multiply. Also compute GELU on the pack thread in the AGMM epilogue (a re-evaluation of #31; the bias add on math and the SFPU work overlap) | LLK header (`ckernel_sfpu_gelu.h`) + `TT_AGMM_ACT_ON_PACK=1` (only active for GELU_TANH with a bias; commit `d28d64afe9d`) | FFN1 2.36 → 2.10 (2 rows at a time) → 2.03 (pack thread) → 2.01 ms (folded coefficient). Steady-state step 1,277 → 1,263 ms | −0.35 ms/block | **51.88** | −1.2% | unchanged (pixel difference 0.0 vs #80, block PCC 99.9647% also the same) | **adopted** (fastest so far, 2.13x) |
| 85 | 2026-10-07 | Use full sync (16 tiles) for the AGMM Dst and larger subblocks (QKV 8×2, FFN1 2×7). More matmuls per unpacked tile in one inner product should lower the unpacker load. Along the way, found a bug: the AGMM factory read `dst_full_sync_en` but did not pass it to the kernel config (setting it overflowed Dst and gave PCC NaN). That was fixed | `TT_AGMM_FULL_SYNC=1` (experimental, reverted; only the factory hand-off fix was kept) | QKV 2.06 → 2.11, FFN1 2.01 → 2.30 ms | worse | — | — | PCC normal (after the fix) | rejected. The unpacker is not the bottleneck; the loss from math and pack no longer overlapping is larger |
| 86 | 2026-10-07 | (Isolating the FFN2 matmul + reduce-scatter [MM+RS]) Block search (K block 3/4/6/9/12, M 4, N 5): 2.52–3.14 ms, no better than the current value (2.58). Matmul grid rows 7/6/5 (RS workers per link 1/2/3): 2.62/2.46/2.92 ms. 2 links with 12 workers: 2.51. Fabric buffers 4/8, chunk width 2, hand-off window 3: 2.57–2.70. Timing-only variants: no compute 2.40, no input reads 2.27, neither (only the output hand-off and RS) 2.11 ms. Accumulating wait times in the RS kernels shows the reader almost never idle (issuing one read per tile, about 90 cycles each, plus barriers), while the writer spends over a third of its time waiting for compute. 16 tiles per group instead of 8 (the compute kernel split into 8-tile pieces) gives 2.68 ms | experimental switches (extended `WAN_MMRS_OVR`, `WAN_MMRS_LINKS`, `TT_SRS_GRANULARITY`, reverted) | no change | ±0 | — | — | PCC unchanged | parked. The bottleneck is the RS reader reading one tile at a time (the matmul output is spread across L1 tile by tile, so it cannot be read in bulk). Fixing it needs a change to how the output is laid out |
| 87 | 2026-10-07 | Compute the ring SDPA softmax exp on the math thread (right after the subtract) instead of the pack thread, to lighten the pack thread that bounds Phase 1 | `TT_RING_SDPA_EXP_MATH=1` (experimental, reverted) | hung both times (detected and reset by the watchdog) | — | — | — | — | rejected. The approximate (fast) exp init appears to overwrite resources shared with the math thread's matmul (such as the replay buffer). The pack-thread version exists to avoid this. The exact exp takes nearly twice the SFPU time, which outweighs the gain |
| 88 | 2026-10-07 | Drop the type cast before the self-attn output projection (SDPA output BF16 → BF8, 0.134 ms/block) and feed BF16 into AGMM directly | `WAN_TOOUT_IN_BF16=1` (experimental, reverted) | output projection 1.19 → 1.80 ms | worse | — | — | PCC 99.9607% | rejected. Doubling the all-gather volume costs far more (the ring SDPA output is fixed to BF16; BF8 would lower the precision of the unnormalized values handed off between ring iterations) |
| 89 | 2026-10-07 | (Isolating the AGMM matmul) Comparing the compute-only lower bounds for LoFi and HiFi2 (QKV 1.78/3.03, FFN1 1.95/3.03 ms) shows that MVMUL itself runs at the theoretical rate, and QKV has a fixed extra of about 0.53 ms on top (the epilogue is 0.03 with bias only, 0.21 ms with GELU). Compute lower bound / data-transfer lower bound: QKV 1.78/1.92 (data-transfer bound), FFN1 1.95/1.69 (compute bound), the 3 output projections 0.86–0.96/1.01–1.13 (data-transfer bound). The same in block tests without FSDP. QKV has N = 15 tiles per core with an N block of 10, so it reads the activations (in0) twice. Also multicast in1 (weights) like #80 (chaining per-row multicasts): the FFN1 data-transfer lower bound drops 1.69 → 1.42 ms, but FFN1 is compute bound, so the measured time does not change. QKV N block 16 (reads in0 only once): 8,10,16 gives 2.06 → 2.00 ms. Subblock 8×1 (fewer hand-offs but more unpacked tiles): QKV 2.28, FFN1 2.44 ms | `TT_AGMM_IN1_MCAST=1` (experimental) + QKV `8,10,16,4,2` | steady-state step 1,263 → 1,265 ms | ±0 | 51.81 | ±0 | pixel difference 11.8 (rounding differences from the block change) | rejected (within noise). in1 multicast helps once FFN1 compute gets faster, so it is kept as a switch |
| 90 | 2026-10-08 | (Pre-evaluation of a ring SDPA Phase 1 rework) Measured the trade-offs of "compute QKᵀ twice: the first time take only the max inside DEST, the second time do the subtract and exp inside DEST and write only P" with timing-only variants. Removing the QKᵀ result write: 17.70 → 17.09 ms (−0.61). Doing the QKᵀ matmul twice: 17.70 → 19.61 ms (+1.91). Both: 18.95 ms. The QKᵀ matmul is about 10.5 million tile multiplies per chip (about 2.6 ms per core with LoFi), so its cost comes from compute volume, not call overhead | `TT_RING_SDPA_HK_SKIP=qkpack / qkdouble` (timing only, reverted) | — | — | — | — | — | **no rework**. In the current configuration Phase 1 is not pack-bound (the earlier measurements predate the LoFi, BFP8 and chunk changes), and the gain from removing the write (0.6 ms + about 1.5 ms for the max reduce) roughly cancels the cost of recomputing (1.9 ms) |
| 91 | 2026-10-08 | (Re-checking the FFN2 RS bottleneck, follow-up to #86) Timing-only variants: shrink the RS reader reads to 64 bytes: 2.52 ms; drop the writer's data sends to the neighbor chip: 2.51; drop the adds and multiplies in the RS compute kernel: 2.52 (normal 2.52–2.58). No matmul compute 2.40, no input reads 2.27, neither 2.11 (#86) | timing only (reverted) | no change | — | — | — | — | no reader rework. The #86 reading that "the reader is the bottleneck" was wrong (the reader was waiting on other stages). The matmul and RS are tightly synchronized through the L1 window (2 blocks) and per-chunk signals between chips, so lightening any single stage does not shorten the whole |
| **92** | 2026-10-08 | Isolating and improving VAE decoding (about 0.93 s). The decoder body is almost all of it (input transfer 21 ms, weight re-reads 0.4 ms, body 926 ms, transfer to host 24–340 ms [host-side jitter]). Tracy on one chunk (3 frames in time): 3D conv 41 ms, neighbor pad (halo exchange) with adjacent chips 14 ms, norms 9 ms, all-gather 7.8 ms. (1) The 3D conv blocking had no entries for the WH 4×8, 480p, 3-frames-at-a-time (#75) shapes: 10 layers used fallback values chosen only by channel count, and the 2 temporal layers (384→768, (3,1,1)) were even hard-coded to 1×1×1. An exhaustive search on one chip added 12 entries to the table (up1_tconv 9,477 → 230 µs, up0_tconv 1,223 → 78 µs, up1_res 1,643 → 1,075 µs ×5, etc.). (2) The mid-block attention padded 29 frames of zeros to split 3 frames over 32 chips, then all-gathered 32 frames over both the H and W axes after compute (5.6 ms of H-axis gather). Switched to the existing setting that splits only over the 8 chips of the W axis (the default on 4×32) | conv3d blocking table (`models/tt_dit/utils/conv3d.py`) + `WAN_VAE_SDPA_W_ONLY=1` (commit `fbfb325d055`) | decoder body 926 → **765 ms** (device-synced timing, same value in 2 runs) | −17% (VAE) | **51.70** | −0.3% | only the VAE rounding changes (pixel difference 2.5 vs #126, same composition). PCC 99.966% in the standalone VAE test | **adopted** (fastest so far, 2.14x) |
| — | 2026-10-08 | (Note: nondeterminism from board state) After the #87 hang, with only the watchdog's single reset, the denoise results changed from run to run even with the same code and settings (the latents did not match even after 2 steps; the same with the #84 code). One more `tt-smi -glx_reset` brought back bit-exact matches. Telemetry (clocks, temperature, DRAM) showed nothing abnormal | checked by saving the latents (`WAN_DUMP_LATENTS=<path>`) | — | — | — | — | the video comparisons (pixel differences) for #128–#136 are invalid; the timings are valid | from now on, after a hang, do a second reset and check that 2 steps match before comparing |
| **93** | 2026-10-08 | Improve the VAE neighbor pad (halo exchange) (an op that copies the conv input into a padded buffer and exchanges the edge rows and columns in H and W with adjacent chips). Tracy on the pipeline VAE (`WAN_PROF_FLUSH_BEFORE_VAE=1` reads out the profiler buffer just before the VAE) over 8 chunks: conv 275 ms, neighbor pad 237 ms, norms 99 ms, concat with the cache 81 ms. In the longest neighbor pad (2.5 ms), most cores finished in 0.4–0.6 ms, and only the 2 cores handling the W direction ran to the end. The WH logical grid has 8 columns, and the H and W cores were placed in the first row, so with 3 links (6 cores) for H, W was cut down to 1 link (2 cores). (1) Place the W cores in the second row, and cap the H and W link counts each by the grid width. (2) The reader/writer for the local copy read, wrote and hit a barrier for every row (stick); batch 16 rows at a time (small effect) | neighbor pad factory and kernels (`TT_NP_W_ROW1=1 TT_NP_LC_BATCH=16`, commit `d3784471c66`) | decoder body 765 → **662 ms** (T=3 chunk 110 → 88–93 ms) | −13% (VAE) | **51.49** | −0.4% | unchanged (pixel difference 0.0 vs #92, PCC 99.966% in the standalone VAE test also the same) | **adopted** (fastest so far, 2.14x) |
| 94 | 2026-10-08 | Decode the VAE 5 latent frames at a time. In #75, 5 at a time (0.99 s) was slower than 3 at a time (0.95 s), but that was because the conv blocking used fallback values. Searched 12 layers at the 5-at-a-time shapes (T 7/12/22, 20) and added 10 layers to the table (for up3_res and up2_spatial the fallback was best). The chunk count drops from 8 to 5, which cuts the per-chunk work (such as the mid-block attention). A T=5 chunk takes 137–140 ms (T=3 takes 88–93 ms). DRAM was sufficient | conv blocking table + `WAN_VAE_T_CHUNK=5` (commit `4830043fff9`) | decoder body 662 → **593 ms** (compared between per-chunk synced timings) | −10% (VAE) | 51.55 (includes host transfer jitter of +75 ms and denoise +30 ms) | — | only the VAE rounding (pixel difference 2.5 vs #93, smooth from frame to frame) | **adopted** |
| **95** | 2026-10-08 | VAE concat with the cache (before each conv, join the last 2 frames of the previous chunk with the current chunk along time; a row-major concat). Both the reader and the writer read or wrote one page (one stick, 192–768 B) at a time and waited on a barrier each time (the hand-off buffer was also only 2 pages). For a row-major concat on a non-last dimension, read and write 16 pages at a time with one barrier (a change to the ttnn concat op itself; the default behavior is unchanged) | concat factory and reader, row-major writer (`TT_CONCAT_RM_BATCH=16`, commit `0bbd7251ab9`) | decoder body 588 → **563 ms** (T=5 chunk 137 → 128 ms) | −4% (VAE) | **51.37** | −0.2% (−0.12 from #93) | unchanged (pixel difference 0.0 vs #94) | **adopted** (fastest so far, 2.15x) |
| 96 | 2026-10-08 | The writer of the tile → row-major conversion (untilize_with_unpadding, multi-core, interleaved output) waited on a barrier after every row (stick). Write a full tile row (32 rows) back to back, then wait once (the L1 source data is released after that, so the semantics are the same; the default behavior is unchanged) | untilize writer and factory (`TT_UNTILIZE_BATCH_BARRIER=1`, commit `d08be4a51ae`) | decoder body 550 → **535 ms** (2-step measurement) | −3% (VAE) | **51.34** (another run with the same output gave 51.62; host-side jitter is about ±0.15 s) | — | unchanged (pixel difference 0.0 vs #95, latents also match) | **adopted** |
| — | 2026-10-08 | (Note: one-time divergence) In #145 (2 steps) and #156 (40 steps), the output differed once from other runs with the same settings (pixel difference 1.55 / 5.87, both right after a new build). All later runs matched bit for bit. There is no record of which stage (denoise or VAE) it happened in | from now on, the run script saves the latents every time | — | — | — | — | — | if it happens again, isolate it with the latents |
| 97 | 2026-10-08 | (Isolating the VAE norms) Re-measuring the VAE in the 5-frames-at-a-time configuration, one decode has conv about 211 ms, norms (RMSNorm + SiLU, the generic LayerNorm op) about 81 ms, neighbor pad about 62 ms, concat 26 ms, layout conversion 37 ms, attention 20 ms. In the largest norm (1.96 ms), all 72 cores run to the end, at about 18 µs per tile row (3 tiles). Tried fidelity HiFi4 → HiFi2/LoFi, approximate math and no fp32 accumulation, but the decoder body stayed at 526–536 ms (reference 542 ms), within noise. The reader and writer already batch 32 rows, so they are fine. The channels are narrow (96–384), and the design that re-initializes each operation for every tile row appears to be what makes it heavy | `WAN_VAE_NORM_FID/APPROX/FP32` (experimental, reverted) | no change | — | — | — | — | rejected. Speeding it up needs a kernel change that processes several tile rows together |
| 98 | 2026-10-08 | W-direction sends in the neighbor pad (halo exchange). For every row it sent one data packet and one "ready to receive" semaphore packet (waiting until the send finished), and then also waited for write completion. The receiver only waits on a cumulative count over all rows, so the semaphore is now sent once at the end with the row count, and the per-row completion wait was removed. A timing-only variant that stops all W-direction data sends, to measure the upper bound, only saves 18 ms (542 → 516 ms), so there is no rework to packets that bundle 4 destinations (scatter) | neighbor pad W writer and factory (`TT_NP_W_FAST=1`) | decoder body 542 → **534 ms** (2 steps) | −1.5% (VAE) | 51.49 (within jitter) | — | unchanged (pixel difference 0.0 vs #96, latents also match) | adopted |
| **99** | 2026-10-08 | Isolating the cost of the VAE norm (RMSNorm + SiLU, the generic LayerNorm op). On a single-chip bench (192 channels), the time per tile row is nearly proportional to the channel count (14.6/25.6/47.4 µs for 96/192/384), not a per-row fixed cost. Varying conditions: without SiLU 1.63 → 0.67 ms; HiFi2, LoFi and approximate settings have no effect; no fp32 accumulation gives 1.22 ms. When DEST is fp32, SiLU computed the sigmoid with the precise exp and 2 Newton iterations for the reciprocal (and ignored the approximate setting). The norm output is BF16, so compute sigmoid(x)=0.5+0.5·tanh(x/2) with the BF16 tanh polynomial (2 rows at a time), the same method as #30 and #84. Bench: 1.64 → 0.82 ms; the max difference from the precise SiLU is 0.031 (smaller than the 0.0625 difference between the original fused version and the separate SiLU op). Note: with row-major input, the norm itself takes 2x because of per-row tilize and hardware re-init, but for the VAE shapes the difference is small (norm2 0.52 ms vs norm1 0.48 ms) | SiLU SFPU implementation (`ckernel_sfpu_silu.h`) and the LayerNorm factory (`TT_SILU_FAST=2`, only affects the LayerNorm op; commit `4dcc6312eb6`) | decoder body 523 → **490 ms** (2 steps) | −6% (VAE) | 51.72 (VAE 0.92 s because of host-transfer jitter; denoise unchanged) | — | only the VAE rounding (pixel difference 2.5 vs #98, latents match, `img/wan22_99_silu_cmp.png`) | adopted |
| 100 | 2026-10-08 | Jitter in the host transfer after the VAE (it moved the total by ±0.15 s). Transfer breakdown: uint8 conversion on device 1.5 ms, DMA 8–10 ms (stable), host-side stitching 9–275 ms (median about 65 ms; often 120–270 ms after 40 steps). Stitching allocated a new buffer of about 97 MB every time (first-touch page handling) and wrote the 32 pieces from one thread through a temporary reordered copy. Now two buffers are allocated and reused alternately (never overwritten while the caller still holds the previous result), and 16 threads write the pieces directly without a temporary copy. Matches the original exactly in a CPU-side comparison | `models/tt_dit/utils/tensor.py` (`WAN_D2H_FAST=1`, commit `ac0ed127d76`) | stitching 9–275 → **17–18 ms**, VAE Decoding 0.57 s | — | 51.34 (includes the syncs for timing the VAE section) | — | unchanged (pixel difference 0.0 vs #99, latents also match) | adopted |
| **101** | 2026-10-08 | CFG handling in cross-attn. The Q with stacked conditional and unconditional rows ([1, H, 2n, E]) was reordered to [2, H, n, E] (transpose 0.10 ms/block) before SDPA because there are two text K/V. As with folding self-attn into heads (#36), reshape Q to [1, 2H, n, E] (no copy), and arrange the text K/V per head (h0 conditional, h0 unconditional, h1 …) at the cache stage, which runs only once (concat along the sequence, then reshape). Only when there is no cross-attn mask. With the configuration up to #100, the total without timing syncs is 51.29 s (VAE 0.52 s) | `attention_wan.py` (`WAN_XATTN_FOLD=1`, commit `70c0be64e39`) | denoise 50.68 → **50.56 s** (steady-state step about 1,263 → 1,259 ms) | −0.2% | **51.17** | −0.2% | unchanged (pixel difference 0.0 vs #100; block PCC 99.9647% also the same) | **adopted** (fastest so far, 2.16x) |
| **102** | 2026-10-08 | Swap the order of "head concat (BF16) → type cast to BF8" before the self-attn output projection (the #59 cast that makes the AGMM input BF8), so the cast comes before the concat. The concat moves half the data, and since the concat only copies, the result is bit-identical. Block test: concat 0.188 → 0.101 ms, type cast 0.136 → 0.139 ms | `attention_wan.py` (`WAN_SDPA_OUT_CAST_FIRST=1`) | denoise 50.56 → **50.38 s** (steady-state step about 1,259 → 1,256 ms) | −0.4% | **50.98** | −0.4% | unchanged (pixel difference 0.0 vs #101, block PCC also the same) | **adopted** |
| — | 2026-10-08 | (Idea to make the VAE conv output tile layout; not pursued) The rows of a Conv3d matmul output block are (t, h, w) output positions; the W-direction block (4–16) is less than 32, and the per-chip W of 104 is not a multiple of 32. Making tiles in tile layout (32 consecutive positions along W at the same (t, h) × 32 channels) would need the blocks to be reordered, which is the same as redoing the current tilize inside the kernel. The best possible gain is also only the tilize plus its inverse, about 37 ms per decode | — | — | — | — | — | — | not pursued |
| — | 2026-10-08 | (Note: follow-up on the one-time divergence) The latents of #167 and #168 (2 steps) also differed from the reference. Both ran right after the single-chip norm bench (`ttnn.open_device(0)`) had been run many times. #145 and #156 also ran right after a build or another run. State left by the previous run (such as L1 semaphores) appears to affect the first 1–2 runs that follow | before comparing, do an idle run or a reset, and check with the latents | — | — | — | — | — | worked around operationally |
| — | 2026-10-08 | (Isolating the DiT q/k norm) The fused distributed RMSNorm + RoPE for q/k (`dit_fused_distributed_rmsnorm`) takes about 0.20 ms per call, on about 64 cores with 4 tile rows (40 tiles each) per core. Per-zone cycle counters in the kernel show, out of about 200k cycles: sum of squares 25k, waiting for the all-gather of the statistics across the 4 TP chips 27k–88k (6k–50k for the first row because of start skew between chips, about 11k per row after that), 1/rms 11k, x·(1/rms) 17k, weight 20k, RoPE (rotation matmul and cos/sin) 67k. HiFi2/LoFi makes no difference (compute speed is not the bottleneck); the op does not accept turning off fp32 accumulation. (a) Do x·(1/rms)·weight in one pass on DEST (broadcast the weight row into a tile and multiply with dest-reuse): PCC dropped to 95% and it was not faster, so it was stopped. (b) Pipeline the rows (compute the sum of squares of row r+1 before waiting for the all-gather of row r, and have the writer send the next row's statistics before writing out the output; the input buffer is exactly 2 rows, so tile indices do not wrap, and the unpack thread temporarily shifts its read pointer to the next row; the input and cos/sin buffers become 3 rows): the wait disappears, but in paired comparisons (2 runs each) the op time shows no difference within the 0.18–0.24 ms jitter (the start skew between chips remains), rejected | experiment (reverted) | no change | — | — | — | — | rejected |
| **103** | 2026-10-08 | The isolation above showed that the intermediate buffers between the stages of the second half of the norm (x·(1/rms), weight, rotation matmul, cos/sin) were fp32 to match the fp32 DEST. Keep fp32 DEST accumulation, but make only the intermediate buffers BF16 (separate switches for RMSNorm and LayerNorm). Block test (2 runs each): q/k norm 0.20 → 0.18 ms, norm1/norm2 0.30/0.33 → 0.28 ms. Block PCC stays at 99.9647% with RMSNorm only, and is 99.9636% with LayerNorm too | DiT fused norm factories (`TT_DRMS_INTER_BF16=1`, `TT_DLN_INTER_BF16=1`, commit `e56e8bb00bd`) | denoise 50.38 → **50.23 s** | −0.3% | **50.84** | −0.3% | the composition changes, picture quality is the same (pixel difference 37.4 vs #102, 27.8 vs #45; `img/wan22_103_norm_inter_cmp.png`) | **adopted** |
| — | 2026-10-08 | (Note: one-time divergence, 3rd time) The first 40-step run of #103 (#176, after a build and a Tracy crash and reset) and the following #177 had latents that differed from other runs with the same settings. After 2 resets, four 4-step runs (2 reference, 2 with the #104 settings) all matched bit for bit, and the 40-step #178 and #179 (#104) and #180 (#103 settings) also matched. The #103 value was re-taken from #180 (50.87 s in #176 → 50.84 s in #180) | before comparing, reset twice and check with the latents | — | — | — | — | — | worked around operationally |
| **104** | 2026-10-08 | The self-attn ring SDPA writer uses the output tensor both to stash the intermediate result of each ring iteration and for the final write, and every access goes through the tile index calculation (`id_of` / `stride2`). Replace only that calculation with a "heads concatenated" layout (in the #36 shape where the CFG pair is folded into heads, the tile of head 2g+p goes to row p·n+r, column g; the output has the same layout as [1, 1, 2n, H·E], and the Python side only re-labels the shape with `ttnn.experimental.view`). For the cross-attn SDPA, add the same pair unfolding to the existing `output_concat_heads=True` (the #101 fold shape). The 2 head-concat ops (0.106 + 0.100 ms/block) go away. Block test without FSDP: 30.12 → 29.99 ms (block PCC 99.9636%, the same) | ring SDPA writer and factory (`TT_RING_SDPA_OUT_CONCAT=2`), SDPA writer and factory (`TT_SDPA_OUT_PAIRFOLD=1`), `attention_wan.py` (commit `3901c7646d9`) | denoise 50.23 → **49.95 s** | −0.6% | **50.57** (#178 gave 50.55) | −0.5% | unchanged (pixel difference 0.0 vs #180, latents also match) | **adopted** |
| **105** | 2026-10-08 | The self-attn V came from the output [1, 1, 2n, H·E] of the fused QKV matmul (AGMM, output split into Q, K, V), was split into heads by an op (`nlp_create_qkv_heads`, 0.098 ms/block) into [1, H, 2n, E], and was then reshaped to [1, 2H, n, E] by the CFG pair folding. In the AGMM split-output writer, change only the tile index of the third output (V) of a 3-way split to a per-head layout ((head × rows + row) × E tiles + column). The Python side only re-labels it to [1, 2H, n, E] with `ttnn.experimental.view`. In the block test without FSDP, the q norm appeared to grow from 0.18 to 0.25 ms (the point that absorbs the start skew between chips moves), and the whole block went 29.99 → 29.98 ms, but it helped over 40 steps | AGMM writer common code and factory (`TT_AGMM_V_HEADS=4`), `attention_wan.py` (commit `40dee3c3a24`) | denoise 49.95 → **49.77 s** | −0.4% | **50.39** | −0.4% | unchanged (pixel difference 0.0 vs #179, latents also match) | **adopted** |
| — | 2026-10-08 | (Idea: remove the 0.14 ms/block BF8 type conversion of the self-attn output.) Passing the output to the output-projection AGMM in BF16 makes the AGMM go from 1.24 → 1.79 ms (2x the all-gather volume), a net loss. Having ring SDPA write BF8 at the end needs an op change that gives it a separate BF8 output tensor, because each core has about 10 Q chunks and spills the intermediate accumulation to the output tensor (the possible gain is 0.14 ms/block). Postponed | `WAN_TOOUT_IN_BF16` (experiment, reverted) | worse | — | — | — | — | rejected / parked |
| **106** | 2026-10-08 | Chunk search for the cross-attn SDPA (20 heads with the CFG pair folded into heads, Q 4096 rows, text K/V 512 rows). (q, k) = (256, 256): 0.400 ms, (128, 512): 0.439 ms, (256, 512): **0.368 ms**, (512, 512): not enough L1; approximate exp: 0.398 ms, no gain. With (256, 512), K is processed in one pass, so online softmax no longer has to redo work | `attention_wan.py` (`WAN_XSDPA_CHUNK=256,512`; also added `WAN_XSDPA_EXP_APPROX`) | denoise 49.77 → **49.74 s** | −0.1% | **50.34** | −0.1% | same composition, small detail differences (mean pixel difference 8.0 vs #105, `img/wan22_106_xsdpa_chunk_cmp.png`) | **adopted** |
| — | 2026-10-08 | (Isolating the matmul bottleneck) Added per-zone cycle counters for the unpack, math and pack threads to the AGMM compute kernel (1 core, about 1.96 M cycles for QKV). Math spends 1.80 M cycles inside the K loop; waiting to acquire DEST is only 4,000 cycles. The pack zone (1.84 M) does not shrink when the number of packs is halved; pack just follows math. Math takes about 23 cycles per single-tile matmul (theoretical LoFi MVMUL: 16). (a) K block 10 → 20 (half the accumulation in L1): 1.96 → 2.13 ms, worse. FFN1 cannot use 16 because of the Ring constraint (the K block must divide 40, the number of K tiles per chip). (b) With the activation (in0) in BF16, math takes 1.5–1.7x as long → unpack matters. (c) With subblock 2×4 (the orientation that re-reads the weights every time) and BFP4 weights, the unpacked bytes drop by about 40% but math only by 5%, and block PCC falls to 99.74%. The bottleneck is the fixed per-tile hand-off cost between unpack and math (about 7 cycles/tile); reducing it needs a rewrite of the LLK MOP (with uncertain benefit). Stopped | experiment (instrumented kernel, `WAN_WEIGHTS_BF4`; reverted) | — | — | — | — | — | stopped |
| — | 2026-10-08 | (Isolating the gap between e2e and the block test) With `WAN_FWD_TIMING=1`: 40 blocks take 1,230–1,262 ms (30.8–31.5 ms per block), 4.4 ms before the blocks, 7.2 ms after, and 2–3 ms in the scheduler. The host finishes issuing the block loop in 213–285 ms, so the host is not the bottleneck. Each of the 40 steps is stable at 1,238–1,241 ms (no thermal drift); only the first step and the expert-switch step take +30 ms (building the cross-attn text K/V). `WAN_PROF_FLUSH_EVERY`, which reads out the device profiler after every block, captured every op of the e2e run (op names on device 0 were matched by op number to the previous Tracy run). Ops that are not in the block test: a chain of slicing ops for the adaLN modulation (about 17 ops, about 0.13 ms/block), and 22–32 µs gaps before the large AGMMs and ring SDPA | `WAN_FWD_TIMING`, `WAN_PROF_FLUSH_EVERY` (measurement only) | — | — | — | — | — | to #107 |
| **107** | 2026-10-08 | For each block, the adaLN modulation adds `scale_shift_table` (a per-block constant, fp32) and the time embedding (per step, shared by all blocks) and splits the result into 6 parts (`ttnn.chunk`). Because this slices rows out of a single tile, a tile → row-major → slice → tile chain ran 5 times per block. Now the time embedding is split into 6 once at the start of the step and passed to all blocks, the table is split once per block on first use and kept, and the block itself only does 6 fp32 additions. The rounding of the split (the row-major conversion goes through tf32) moves before the addition, so the results are not bit-identical | `transformer_wan.py` (`WAN_TEMB_PRESLICE=1`, commit `f125ff57522`) | denoise 49.74 → **49.64 s** (steady-state step about 1,240 → 1,236 ms) | −0.2% | **50.25** | −0.2% | looks the same (mean pixel difference 3.1 vs #106, `img/wan22_107_temb_preslice_cmp.png`) | **adopted** (fastest so far, 2.20x) |
| — | 2026-10-08 | (Checking time outside the blocks and the clock) Capturing a normal step with the device profiler: one block on device 0 is 30.0M cycles (op total 29.88M), the same as in the block test. The profiler's ms values assume 1 GHz; with an aiclk under load of about 977 MHz the real time is 30.7 ms, which matches the 1,227 ms for 40 blocks from `WAN_FWD_TIMING`. There is no hidden time besides the about 14 ms outside the blocks. Adding syncs inside the block loop (`WAN_SYNC_EVERY`) slows things down: 1,245 ms with a sync every 5 blocks, 1,302–1,358 ms with a sync every block | measurement only (reverted) | — | — | — | — | — | — |
| 108 | 2026-10-08 | To remove the BF8 type conversion of the self-attn output (0.14 ms/block): only in the last ring iteration of ring SDPA, pack the normalized output into a dedicated BF8 CB and write it, with heads concatenated, into the unused joint output tensor (made BF8 with the same shape). The intermediate accumulation is still handed off through the BF16 output tensor as before. In the block test the type conversion disappears, 29.97 → 29.64 ms, and PCC goes 99.9635 → 99.9650% (no more double rounding through BF16). The device profiler also shows a normal-step block dropping from 30.00M → 29.79M cycles. But both 40-step runs took 50.39 s (steady-state step about 1,240.5 ms), slower than the re-measured baseline (#197, 50.23 s, about 1,237 ms). Fewer cycles but more time suggests that the clock under load drops (power limit), but tt-smi during the run only returns idle values, so this could not be confirmed. Passing BF16 directly to the output-projection AGMM makes the AGMM worse, 1.24 → 1.79 ms | ring SDPA op, factory, compute and writer, `attention_wan.py` (`TT_RING_SDPA_FINAL_BF8=1`, reverted) | 50.39 (worse) | — | — | — | same composition (mean pixel difference 11.5 vs #107) | rejected → re-adopted in #117 |
| — | 2026-10-08 | (Checking the clock drop) `tt-smi -s` at denoise step 10 showed that only 4–5 specific chips out of the 32 had their aiclk lowered to 925–981 MHz (example: one chip at 925 MHz, 154 W, 47.9 °C, 0.90 V; the others at 1000 MHz, 122–150 W). None reached the limits (TDP 170 W, TDC 240 A, temperature 75 °C), and the chips that slow down have a higher core voltage (896 mV and 859 mV; their ring-oscillator values also differ). Because the chips synchronize with each other, the slowest chip (925 MHz) sets the overall speed. This looks like firmware voltage/frequency control and cannot be changed from software. As #108 shows, fewer cycles can still mean more time if power goes up, so decisions are made on measured 40-step runs | measurement only | — | — | — | — | — | — |
| **109** | 2026-10-08 | Investigation of the gaps between ops from section 9.8. In the 32 µs gap between the cross-attn SDPA and the output-projection AGMM, all 72 AGMM cores start within 0.4 µs of each other, so it is a dispatch wait. The Tensix kernel config region is 69 KB by default on WH (`default_l1_kernel_config_size` in `wh_hal_tensix.cpp`), and dispatch cannot write a program that does not fit together with the previous op until that op finishes. Kernel binary sizes (ELF load segments): ring SDPA 60.5 KB, cross-attn SDPA 49.7 KB, normalization about 29 KB, AGMM about 18 KB. Reduce `worker_l1_size` by the same amount to enlarge the config region (default 1,393,440 B = L1 1,499,136 B − 105,696 B). +64 KB: 50.17 s; +128 KB: **50.14 s** (steady-state step about 1,236–1,238 → 1,234–1,235 ms); +256 KB: error because the ring SDPA CBs no longer fit in L1 (about 22 KB is left at +128 KB). In the device profiler (#207, 4 steps), the 32, 22 and 5–7 µs gaps all become 0.6 µs, and a steady-state block goes 29.996 → 29.935 M cycles | `test_performance_wan.py` (`WAN_KCFG_EXTRA_KB=128`, commit `9fc09e3f42c`) | denoise 49.63 → **49.54 s** | −0.2% | **50.14** | −0.2% | no change (mean pixel difference 0.0 vs #197) | **adopted** (fastest so far, 2.20x) |
| 108b | 2026-10-08 | In #108 (write the final ring SDPA output directly as BF8 and remove the type conversion), the output-projection AGMM comes right after ring SDPA, so the #109 config-region wait (32 µs/block) also overlapped. Re-measured on top of #109: **50.19 s** (steady-state step about 1,235–1,240 ms), slower than #109 alone at 50.14 s. It is slower even without the wait, so the loss is attributed to the clock dropping under load (power) | reverted again | 50.19 (worse) | — | — | — | same composition (mean pixel difference 11.5 vs #109) | rejected → re-adopted in #117 |
| — | 2026-10-08 | (Re-measuring the ring SDPA bottleneck, #109 configuration, block test) Ring SDPA time with timing-only variants that skip work: normal 17.72 ms, no matmul 15.44, no exp 14.77, neither 12.69, **no K/V transfer (DRAM reads and chain receive/forward) 17.52**, no transfer, matmul or exp 12.02. With the #78 multicast the transfer is almost fully hidden, and **ring SDPA is bound by compute (hand-offs among the 3 compute threads)** (in #77 the DM-only lower bound was 14.8 ms). Skipping one more thing at a time: no pack that accumulates the P row sums in L1 (once per score tile) 16.25 (−1.47), no P write-out 15.28 (−2.44), no subtraction 17.32 (−0.40), no correction (SALAD) multiply 17.66 (−0.06). One core holds about 41,000 score tiles, about 430 cycles per tile (theoretical matmul: 128 cycles). The sum of the individual savings is larger than the saving from skipping everything: the 3 threads synchronize at a fine grain, so lightening one of them exposes another wait | timing-only switch `TT_RING_SDPA_HK_SKIP=kv/sum/ppack/sub/salad` | — | — | — | — | — | measurement only |
| 110 | 2026-10-08 | Compute the P row sums with a matmul instead of pack accumulation in L1 (P × "a tile with 1s only in column 0" (`cb_col_identity`, already used by normalization), accumulated along K inside DEST and written out once per row. Column 0 holds the sum and the other columns are 0, so the correction and normalization work unchanged. The denominator is built from the same LoFi-rounded P as the numerator). The first result was PCC nan: `cb_col_identity` wrote only column 0 and left the rest uninitialized; normalization only used column 0 of the product, so it had never mattered. Filling the whole tile with 0 fixed it. Single-chip SDPA (Q 4,096, K 32,768, 10 heads): 11.63 → 11.47 ms (per row group) / 11.31 ms (4 Q rows at once), PCC 99.9390 → 99.9375%. But ring SDPA gets slower: 17.69 → **18.15 ms** / **17.84 ms** (PCC 99.9647 → 99.9618%). The added matmul costs about 39 cycles per tile without a MOP and cannot win back the removed pack (about 36 cycles/tile). **The same idea was tried in #81** (the PCC collapse in (a) back then was probably caused by this uninitialized column). The conclusion (the fixed cost of a matmul call is heavier than writing out the sum) is the same as in #81 | ring SDPA compute, factory and writer (`TT_RING_SDPA_SUM_MM=1/4`, reverted) | ring SDPA 17.69 → 17.84 (worse) | — | — | — | PCC 99.9618% | rejected |
| **111** | 2026-10-08 | The post-processing outside the blocks (about 7.2 ms of the about 14 ms per step) consists of: adding and splitting `scale_shift_table`, `norm_out` (LayerNorm distributed over TP, fp32 output), the TP all-gather (dim 3), `proj_out` (HiFi4, 5120 → 64), and the SP all-gather of the output. After the all-gather, `norm_out` works on 8192 × 5120 × 4 B ≈ 168 MB per chip, of which about 126 MB is received from other chips (about 2.5 ms at 50 GB/s). On the first call, build once a copy of the `proj_out` weights (5120 × 64) split 4 ways along K over the TP axis (read the replicated weights back to the host and distribute them with `ShardTensor2dMesh`); each chip computes partial sums over its own 1280 columns (`minimal_matmul`, HiFi4, fp32). The partial sums (8192 × 64 × 4 B ≈ 2 MB) are summed with a row-wise `reduce_scatter` (2048 rows per chip) and an `all_gather`; finally the bias is added and the result is cast to the output type | `transformer_wan.py` (`WAN_PROJ_OUT_TP=1`, commit `96e25843226`) | denoise 49.54 → **49.38 s** (steady-state step about 1,234–1,235 → 1,230–1,233 ms) | −0.3% | **49.99** | −0.3% | same composition; the butterfly's wing angle and position differ slightly (mean pixel difference 12.2 vs #201, `img/wan22_111_projout_tp_cmp.jpg`) | **adopted** (fastest so far, 2.21x) |
| — | 2026-10-08 | (Re-checking time outside the blocks, #111 configuration) Measured with a sync per section: pre-processing 4.4–5.2 ms, post-processing 7.2 → **3.3 ms** (the about 3.9 ms from #111 is gone), and the difference between the step total and the blocks plus pre/post-processing (scheduler etc.) about 4 ms. Removing the per-step sync (the measurement-only `WAN_STEP_TIMING=1`, used in e2e since #13 and not part of the baseline measurement) changes denoise from 49.377 → 49.363 s, only about 0.35 ms per step (comparisons keep it on from now on). In the device profile (#207), the ops at the steady-state step boundaries total about 4 ms | `WAN_STEP_TIMING=0` (#210) | denoise 49.36 s | — | 49.99 | — | no change (mean pixel difference 0.0) | measurement only |
| **112** | 2026-10-08 | Every step created the timestep tensor (1 element, fp32) and the CFG coefficient tensor (constant) with `float32_tensor(..., device=mesh)`. At the write to the device the host waits for the previous step to finish, so the device sits idle while the first ops of the next step are issued. Now the timestep tensors for all steps are created before the loop, and each coefficient value is created once and reused | `pipeline_wan.py` (`WAN_PREUPLOAD=1`, commit `96fd2035595`) | denoise 49.38 → **49.33 s** (steady-state step about 1,230–1,233 → 1,228–1,229 ms) | −0.1% | **49.94** | −0.1% | no change (mean pixel difference 0.0 vs #208) | **adopted** (fastest so far, 2.21x) |
| — | 2026-10-08 | (Re-measuring the ring SDPA chunk) (128, 1024) was chosen in #69, before the #78 multicast (when transfer was the bottleneck). Block test on the current, compute-bound configuration: (128, 1024) 17.72 ms, (256, 512) 18.89, (128, 512) 19.25, (64, 1024) 22.49; (64, 2048) and (128, 2048) do not fit in L1 | `WAN_SDPA_CHUNK` | no change | — | — | — | — | keep as is |
| 113 | 2026-10-08 | (Post-processing of the output-projection-type AGMM) With per-thread cycle counters in the AGMM compute and the in0/in1 transfers removed, the two output projections with fused residual (self-attn output, cross-attn output) spend about 195,000 cycles per core on post-processing (bias + gate × result + residual): about 1,200 cycles per tile, about 23% of the op's compute. About 160,000 of that is the unpack thread **waiting for the residual to arrive**: after the K loop, the in1 DM kernel reads the residual and gate one row (5 tiles) at a time, with a barrier per row (because with `force_transpose`, in0 is the one that writes the output). (a) Post-processing in 2 passes (bias batched over half of DEST; gate multiply and residual add in one pass with `add_reuse_dest`; skip the SFPU multiply by the scalar 1.0). (b) The in1 DM starts reading residual, gate and bias asynchronously at the start of the second K block and pushes them with a single barrier after the K loop. Without transfers: post-processing 195,000 → 34,000 cycles, op compute 854,000 → 761,000 cycles (−11%). But in the real block test the self-attn output gets worse, 1.22 → 1.32–1.34 ms, and the cross-attn output goes 1.22 → 1.19–1.36 ms (PCC unchanged at 99.9647%). Supply (in1 DM) is the bottleneck: the early read delays the weight supply, and the faster post-processing exposes the residual wait that the original 3 passes hid | `TT_AGMM_TERNARY_FAST`, `TT_AGMM_TERN_EARLY` (reverted) | worse | — | — | — | PCC 99.9647% | rejected |
| **114** | 2026-10-08 | Even after `WAN_TEMB_PRESLICE` (#107), each block still had 10 small modulation ops (6 fp32 additions of the table chunk + time-embedding chunk, 2 × `1.0 + scale` passed to norm1 and norm3, 2 BF16 type conversions of the gates; 4–5 µs each). Now the "1 +" is added once, on first use, to the cached per-block table chunks (the 2 scales), and the 2 gates go directly to BF16 with `ttnn.add(..., dtype=bfloat16)`. Gate values are the same as before (the fp32 sum rounded to BF16); for scale, (table + 1) + time versus 1 + (table + time) changes the fp32 rounding | `transformer_wan.py` (`WAN_TEMB_FOLD=1`, commit `a08cbdfe75a`) | denoise 49.33–49.35 → **49.32 s** | −0.03 to −0.05% | **49.93** | −0.03% | same composition; the butterfly's wing angle differs slightly (mean pixel difference 12.6 vs #211, `img/wan22_114_temb_fold_cmp.jpg`) | **adopted** (fastest so far, 2.21x) |
| 115 | 2026-10-08 | (Groundwork for a ring SDPA rewrite) (1) Ported the #32 prototype (no row max, subtraction or correction, no write of S to L1; QKᵀ → exp inside DEST → write out P and the row sums → accumulate P·V in L1) to the current kernel and measured a timing-only single-chip SDPA (Q 4,096, K 32,768, 10 heads, chunk (128, 1024), BFP8): 11.6 → **11.9 ms, slower** (still 11.9 ms even without copying the previous result per K chunk). It was −28% in #34, but the original kernel has improved since then, and the naive structure can no longer win. (2) Added per-thread cycle counters to the real ring SDPA compute (18.3 M cycles per core). DEST hand-off waits (math `tile_regs_acquire`, pack `tile_regs_wait`) are all under 1%. Time per function: unpack: QKᵀ 4.61 M, P·V 5.01 M, subtract + exp 3.05 M, max 0.77 M, correction 0.73 M; math: 3.56, 3.92, 6.42, 0.51, 0.64 M; pack: 3.08, 3.09, 8.45, 0.40, 0.54 M. All 3 threads are 80–90% busy and balanced, and unpack spends 9.6 M on matmuls alone (expanding BFP8 tiles, about 234 cycles per score tile). The QKᵀ and P·V subblocks are already 2×4, and reuse of unpacked tiles is at its limit. Removing the L1 round trips still leaves the unpack cost of the matmuls, so the premise of the rewrite (−15 to 20%) no longer holds | prototype and measurement (`TT_SDPA_RAW_EXP`, `TT_RING_SDPA_THR_PROF`, reverted) | 1 chip 11.6 → 11.9 ms | — | — | — | — | **rewrite cancelled** (a big reduction needs fewer matmul unpacks, which means lowering precision or fundamentally changing how DEST is used) |
| 116 | 2026-10-08 | (Follow-up to #115: is unpack bound by bytes?) Changed the ring SDPA inputs q, k, v (and the dummy joint) to BFP4 (576 B per tile, about half of BFP8) and measured in the block test. Ring SDPA only goes 17.70 → 17.35 ms (−2%), and block PCC drops 99.9647 → 99.51%. Unpack time is set by the number of tiles (instructions and hand-offs), not by the number of bytes | `WAN_SDPA_KV_BF4=1` (experiment, reverted) | −0.35 ms (excluding the type-conversion ops) | — | — | — | PCC 99.51% | rejected |
| — | 2026-10-08 | (Checking the LLK matmul unpack) On WH, `_llk_unpack_AB_matmul_mop_config_` is already pipelined: per tile it replays 1 `UNPACR` instruction + 3 instructions that write back the base address of the next tile (`RDCFG`, `ADDDMAREG`, `REG2FLOP`) + 1 `NOP`, alternating between 2 config contexts. Issuing the instructions takes a few cycles; the gap to about 23 cycles per tile (theoretical LoFi MVMUL: 16) probably comes from the unpacker's expansion speed and the hand-off wait with math. There is no visible room to gain by reordering instructions (same conclusion as #89) | reading only | — | — | — | — | — | not pursued |
| **117** | 2026-10-09 | Re-measured #108 (rejected earlier: "write the final ring SDPA output as BF8 into the joint output") on the current configuration (#114). In the device profile (4 steps, chip 0), a steady-state block goes 29.935 → **29.675 M cycles** (the type-conversion op is gone; ring SDPA also 17.68 → 17.63). Both 40-step e2e runs took **49.728 s** (steady-state step about 1,228–1,231 → 1,223 ms; the two videos are identical), so the slowdown of #108 (50.39 s, 2 runs) and #206 (50.19 s) did not reproduce. The cause of that slowdown was not identified (in #204 a missing install step, needed for C++ changes to reach the Python side, was found, and the earlier #108 measurements may have been affected by it) | ring SDPA op, factory, compute and writer, `attention_wan.py` (`TT_RING_SDPA_FINAL_BF8=1`, commit `2b5c89afefe`) | denoise 49.32 → **49.11 s** | −0.4% | **49.73** | −0.4% | same composition (mean pixel difference 7.6 vs #213, `img/wan22_117_sdpa_final_bf8_cmp.jpg`) | **adopted** (fastest so far, 2.22x) |
| — | 2026-10-09 | (Re-measuring the #89 AGMM in1 multicast on top of #117) 49.96 s (steady-state step about 1,229 ms). #117, re-measured at the same time, gave 49.728 s for the third time (about 1,224 ms), so the in1 multicast is 0.23 s slower (identical video) | `TT_AGMM_IN1_MCAST=1` | 49.96 (worse) | — | — | — | no change (mean pixel difference 0.0) | rejected |
| **118** | 2026-10-09 | (Re-adopting #28; the per-thread measurement of #115 shows that the pack thread is the busiest one in ring SDPA) The SDPA compute (`compute_streaming.hpp`) uses batched writes (blocked pack) only when one write is at least `MIN_BLOCKED_PACK_TILES` tiles. On WH this is 8 (with a comment saying that at width 4 the reconfiguration overhead outweighs the gain), so Wan's QKᵀ rows, P and P·V outputs (subblock 2×4) were written one tile at a time. Adding a switch that sets the threshold to 4 for the ring SDPA op only: in the block test ring SDPA goes 17.73 → **17.19 ms** (PCC unchanged at 99.9647%); threshold 2 gives 17.16 ms, no further change. Applying it to the cross-attn SDPA (a different op) as well does not change the time but changes the video (mean pixel difference 5.0 vs #219), so it is applied to ring SDPA only | ring SDPA compute and factory (`TT_SDPA_BLOCKED_PACK_MIN=4`, commit `f0704202c2f`) | denoise 49.11 → **48.32 s** (steady-state step about 1,223 → 1,204 ms) | −1.6% | **48.92** | −1.6% | no change (mean pixel difference 0.0 vs #218) | **adopted** (fastest so far, 2.26x) |
| — | 2026-10-09 | (Re-measuring the bottleneck after #118) Timing-only ring SDPA with work skipped (normal 17.19 ms): no exp 14.32 (−2.87), no matmul 14.84 (−2.35), no P row-sum accumulation 15.22 (**−1.97**; −1.47 before #118), no P write-out 15.55 (−1.64; −2.44 before), no subtraction 16.72 (−0.47). The per-thread measurement also shows that batched writes reduced the pack thread's QKᵀ and P·V write-outs from 3.08 → 2.45 M and 3.09 → 0.60 M, and pack now waits for math during the correction-factor computation. A timing-only variant that accumulates the sum only once per row (wrong values): 17.19 → 16.03 ms | timing-only switch | — | — | — | — | — | measurement only |
| **119** | 2026-10-09 | The P row sums were computed by writing the P tiles one at a time into the row-sum tile with accumulation in L1 (4 tiles per row × 2 rows = 8 writes). The preceding P write-out has already set up the width-4 batched-write MOP, so setting the pack general-purpose register `OUTPUT_ADDR_OFFSET` (the value added to the output address per tile in batched writes) to 0 makes the 4 tiles accumulate one after another into the same L1 tile. That takes one write per row and no MOP reconfiguration. Afterwards the register is restored to its original value (the size of one tile). The first subblock of a row (which must overwrite) is still written one tile at a time. The first version (reconfiguring the MOP twice per call) was slower at 17.35 ms; the version that reuses the MOP as is gives 17.19 → **16.84 ms** in the block test, PCC unchanged at 99.9647% | ring SDPA compute and factory (`TT_RING_SDPA_SUM_STRIDE0=1`, commit `fda8370cc5d`) | denoise 48.32 → **47.68–47.71 s** (steady-state step about 1,204 → 1,188 ms) | −1.3% | **48.29 / 48.31** | −1.3% | no change (mean pixel difference 0.0 vs #219) | **adopted** (fastest so far, 2.29x) |
| **120** | 2026-10-09 | In the per-thread measurement after #119, the pack thread's "subtract + exp" is the largest item at 8.35 M cycles (about 204 cycles per score tile). Exp calls `exp_packthread_tile` once per P tile, and inside it `_llk_math_eltwise_unary_sfpu_params_` sets the DEST write address and, at the end, waits for the SFPU to finish (`STALLWAIT … WAIT_SFPU`) every time, so the SFPU pipeline drained between tiles. The body of the approximate exp (32 iterations, replay and `SFPLOADMACRO`) advances the DEST address by one tile on its own, so the bodies for all tiles of the subblock (2×4 = 8 tiles) now run back to back inside one call (one setup and one completion wait). The computation is the same; in the block test ring SDPA goes 16.84 → **16.36 ms**, PCC unchanged at 99.9647% | ring SDPA compute and factory (`TT_RING_SDPA_EXP_BATCH=1`, commit `6b87c776e71`) | denoise 47.71 → **46.93 s** (steady-state step about 1,188 → 1,169 ms) | −1.6% | **47.54** | −1.6% | no change (mean pixel difference 0.0 vs #222) | **adopted** (fastest so far, 2.32x) |
| 121 | 2026-10-09 | Same idea as #120 for the AGMM post-processing (bias + GELU; GELU runs on the SFPU of the pack thread): add in batches of half of DEST, and compute GELU back to back in a single SFPU call. In the block test FFN1 stays at 2.00 → 2.00 ms (PCC also unchanged). In the real FFN1 the SFPU wait in post-processing is hidden behind other work | `TT_AGMM_ACT_BATCH=1` (reverted) | no change | — | — | — | PCC unchanged | rejected |
| 122 | 2026-10-09 | In ring SDPA Phase 1, alternate "subtract + exp of the previous row" and "QKᵀ of the current row" two subblocks at a time instead of one, halving the subtraction and matmul initializations. In the block test ring SDPA gets slower, 16.36 → 16.60 ms (PCC unchanged). The loss from breaking the overlap of exp (pack) and QKᵀ (math) is larger than the gain from fewer initializations | `TT_RING_SDPA_PAIR=1` (reverted) | worse | — | — | — | PCC unchanged | rejected |
| — | 2026-10-09 | (Ring SDPA time per phase after #120; 17.0 M cycles/core, slightly slower because of the instrumentation) Phase 1 (QKᵀ and softmax): unpack 7.86, math 7.93, pack 8.06 M; Phase 2 (P·V and correction): 9.04, 9.02, 8.83 M. For each K chunk the order is "QKᵀ (row group 0, math only) → QKᵀ (row group 1) + exp (row group 0) → P·V (row group 0) + exp (row group 1) → P·V (row group 1, math only)", so pack work (exp) exists only in the middle two zones (at the end of Phase 2, pack waits 2.3 M for the correction-factor computation). This is a structural property. Chunk re-measurement (on top of #120): (128, 1024) is best at 16.44 ms, (256, 512) 17.73, (128, 512) 17.91 | timing only | — | — | — | — | — | measurement only |
| **123** | 2026-10-09 | With cycle counters in the fused distributed LayerNorm (adaLN norm1, norm2, norm3, about 0.3 ms each): of about 300,000 cycles per call, PRE (per-chip mean and variance with Welford's method) takes about 120,000 (about 750 cycles per tile; each tile is transposed and updated sequentially on the SFPU), combining the statistics (including the wait for the cross-chip all-gather) 75,000–105,000, and POST (4-stage affine transform) 75,000–86,000. Replaced PRE for the case where the input rows are in L1 and TP > 1: the sum accumulates the x tiles into DEST (fp32) with a row-wise REDUCE_ROW SUM; the sum of squares computes x × x per tile, accumulates it in L1 into one fp32 tile (`combine_cb`, which is unused during PRE) and then does one REDUCE_ROW; mean = sum ÷ 1280, variance = max(sum of squares ÷ 1280 − mean², 0). The result lands in column 0, so it is transposed into row 0 to match Welford's layout (the statistics all-gather sends only row 0, so leaving it in column 0 gave nan). Block PCC 99.9647 → 99.9646% | fused distributed RMSNorm/LayerNorm factory and LayerNorm compute (`TT_DLN_FPU_PRE=1`, commit `4ee770bbb48`) | denoise 46.93 → **46.72 s** (steady-state step about 1,169 → 1,164 ms) | −0.4% | **47.33** | −0.4% | same composition; the butterfly's wing angle differs slightly (mean pixel difference 12.4 vs #223, `img/wan22_123_ln_fpu_pre_cmp.jpg`) | **adopted** (fastest so far, 2.33x) |
| 124 | 2026-10-09 | In ring SDPA Phase 2, move the P·V of the last row group (over all of K, a math-only zone) into the drain (the part that alternates row group 1's subtract + exp and row group 0's P·V per K subblock), one K subblock at a time, to overlap it with pack's exp (the output accumulates in L1, and before reading P it waits for a pack → unpack signal). In the block test ring SDPA **gets worse, 16.36 → 16.92 ms**, and PCC drops 99.9647 → 99.9639%. The per-subblock signal waits and the accumulating writes to L1 outweigh the gain from the overlap | `TT_RING_SDPA_DRAIN_BOTH=1` (reverted) | worse | — | — | — | PCC 99.9639% | rejected |
| — | 2026-10-09 | In the ring SDPA softmax, enable pack ReLU after exp (right before the P write-out) instead of before it, so that the pack wait for switching the ReLU setting overlaps with the SFPU exp. In the block test ring SDPA 16.47 → 16.49 ms (no change), PCC unchanged | `TT_RING_SDPA_RELU_LATE=1` (reverted) | no change | — | — | — | PCC 99.9647% | rejected |
| **125** | 2026-10-09 | With the #123 counters, the second half (POST) of the fused distributed LayerNorm takes about 85,000 cycles per call. On the rows in L1 it ran 4 stages, x − mean (column broadcast) → × 1/σ (column) → × weight (row) → + bias (row), each with its own unpack, compute and pack round trip through a BF16 intermediate buffer. The WH LLK supports binary ops that move the DEST value to srcA and use it ("DEST reuse", `DEST_TO_SRCA`) combined with broadcast (the math MOP does `MOVD2A` and `ZEROACC` per face). So within one DEST acquire per block, stage 1 is the normal `sub_tiles_bcast_cols`, stages 2–4 use DEST reuse, and pack happens once. DEST-reuse math waits for srcA dvalid once per face, so unpack keeps the COL/ROW MOP shape of `unpack_AB` and only replaces the srcA read with `ZEROSRC` + `SET_DVALID` (`SET_DVALID` alone hangs; the COL/ROW variants of `unpack_A` give one dvalid per tile, which does not match). POST goes from about 85,000 → about 71,000 cycles (about 69,000 with the srcA read skipped), block PCC 99.9646 → 99.9660% (only one rounding to the BF16 intermediate) | fused distributed RMSNorm/LayerNorm factory and LayerNorm compute (`TT_DLN_POST_FUSE=1`, commit `28625819a2f`) | denoise 46.71 → **46.67 s** | −0.1% | **47.28** | −0.1% | the composition changes, picture quality is comparable (mean pixel difference 36.6 vs #225, `img/wan22_125_ln_post_fuse_cmp.jpg`) | adopted (together with #126) |
| **126** | 2026-10-09 | After #125, POST takes about 430 cycles per tile. LayerNorm runs in HiFi4, and the 2 DEST-reuse multiplications (× 1/σ, × weight) take 4 passes per face. Switching only these 2 to HiFi2 brings POST from about 69,000 → **about 55,000 cycles**, block PCC 99.9670%. LoFi gives about 53,000 cycles but fails the block PCC check | same as above (`TT_DLN_POST_FID=2`) | denoise 46.67 → **46.62 s** (steady-state step about 1,162 ms) | −0.1% | **47.22** | −0.1% | the composition changes, picture quality is comparable (mean pixel difference 49.4 vs #225, `img/wan22_126_ln_post_fuse_hifi2_cmp.jpg`) | **adopted** (fastest so far, 2.34x) |
| 127 | 2026-10-09 | Applied the same method as #125 (after the COL-broadcast multiply, multiply by the weights with DEST reuse + ROW broadcast) to the second half of the q/k RMSNorm + RoPE, combining x × (1/rms) and × weight into one DEST acquire. The attempts around #103 expanded the weights to all tiles and then multiplied with DEST reuse, and PCC collapsed to 95%. With this method PCC is 99.9665%. The first version also applied to the RMSNorm without RoPE (the cross-attn Q normalization), skipped the weight stage that should write to `output_cb`, and hung (fixed by adding "has RoPE" to the condition). In the block test the q/k normalizations stay at 0.20 / 0.20 ms (the statistics all-gather wait dominates). e2e, including HiFi2 for the multiply: 47.22 → 47.26 s | `TT_DRMS_WFUSE2=1 TT_DRMS_POST_FID=2` (reverted) | no change | — | 47.26 | — | the composition changes (mean pixel difference 44.7 vs #227) | rejected |
| — | 2026-10-09 | (Breakdown of a steady-state step after #126, device profile, chip 0) One block is 28.24 M cycles: ring SDPA 16.33, 6 matmuls 10.17 (QKV 2.01, FFN1 2.01, FFN2 2.61, self-attn output projection 1.23, cross Q 1.09, cross output projection 1.22), 6 normalizations 1.26, cross SDPA 0.37, small modulation ops 0.03. One step of about 1,162 ms ≈ 40 × 28.24 M cycles ÷ about 0.97 GHz, so there is almost nothing outside the blocks. The 6 `AllGatherAsync` ops in the block test (about 1.7 ms total) are the FSDP weight gather and do not exist in the fully resident (#72) e2e. Removing the measurement-only `WAN_STEP_TIMING=1` (device sync before and after each step) gives 47.22 → 47.20 s | measurement only (`WAN_STEP_TIMING=0`) | — | — | 47.20 | — | identical (mean pixel difference 0.0) | measurement (the measurement procedure is kept unchanged for comparability) |
| **128** | 2026-10-09 | The #115 measurement showed that ring SDPA unpack spends 9.6 M cycles on matmuls alone (about 234 cycles per S tile), and #116 showed that the bottleneck is the number of tiles, not bytes. When ct ≥ rt, `unpack_AB_matmul` reads the in0 (Q) tiles into srcB once per row and reads an in1 (K) tile into srcA for every matmul. The factory's `determine_largest_subblock_size` picked the first candidate, 2×4, so one step along K reads 2 + 8 = 10 tiles. With 1×8 it is 1 + 8 = 9 tiles (the streaming compute supports single-row subblocks, and the Sk chunk of 32 tiles is a multiple of 8). For P·V the head dimension is 4 tiles, so every shape gives 10 tiles for 8 matmuls. In the block test ring SDPA goes 16.36 → **16.12 ms**, block 31.02 → 30.53 ms, PCC 99.9670 → 99.9672% | ring joint SDPA factory (`TT_RING_SDPA_QK_SB=1,8`, commit `e5c7b1d3b0d`) | denoise 46.62 → **46.35 s** (steady-state step about 1,162 → 1,154 ms) | −0.6% | **46.96** | −0.6% | same composition (mean pixel difference 16.0 vs #227, `img/wan22_128_qk_sb18_cmp.jpg`) | **adopted** (fastest so far, 2.35x) |
| 129 | 2026-10-09 | The same override as #128 for the cross-attn SDPA (Q chunk 8 tiles, text K 16 tiles, default 2×4), set to 1×8 | `TT_SDPA_QK_SB=1,8` (reverted) | no change | — | 46.96 | ±0 | same composition (mean pixel difference 13.9 vs #231) | rejected |
| — | 2026-10-09 | (Reviewing how experiments are run) One e2e run takes about 81 s (47 s inference + about 30 s for startup, initialization, etc.), a block test about 2.5 minutes, and a C++ rebuild about 30 s. Speed is compared with a short 8-step e2e (median steady-state step, about 42 s per run, 1.2 ms difference between 2 runs of the same setting), and only candidates are checked for quality with 40 steps. Short e2e comparisons: batched-write threshold 2: 1,154.4 ms; AGMM in1 multicast: 1,157.3; QKᵀ subblock 1×4: 1,205.6 (fixed overhead of about 150 cycles per subblock); without HiFi2 in the second half of the normalization: 1,155.6 (baseline 1,155–1,156). QKV with N block 16 does not fit in L1. Batched exp in the cross-attn SDPA: 46.96 → 46.94 s (identical video) | screening only | — | — | — | — | — | rejected |
| 130 | 2026-10-09 | (Investigating the overlap of exp and matmul) Timing-only: skipping the ring SDPA exp brings the steady-state step from 1,155 → 1,080 ms, skipping the matmuls to 1,045 ms. About one third of the serial estimate for exp (about 2.9 ms/block) is hidden. Prototypes to make exp itself lighter: the standard approximate exp for SDPA takes 2 instructions per row, LOADMACRO + a separately issued SHFT2 (because the macro's Simple unit is used to restore the sign). (1) Writing the rounded integer bit pattern directly into DEST does not work, because 16-bit DEST stores BF16 reordered as "sign, mantissa, exponent" (`DstEncodeBF16`). (2) Run only the compute macro (LD, multiply-add, round to UInt16, shift, store) of the standard "clamp the input to ≥ −88.5" version at 1 instruction per row, keeping the standard exp for masked K chunks: in the block test ring SDPA 16.12 → 15.48 ms, PCC still 99.9672%, 40 steps 46.96 → 45.71 s. But the video becomes hazy with lower contrast. Real Wan logits reach x < −88.5, and the rounding to UInt16 takes the absolute value, so weights that should be 0 become large. Clamping correctly needs a compare (SWAP), which uses the same Simple unit as the shift, giving 3 cycles per row, slower than the standard | `TT_RING_SDPA_EXP7=1` (reverted) | — | — | 45.71 | −2.7% | **picture quality drops** (hazy, mean pixel difference 47.4 vs #231, `img/wan22_130_exp_fast_cmp.jpg`) | rejected |
| 131 | 2026-10-09 | (Re-checking items under 40% efficiency) cross-attn SDPA: timing-only, skipping matmul / exp / both gives steady-state steps of 1,152.9 / 1,154.6 / 1,152.6 ms (baseline 1,155). K/V is already shared through the non-causal chain. A version that, when consecutive Q chunks of a core belong to the same head, drops the chain and copies K/V within the core from the previous CB face gives identical results (the path was confirmed by the output changing when the copy is skipped) but takes 1,155.5 ms, no gain. Per-RISC times: all RISCs run until the end, median 323 µs / max 398 µs (4–5 units over 72 cores), about 80 µs per unit, about 500 cycles per S tile | `TT_SDPA_KV_REUSE=1` (reverted) | no change | — | — | — | identical | rejected |
| — | 2026-10-09 | (Re-checking items under 40% efficiency) The TP-direction all-gather alone (`CCLManager`, 4 links, Ring) on an 8192 × 5120 activation: BF8 0.586 ms, BF16 1.064 ms (saturates at about 58 GB/s received per chip). The output-projection-type AGMM (1.09–1.23 ms) is set by the lower bound of the data transfer inside the op (1.0–1.15 ms) | measurement (`models/tt_dit/tests/models/wan2_2/test_hk_ag_bw.py`, not committed) | — | — | — | — | — | measurement |
| **132** | 2026-10-09 | Added wait-time counters to the AGMM data-movement kernels (extending the #89 measurement to the multicast path). In each of the 3 output-projection-type AGMMs, the injector of each row (8 cores) does all of the following serially: reading the block from DRAM about 310 µs, waiting for free CB space in the row's cores about 300 µs, multicasting to the row about 235 µs, and writing out its own output about 290 µs; the total ≈ the op time. Waiting for data from other chips is about 2 µs, so **the fabric is not the bottleneck**. The row's receivers wait about 850 µs for data, and the in1-side kernel waits 520–630 µs for CB space, so both have slack. When the grid is transposed (M > N, which holds for all Wan AGMMs), writing the output was the in0-side kernel's job. Now the in1 side writes the output and the in0 side reads the post-processing inputs (residual, gate) | AGMM factory (`TT_AGMM_OUT_IN1=1`, commit `417267b44c5`) | denoise 46.35 → **45.56 s** (steady-state step about 1,155 → 1,134 ms) | −1.7% | **46.17** | −1.7% | no change (mean pixel difference 0.0 vs #231) | adopted |
| **133** | 2026-10-09 | Measurement after #132: the injector spends 120–270 µs "waiting for free CB space in the row's cores", and the 2 forwarding cores spend up to about 460 µs sending to the fabric. A forwarding core did "receive → send to the fabric (about 11 µs) → reserve the CB for the next block and signal ready", and the injector waits for all cores in the row to be ready before it multicasts, so the send time became the injector's wait. Now the core signals ready first, then sends the previous block (still in the other half of the CB), and sends the rest at the end | AGMM in0 kernel and factory (`TT_AGMM_FAB_DEFER=1`, commit `9721a618a08`) | denoise 45.56 → **45.31 s** (steady-state step about 1,134 → 1,129 ms) | −0.5% | **45.93** | −0.5% | no change (mean pixel difference 0.0 vs #235, latents also identical) | **adopted** (fastest so far, 2.40x) |
| 134 | 2026-10-09 | Right after the injector issues the multicast of the current block, issue the DRAM read of the next K block (after its arrival wait), then wait for the multicast to complete. The results are identical, but the steady-state step goes 1,129.2 → 1,130.0 ms, no gain. Per-RISC times on top of #133: cross Q projection 0.93 ms (compute about 860 µs, NCRISC 894–931 µs), cross output projection 0.99 ms (compute about 920 µs, NCRISC 959–991 µs); the output-projection-type AGMM has reached the compute lower bound (0.86–0.96 ms, #89) | `TT_AGMM_IN0_PREFETCH=1` (reverted) | no change | — | — | — | identical | rejected |
| — | 2026-10-09 | (Re-measurement on top of #133) Output-projection-type AGMM: compute about 860–920 µs (math about 30 cycles per single-tile matmul; FFN1 about 20), residual post-processing down to 14–65 µs. The injector: read about 305, multicast about 265, row free-space wait 63–66 (120–270 before #133), residual read about 95, wait for data from other chips 97–128 µs (skew between chips); compute and injector are about balanced. Output-projection-type subblocks 1×5 / 4×1 / 2×1: 1,130.1 / 1,129.9 / 1,133.1 ms, no change. QKV reads in0 twice (N of 15 tiles per core, N block 10), but a setting with M block 8 that allows N block 15–16 was worse at 1,155.7–1,157.3 ms | screening, measurement | — | — | — | — | — | rejected |
| 135 | 2026-10-09 | On injector cores only, move the residual and gate reads to the in1-side kernel (the 2nd bit of the first in1 argument says whether the core is an injector). The results are identical, but 1,129.2 → 1,129.9 ms, no gain (injector and compute are balanced) | `TT_AGMM_TERN_OFFLOAD=1` (reverted) | no change | — | — | — | identical | rejected |
| — | 2026-10-09 | (Upper bound of further matmul gains, timing-only, all 5 AGMM types) Skipping the injector's DRAM read: steady-state step 1,129 → 1,120 ms (the upper bound of a read rewrite is 0.36 s over 40 steps). Skipping the matmul compute: 1,094 ms (−1.4 s even if compute were zero; realistically 0.3–0.5 s from an LLK change to the per-tile unpack-to-math hand-off of #118). A version of the #134 early read on the other NoC hangs (NCRISC and NoC command buffers / counters collide) | timing only (reverted) | — | — | — | — | — | — |
| — | 2026-10-09 | (Checking the per-subblock fixed cost in ring SDPA, timing-only) With counters inside the LLK matmul unpack function: 41,000 calls per core, and of the 7.27 M cycles spent inside the function (16.16 M for the whole kernel), 5.41 M is waiting for the next unpack config context to become free. The about 45 cycles of TRISC setup per call are hidden behind the hardware unpacker, and the bottleneck is the unpacker's own processing of about 18–20 cycles per tile. Batching the calls does not help; the only way is to unpack fewer tiles (QKᵀ is at its lower bound since #128; P·V cannot go lower because of the 4-tile head dimension and the 4-tile Q chunk) | timing only (reverted) | — | — | — | — | — | — |
| 136 | 2026-10-09 | Revisited the #130 fast exp as an off-by-default feature. Raising the additive constant of the approximate exp by 256·M multiplies all of P by 2^M (cancelled by the final division in softmax), which widens the correctly computed range from x ≥ −88.5 to x ≥ −(88.5 + M·ln2). Masked K chunks use the standard "clamp the input at the lower bound" version with the same constant. With M = 64, block PCC stays at 99.9672%, ring SDPA 16.12 → 15.47 ms, 40 steps 45.93 → 44.91 s. But the video is hazy, as in #130. A version that applies the lower-bound clamp to every chunk (slow, 49.88 s) is not hazy (same composition, mean pixel difference 16.4). So real Wan logits go far below x = −132 (a pre-scale difference of about 1,500), and no M allowed by the BF16 exponent range can cover that. Correct clamping competes with SHFT for the Simple unit and takes 3 or more cycles per row, slower than the standard exp (about 2.1). Quality cannot be kept, so it is not adopted even as an option | `TT_RING_SDPA_EXP_FAST=64` (reverted) | — | — | 44.91 | −2.2% | **picture quality drops** (hazy, mean pixel difference 47.4 vs #236; not hazy when every chunk goes through the lower-bound clamp. `img/wan22_136_exp_fast64_cmp.png`, `img/wan22_136_exp_safeall_cmp.png`) | rejected |
| 137 | 2026-10-09 | A workaround for the #136 problem that "rounding to UInt16 takes the absolute value". Instead of using the rounding unit, add 1.5 × 2²³ (a magic number) to the multiply-add constant; the low bits of the fp32 result then hold the rounded integer k in two's complement. Shifting it left by 15 bits gives: k ∈ [0, 65535] → the same BF16 bit pattern as before; k ∈ [−65536, −1] → a negative number (0 after the pack ReLU at the P write-out); −∞ → 0 (still 1 instruction per row). With M = 100 the correctly handled range is x ≥ −335 after scaling (a pre-scale difference of about 3,800, about 2.5x that of #136). Block PCC 99.9672%, ring SDPA 15.48 ms, 40 steps 44.91 s, but the video is hazy. Real Wan logits go below this window. The window is capped by the 16-bit output width (about 177 octaves), and M by the BF16 exponent range. Full clamping needs a compare or a sign restore; both use the same Simple unit as the shift, giving 2–3 instructions per row, which is no faster than the standard exp. Conclusion: a faster exp that keeps quality is not possible | `TT_RING_SDPA_EXP_FAST=100 TT_RING_SDPA_EXP_MAGIC=1` (reverted) | — | — | 44.91 | −2.2% | **picture quality drops** (hazy, mean pixel difference 41.2 vs #236, `img/wan22_137_exp_magic100_cmp.png`) | rejected |
| **138** | 2026-10-09 | Fixed the cause of the haze in #136 and #137 (when the input falls below the correctly computed range, weights that should be 0 become large bogus values) by removing the range altogether. The standard approximate exp (Schraudolph) builds the bit pattern of exp(x) as k ≪ 15 (k = round(A·x + (B−C))). #138 instead places the multiply-add result at the top of the FP32 range as y = 2¹²⁷ × (1 + (M1 − k)·2⁻²³). The smaller k is (the further the score is below the row max), the larger y gets; for k ≤ M1 − 2²³, y overflows to ∞ (a −∞ input also gives ∞). Shifting y left by 15 bits with the Round unit's shift (SFPSHFT2) drops the sign and exponent and keeps only the low 17 bits of the mantissa. The Simple unit's integer subtraction (SFPIADD) subtracts this from the bit pattern of −1.0. M1 = 8,355,584 is chosen so that −1.0 ≡ M1 ≪ 15 (mod 2³²), so the result is exactly k ≪ 15. k ≥ 0 gives the same bit pattern as the standard; k ∈ (M1 − 2²³, 0) gives a negative number and ∞ gives −1.0, and both become 0 through the pack ReLU at the P and row-sum write-outs. Each of the macro's 4 units (Load, MAD, Round, Simple) gets one instruction, so it is still 1 instruction per row (the standard takes 2 instructions, about 2.1 cycles), there is no input range limit, and masked chunks are handled as is. An earlier idea, "flush to 0 through denormals", does not work because intermediate products below 2⁻¹²⁷ are discarded (according to the bit-accurate FMA model `fma_model_wh` in the ISA documentation). A simulation with that model matches the standard exp for all 32,640 negative BF16 values and −∞ (relative difference within 0.8%; everything outside the range is 0). Block PCC stays at 99.9672%, ring SDPA 16.12 → 15.40 ms | `compute_streaming.hpp` and the ring SDPA factory (`TT_RING_SDPA_EXP_OVF=1`, commit `2062e93228e`) | denoise 45.31 → **44.15 s** (steady-state step about 1,129 → 1,100 ms) | −2.6% | **44.76** | −2.5% | not hazy; same composition with small detail differences (mean pixel difference 14.6 vs #236; the #136 diagnostic version with the lower-bound clamp on every chunk is 16.4, the hazy #136 and #137 are 47.4 and 41.2. `img/wan22_138_exp_ovf_cmp.png`) | **adopted (optional, off by default)**. 2.47x when enabled |
| — | 2026-10-09 | (Measurement) Per-thread breakdown of ring SDPA on top of #138 (the #115 measurement re-applied; M cycles per core, total 15.98). Unpack: QKᵀ 4.70, P·V 3.69, sub+exp 1.80, max 0.65, correction (SALAD) 0.73, correction factor 0.20; math: 3.51, 4.21, 3.46, 0.84, 0.65, 0.10; pack: 1.88, 0.32, 7.06, 0.33, 0.41, 2.32. With the standard exp the total is 16.69 (pack sub+exp 7.86). Making the correction-factor exp approximate only reduces 2.32 → 2.24, so the pack thread's correction-factor time is waiting, not compute (the SFPU waits for the FPU to become free). Measurement-only variants that skip work: ring SDPA 15.40 ms → no correction factor 15.21, no correction factor and no correction 14.70, no max for the 2nd and later chunks 14.35 (PCC 99.9656%), no max, correction factor or correction 13.54 (−12%) | measurement-only switch (reverted) | — | — | — | — | — | measurement |
| 139 | 2026-10-09 | Revisited #82's "reuse the max" on top of the #138 exp (which also handles positive inputs). The first ring iteration (the 4 local K chunks) computes the true max; later iterations fix that value as the reference (no max computation; the correction factor is exp(0) = 1). Block PCC 99.9635%, ring SDPA 15.40 → 14.49 ms, 40 steps 44.76 → 43.25 s, but the video breaks (a version that scales the P output by 2⁻⁶⁴ to widen the headroom for positive inputs shows blocky speckles; the unscaled version shows haze and horizontal streaks). Saving the ring SDPA input Q and K and analyzing them on the CPU: in most layers the true row max is within 15 nat of the first chunk's max and within 11 nat of the local K's max. But layers 39 and 40 are off by orders of magnitude: in layer 40 the max is up to 280 nat above the first chunk and up to 44 nat above the local K; in layer 39 at step 20 it is 142 nat above the local K, and the spread within a row reaches 1,200 nat. The #138 exp computes positive inputs correctly only up to 88 nat (scaling by 2⁻⁶⁴ widens that to 133 nat, but then the row sum overflows past 2¹²⁸), so a fixed reference does not work. The haze in #136 and #137 was also caused by this range in layers 39 and 40 (wider than the 132–335 nat window) | `TT_RING_SDPA_FIXED_MAX=1` (reverted) | ring SDPA 15.40 → 14.49 ms | −5.9% | 43.25 | −3.4% | **broken** (2⁻⁶⁴ version: speckles; unscaled version: haze and horizontal streaks. `img/wan22_139_fixed_max_cmp.png`, `img/wan22_139_fixed_max_s0_cmp.png`) | rejected |
| 140 | 2026-10-09 | Move the exp of the correction factor exp(previous max − current max) from the pack thread to the math thread (computed right after the subtraction, inside the same DEST acquire). The pack thread's SFPU was waiting for a free FPU (after the P·V matmul), and the aim was to remove that wait. Block PCC is unchanged (99.9672%), but ring SDPA gets slower, 15.40 → 15.76 ms. The math thread is busier (about 80% utilization), and the pack thread's wait was not on the critical path | `TT_RING_SDPA_CORR_MATH=1` (reverted) | ring SDPA 15.40 → 15.76 ms | +2.3% | — | — | — | rejected |
| — | 2026-10-09 | (Measurement) Checking the FFN2 bottleneck (fused matmul + reduce-scatter + residual op, 2.55 ms). Running only a matmul of the same shape on the same 8×7 grid as the fused op (56 cores; the top 2 rows are RS workers) takes 2.548 ms, the same as the whole fused op. On 8×8 it takes 2.24 ms, on 8×9 (72 cores) 2.00 ms. RS alone as a standalone op takes 0.75 ms in BF8 (3–4 workers/link; BF16 1.35 ms). So FFN2 is bound by the 56-core matmul, and RS is hidden. Giving the fused op's matmul 8 rows leaves too few cores for the RS workers and mux, so it does not run; RS with 2 links and 1 worker takes 3.99 ms; 6 rows take 2.73 ms. Splitting it into "72-core matmul + standalone RS" is slower at 2.00 + 0.75. Reducing it needs a big change that moves RS onto the data movement of the matmul cores (the reverse of AGMM) | measurement (`models/tt_dit/tests/models/wan2_2/test_hk_ffn2_mm.py`, `test_hk_rs_bw.py`, not committed) | — | — | — | — | — | measurement |
| **142** | 2026-10-10 | Replaced the FFN2 fused op (#2, #76: matmul on the 56 cores of 8×7, RS workers and mux on the top 2 rows) with a new program in which the matmul cores also do the reduce-scatter themselves. The matmul runs on all 72 cores of 8×9. In each of the 4 ring iterations, each core computes its own M rows × the columns of slice s(k) that it owns (one K loop); from the 2nd iteration on, it adds the partial sums that the neighboring chip computed for the same tiles in the previous iteration. The last iteration is the core's own slice, and it writes out = residual + (matmul + received sum + bias) × gate. The partial sums are relayed by the core at the in1 end of each column (row 9). It reads the blocks of the 9 cores in the column and writes them, over its own fabric connection (4 links × forward/backward, one per core), directly into the receive CB of the core at the same position on the neighboring chip, and advances the receiver's semaphore with the last packet of the block. The receive CB holds 3 iterations' worth, so the sender does not wait for the receiver to consume. To avoid writing into the L1 of another program, before the first send the relay core exchanges an "entered the op" signal with the opposite-direction relay core on the neighboring chip. Speed tuning: (1) Split the 8 cores of a row into 2 groups of 4, each handling a different M range (15 rows) (N is 10 tiles wide per core). Twice as many cores read in0 from DRAM: 2.60 → 2.39 ms. (2) Stagger the position of the in0 reader per row to spread the readers over all columns (2.35 → 2.30 ms). Findings: the bottleneck is the in0/in1 DRAM reads and the chained hand-off (skipping matmul and send still gives 2.16 ms; the matmul alone (transposed grid) is 1.92 ms). The same read size differs 2–3x in speed depending on the reader's position and NOC, and 9 readers in one column is slow. Versions that swap the in0 and in1 RISCs/NOCs (2.6–3.5 ms), share in1 between the 2 groups (2.49 ms), or use 1 group (2.60 ms) are slower. Precision: the first version had PCC 99.9685% (default 99.9790%), and the video composition changed (mean pixel difference 22.9). Comparing with inputs that bf8 and bf16 represent exactly showed that the 2⁻⁶ digit was lost where values are 0.5–1. The cause: the gate multiply, done in the same compute kernel as the matmul, used the matmul's LoFi (5 significant bits for the partial products). The default RS compute is HiFi4, so making only this multiply HiFi4 gives zero error, like the default, and PCC 99.9792% with random inputs. In L1 the CBs take 1.33 MB; in the e2e with the enlarged kernel config region (#109) about 6 KB is left (it fits by shrinking the partial-sum send CB to one block) | new program `hk_ring_mmrs_program.cpp` and 3 kernels (`TT_MMRS_RING=1`, commit `e9431ff8638`) | FFN2 2.55 → 2.30 ms (device time) | −10% | **45.43** | −1.1% | same composition (mean pixel difference 14.6 vs #236. Combined with the fast exp, #240: 44.34 s, mean pixel difference 11.3. The version that kept LoFi: 22.9, composition changed. `img/wan22_142_ring_mmrs_cmp.png`, `img/wan22_142_ring_mmrs_lofi_gate_cmp.png`) | **adopted**. Combined with the fast exp (#138): **44.34 s** (2.49x) |
| **143** | 2026-10-10 | Tightening the #142 bottleneck (in0 DRAM reads and the chained hand-off). (1) Switch to dynamic NOC mode (each RISC can use both NOCs) and do only the DRAM reads of the in0 kernel (RISCV_1, which writes on NOC1) on NOC0. Earlier measurements showed the same DRAM read is about 3x slower on NOC1 than on NOC0 (the same trend for in0 and in1 in the swap experiment), and #142's in0 read on NOC1. Still chained: 2.30 → 2.07–2.13 ms. (2) Each of the 4 cores in a row group reads only its own 2 rows and writes them into the same CB position on the other 3 cores. For each K block, the cores tell each other "my CB is free" and "I wrote my rows" by block number (the per-member flags live in program semaphores, so they start at 0). Combined with dynamic NOC: 1.96–1.97 ms (about the same as the matmul alone (transposed grid), 1.92 ms). Without dynamic NOC the shared reads were slower (2.55 ms). The 1-group configuration takes 2.30 ms even with dynamic NOC, and M block 15 (half the in1 reads) is slower at 2.42 ms. 40 calls with the same input give bit-identical outputs. Two consecutive 4-step e2e runs also give identical latents (the difference in the #142 vs #143 comparison was the divergence of the first run right after a rebuild) | `hk_ring_mm_in0.cpp`, `hk_ring_mmrs_program.cpp` (on by default; `TT_MMRS_RING_IN0_GATHER=0` and `TT_MMRS_RING_DYN=0` restore the old behavior; commit `765dad46ac0`) | FFN2 2.30 → 1.97 ms | −14% | **45.00** | −0.9% | same composition (mean pixel difference 3.6 vs #247. Combined with the fast exp: 43.88 s, mean pixel difference 9.3 vs #248. `img/wan22_143_ring_gather_cmp.png`) | **adopted**. Combined with the fast exp (#138): **43.88 s** (2.52x) |
| **148** | 2026-10-10 | (Fix) A race in the #143 in0 shared reads. Each core in a row group writes its own rows into the same CB position on the other 3 cores and then writes a "written" flag (block number). The rows go out through the write command buffer and the flag through the register command buffer, so the flag could arrive before the tail of the row write (about 20 KB, split into several packets), and the receiver did the matmul on stale rows. How it was found: run the block test 600 times in a row, mixing a little of the previous output into the next input (so an error persists to the end), and compare the hash of the final output. With the default configuration two runs disagreed; with `TT_MMRS_RING=0` they matched; in the #142 form (no shared reads, no dynamic NOC) they matched; enabling only the shared reads gave a different value every time. The 4-step and 40-step e2e runs happened to be bit-identical every time. Fix: insert `noc_async_writes_flushed()` after writing the rows for all members, then write the flag (on the same VC and route, the later flag cannot overtake the rows). Also, the shared DRAM reads now wait for completion on a dedicated transaction ID (in dynamic NOC mode, the other RISC of the same core also reads on the same NOC). After the fix, the 600-iteration test always gives the same value as the #142 form (3 out of 3). FFN2 unchanged at 1.94 ms, and the 40-step e2e matches the pre-fix result hash at every step | `hk_ring_mm_in0.cpp` (commit `78f4f5012f6`). Test: the measurement-only `HK_BLOCK_ITERS`, `HK_BLOCK_VARY=chain` and `HK_BLOCK_HASH` in `test_transformer_wan.py` (not committed) | FFN2 1.94 ms (unchanged) | 0 | 45.00 | 0 | same (hashes match at every step) | **fix** (correctness) |
| **149** | 2026-10-10 | (Fix) Even after #148, the 2000-iteration block test that mixes in the previous output gave a different value in about 1 of 5 runs with the default configuration. Two more places had the same kind of gap: the #142 in1 chain forward (each core in a column writes the 9×10-tile in1 block to the next core and then writes a "valid" flag), and the in0 chain forward used when shared reads are off. Both wrote the flag with the register command buffer right after writing the block with the write command buffer, and lacked the `noc.async_writes_flushed()` that minimal_matmul has (if the flag arrives before the tail of the block, the next core does the matmul on stale in1; also, in1 is forwarded after it is handed to the core's own compute, so there was a window in which the next block could arrive in the CB slot that was still being sent). After adding the flush, the 2000-iteration test gave the same value 10 out of 10 times (before the fix, 1 in 5 differed). FFN2 unchanged at 1.94 ms | `hk_ring_mm_in1.cpp`, `hk_ring_mm_in0.cpp` (commit `4e4aa653474`) | FFN2 1.94 ms (unchanged) | 0 | 45.00 | 0 | same | **fix** (correctness) |
| 150 | 2026-10-10 | Remove the dedicated cores for the ring SDPA K/V all-gather; instead, the 8 writer cores of the last SDPA row relay K/V to the neighboring chip one hop at a time (store-and-forward through DRAM, heads split across the links). SDPA can then use all 72 cores of 8×9, and each core has 9 Q chunks instead of 10. Ring SDPA 15.39 → 13.89 ms, block 32.97 → 31.68 ms, 40 steps 45.00 → 43.48 s (with the fast exp `TT_RING_SDPA_EXP_OVF=1`: 43.88 → 42.57 s). However, in the 2000-iteration block test that mixes in the previous output, about 30% of the runs give a different value (the base gives the same value in all of 20+ runs). The cause was not identified. A version that finishes the relay before the writer's own work gives the same value every time but is slower than the base (ring SDPA 16.79 ms). See "#150 details" | `TT_RING_SDPA_RELAY=1` (not committed) | 31.68 (ring SDPA 13.89) | −3.9% | (43.88 → 42.57) | −3.0% | PCC unchanged. About 30% of the 2000-iteration tests give a different value | **parked** (not deterministic) |
| 151 | 2026-10-10 | Changed how FFN2 (#142) distributes in1 (the weights) to the same shared reads as the #143 in0. Each of the 9 cores in a column reads one K row (10 tiles) of the in1 block from DRAM and writes it directly into the same CB position on the other 8 cores of the column (before, the head of the column read 90 tiles and forwarded them along the chain). The flags live in a small CB, and the block number written includes a tag that changes per call (a runtime argument), so they are not confused with leftovers from a previous call or from other ops (the atomics and write barriers that made the #145 shared reads slow are not used). Results are correct (PCC unchanged), but FFN2 goes 1.97 → **3.75 ms** (about 39 µs per K block, against about 15 µs of compute). With all-to-all writes inside a column, the column's vertical NOC links carry about 2x the traffic of the chain. Multicast would carry the same amount as the chain, but dynamic NOC mode (needed for the fast in0 DRAM reads) combined with multicast hangs (#144, #147) | `TT_MMRS_RING_IN1_GATHER=1` (reverted) | FFN2 3.75 ms | worse | — | — | PCC unchanged | rejected |
| — | 2026-10-10 | (Measurement) In the ring SDPA QKᵀ, remove the transpose of the K tiles during unpack (measurement only; the result is wrong, PCC 98.80%). Ring SDPA goes 15.39 → 15.43 ms, unchanged, so the transpose costs nothing. Storing K pre-transposed would not speed up SDPA, which is bound by unpack, so this is not pursued | measurement only (kernel edited directly, reverted) | — | — | — | — | — | measurement |
| — | 2026-10-10 | (Measurement) Switch the FFN2 (#142) matmul to the no-MOP version that ring SDPA uses (`matmul_block_no_mop`, streamed directly from the replay buffer). The op goes 1.94 → 1.97 ms (unchanged). A compute-only bench also gives the same 20.22 → 20.21 cycles per tile matmul. The gap to the LoFi MVMUL floor of 16 cycles (about 4 cycles) is explained by unpack, not by MOP startup (subblock 4×2 unpacks 10 tiles for 8 matmuls; at about 16 cycles per tile that is about 20 cycles per matmul). For this shape, rewriting the LLK has no headroom | measurement only (`TT_MMRS_RING_NOMOP=1`, `TT_MMRS_RING_BENCH=1`, reverted) | — | — | — | — | — | measurement |
| 153 | 2026-10-10 | Retry of #29 (with the current ring SDPA kernel). Store QKᵀ and P (`cb_qk_im`) in BFP8 to halve the bytes that softmax and P·V write and read. In the broken state ring SDPA goes 15.40 → 12.82 ms (PCC nan). Causes of the nan found: the padding mask is added into `cb_qkt_im` with L1 accumulation (block-float cannot be L1-accumulated); the max and row-sum packs write to bf16 CBs with the packer still set to BFP8 (the spot found in #29); the row-sum trick of #119 reuses the P pack MOP for the bf16 row sum. Even after dropping the mask (padded K/V are zero), switching the packer format back and forth around the max and row sum, and removing #119, the result stays nan, at 16.21 ms, slower than base (switching the packer config per subblock costs more than the halved bytes save). The history was not checked before the experiment, and #29 was repeated | `TT_RING_SDPA_QKIM_BF8=1` (reverted) | ring SDPA 16.21 ms | worse | — | — | PCC nan | rejected |
| 144 | 2026-10-10 | Apply the #143 finding "DRAM reads are slow on NOC1" to AGMM too (the QKV, FFN1 and output projection matmuls). Wan's AGMM uses a transposed grid, and the RISC on NOC1 reads the weights (in1) from DRAM. A version that puts the data movement kernels in dynamic NOC mode and reads the weights on NOC0 hangs in the 8-step short e2e. The #132 measurement shows the in1 side of AGMM waits 0.5–0.6 ms for free CB space, so it is not the bottleneck; a fix is unlikely to help, and this is not pursued further | `TT_AGMM_DYN=1` (reverted) | — | — | — | — | hang | rejected |
| — | 2026-10-10 | (Measurement) Headroom of candidate 4 (the matmul LLK). Added a measurement-only compute-only mode to the #142 op (keep one block of in0 and in1 in L1 and run the same K loop as the op) and derived cycles per tile matmul from the math thread time (bf8, LoFi). Subblock 4×2 takes **20.2 cycles**, 2×2 22.2, 1×5 23.7, 8×1 25.3. Against the MVMUL floor (16 cycles with LoFi), 4×2 adds about 4 cycles (26%), so rewriting the LLK can shrink compute by at most about 20%. Also, FFN2 compute alone is 69,120 tiles × 20.2 = about 1.40 ms, 0.57 ms away from the op's 1.97 ms, so it is not compute-bound (the rest is the per-K-block hand-off waits and synchronization) | measurement only (`TT_MMRS_RING_BENCH=1`, reverted) | — | — | — | — | — | measurement |
| 145 | 2026-10-10 | An attempt to cut the bottleneck left after #143 (the top core of each in1 column reads 90 tiles per K block and forwards them down; receivers spend 1.05–1.1 ms of 1.9 ms waiting for in1). (1) Share in1 across the 9 cores of the column (each core reads one K row of 10 tiles and multicasts it to the whole column; free space and arrival are counted with two counter semaphores, and arrival is added atomically after the write barrier): works correctly but takes 2.84–2.88 ms (18 atomics and barrier syncs per K block are expensive). (2) M block 15 (the whole M range of each core; in1 reads and forwards are halved) with K block 3 or 4: 1.99–2.13 ms (K block 6 does not fit in L1). Both are slower than 1.97 ms | `TT_MMRS_RING_IN1_GATHER=1`, `TT_MMRS_RING_BLK=15,3,3,2` and others (reverted) | FFN2 1.97 → 2.84–2.88 / 1.99–2.13 ms | worse | — | — | — | rejected |
| 146 | 2026-10-10 | Step 1 of candidate 1 (lazy rescale: apply the ring SDPA correction only when needed). Update a row's reference max only when the chunk max exceeds it by τ nat or more (rows that do not exceed it keep the old value and get a correction factor of exp(0); the same factor multiplies O and the row sum, so it cancels in the final normalization and the result is mathematically the same). Right after the max reduce, the math thread copies the previous max into DST, and the SFPU selects "the previous value if new ≤ previous + τ". Block PCC: fast exp (#138) + τ 10 nat gives 99.9681% (fast exp alone 99.9692%); the default exp + τ 1 nat gives 99.9689%, but the default exp + τ 10 nat gives 99.65% (the default exp cannot compute positive inputs correctly). However, the check overhead slows ring SDPA by 0.5–0.6 ms: 15.41 → 16.02 ms (fast exp), 16.14 → 16.66 ms (default). Step 2 (for chunks where every row keeps its reference, skip the correction and SALAD and add P·V directly into the previous O) can save at most about 0.7 ms per the #139 measurement, and the last chunk of each ring pass cannot be skipped because it spills to DRAM, so the net gain is almost nothing. Making the check free, e.g. by storing the max shifted by τ, does not work either, because the update frequency goes back to that of the standard path. Stopped | `TT_RING_SDPA_LAZY_MAX=<nat>` (reverted) | ring SDPA 15.41 → 16.02 ms | +4.0% | — | — | — | rejected |
| 147 | 2026-10-10 | Follow-up to #145. The top core of each in1 column sends the 90 tiles it reads per K block to the other 8 cores of the column with one multicast instead of a chain of forwards (receivers report free space to the top core, which waits for all of them before sending). Without dynamic NOC (each RISC's NOC fixed), it takes 2.57–2.63 ms, no faster than the chained version under the same conditions (2.55 ms). Combined with dynamic NOC mode (the #143 default), it hangs on the first call (no device timeout; like #144 for AGMM, the combination of dynamic NOC and multicast hangs). Also re-measured the blocking on top of #143: K block 6 gives 1.98–2.08 ms, 12 gives 1.99–2.07 ms, 4 gives 2.14–2.21 ms, subblock 2×2 gives 2.03–2.07 ms; the default (M 8, K 9, 4×2, 1.96–1.97 ms) stays the fastest | `TT_MMRS_RING_IN1_MCAST=1` (reverted) | FFN2 2.55 → 2.57–2.63 ms (no dynamic NOC); hang with dynamic NOC | worse | — | — | — | rejected |
| — | 2026-10-10 | (Measurement) Upper bound of candidate 3 (merge the q and k RMSNorm + RoPE into one op). A measurement-only version that skips the k normalization and uses the q output directly as K shrank the steady-state step a lot in the short e2e (8 steps): 1,078.0 → 1,058.3–1,059.3 ms (−0.49 ms/block). But replacing the k normalization with a copy of q (ttnn.clone) gives 1,073.5 ms (−0.11 ms/block), which matches the device time of a single block (28.63 → 28.51 ms). The large gain was a side effect: passing the same tensor as Q and K made the ring SDPA reads favorable. Removing one normalization is worth only about 0.1 ms/block. Even when merged, the k-row processing remains, so the expected gain is 0.05–0.1 s in e2e. Not worth a large rework of the op internals (the stats buffer and data movement kernels); not pursued | measurement only (`HK_SKIP_KNORM`, reverted) | — | — | — | — | — | measurement |
| **154** | 2026-10-10 | (Optional configuration) Run dispatch on Ethernet cores and grow the compute grid from 8×9 to 8×10. Standard Tensix dispatch reserves the 8 cores of the 10th row. On Galaxy all 16 Ethernet links of each chip are connected and used by the fabric, so standard Ethernet dispatch stops with "no free cores". On the SP axis (the ring of 8 chips), 2 of the 4 links on each edge are removed from the fabric at both ends (the same mechanism used to drop unstable UBB links, `HK_DISABLE_ETH_CHANS`). The Ethernet cores of the removed links become "free" and run prefetch, dispatch and dispatch_s. On 8×10, ring SDPA uses the 72 cores of 8×9 (the #150 zone split: 64 cores × 9 chunks + 8 cores × 8 chunks, no relay) plus the all-gather on the 10th row (2 links), and the QKV and FFN1 matmuls split N over 9 rows (1.97 → 1.78 ms, 2.01 → 1.75 ms). Other matmuls keep the 8×9 blocking (`WAN_MM_GRID=8,9`). In the block test without FSDP, one block goes 26.47 → 25.19 ms (ring SDPA 15.74 → 15.01, cross-attn SDPA 0.37 → 0.31). Ethernet dispatch itself adds no overhead (limiting the grid to 8×9 gives the same op times and PCC). The 2000-block chained block test matched in 6 of 7 runs (the one mismatch was the first run right after switching from Tensix dispatch) | `source models/tt_dit/tests/models/wan2_2/eth_dispatch/env.sh` (`eb7c75c149f`) | 25.19 | −1.28 (no FSDP) | 43.01 (fast exp: 42.14) | −1.99 (−1.74) | same composition (mean pixel difference 12.2 vs #249, 8.4 vs #250) | **adopted (optional)** |
| — | 2026-10-10 | Alternatives to #154. (1) Remove only 1 link per SP-axis edge (2 free cores per chip; a new 1-CQ configuration without dispatch_s that runs only prefetch and dispatch, `HK_ETH_NO_DISPATCH_S=1`). With 3 SP-axis links, ring SDPA is 14.17 ms, one block 24.34 ms, 40 steps 41.01 s (fast exp), but the 2000-block chained block test gave a different value in all 4 runs (also 2 of 2 differed without widening the matmuls, and 1 hung). Isolating it: the configuration without dispatch_s is innocent (3/3 match with 2 links). 3 SP-axis links differ in 1 of 8 runs even with Tensix dispatch, and this does not happen with 2 links (6/6) or 4 links (10/10). It is a timing-dependent race when the ring SDPA all-gather runs on 3 links; Ethernet dispatch only changes the timing and makes it show up more often (cause not identified; the per-link worker sync and the atomic notifications with flush look correct). (2) Run FFN2 (#142) on the full 8×10 (`TT_MMRS_RING_FULL=1`, M per core 15 → 13): 1.95 → 1.98–2.02 ms, no faster (bound by the top core of the in1 column, #145). (3) The output projection type matmul (N 1280) keeps N 5 tiles per core, the same as on 8×9, and a non-transposed 7×10 only cuts tiles per core from 160 to 156, so not pursued | `HK_ETH_NO_DISPATCH_S=1` / `WAN_AXIS_NUM_LINKS=1:3` / `TT_MMRS_RING_FULL=1` (included in `eb7c75c149f`, off by default) | 24.34 | — | 41.01 (fast exp) | — | same composition (mean pixel difference 8.4 vs #250) | rejected (3 links are not deterministic) |
| **155** | 2026-10-10 | Isolating and fixing the race in the #154 "keep 3 SP-axis links" layout. In the 2000-block chained block test: (1) a measurement layout where SDPA waits for the whole all-gather before reading (`TT_RING_SDPA_WAIT_ALL=1`, ring SDPA 20.39 ms) still differs 3/3 → it is not SDPA reading early. (2) Really alternate the K and V all-gather buffers per call (`get_ag_ping_pong_buffer` is keyed by shape, so K and V of the same shape took buffers 0 and 1 every time, and there was no alternation across calls; `WAN_AG_PP_FIX=1`): differs 3/3. (3) Stop splitting the transfer of the diametric slice (`TT_RING_SDPA_NO_SPLIT=1`): differs 3/3. (4) Change the removed channels from 7, 11 to 4, 8: differs 3/3 → the link count matters, not the channels. (5) The WH all-gather splits the 512 pages of each head into a contiguous range per link and sends them in scatter packets of 2 pages (Blackhole splits the work by DRAM bank, 4 pages per packet). With 3 links the ranges are 171, 171 and 170, which includes odd sizes (2 or 4 links give even sizes). Using the per-bank scheme on WH too (`TT_RING_AG_BANK_OWNED_WH=1`) gives 6/6 match (the same hash as the Tensix dispatch baseline). Speed with 2 links is unchanged (ring SDPA 15.01 → 14.96 ms; packet size is not the bottleneck). With 3 SP links + per-bank, ring SDPA is 14.13 ms and one block 24.91 ms (#154: 25.19). Adding the layout where QKV and FFN1 use up to the 10th row differs 2/4 (same with the #133 deferred relay turned off; being isolated) | `source models/tt_dit/tests/models/wan2_2/eth_dispatch/env.sh` (`8f988b8187a`; 1 link by default, `HK_ETH_LINKS_TAKEN=2` for the #154 layout) | 24.91 | −0.28 | 42.68 (fast exp: 41.85) | −0.33 (−0.29) | nearly the same (mean pixel difference 3.6 vs #249, 9.3 vs #250) | **adopted (optional)** |
| — | 2026-10-10 | (Measurement) Why 2 and 3 SP-axis links differ. In the wait-for-all layout, ring SDPA takes 23.40 ms with 2 links and 20.39 ms with 3, and the all-gather about 9.2 ms and 6.2 ms (about 4.2 GB/s per channel, nearly proportional to the link count). One slice (K and V, about 22 MB) reaches the neighbor in about 2.65 ms with 2 links and about 1.77 ms with 3. SDPA first computes its own slice (about 1.78 ms of work), so with 2 links it waits about 0.87 ms for the first neighbor slice (matching the 0.84 ms ring SDPA difference between 2 and 3 links). Signaling arrival per head would likely remove the wait even with 2 links | measurement | — | — | — | — | — | measurement |
| **156** | 2026-10-11 | (Fix) On top of #155, when QKV and FFN1 use up to the 10th row (8×9 workers), the 2000-block chained block test differs in 2 of 4 runs. QKV alone 2/4, FFN1 alone 1/4. Synchronizing all chips from the host before starting AGMM still differs 1/4 (it is not skew in chip start times). Turning off #80 (in0 multicast to the whole column) still differs (1 run hung); turning off #132 (write the output from the in1-side kernel) gives 4/4 match. #132 only swaps which kernel writes the output, and the in1 side uses the existing `write_block_sync_granular` (writes row by row). That function freed the CB with `cb_out.pop_front` right after issuing a row's NOC write, and waited for completion (`async_writes_flushed`) only at the end of the block. When the compute side writes the next block into the freed CB, it overwrites rows the NOC has not read out yet (the same function in minimal_matmul already had the fix "finish reading the row before pop"). Waiting for completion per row before freeing gives 4/4 match at the same speed (one block 24.31 ms). (Note: the test that turned off the #133 deferral was invalid, because the switch was later overwritten by the fastest-settings string) | `matmul_dataflow_common.hpp` (`57d8f06958c`, always on) | 24.31 | −0.60 | 41.87 (fast exp: 41.02) | −0.81 (−0.83) | same composition (mean pixel difference 12.2 vs #249, 8.4 vs #250) | **adopted** |
| — | 2026-10-11 | Retest of #150 (relay the ring SDPA K/V through the SDPA writer). The #150 finding "the SDPA output of non-relaying cores is off by 1–2 ULP" could also be explained by the #156 matmul write-out race (rows of the QKV output get corrupted), so it was reapplied on top of #156 (3-way merge of the diff based on `4e4aa653474`). (1) 2000-block chained block test with Tensix dispatch, 4 SP links, 72 cores: 1 of 6 runs differs (about 30% in #150). The relay has another race left. (2) SDPA on all 80 cores of the 8×10 grid (8 chunks per core; zones of 64 + 16 cores; with 3 links the 20 heads are split 7, 7, 6 across links): ring SDPA 14.20–14.88 ms, no faster than 72 cores + all-gather on the 10th row (14.16 ms) (bound by the transfers of the 6 relay cores) | experiment (reverted) | — | — | — | — | same PCC | rejected |
| **157** | 2026-10-11 | (Fix) Running the 40-step e2e twice in the #156 layout, the final latents did not match (max difference 1.79, mean pixel difference of the video 4.0). Per-step latent hashes (`WAN_STEP_HASH=1`) match for steps 0–31 and diverge from step 32. The chained block test (2000 blocks) matched, so a condition present only in e2e was suspected. Turning off #109, which grows the kernel config region by 128 KB (`WAN_KCFG_EXTRA_KB=128`, enabled only in the e2e test), gives all 40 steps matching in 3 of 3 runs (one step 1,027.9 → 1,031.6 ms). The standard 1-CQ Ethernet dispatch with dispatch_s (`HK_ETH_LINKS_TAKEN=2`) matches 3 of 3 even with the larger region. The problem lies somewhere in the combination of the dispatcher doing the go signal and completion wait itself (no dispatch_s) and the reuse of a large config region (not identified). Without dispatch_s, #109 is not used in performance tests. The host finishes enqueuing the ops for 40 blocks in about 0.9 s, ahead of the device, and trace (`WAN_TRACED=1`) is slower at 1,060.7 ms per step (trace cannot use the cross-attn K/V cache) | `test_performance_wan.py` (`3cff4729ced`) | — | — | 42.00 (fast exp: 41.19) | +0.13 (+0.17) | same composition (mean pixel difference 12.2 vs #249, 8.4 vs #250). All steps match in all 3 runs | **adopted** |
| — | 2026-10-11 | (Measurement) Breakdown of one #157 step (`WAN_FWD_TIMING=1`, 8-step short e2e): preprocessing 4.5 ms, 40 blocks 1,020–1,050 ms (one block 25.5–26.3 ms; the block test without FSDP is 24.3 M cycles, consistent given the loaded clock of about 0.98 GHz and run-to-run variation), postprocessing 3.4 ms, scheduler 2–3 ms. The host finishes enqueuing the ops for 40 blocks in 895–950 ms (about 0.45 ms per op on 32 chips), ahead of the device. Trace (`WAN_TRACED=1`) is slower at 1,060.7 ms per step (trace cannot use the cross-attn K/V cache) → the host is not the bottleneck. Removing the per-step sync (`WAN_STEP_TIMING=1`, for measurement) gives 42.00 → 41.89 s (same video; the sync stays on in recorded numbers to compare with earlier results). VAE decode on 8×10: 0.506 → 0.514 s | measurement | — | — | — | — | — | measurement |
| — | 2026-10-11 | Run FFN2 (#142) on the full 8×10 and re-tune the blocking (`TT_MMRS_RING_FULL=1`, M 13 per core): 7,12,7,1 gives 1.98 ms, 8,12,4,2 gives 2.04 ms, M block 13 does not fit in L1 → stay on 8×9 (1.95 ms) | block test | — | — | — | — | — | rejected |
| 158 | 2026-10-11 | The top core of each FFN2 (#142) in1 column issues the DRAM read of the next in1 block before forwarding the current block (if the next CB slot is free). in1 DRAM reads are already on NOC0 (dynamic NOC). FFN2 goes 1.97 → 2.02 ms, no faster. An in1 CB of 3 blocks does not fit in L1 | `TT_MMRS_RING_IN1_PF=1` (reverted) | — | — | — | — | same PCC | rejected |
| — | 2026-10-11 | (Measurement) Lower bound of ring SDPA (72 cores, measurement-only skips): everything 14.2 ms, no matmul 12.00, no matmul and no exp 11.08, additionally no injector K/V DRAM reads (result wrong) 11.85, only the K/V reads skipped 14.09 ms. The injector DRAM reads (NOC1) are not the bottleneck (moving them to NOC0 as in #143 does not help). The floor of about 11 ms is the softmax tile processing on the compute side (writing S, reading for the max and the subtraction, writing P, row sum, O correction; about 38 µs per Q/K chunk pair, about 60 cycles per tile operation) | measurement only (reverted) | — | — | — | — | — | measurement |
| 159 | 2026-10-11 | The per-step timestep embedding (fp32 MLP; 5120×5120 and 5120×30720, row-parallel, M = 1) takes about 4.1 ms of the 4.9 ms before the blocks (measured with synced section markers; the rest is patch embedding 0.3 and CFG concat 0.3; after the blocks: output norm 2.0, the rest 1.7, scheduler 2.2–2.5 ms). It depends only on t, so compute all steps in one forward with the steps stacked along rows (M = 40), and have each step slice out its row (`WAN_TEMB_BATCH=1`). The results match bit for bit. But if the short warm-up run differs in shape or slice position, compilation runs at the start of the real run and it takes 45.27 s. Keeping 40 rows in both gives 42.01 s (#157 is 42.00 s; one step 1,031.6 → 1,030.0 ms, offset by the precompute). In one run without hashes the final latent differed (#157 had matched in 15+ runs so far). The gain is 0.07 s or less and the divergence cannot be explained, so it is not adopted. Also, #157 and the standard 1-CQ Ethernet dispatch were each run 4 more times without hashes, and all matched | `WAN_TEMB_BATCH=1` (reverted) | — | — | 42.01 | +0.01 | same latent in 3 of 4 runs | rejected |
| 160 | 2026-10-11 | How in0 is distributed in the out-proj type AGMM (self-attn output, cross-attn Q and output; 8192×5120×1280, N block 5). Compute-side measurement (`TT_AGMM_THR_PROF=1`, timing-only variant) shows unpack waits 130–210 µs per op for in0. Adding zone measurement to the injector (the core that reads in0 from DRAM and multicasts it along the column) gives about 21 µs per block for the cross-attn Q: DRAM read 7.6, multicast (issue and completion wait) 6.5, wait for all-gather arrival 3.7, wait for receivers to be ready 1.4 µs. (1) Move the multicast completion wait to just before the next multicast (`TT_AGMM_IN0_LATE_FLUSH=1`): only the multicast zone goes 262 → 234 µs; the op time is the same. A 139 KB multicast spends transfer time in the issue itself (the #134 prefetch also issued its read after the multicast issue, so it barely overlapped). (2) Issue the next block's read before the multicast issue. Check CB space and arrival without waiting, and fall back to the old order if they are not ready (`TT_AGMM_IN0_PF2=1`, in0 CB with 3 stages): the prefetch happens in only 4–13 of 40 blocks (the next block has not yet arrived from the other chips; the injector waits 166–598 µs per op for arrival). With subblock 8×1 or 1×5, the cross-attn Q is 0.95 ms, and the 2 blocks of the block test take 46.84–46.93 ms (now 46.51 ms). With 1×5, unpack drops by about 50 µs, but the in0 wait grows by the same amount. The bottleneck is not in0 distribution within a chip but arrival between chips (the pace of the whole ring and skew in op start times across chips). Also, deeper CBs after #157 (`TT_AGMM_CB_DEPTH_SMALLN=3/4`) leave the cross-attn Q at 0.93–0.94 and the output at 1.00 ms, unchanged, and 16×1, which uses the whole DEST, gives nan and is slower | experiment (reverted) | — | — | — | — | same PCC | rejected |
| — | 2026-10-11 | (Measurement) Medium-size ops in the block (about 1.7 ms per block): 3 fused distributed LayerNorm + adaLN ops (230–250 µs), q/k RMSNorm and RoPE (186–295 µs; the first one absorbs the chip skew at the end of QKV), cross-attn q RMSNorm (168 µs), cross-attn SDPA (314 µs). Adding per-row zone measurement to the LayerNorm compute kernel (`TT_DLN_PROF=1`, timing-only variant, 4 rows per core): input wait 27, PRE 50, stats all-gather wait 50–73, combine (a series of small SFPU ops) 30, POST 54 µs. Each row runs PRE → wait for the all-gather round trip → combine → POST in series, with 4 waits. Effective bandwidth is 120–140 GB/s, not DRAM-bound | measurement only (reverted) | — | — | — | — | — | measurement |
| 161 | 2026-10-11 | Pipeline the LayerNorm rows (`TT_DLN_PIPE=1`): compute runs the next row's PRE before the current row's combine and POST, the worker writer sends the next row's stats before writing out the current row, and the reader reads input up to 2 rows ahead (the input CB holds all assigned rows; wrapping around made the next row's reference run past the end of the CB and gave nan). LayerNorm 231–252 → 210–216 µs (all-gather wait 3–12 µs), 2 blocks of the block test 46.5 → 46.3 ms, PCC and chained block test hashes the same as the default. While investigating, found a latent race: the stats forwarder counted arrivals with one cumulative counter across rounds, so when rounds overlap it can mistake an arrival from the previous round. Added a layout that counts 4 bits per round (`TT_DNORM_ROUND_FIELDS=1`; recording the values at start and end confirms the counting is correct). However, the chained block test of this layout hung twice (other layouts had no hangs except the first run right after a reset), and pipelining had 1 hash mismatch in 6 runs. Redoing the chained block test after #162: pipelining 4/4 match, no hangs (the per-round counting alone differs in 1 of 4). e2e 41.80 / 41.79 s (#162: 41.81 / 41.82 s, latent bit-identical), only −0.02 s: the LayerNorm saving in the block test (about 95 µs per block) is absorbed into other ops as inter-chip waits in e2e (the same result as #103b, the same pipelining for the q/k RMSNorm) | experiment (reverted) | 46.3 (2 blocks) | −0.2 | 41.79 | −0.02 | PCC and latent the same | rejected |
| — | 2026-10-11 | (Investigation) With the current default (#157, 3 SP links, Ethernet dispatch without dispatch_s), the 2000-block chained block test matched only about 40% of the time, even with committed code (on 2026-10-10 it was 8/8). Same after a reset. Isolation: k2, which removes 2 links per edge (2 SP links, with dispatch_s), matches 5/5. Keeping k1 and running the SP CCL on 2 links gives 7/8, but ring SDPA goes 14.13 → 14.91 ms (+0.8 ms per block). With the k1 link state, Tensix dispatch (with dispatch_s, 3 SP links) also gave 3 different hashes in 4 runs → the cause is the 3-link SP CCL, not the lack of dispatch_s. Calling ring SDPA alone 3000 times with alternating inputs never diverges (it shows up only combined with other ops in the block). The cause is #162 | investigation | — | — | — | — | — | investigation |
| **162** | 2026-10-11 | (Fix) The check that enables the bank-owned scheme (#155), `TT_RING_AG_BANK_OWNED_WH`, was `e[0] == 1` (the number 1, not the character '1'), so it was always off even when set. With 3 SP links it stayed on the standard scheme that splits the 512 pages of a head into a contiguous range per link (171/171/170) and sends 2 pages at a time. Fixing it to '1' gives 5/5 match in the chained block test (hash the same as the 2-SP-link baseline), ring SDPA 14.12 ms (same), same PCC. e2e gets a little faster with 4-page packets: 41.81 / 41.82 s (latent bit-identical to #157); with fast exp 41.02 / 41.01 s (the 2 runs' latents match; they differ from the #157 fast exp run, which was probably affected by the race) | `ring_attention_all_gather_async_multi_core_with_workers_program_factory.cpp` (`ef42272cf85`) | 23.2 | — | 41.81 (fast exp: 41.01) | −0.19 (−0.18) | default: same video as #157 (mean pixel difference 12.2 vs #249). Fast exp: 8.4 vs #250 | **adopted** |
| — | 2026-10-11 | On top of #162, retest #109 (grow the kernel config region) with it enabled also without dispatch_s (the reason it was dropped in #157 may have been the 3-SP-link race): e2e 41.68–41.69 s (−0.13 s), but the latent diverges in 1 of 3 runs (max difference 4.43) → another race remains in the combination of no dispatch_s and the larger config region. Not used, as in #157 | temporary switch in the performance test (reverted) | — | — | 41.68 | −0.13 | diverges in 1 of 3 runs | rejected |
| — | 2026-10-11 | (Measurement) Clock of each chip at step 10 of the #162 e2e (`tt-smi -s`): 29 of 32 chips run at 1000 MHz (118–164 W); one chip hits the 170 W TDP and runs at 907 MHz; two others run at 965 MHz (164 W) and 972 MHz (160 W). Chips that hit the power limit slow down (which chip varies by run and moment; in the next run the chip that was at 907 MHz ran at 1000 MHz and 134 W, and another chip ran at 922 MHz and 164 W), and because CCLs keep the chips in lockstep, the whole mesh is held back by that chip. This is why the changes that hide waits (#160, #161) did not help e2e (the slow chip has almost no waits). Changes that reduce compute or data volume help both time and power (the clock of this chip) | measurement | — | — | — | — | — | measurement |
| — | 2026-10-11 | Retest #159 (compute the timestep embedding for all steps at once) on top of #162: e2e 41.80 / 41.82 s (#162: 41.81), no effect; the final latent of the second run differs (max difference 1.57, about the same as 1.53 in #159, so it reproduces) → stays rejected | experiment (reverted) | — | — | 41.80 | −0.01 | diverges in 1 of 2 runs | rejected |
| — | 2026-10-11 | (Measurement) e2e of a timing-only variant that skips the SFPU ops of the LayerNorm combine (`TT_DLN_SKIP_COMB`, result wrong): 41.81 → 41.40 s. The combine compute is only about 60 µs per block (about 0.1 s in e2e); the rest is likely because the stats became garbage, changing the activation values, so the clock of the power-limited slow chip went up (the final latent is close to noise, standard deviation 1.00) → while performance is bound by power, timing-only upper bounds that break the result are not reliable | timing-only variant (reverted) | — | — | 41.40 | −0.41 | result wrong | measurement |
| — | 2026-10-11 | If the power-limited slow chip sets the pace, would reducing the power of polling (waiting cores reading L1 nonstop) raise its clock? In the loops for 2 compute-side CB waits, 2 DM-side CB waits and 2 semaphore waits, spin 16 NOPs between reads (header change, JIT cache rebuilt). e2e 41.81 → 41.89 s (latent the same); at step 10 the slowest chip is still at 922 MHz (164 W) → polling power does not matter | experiment (reverted) | — | — | 41.89 | +0.08 | latent the same | rejected |
| — | 2026-10-11 | Retest without dispatch_s with the config region growth limited to +64 KB: e2e 41.65–41.68 s (−0.15 s, faster than 41.68 s with +128 KB). The first 3 runs had latents bit-identical to #162, but the next 3 all differed from the baseline and from each other (max difference 1.6–1.7) → 3 of 6 runs diverge. The divergence in #159 (batched timestep embedding) looks like the same kind; a race seems to remain in the no-dispatch_s path that surfaces when dispatch timing changes → rejected | temporary switch in the performance test (reverted) | — | — | 41.66 | −0.15 | diverges in 3 of 6 runs | rejected |
| — | 2026-10-11 | (Isolation) Is the no-dispatch_s path the cause of the divergence with a larger config region? With the k1 link state (3 SP links, bank-owned) but Tensix dispatch (with dispatch_s, 8×9) and +128 KB, 3 e2e runs: 44.95 s, 1 of 3 diverges (max difference 1.65) → it is not about dispatch_s but the combination of the 3-SP-link CCL and the timing change from the larger config region (with k2's 2 SP links, +128 KB matches 3/3, #157). Without growing the config region, e2e keeps matching even with 3 SP links (7/7 since #162). There is one CCL sub-device, so it is not op overlap. The bank-owned handling of partial packets (WH has 12 banks, so the 512 pages of a head become 42–43 pages per bank) is consistent between the reader and writer sides. Cause not identified | investigation | — | — | 44.95 | — | diverges in 1 of 3 runs | investigation |
| — | 2026-10-11 | (Isolation, continued) With the k1 link state, run the SP CCL on 2 links (no bank-owned) with +128 KB: e2e 43.02–43.05 s, 1 of 4 diverges (max difference 1.56) → regardless of the number of links the CCL uses (2 or 3) or the dispatch (Ethernet or Tensix), it happens with the combination of a fabric that physically keeps 3 links per SP edge and the timing change from the larger config region (with k2's 2 physical links, 3/3 match). The default without growth keeps matching in e2e | investigation | — | — | 43.03 | — | diverges in 1 of 4 runs | investigation |
| — | 2026-10-11 | Output projection type matmul (8192×5120×1280) on the 8×10 grid (1280 in `WAN_AGMM_WIDE_N`, workers 8×9): blocking 16,8,5,8,1 gives cross-attn Q 0.93 ms (same), M block 8 gives 1.10 ms (slower). The 40 tiles of N do not split evenly across 9 rows. Disassembling the matmul compute kernel (TRISC1) shows 3 instructions per tile product: the DEST write target setup, MOP and SETRWC, so the RISC-V side is not the bottleneck. Together with the compute-only timing of #89 (MVMUL theoretical + a fixed cost of about 7 cycles per tile product), the rest is likely the per-tile overhead on the unpack side (it needs a rebuilt LLK unpack MOP and the gain is unclear, so not pursued). The `test_compute_mm` micro-benchmark fails because its verification DPRINT readout does not match the current code | block test (env vars only) | — | — | — | — | — | rejected |
| 163 | 2026-10-11 | Process the SFPU ops of the LayerNorm stats combine only for row 0, where the stats live (`VectorMode::R`, 2 iterations per face. The WH SFPU handles 4 rows × 8 columns, the even or the odd columns, per iteration, so all columns of row 0 need 2 iterations; 1 iteration broke PCC to 52.7%). The PRE stats op on column 0 uses `VectorMode::C` (faces 0 and 2). The row 0 / column 0 math is the same, so PCC and the e2e latent are bit-identical. Block test 46.17 ms (2 blocks), e2e 41.79 / 41.81 s (#162: 41.81 / 41.82), about −0.015 s. The q/k RMSNorm combine has only 3 SFPU ops, so the effect would be a few µs | experiment (reverted) | 46.17 (2 blocks) | — | 41.80 | −0.015 | latent the same | rejected |
| — | 2026-10-11 | Check whether the 3 output projection type matmuls (self-attn output projection, cross-attn Q projection and output projection, 0.93–1.0 ms each) can be changed from input all-gather + matmul to the same "local matmul + ring reduce-scatter" form as the #142 FFN2. The input is already TP-split along the head or hidden dimension, so building partial sums with row-split weights (1280×5120) and reduce-scattering them would remove the in0 distribution that is the bottleneck (#132–#135, #160). Unit test of the #142 op in the output projection shape (M 8192, K 1280, N 5120, partial sums in bf8): 1.25 ms per call on 8×9 (host timing; the FFN2 shape's 2.18 ms corresponds to the op's 1.95 ms, so about 1.05 ms on device), 1.31 ms on 8×10, 1.43 ms with M block 4. Even the matmul alone with the same blocking (`minimal_matmul`) takes about 1.0 ms on device: K is only 40 tiles, so there are 4 times as many output (partial sum) tiles as in the current form (4× the in0 transfer is just traded for 4× the pack and write-out), and with K block 10 each output is accumulated into L1 4 times. K block 20 is 4 KB short of L1 (M block 4). No prospect of beating the current AGMM, so stopped | measurement (added the shape and grid options `HK_RING_MKN`, `HK_RING_GRID_Y` to `test_hk_mmrs_ring.py`, uncommitted test) | — | — | — | — | — | rejected |
| **164** | 2026-10-11 | Compact the cross-attn text padding exactly. The text encoder zero-pads the prompt to 512 tokens and the model attends to all of them without a mask, so the K and V at the padding positions all have the same value, and n copies of the same key are equivalent to "that key's softmax weight times n". Compact K/V to 8 tiles (256 entries): in each tile, columns 0–15 hold real tokens (or padding if the prompt is short) with weight 1, and columns 16–31 hold padding with weight w. The condition that the total weight equals the original 512, 8×(16 + 16w) = 512, gives w = 3 (exact in bf16, bf8 and LoFi). The SDPA row sum is formed by accumulating P tiles column-wise in L1 and then multiplying by `col_identity` (a tile with ones in column 0), so set the elements of columns 16–31 of `col_identity` to 3 (added `sum_col_weight_start` and `sum_col_weight` to the op) and cache the V rows of those columns multiplied by 3. The prompt goes from 512 → 256 rows by multiplying with a selection matrix (0/1, HiFi4) (padding positions always get the last padding row). Up to 128 real tokens w = 3, up to 192 w = 5, ... up to 248 w = 33. The test's negative prompt is 126 tokens. In the block test (prompts with aligned padding rows), the output difference with and without compaction is PCC 0.99946, mean difference 0.087. Two other mathematically identical computations (the w = 5 layout, or no compaction with k chunk 512 → 256) also give 0.99949 and 0.99948, the same size, so the results match within rounding noise. Deliberately leaving the column weights at 1 breaks it to 0.99474, mean difference 1.38 (a check of the test's sensitivity). Cross SDPA 0.31 → 0.18 ms (q chunk 256; 0.19 with 512, 0.20 with 128). e2e 41.60 / 41.62 s (#162: 41.81 / 41.82, −0.20 s). However, the 2 runs with compaction had latents that did not match (max difference 1.99). 2 runs with per-step hashes (host sync every step) matched at every step. (At this point the 3-SP-link configuration was thought to be the cause, but the candidate 5 investigation showed it is a race that already exists in the default configuration, and its frequency varies with the state of the machine.) In a re-evaluation a few hours later, all 10 runs with compaction matched (41.61–41.65 s), and with fast exp 40.80 / 40.80 s also matched. The video has the same composition as #162 (mean pixel difference 8.9); adopted | `WAN_XATTN_COMPACT=1` (commit `627c534df9a`) | — | — | **41.64** | −0.17 | matches within rounding noise; 10/10 runs match | **adopted** |
| — | 2026-10-11 | (Investigation, candidate 5) The issue where the e2e latent rarely diverges when timing changes. (1) Even in the configuration with 2 physical SP links (k2), 2 of 6 runs with compaction (#164) diverged. The earlier "the configuration that keeps 3 SP links is the cause (k2 matches 3/3)" was chance and wrong. It also diverges with Tensix dispatch, so it is an op race common to all configurations. (2) Reading the base FW counters of all 32 chips and all Ethernet links (retraining, CRC, FEC corrected and uncorrected codewords) before and after the run: CRC and uncorrected are always 0, and corrected only rises by 1–2 per run, with no difference between diverged and matching runs → data is not corrupted on the links. (3) A measurement that accumulates the fp32 sum of each block's output on the device and reads it at the end of generation (no host sync) shows the divergence starts at a different place each time: step 38 block 21 (the 4 TP devices at SP position 6), step 17 block 2 (3 of the 4 TP devices at SP position 5, relative difference of the sum 3e-6). Adding measurement points inside the block as well (6 spots, or adding the ring SDPA output) makes all 9 runs match with no divergence. (4) The distributed norm forwarder waits on the cumulative arrival count across all rounds, so in theory a fast peer's addition for the next round can satisfy the condition and the reader can read another peer's page for this round (the value from the previous call). Making the version that counts in a 4-bit field per round the default gave 6 matching runs among those that did not hang, but 2 of 9 runs hung mid-denoise (the added value stays 32-bit in the router; cause not identified), and the run right after a hang diverged. A version that also applies it to GroupNorm (VAE) hangs in the VAE. The cause was not found; time-boxed and stopped | investigation (reverted) | — | — | — | — | — | stopped |
| — | 2026-10-11 | (Candidate 4) Breakdown of the VAE (0.51 s): time per section (with sync) is input preparation 15–33 ms, decoder 484 ms, D2H 30–35 ms. The decoder is a 1-frame chunk of 29.5 ms + 4 chunks of 5 frames (110–113 ms each). For the 5-frame chunks the host finishes enqueuing ops in 30–44 ms, so they are device-bound (trace does not help). Capturing a 2-step e2e with tracy and splitting the 467 ms op total on device 0 by type: conv3d about 45%, neighbor pad about 14%, RMSNorm about 8%, tilize/untilize about 8%, concat about 5%, VAE SDPA 3%. Tried: conv3d on the 8×9 grid (with blocking searched for 8×9) makes the decoder 484 → 498 ms (slower); converting the input to bf16 and a contiguous layout in torch before sending it (difference within noise). The fused neighbor pad + conv3d op is not in this tree. What remains is conv3d kernel work or new fused ops, about 0.1 s for the effort, so not pursued | measurement (reverted) | — | — | — | — | — | rejected |
| — | 2026-10-11 | (Measurement conditions) The e2e script synced the host and device every step to record per-step times (`WAN_STEP_TIMING=1`). The pipeline itself runs the solver on the device too and does not sync between steps, so the time without the sync is the true time. Removing it in the #162 configuration gives 41.68–41.73 s (with sync 41.79–41.82 s, about −0.1 s), with latents bit-identical to the synced runs. But without sync, even the default configuration hung in 1 of 5 runs (at denoise step 17), and 1 of 9 runs gave a latent different from the others. All e2e numbers so far are with sync, and comparisons are aligned on this condition | measurement (script setting) | — | — | 41.70 | −0.1 | without sync, even the default rarely diverges | recorded |
| — | 2026-10-11 | (Investigation, candidate 5 continued) Switching FFN2 from the ring (#142) back to the standard fused MM+RS (`TT_MMRS_RING=0`, 42.54 s) and running 10 times with compaction: only the first run right after a rebuild differed; 9 matched. Then 10 runs with the ring under the same conditions: all 10 matched (a few hours earlier, 3 of 8 runs diverged under the same conditions). The divergence frequency varies more with the state of the machine (right after a reset or not, temperature, power and so on) than with the settings, so comparing counts taken at different times cannot narrow down the cause. The persistent CCL buffers (norm stats, AGMM in0, ring SDPA K/V) are all pools of 2, and for a peer to move 2 ops ahead it needs the slow chip's contribution through the collective in between, so the design does not allow overwrites across ops. Cause not identified | investigation | — | — | — | — | — | recorded |
| **165** | 2026-10-11 | Re-compared the config region growth (+64 KB) by alternating it with the baseline in 6 pairs in the same session (with #164, with sync): baseline 41.62 s, 1 of 6 diverged and 1 hung; +64 KB 41.46–41.51 s, all 6 matched → the growth does not increase divergence; it is a pre-existing race, so +64 KB is adopted (the cap without dispatch_s is 64 KB). The hang triage (3 times, `HANG_TRIAGE=1`) always showed the same shape: only 1 device (a different one each time) stays in the ring SDPA of block b (32 reader cores wait on the semaphore that signals the on-chip all-gather, and that device's all-gather cores have finished), the 3 other devices of that TP group wait in the output projection AGMM, and the remaining 28 devices are in the ring SDPA of block b+1. During that investigation, hit again the flaw where the ring SDPA K and V all-gather buffers have the same shape, so they fight over the same ping-pong pair and always use the same 2 buffers (no alternation across calls) (`WAN_AG_PP_FIX` was made in #155 but not adopted because it did not fix the chained block test divergence at the time). SP columns are connected only through ring SDPA, and a fast column can get 1 block ahead, so it can write the next K/V into a buffer that a slow device is still reading. Fixed by default. 8 pairs alternating with the baseline without sync: baseline 1 diverged and 1 hung; fix: the 7 runs that did not hang matched, 1 hung → this does not remove the hangs. With fast exp 40.70 s (another run hung in the VAE; latent matched) | `WAN_KCFG_NODS_MAX_KB=64`, `WAN_AG_PP_FIX` (commit `c737cad24ea`) | — | — | **41.49** | −0.15 | latent the same | **adopted** |
| **166** | 2026-10-11 | (Candidate 3) Work outside the blocks is about 5 ms per steady-state step in a 2-step tracy capture: timestep MLP 3.1 ms (the 5120→30720 fp32 matmul 2.48 ms), around proj_out 0.9 ms, norm_out 0.44 ms, solver 0.25 ms, CFG concat 0.21 ms, patch embedding 0.15 ms (in a 2-step trial run the expert model switches every step, so text embedding 3.5 ms and K/V cache creation 0.9 ms are also included). With the script's sync on, the head of each step is host-bound and temb hides behind it (why #159 showed no effect). The #159 batched computation (`WAN_TEMB_BATCH=1`) alternated with the baseline in 4 pairs without sync: baseline 41.32–41.36 s, batched 41.25–41.27 s (−0.07 s); the latents of runs that did not diverge are bit-identical to the baseline; both diverge in 1 of 4 runs (the pre-existing race). With sync also 41.49 → 41.43 s. Without sync and with fast exp, 40.43 s. The rest (fusing the proj_out bias, bf16 norm_out, moving the concat earlier) adds up to less than 0.02 s over 40 steps, so not pursued | `WAN_TEMB_BATCH=1` (commit `a85a84b6be5`) | — | — | **41.43** | −0.06 | latent the same | **adopted** |
| — | 2026-10-11 | (Investigation, cause of the hang) Investigated the ring SDPA hang that happens 1–2 times in 8–16 e2e runs without sync, with triage and debug bus readout (`HANG_DUMP=1`). Each time 1 core on 1 device stops, in the first row (row 4) of group 1 of the Q interleave multicast (odd heads = uncond, rows 4–7). On that core, the unpacker is stuck while executing an UNPACR with SrcA Z increment (0x420080C1, an instruction of the unpack MOP for the max reduce), the unpack instruction buffer is full, math waits after SETRWC, and the last pack instruction is ZEROACC. All 3 TRISCs are stuck on instruction writes (config register writes and the like). The reader waits for free space in the K/V CB, and the group's injector and other cores wait on the chain hand-off semaphore; as a result the other devices move on to the next ring SDPA, and the 3 other devices of the same TP group move on to AGMM and wait. Likely an inconsistency inside the coprocessor: in the preceding stage (the max reduce or the sub bcast), the counts of SrcA unpacks and consumes disagree. Isolation (e2e without sync, hang counts): placing the injector at the group's corner core (so the multicast path does not cross other groups) 2/5, removing reduce_trigger 2/16 (41.90 s), removing only the first-half overlap of reduce_trigger 1/12 (41.79 s), removing all custom softmax changes (merged exp, row-sum MOP, blocked pack threshold 4) 1 of 4 (42.20 s) → it happens in every case. Stopped trying to find the cause | investigation (reverted) | — | — | — | — | — | stopped |
| — | 2026-10-11 | (Measurement) Which ring SDPA steps matter, measured with measurement-only skips (block test, standard exp, 2-SP-link configuration, device 0 / 5): baseline 15.53 / 15.74 ms, no exp 14.18 / 14.04, no row-sum pack 14.70 / 14.28, no P pack 14.47 / 14.52, no S pack 15.23 / 15.01, no matmul 13.40 / 13.73, no row sum and no exp 13.72 / 13.68 → the pack thread is the bottleneck, and exp, the P pack and the row-sum pack (accumulating into L1 per tile) weigh about the same | measurement only (`TT_RING_SDPA_HK_SKIP=sum/ppack/spack`, reverted) | — | — | — | — | — | measurement |
| — | 2026-10-11 | (Measurement) Time per thread (measurement-only counters, median over 72 cores, ms). Baseline (total 15.0): unpack is QKᵀ 4.72, P·V 3.29, sub and exp 1.66, max 0.55, SALAD 0.66; math is 3.12, 3.80, 3.64, 0.87, 0.59; pack is 1.54, 0.31, 6.49, 0.29, 0.38, plus correction factor 2.07 (its content is `mop_sync` for reconfiguring the MOP, i.e. waiting for the previous row-sum pack to finish). After #167 (total 14.8): unpack is QKᵀ 3.63, P·V 4.65, sub and exp 1.46, matmul re-init 0.52; math is 3.09, 4.52, 2.71, 0.26; pack is 1.52, 3.82, 4.33, correction factor 0.60. In all 3 threads 75–80% of the time is inside functions, and 20–25% is outside any function (CB waits and config switches) | measurement only (`HK_THR_PROF`, reverted) | — | — | — | — | — | measurement |
| 168 | 2026-10-11 | In Phase 1, batch the previous row's sub_exp for 4 column subblocks and then run the current row's 4 QKᵀ (matmul re-init, sub init and the unpack format switch happen once per row), and in Phase 2 run the first row's P·V as one matmul too (on top of #167). PCC is the same (99.9682%), but ring SDPA is slower, 14.83 → 15.59 ms. The current form, alternating sub_exp and QKᵀ every 8 tiles, keeps the 3 threads running at the same time | `TT_RING_SDPA_GROUPED=1` (reverted) | — | — | — | — | same | rejected |
| 169 | 2026-10-11 | Reorder the computation (same math): in Phase 1, per column subblock, do "current row QKᵀ → previous row sub_exp" (so at the start of the next row the QKᵀ unpack can begin without waiting for this row's max); at the start of Phase 2 run P·V (row 0) before sub_exp (row 3); and if the P·V row group is 1 row (#167), delay the pack → unpack handshake until just before row 3 is read. PCC is the same (99.9682%; 99.9694% without #167). The block test is within noise (with #167 14.80 → 14.56 ms; alone 15.07 → 15.28 ms), and the 8-step e2e is slower: 1,005.2 / 1,007.2 ms vs 1,003.3 ms for #167 | `TT_RING_SDPA_REORDER=1` (reverted) | — | — | — | — | same | rejected |
| 170 | 2026-10-11 | Compute the softmax exp with one LOADMACRO per row, giving the same bits as the standard approximate exp (the standard version rounds fp32 → int16 and then restores the sign with SHFT2 and SETSGN, so it issues twice per row). Use the single chain MAD → fp32 → uint16 rounding → 15-bit left shift → write-back from the standard version that "clamps input at −88.5", without the clamp (x = s − max ≤ 0, so if uint16 turned negatives into 0 the bits would be the same). Ring SDPA 14.26 → 13.60 ms, but PCC 79.0%: uint16 rounding does not turn negative values into 0; it returns the magnitude (which is why the standard version clamps the input first). No other combination produces the same bits in one issue | `TT_RING_SDPA_EXP_U16=1` (reverted) | — | — | — | — | PCC 79.0% | rejected |
| **167** | 2026-10-11 | Compute the ring SDPA softmax row sum with the P·V matmul. Isolation (the measurement row above) showed the pack thread is the bottleneck, and the row sum was built with packs that accumulate each P tile into L1 (128 per chunk pair). In P·V, multiplying by a "tile with 1 only in column 0 and 0 elsewhere" (bfp8_b like V, built once by the writer) as a 5th column after the 4 V tiles reuses the P already in srcB, and the row sum comes out in column 0 of the 5th DEST tile. Unpack reads the 4 V tiles with the standard reuse MOP, then issues one more UNPACR with SrcA pointed at the ones tile in the same config context; math uses the standard no-MOP 1×5 matmul (the t_dim = 1 replay frees SrcA per tile). P·V works one row of Q tiles at a time (5 DEST tiles), and the row sum takes one pack per row. The `col_identity` matmul in the normalization takes column sums, so the sum tile with only column 0 can be used as is. Unpack and matmul work for P·V grows by about 1/4. Block test ring SDPA 15.53 / 15.74 → 15.15 / 15.18 ms (2 SP links). But this alone hung in 3 of 9 e2e runs, with a new shape: math waits at the DEST commit (`t6_semaphore_post<MATH \| WAIT_SFPU>`, waiting for the SFPU to be free), the pack thread waits at the SFPU prologue `STALLWAIT(STALL_SFPU, MATH)` (waiting for math to be free), each waiting on the other, and the unpacker is stuck on the P·V SrcA UNPACR. The pack thread's DEST wait only stalls pack instructions, so this prologue was what stalled the SFPU instructions. Replacing the prologue with a SEMWAIT on MATH_PACK (proceeds only after the commit) still hung in 2 of 3 runs as long as the stall is done in hardware (a stalled SFPU instruction seems to count as the SFPU being busy). So math raises a dedicated semaphore (the unused UNPACK_TO_DEST) after the sub commit, and the pack TRISC reads it, waits, lowers it and then issues the SFPU instructions (`TT_RING_SDPA_SFPU_GO`, with the hardware wait in the prologue removed); this hang shape no longer appeared. 5 alternating pairs in the same session: baseline 41.24–41.29 s, PV_SUM only 40.77–40.78 s. PV_SUM + SFPU_GO without sync 40.79–40.90 s, with sync 41.00 s (#166: 41.43 s). With PV_SUM + SFPU_GO, the latents of 13 runs, excluding the first run right after a rebuild, matched exactly (in the same session the baseline diverged in 3 of 5, baseline + SFPU_GO in 1 of 4, PV_SUM only in 3 of 4) → in that session both sources of divergence seemed gone, but in later sessions PV_SUM + SFPU_GO also diverged (2 of 3, 1 of 6 and so on), so a source of divergence remains. The pre-existing hang that stops in the max reduce unpack remains (in the same session 1 in 5 runs for both baseline + SFPU_GO and PV_SUM + SFPU_GO). Block PCC 99.9692 → 99.9682%. With fast exp (#138, optional), 40.07 s without sync (mean pixel difference 13.3 vs #167; as before, only details change) | `TT_RING_SDPA_PV_SUM=1 TT_RING_SDPA_SFPU_GO=1` (commit `910ca50e9e6`) | — | — | **41.00** | −0.43 | same composition (mean pixel difference 9.4 vs #166, `img/wan22_167_pv_sum_cmp.jpg`), latent matches in 13 runs | **adopted** |
| — | 2026-10-11 | (Measurement, on top of #167) Added to the per-thread times the whole of one computation (one K chunk) and zones for waiting on K, V and Q CB arrival: outside the computation (the ring loop) is 0.05–0.09 ms, and the K/V/Q arrival wait in unpack is only 0.19 ms (data supply is not the bottleneck). Within one computation, unpack spends about 2.9 ms (20%) outside any function, mainly on switching unpack data formats (between bf8 K, Q, V and bf16 S, P; 24 per pair in Phase 1 alone; each switch waits for the unpacker to go idle) and on init. In an e2e tracy capture, the op total of one block, 25.1–25.3 ms, is about 3% longer than the block test (about 24.4 ms) (ring SDPA 14.41 ms); this is the clock difference under load, not gaps between ops | measurement only (reverted) | — | — | — | — | — | measurement |
| 171 | 2026-10-11 | Reordering to reduce format switches (on top of #167, same math): (1) alternate Phase 1 sub_exp and QKᵀ every 2 column subblocks (switches 24 → 12): ring SDPA 14.83 → 15.02 ms (matmul re-init drops 0.52 → 0.32 ms, but the QKᵀ wait grows: unpack 3.65 → 4.48, math 3.07 → 4.33 ms); (2) at the start of Phase 2, finish all of row 3's sub_exp and then run row 0's P·V as one matmul: 15.09 ms. Both are slower. The current form, alternating the pack-heavy sub_exp and the unpack-heavy matmul every 8 tiles, keeps the 3 threads busiest at the same time | `TT_RING_SDPA_PAIR=2` / `TT_RING_SDPA_DRAIN_FULL=1` (reverted) | — | — | — | — | same | rejected |
| 172 | 2026-10-11 | Batching in ring SDPA Phase 2 (on top of #167, same math): (1) compute the SALAD correction factor exp(previous max − current max) for all rows at once (`TT_RING_SDPA_CORR_ALL=1`; the sub, exp and pack setup done 4 times, once per row, is done once); (2) run the P·V of rows 1–3 back to back (one pair of unpack format switches), with SALAD and push / normalization afterward (`TT_RING_SDPA_SALAD_LATE=1`); (3) SALAD for all rows in one call (`TT_RING_SDPA_SALAD_MERGE=1`). 2 alternating pairs in the same session (with the config region growth, without sync): default 40.91 / 40.82 s, batched 40.41 s (−0.45 s). But hangs and divergence increase: the batched version had 4 hangs and 2 divergences in about 10 runs (mean pixel difference 17.3 / 6.52; default 9.39), while the default had 1 hang and no divergence in 5 runs. The divergence is also likely the same SrcA/SrcB valid mismatch (the case where it recovers without stopping), so it will not become the default until the cause is fixed | `TT_RING_SDPA_CORR_ALL=1 TT_RING_SDPA_SALAD_LATE=1 TT_RING_SDPA_SALAD_MERGE=1` (optional, off by default; commit `29d41c39113`) | — | — | 40.41 | −0.45 | same math (except diverged runs) | parked |
| — | 2026-10-11 | (Investigation) Difference per block between e2e and the block test: in a 4-step e2e that reads out the profiler every step (`WAN_PROF_FLUSH_STEP=1`), medians by op type over 30 steady-state blocks are ring SDPA 14.33, AGMM 6.53, FFN2 1.95, normalization 1.32 ms, gaps between ops 0.15 ms. The cross-attn text K/V projection (selection matmul, to_kv, V weights, concat and typecast, about 0.67 ms) showed up in every block, but the cache works as designed (the last of the 4 steps is the first step of the low-noise expert model; over 40 steps the cache is rebuilt in only 2 steps). For blocks that hit the cache, the op total is about 24.4 M cycles, the same as the block test, and the gap to the 25.1 ms wall clock (about 2.7%) is the clock under load. The ring SDPA zone imbalance (9 or 8 per core over 72 cores) has no room to improve: however the 2,560 Q tiles are split, the per-core maximum is 36 tiles, the same as now | measurement only (reverted) | — | — | — | — | — | measurement |
| — | 2026-10-11 | (VAE) Output path: when outputting uint8, skip the decoder's final clamp(−1,1) and the tile conversions around it (the uint8 conversion clamps), and concatenate the chunk outputs once at the end: the video is bit-identical and the decoder goes 484 → 481 ms (within noise), so not pursued. A version that keeps chunks in BTHWC, reshapes them to [B,T,H,W·C], reads them as uint8 and drops the permute breaks, because the reshape of the row-major C = 3 layout mixes values (mean pixel difference 18). Changing conv3d precision (now HiFi2 with fp32 DEST) to LoFi times out, and a bf16 DEST hits a TT_FATAL on the subblock condition (the per-layer blocking would need to be searched again). None of these would be more than an optional, quality-affecting feature, so not pursued | `WAN_VAE_FAST_OUT` / `WAN_VAE_CONV_FID` / `WAN_VAE_CONV_FP32` (reverted) | — | — | — | — | same / broken | not pursued |
| — | 2026-10-11 | (Investigation, cause of the hang 2) The triage text keeps the call stacks of the 3 TRISCs, and the debug bus also shows the owner of each SrcA/SrcB bank (unpacker or math) and which bank each of them points to now. There are 3 places where it stops, and in all of them math waits for SrcB inside an op that uses SrcB, while the unpacker has moved ahead and waits for free SrcA: the max reduce (custom MOP, scaler into SrcB once), SALAD (multiply with a SrcB column broadcast), and the correction factor subtraction (stops at its commit). At the stop, SrcB shows states such as "unpacker and math point to the same bank and only one is valid", so the ordering between the 2 banks is broken. (1) Making the dummy valid that `copy_tile` sets on SrcB before the reduce (an UNPACR_NOP that does not wait for a free bank) wait for a free bank first still hangs on the first run (`TT_RING_SDPA_FENCE=256`). (2) A fence where the unpacker waits for math to finish at every op boundary (`TT_RING_SDPA_FENCE=63`, 40.63 s, same hash) still hangs, stopping inside SALAD → it is not running ahead across ops; the mismatch happens inside an op. (3) A diagnostic version that runs exp on the math thread (so the pack thread's SFPU and math's FPU do not overlap) had all cores stop at the same place, an issue on the implementation side, so it was inconclusive. The default #167 (no config region growth) also hung in 1 of 3 runs, and the batched version in 2 of 2. The 21 hangs so far (excluding 2 caused by implementation bugs) concentrated on a few specific chips (and almost always on the same core, 9-5, logical (7,4)), so a hardware-specific cause is possible. The next isolation step is to compare the hang rate of the same configuration on another Galaxy | investigation (`TT_RING_SDPA_FENCE`, reverted) | — | — | — | — | — | unresolved |
| 173 | 2026-10-12 | (VAE) Make the conv3d matmul DEST full-sync (fp32 DEST grows from 4 to 8 tiles and the subblock from 1×3 to 2×3, which should reduce unpack per tile product; the K accumulation order is the same). Layers with a C_out block of 12 tiles do not divide by 8, so the factory's subblock width was changed to "the largest divisor that fits in DEST" (same result in the current configuration). On a 5-frame chunk (no cache), the conv3d total goes 48.6 → 56.1 ms, slower in every layer (+15%), PCC the same 99.9636%. The main cause is that full-sync cannot use the fast compute tilize (`can_use_fast_tilize` excludes full-sync) and falls back to the slow tilize (tilize is about 22% of conv3d in the default configuration). Losing the alternation of DEST between math and pack adds to it | `WAN_VAE_CONV_FULLSYNC=1` (reverted) | conv3d 48.6 → 56.1 ms | +15% | — | — | same | rejected |
| — | 2026-10-12 | (Measurement, VAE) Break down conv3d by skipping compute steps for measurement only (`TT_C3D_HK_SKIP`, 5-frame chunk, no cache, sum over 35 convs): default 48.6 ms, no tilize 38.0, no matmul 42.3, neither 34.8 ms. The read (vol2col: gather the receptive field shard from DRAM and lay it out by copying kW×C bytes 9 times inside L1 for each output position) alone takes 72%, the matmul mostly hides behind it, and tilize adds about 10 ms on top. For the 96-channel layers, no tilize gives 2080 → 1678 us, and no matmul still 2000 us. In the production shape (after the first 1-frame chunk, 5-frame chunks with cache), of 117.6 ms per chunk, conv3d is 48.3, neighbor pad 21.1, RMSNorm 9.4, concat with the cache 7.2, tilize / untilize 9.7 ms | measurement-only switch (`TT_C3D_HK_SKIP`, to be reverted) | — | — | — | — | — | measurement |
| — | 2026-10-12 | (VAE) Ran the Wan VAE through the conv3d halo mode (`neighbor_pad_halo` exchanges only the H/W boundary rows and puts them in a small buffer, and the conv3d reader reads only the boundary positions from there; no internal copy in neighbor pad is needed; the path used by the LTX VAE). The zero frames before the causal conv were put into the concat with the cache (conv3d's T padding is symmetric). It hangs in the first conv's `neighbor_pad_halo`: 9 of 32 chips stop at the startup barrier for the H direction (`np_writer.cpp` line 213), and the rest wait to receive in the W direction. Same with Ethernet dispatch and standard dispatch. The barrier code is almost the same as the normal neighbor pad, so the problem is likely in the host-side setup or in core placement on this 4×8 mesh. The expected gain is about 15 ms per chunk (about 60 ms in e2e), and it needs an op investigation, so parked | `WAN_VAE_HALO_CONV=1` (uncommitted) | — | — | — | — | hang | parked |
| **174** | 2026-10-12 | (VAE) Keep the input and output of residual blocks without a 1×1 shortcut in row-major. RMSNorm (with SiLU) and the residual add can read row-major directly, and on 1 chip at the production shapes ([1,20,120,104,96], [1,17,60,52,192]) they give the same bits as the tile layout. This removes the untilize after norm1, the tilize before the add, and the tilize after the upsampler; only the first block of up1, which has a shortcut, goes back to tiles. The output of the T = 6 VAE test (the first 1-frame chunk and a 5-frame chunk with cache) is bit-identical, and the 5-frame chunk goes 118.4 → 111.9 ms (tilize / untilize 10.1 → 2.3 ms). 40-step e2e, alternating (no sync): off 40.91 / 40.88 s (VAE 0.516 / 0.509 s), on 40.78 s (VAE 0.490 s); one more run hit the known ring SDPA hang in denoising. Only the first run with tracy diverged, and 79 profiler start/end markers did not match on a specific chip (no divergence and no marker errors from the second run on), so this was judged a transient issue on the chip side | Default on (`WAN_VAE_RM_RESID=0` reverts; commit `c17778dcac0`) | VAE 0.509–0.516 → 0.490 s | −4% | **40.78** | −0.02 | Bit-identical | **adopted** |
| **175** | 2026-10-12 | (VAE) Keep the temporal cache of the causal conv (3×3×3) together with its halo. Before, the cache was the last 2 frames of the previous chunk's input; the next chunk concatenated them in front of its input (a copy of the whole input) and then added the halo with neighbor pad. The last 2 frames of the neighbor pad output, halo included, are exactly the leading frames of the next chunk (the first chunk uses zeros; a 1-frame chunk uses "last frame of the previous cache + input", as before). Add an optional `front_tensor` to `neighbor_pad_async`: instead of filling the leading frames of `t_front_pad` with zeros, the local copy cores copy them page by page (the reader reads, the writer writes). The W-direction exchange skips the leading frames (they already have the halo). The conv slices the last 2 frames of its output and returns them as the next cache. The output of the T = 6 VAE test is bit-identical. 5-frame chunk 111.9 → 103.5 ms (concat 7.25 → 0.28 ms, neighbor pad 21.2 → 19.7 ms). 40-step e2e, alternating (no sync): off 40.87 s (VAE 0.489 s), on 40.76 s (VAE 0.458 s). The other 2 runs hung (the off run stopped inside the normal VAE neighbor pad, a rare hang that existed before; the on run in the denoising ring SDPA) | Default on (`WAN_VAE_NP_CACHE=0` reverts; commit `32533e8f0be`) | VAE 0.489 → 0.458 s | −6% | **40.76** | −0.03 | Bit-identical | **adopted** |
| — | 2026-10-12 | (Measurement) When ring SDPA switches the unpack data format (between bf8 Q/K/V and bf16 S/P, 24 times per pair in Phase 1 alone), the LLK reconfig waits with `STALLWAIT(STALL_CFG, UNPACK0/1)` until the unpacker is idle. To check whether this wait is the "20% outside functions" of the unpack thread, the wait was removed for measurement only (`TT_RING_SDPA_RECONFIG_NOSTALL`). 8-step e2e, alternating: steady-state step default 1,081.8 ms, no wait 1,081.3 / 1,081.8 ms, so no difference, and the block test error (RMSE/σ 3.3%) is the same → the unpacker is already idle at the switch, and this wait is not the bottleneck. The WH unpacker also has per-context format registers (`THCON_SEC0_REG7_Unpack_data_format_cntx*`, enabled by `Ovrd_data_format`), but there is no longer a reason to use them | Measurement only (reverted) | — | — | — | — | — | measurement |
| — | 2026-10-12 | (VAE) Raise the number of sticks that the neighbor pad interior copy groups under one NoC wait (`TT_NP_LC_BATCH`, 16 since #93): on the T = 6 VAE (on top of #175) the neighbor pad total is 23.67 ms at 16, 24.54 ms at 32, and 25.05 ms at 64, so it gets slower. The output is the same | `TT_NP_LC_BATCH=32/64` (kept at 16) | — | — | — | — | Same | rejected |
| **176** | 2026-10-12 | (VAE) Split the queries of the mid-block attention. With `WAN_VAE_SDPA_W_ONLY`, the frames (5, padded to 8) are spread over the 8 chips of the W axis, and the 4 chips of the same column on the H axis all computed the same 6,240 queries of the same frame. Each chip now computes only the 1,560 queries of its own H row (`mesh_partition` on the H axis in row-major; 1,560 is not a multiple of a tile, so the split is done in row-major) against the full keys and values. The output is then already split along H, so the H-direction `mesh_partition` is not needed, and the W-direction all-gather is 1/4 the size. In the 1-frame chunk (no split in T), all 32 chips computed the same frame; now each chip computes only the 195 queries of its own H×W share, and no regather of the output is needed. The output of the T = 6 VAE test is bit-identical. VAE test 180.2 → 173.3 ms (SDPA 7.87 → 2.31 ms, all-gather 2.55 → 1.28 ms). 40-step e2e, alternating (no sync): off 40.78 s (VAE 0.460 s), on 40.84 / 40.77 s (VAE 0.445 / 0.446 s). The other off run hung (the first run right after a reset; it stopped in the text encoder all-gather (`all_gather_async`, `minimal_default_writer.cpp` line 290), unrelated to the VAE) | Default on (`WAN_VAE_ATTN_QSPLIT=0` reverts; commit `bc07569a9a8`) | VAE 0.460 → 0.445 s | −3% | 40.77 | −0.015 | Bit-identical | **adopted** |
| — | 2026-10-12 | (Investigation, VAE neighbor pad hang) Today's e2e runs had 1 hang inside the VAE neighbor pad (with #174. 2 other runs also stopped in a file named `minimal_default_writer.cpp`, but that was the all_gather_async kernel: the text encoder all-gather in the first run right after a reset). A stress test repeats only the VAE 40 times in one process and compares each output with the first (`WAN_VAE_TEST_REPEAT`, T = 6): default (#174–#176), 7 runs, 5 divergences in 272 iterations (4 in the first run, which then hung; the stop point differed per chip: the neighbor pad H receive, the W barrier, the W receive); without #174–#176, 2 runs, 80 iterations, no divergence; with the W-direction sender reverted (`TT_NP_W_FAST=0`, before #98), 5 runs, 200 iterations, no divergence. Fewer ops between neighbor pad calls may make an existing rare race more likely, but the counts are small and the cause is not identified (the atomic increment uses a flush that waits for the preceding writes at the destination, so the ordering of #98 holds). The default stays as is | Investigation (`WAN_VAE_TEST_REPEAT` is a test-only switch) | — | — | — | — | — | under investigation |
| — | 2026-10-12 | (Measurement) Today's default (with #174–#176) combined with optional features. 40-step e2e (no sync): default **40.75 s** (denoising 40.21 s, VAE 0.443 s; one more run, the first run right after a reset, hung in the text encoder all-gather), default + fast exp (#138) **40.10 / 40.04 s** (denoising 39.54 / 39.49 s; latent difference 12.72 / 12.38, one diverged), default + fast exp + Phase 2 merge (#172) **39.67 s** (denoising 39.12 s; latent difference 12.72, the same as the first fast-exp-only run). One more run hung in ring SDPA at step 25 | `TT_RING_SDPA_EXP_OVF=1`, `TT_RING_SDPA_CORR_ALL=1 TT_RING_SDPA_SALAD_LATE=1 TT_RING_SDPA_SALAD_MERGE=1` | — | — | **39.67** | — | Fast exp changes fine details (#138). #172 is the same computation | measurement (optional features combined go under 40 s) |
| — | 2026-10-12 | (Measurement, input for a decision) 4 alternating rounds of 3 configs (40 steps, no sync): default 40.83 / 40.84 / 40.84 s (1 run hung at step 6, the latent of 1 run diverged), default + fast exp (#138) 40.01 / 40.01 / 40.11 / 40.01 s (no hangs; the latent differs slightly per run, 12.43–13.94), default + fast exp + #172: all 4 runs hung (with the 2 runs just before, 5 of 6 hung; the only finished run is the 39.67 s one) → #172 cannot be used at the current hang rate. The fastest config that does not raise hangs is 40.01 s with fast exp | `TT_RING_SDPA_EXP_OVF=1` (optional) and others | — | — | 40.01 | — | Fast exp changes fine details | measurement |
| — | 2026-10-12 | (Investigation, closed) The chained 2000-block test (fast exp config) hung in 4 of 10 runs: 2 were a SrcA/SrcB divergence in ring SDPA (core 9-5 of a specific chip); in the other 2, the data sender of AGMM (`all_gather_minimal_matmul_async`) on a specific chip stopped while closing its fabric connection (`WorkerToFabricEdmSenderBase::close`), and the 3 other chips of the same TP group waited in the next distributed RMSNorm. The two different hang types concentrated on a few specific chips (not the power-capped slow chips), so a hardware-specific cause is possible. Partial rates for the parts of #172: `CORR_ALL` alone 3 of 3, with `SALAD_LATE` 0 of 2, all 3 parts 0 of 2. Treating this as hardware-specific instability, the hang investigation stops here, and changes that raise hangs do not count as improvements. Known LLK bugs (tt-metal #58717, the matmul reuse direction; #52691, the DVALID count of WH broadcast) do not apply to the current ring SDPA calls | Investigation | — | — | — | — | — | closed |
| **177** | 2026-10-12 | (VAE) Write the norm output straight into the conv's halo buffer. Most inputs of the causal conv (3×3×3) are RMSNorm (+SiLU) outputs, and neighbor pad copied each whole input into the halo ping-pong buffer before exchanging the halo (one extra read and write of every conv input). Add an optional output tensor and `output_pad` (t, h, w) to `prim::layer_norm` (row-major RMSNorm, `dit_rms_norm_unary_fused`); the row-major writer writes each stick at its interior position in the large buffer. Add `interior_in_place` to `neighbor_pad_async`: the output is the input itself, the reader reads the interior of the halo buffer, and the interior copy is skipped. norm1 and norm2 of the residual blocks and the decoder head norm use this path. The output of the T = 6 VAE test is bit-identical. 5-frame chunk 99.7 → 97.1 ms (neighbor pad 19.7 → 17.0 ms) | Default on (`WAN_VAE_NP_NORM=0` reverts; commit `1fa87c13974`) | 5-frame chunk 99.7 → 97.1 ms | −3% (VAE) | (measured together with #178) | — | Bit-identical | **adopted** |
| **178** | 2026-10-12 | (VAE) How neighbor pad sends over fabric. Per-core timing showed the op was bound by the W-direction cores: 1 packet per stick (192 B at 96 channels), about 1 us per stick (0.45–1 us depending on direction; the same for H). Making the reader read 8 sticks at a time changes nothing (neighbor pad 17.00 → 16.92 ms), so the conclusion was that the sender can only push as many packets as the EDM buffer slots divided by the round-trip time. (1) The H and W writers pack 4 sticks for the neighbor into one fabric scatter-write packet (add a public function to `WorkerToFabricEdmSender` that writes into the middle of a slot payload; the CB pages are held until the packet is sent). The H corner sticks, which land in the L1 of the receiving core, stay single unicasts because they arrive at the same core as the semaphore (EDM sends scatter writes on the default VC, so a semaphore increment can overtake them; the first version had output differences up to 0.0049). (2) The fabric reader reads 8 sticks at a time. (3) On the W edge chips, the reader built a zero stick for every row and the writer wrote it to the local chip (about 0.3 us per row, 840 us on the edge chips only), so the W reader now writes directly from a dedicated zero stick (CB c_in2). Top-level neighbor pad 1,200 → 500 us (checked with per-device timing zones, `TT_NP_ZONES=1`). The output of the T = 6 VAE test is bit-identical, 5-frame chunk 97.1 → 88.1 ms (neighbor pad 17.0 → 8.1 ms), test device time 173.3 → 159.6 ms (with #177). Stress test repeating only the VAE 40 times per run: old, 2 runs, no divergence in 80; new, 2 runs, 1 divergence in 80 (about the same as the existing rare divergence: the investigation row above, 5 in 272 with the default). 40-step e2e, alternating (no sync): old 40.82 / 40.77 / 40.78 s (VAE 0.448 / 0.444 / 0.449 s), new 40.72 / 40.74 s (VAE 0.411 / 0.410 s; one more run hung in ring attention at denoising step 12, unrelated to the VAE), new + fast exp 39.999 / 40.006 / 40.064 s (VAE 0.413 / 0.410 / 0.409 s, no hangs) | Default on (`TT_NP_SCATTER=0` and `TT_NP_RD_BATCH=1` revert; `TT_NP_ZONES=1` is for timing; commit `1fa87c13974`) | VAE 0.447 → 0.410 s (with #177) | −8% (VAE) | **40.72** | −0.06 | Bit-identical | **adopted** |
| — | 2026-10-12 | (VAE, #179) Cut the fixed cost per tile row in the row-major RMSNorm (+SiLU). Of the 10.5 ms of RMSNorm in a 5-frame chunk, about 7 ms is the 96-channel layer (20×120×104×96, 1,004 us per call, 80 cores, about 98 tile rows per core). That is about 10 us per tile row (about 16 us at 192 channels), which suggests a fixed cost of about 4.3 us. (1) With TILIZE_IN, `compute_kernel_hw_startup` is called for every row; reduce it to the DST sync and the pack init (`TT_LN_NO_RESTART=1`) or skip it entirely (`=2`): both bit-identical, RMSNorm 10.47 → 10.33 / 10.27 ms. (2) block_size (4 with fp32 DEST) tilizes and untilizes 4 tiles even when Wt = 3, so use the largest value at or below the limit that divides Wt (`TT_LN_BLK_DIV=1`, Wt = 3 → 3, 6 → 3): bit-identical, 10.47 → 10.29 ms (1,004 → 993 us). Both gain under 2%. RMSNorm runs 4 FPU ops per tile row at HiFi4 with fp32 DEST (square, row sum, x × rsqrt, × gamma) plus SiLU, so it looks compute-bound (little room to shrink it without changing the precision settings) | `TT_LN_NO_RESTART`, `TT_LN_BLK_DIV` (reverted) | RMSNorm 10.47 → 10.27–10.33 ms | −2% | — | — | Bit-identical | rejected |
| **180** | 2026-10-12 | (VAE host side) Timing the VAE sections in e2e with sync (`WAN_VAE_TIMING=1`): input 25.9 ms, decoder 371.2 ms, readback to the host 30.6 ms (uint8 on device 1.5, DMA 10.1, assembly 17.4 ms). The assembly (reordering each chip's BCTHW piece into BTHWC in a 97 MB array, 16 threads) takes 5 ms in a standalone bench. The second of the 2 alternating output buffers was only created and zero-filled on the second call (the real run after the allocation run), so both are now prepared on the first call: assembly 17.4 → 8.6 ms. The input is scaling 2.3, reorder 2.1, and `from_torch` (distribution to and writes into 32 chips) 21.1 ms, and the float32 → bf16 conversion is slow (7.7 ms on the host; 0.4 ms if torch first makes a contiguous bf16 tensor, with bit-identical rounding), so torch now converts before handing the tensor over: 21.1 → 13.1 ms. 40-step e2e (fast exp, no sync): assembly fix only 39.985 / 39.986 s (VAE 0.407 / 0.398 s), with the input fix too 40.048 / 39.951 s (VAE 0.403 / 0.391 s). One more run hung in ring attention at denoising step 19 (unrelated to the VAE) | Default on (`WAN_VAE_IN_BF16=0` reverts the input; commit `f3722c34766`) | VAE 0.410 → 0.391–0.407 s | −2% (VAE) | **39.951** (fast exp) | −0.01 | Same values | **adopted** |
| — | 2026-10-12 | (VAE input, continued) Timing the host-side split and the write separately: `ShardTensor2dMesh` split 8.2 ms, write to the device 0.7 ms. (1) Merge W and channels and send rows as 832 B pages: 13.6 ms, no change (small page writes were not the cause). (2) Make 32 pieces with torch views and combine them with `from_host_shards`: 9.0–9.6 ms, no change (the fixed cost of each `from_torch` call). (3) Arrange the pieces in torch as a contiguous block [32, T, h, w, C] and distribute it along dim 0 (#182): 1.6 ms | (1)(2) reverted | Input 8.2 → 1.6 ms ((3)) | — | — | — | Same values | (3) adopted (#182) |
| **181** | 2026-10-12 | (VAE) Hand-off for the mid-block attention. In a 5-frame chunk, before the frames were spread one per chip over the 8 chips of the W axis (T padded to 8, `mesh_partition`), every chip gathered all 5 frames of the chunk along H (0.70 ms) and W (AllBroadcast 2.57 ms, about 21 MB per chip including tile padding). Now each chip's (T, 1, h×w, C) goes through `all_to_all_async_generic` on the W axis (gather the W blocks, split T): each chip receives the W blocks of its own frame, all-gathers only along H, and a row-major permute (4D, with W and channels merged into the last dimension) restores the frame's raster order (6,240 tokens). norm, qkv, and SDPA stay the same. The output (1, h, W, C) is reordered into W blocks, and a reverse all-to-all returns each chip's own (5, 15, 13). The output of the T = 6 VAE test is bit-identical, 5-frame chunk 88.2 → 83.9 ms (AllBroadcast 2.57 → 0, all-gather 1.15 → 0.29, all-to-all 0.48; untilize, tilize, and pad also shrink), whole test 159.6 → 155.7 ms. Stress test repeating only the VAE 40 times per run (with and without, alternating, 2 runs each): no divergence in 80 for either. 40-step e2e (fast exp, no sync, with #182): 40.021 / 39.934 / 39.943 s (VAE 0.371 / 0.373 / 0.368 s, no hangs) | Default on (`WAN_VAE_ATTN_A2A=0` reverts; commit `c18627970be`) | VAE 0.391–0.403 → 0.368–0.373 s (with #182) | −6% (VAE) | **39.934** (fast exp) | −0.02 | Bit-identical | **adopted** |
| — | 2026-10-12 | (VAE, #183) In the first chunk (1 frame), the temporal cache of the causal conv is all zeros, so the conv becomes a (1, 3, 3) conv with only the last temporal weight (kt = 2) (the rows of the prepared weights are (C_in block, kT, kH, kW, c), so the last 1/3 of each block is sliced on device; the zero terms only add an exact 0 to the sum, and the order of the remaining terms is unchanged). The output of the T = 6 VAE test is bit-identical, and the device time of the first chunk goes 23.1 → 12.2 ms (conv3d 15.2 → 3.3 ms). In e2e, however, the first chunk goes 29.2 → 34.2 ms (with sync), and the VAE stays at 0.369 s (the first chunk is bound by the host-side op enqueue, and slice and other ops are added). The stress test showed divergences and hangs, but alternating with and without gave 3 and 2 in 80, so kT = 1 was not the cause (next row) | `WAN_VAE_FIRST_KT1=1` (reverted) | First chunk 23.1 → 12.2 ms (device), no effect in e2e | — | — | — | Bit-identical | rejected |
| — | 2026-10-12 | (Investigation, VAE divergence) In the stress test, the VAE output rarely diverges from the first iteration (max 0.0039–0.0117, in small regions of about 7–60 pixels, only in frames of 5-frame chunks). The diverging spots are not limited to the chip boundaries in H and W; in each run they cluster in the area of one specific chip. Before today's changes (#177, #178, #181) vs. the current default, alternating 3 runs each (40 iterations per run): before 3 in 120, now 3 in 120, the same rate, no hangs. The rate swings between 0 and 5% depending on the time (2 hours earlier, the current config had no divergence in 160). During the kT = 1 experiment the VAE hung twice (both times one core of one chip stayed inside conv3d, and the other chips waited at the start barrier of the next neighbor pad; cores (2,9) and (3,9) of specific chips) | Investigation (`WAN_VAE_TEST_REPEAT` and a test change that prints the divergence location) | — | — | — | — | — | existing rare divergence (not increased by today's changes) |
| **184** | 2026-10-12 | (Pipeline) Time the hand-off of the final latent from denoising (more sections in `WAN_VAE_TIMING=1`): all-gather on the SP axis 1.3 ms, read from 1 chip 9.3–9.8 ms (4 MB, about 0.4 GB/s on that chip's command queue), host unpatchify 1.1 ms. (1) Convert to row-major on device first: read 9.8 ms, no change (the host untilize is not the bottleneck). (2) Drop the all-gather, read the SP-direction pieces of all chips in one async read, and join replica 0 of the other axis in SP order: 7.8 ms (matches the all-gather path with `WAN_VAE_IN_CHECK=1`). (3) Read only the chips of one column individually: not faster. 40-step e2e (fast exp, no sync, (2)): 39.940 / 39.952 / 39.943 s (denoising 39.467 / 39.478 / 39.471 s, within noise). The other 2 runs hung in the denoising ring SDPA (unrelated to the VAE) | Default (2) (`WAN_LAT_SHARDS=0` reverts; commit `849a100d9ae`) | Hand-off 11 → 7.8 ms | — | 39.940 (fast exp) | −0.003 | Same values | **adopted** |
| **185** | 2026-10-12 | (VAE) Re-search the conv3d blocking. The e2e tracy (from `cpp_device_perf_report.csv`: a 5-frame chunk takes about 77 ms on device, 63% of it conv3d) shows that the 96→96 3×3×3 conv at 120×104 (6 per chunk) and the 192→96 spatial conv were not in the table and ran with default values (the channel_key fallback) (the 2026-10-08 search ran on the 8×9 grid before Ethernet dispatch, and the fallback won for these two). Re-run `bruteforce_conv3d_sweep` on the 8×10 grid (1 chip, timed with trace): conv_out 2408 → 2067 us, up2_res 2722 → 2600, up1_res 1701 → 1608, up3_res 2178 → 2110 (new), up2_spatial 1525 → 1501 (new), up1_spatial 2339 → 1473 us. The C_in blocking (the K order) does not change, so the output of the T = 6 VAE test is bit-identical. 5-frame chunk 83.9 → 81.0 ms (conv3d 49.0 → 46.1 ms), conv3d of the first chunk 15.2 → 14.1 ms. The stress test had 2 hangs where one core of row 9 (the row that became usable with Ethernet dispatch) stopped inside conv3d (core (3,9) of a specific chip both times; math waits in the next fast tilize init, unpack in the bias untilize). Both were convs with C_in / C_out blocks of 384 / 96, and the second was up0_spatial, whose table entry was not changed. To avoid adding more convs of that shape, the new value for up1_spatial (384 / 96) is not used. Alternating after 2 resets (40 iterations per run): old 5 divergences in 80, new 1 in 81, no hangs. 40-step e2e (fast exp, no sync): 39.995 / 39.922 / 39.938 s (VAE 0.359 / 0.361 / 0.360 s, no hangs) | Default on (override table `_BLOCKINGS_8X10_RESWEEP`; `WAN_C3D_RESWEEP=0` reverts; commit `2d210b9eb8a`) | VAE 0.366–0.371 → 0.359–0.361 s | −2% (VAE) | **39.922** (fast exp) | −0.008 | Bit-identical | **adopted** |
| — | 2026-10-12 | (VAE, #186) Lower the math fidelity of the residual block RMSNorm (+SiLU) from HiFi4 (`WAN_VAE_NORM_FID`): on the T = 6 VAE test, HiFi2 gives PCC 99.951% (max difference 0.0625, mean 0.0080), HiFi3 99.964% (mean 0.0068), LoFi 99.379% (mean 0.027); none is bit-identical. Even at HiFi2, RMSNorm only goes 10.47 → 10.37 ms, so FPU fidelity is not the bottleneck. Removing the fused SiLU for timing only (`TT_HK_LN_NOACT=1`, wrong output) gives 10.47 → 8.27 ms (96-channel layer 1,004 → 743 us), so SiLU costs about 2.2 ms per chunk | `WAN_VAE_NORM_FID`, `TT_HK_LN_NOACT` (reverted) | — | — | — | — | Values change | rejected |
| — | 2026-10-12 | (VAE, #187) Timing only: when the conv3d reader skips the vol2col copy (`TT_C3D_SKIP_V2C`, wrong output), conv3d on the T = 6 test goes 46.08 → 43.94 ms per 5-frame chunk. Most layers drop 3–6%; only conv_out (Cin 96, Cout 3) drops 43%. The bottleneck is the compute side (tilize and the HiFi2 matmul), so rewriting the reader as an implicit GEMM gains nothing | `TT_C3D_SKIP_V2C` (timing only, reverted) | — | — | — | — | Values change | rejected |
| 188 | 2026-10-13 | (VAE) Host side of the decode input and output: (a) fold the latent scaling, the bf16 conversion, and the reorder into per-device blocks into one strided copy (`WAN_VAE_IN_FAST`); (b) do the RGB reorder on device (matmul with a 0/1 matrix, HiFi4 with fp32 accumulation, exact), each device sends (T, h, w×3) uint8 rows, and the host only copies rows instead of reordering planes (`WAN_VAE_OUT_RGB`). The uint8 video is bit-identical (3/3, 6/6). Timing with a sync per section suggested −5 to −7 ms (d2h 23–24 → 18 ms, DMA 10 → 6.3 ms), but timing only the VAE without sync, alternating 5 times each, gives old 358.3, input side only 358.4, output side only 357.2, both 357.8 ms: no difference. The on-device reorder compute (about 3.4 ms) cancels the host-side savings. e2e: default 0.3602 s, new 0.4125 s (1 run). In this e2e, 3 of the 5 runs hung in the denoising ring SDPA (fast exp) | `WAN_VAE_IN_FAST`, `WAN_VAE_OUT_RGB` (reverted) | — | — | — | — | Bit-identical | rejected |
| **189** | 2026-10-13 | (VAE) Fuse the final add of the residual block into the second conv. Add `residual_tensor` (row-major bf16, the same shape as the output) to conv3d: for each output block, the reducer's writer reads the residual rows into a CB in the same order as the output rows, and compute tilizes them and adds them after the bias and before the untilize (an FPU add on the bf16 intermediate tiles). It is used when the second conv has a single C_in block, a C_out block larger than 32, and a patch count that is a multiple of a tile (the top-level 96 channels). T = 6 tracy: the BinaryNg of about 0.9 ms × 3 in a 5-frame chunk disappears, that conv2 goes 2018 → 2074 µs, and the total goes 151.70 → 149.82 ms. Bit-identical (the T = 6 output, the 81-frame uint8 decode, and a stress test of 3 runs × 40 (each after 2 resets): all 120 match, no hangs). Timing only the VAE (alternating 5 times): 357.8 → 348.3 ms. 40-step e2e (no sync): fast exp 40.005 / 39.915 s (VAE 0.3555 / 0.3475 s, the video is the same as #185), default 40.662 / 40.677 s (VAE 0.3477 / 0.3479 s), no hangs | Default on (`WAN_VAE_C3D_RESID=0` reverts; commit `1d717855da4`) | VAE 0.359–0.361 → 0.348 s | −3% (VAE) | **39.915** (fast exp) | −0.012 | Bit-identical | **adopted** |
| **190** | 2026-10-13 | (VAE) Extend the #189 fusion to conv2s with several C_in blocks (fp32 partial sums). After the reduce in conv3d, the fp32 result is first rounded into a bf16 CB with the same read pattern as the untilize (a copy to srcA), and then the tilized residual rows are added. This matches the result of the conv followed by a separate add. All row-major residual blocks whose conv2 has full-block output are fused (the 192 and 384 channels too). T = 6: bit-identical. The 192-channel conv2 goes 2523 → 2613 µs (+90 µs), 3 adds of about 378 µs disappear, and the T = 6 total goes 149.82 → 148.17 ms. Timing only the VAE (alternating 5 times): old 357.5, #189 348.6, #190 344.9 ms. Stress test: 79 of 80 match (1 is the known kind of divergence, max 0.0117, frames 12–20 of a 5-frame chunk), no hangs. 40-step e2e (no sync): fast exp 39.998 / 39.924 s (VAE 0.3469 / 0.3468 s, the video is the same as #185), default 40.656 / 40.665 s (VAE 0.3454 / 0.3445 s), no hangs | Default on (`WAN_VAE_C3D_RESID_ALL=0` limits it to convs with a single C_in block; commit `1860a57323d`) | VAE 0.348 → 0.346 s | −1% (VAE) | **39.924** (fast exp) | −0.002 | Bit-identical | **adopted** |
| **191** | 2026-10-13 | (VAE) Three fixes to the row-major RMSNorm (tilize in, untilize out, gamma but no beta): (1) the gamma (+ fused SiLU) pass writes from DEST straight to the row-major output with `pack_untilize` (removes the separate untilize pass; the tile output CB is bf16, the same format as the output, so the values are the same); (2) the block width is the largest value at or below the DEST limit that divides Wt (Wt 3 → 3, 6 → 3), so padding tiles are not processed (`TT_LN_BLK_DIV` from #179); (3) after the tilize of each tile row, only the DST sync and the packer are re-initialized (`TT_LN_NO_RESTART=1` from #179). T = 6: bit-identical. norm 11.69 → 11.16 ms ((1) only) → 10.78 ms (all 3; 96 channels 7462 → 6791 µs, 192 channels 2631 → 2498 µs). Stress test 79 of 80 match (1 is the known kind of divergence: frames 15–20 of a 5-frame chunk, about 56×56 pixels, max 0.0186), no hangs. 40-step e2e (no sync): fast exp 39.982 / 39.982 s (VAE 0.3413 / 0.3412 s, same video). One run of the default config hung in the denoising ring SDPA (`ring_joint_reader`, compute; this also happens with the default config: on 2026-10-13, 3 of 10 fast-exp runs and 1 of 5 default runs) | Default on (`TT_LN_FUSED_UNTILIZE=0`, `TT_LN_BLK_DIV=0`, `TT_LN_NO_RESTART=0` revert; commit `c088a418610`) | VAE 0.346 → 0.341 s | −1.5% (VAE) | **39.982** (fast exp) | −0.005 | Bit-identical | **adopted** |
| **192** | 2026-10-13 | (VAE) The (1,3,3) spatial conv after the upsample (nearest, 2×) reads the low-resolution tensor directly. Added `input_upsample=2` to conv3d: the input is the low-resolution tensor with a halo of 1 in H and W, the conv runs on the high-resolution shape with halo (2H−2 × 2W−2), and when the reader gathers the input pieces it maps row/column Y of the high-resolution shape with halo to (Y+1)/2 at low resolution (the vol2col assembly is unchanged; coalesced reads are not used in this mode). The up blocks drop `ttnn.upsample` and run neighbor pad at low resolution. T = 6 tracy: the 6 upsample ops disappear; in a 5-frame chunk, upsample 1.03 ms plus the high-resolution neighbor pad 1.71 ms become a low-resolution neighbor pad of 0.69 ms. The top-level spatial conv is +49 µs. T = 6 total 148.10 → 144.29 ms. Bit-identical (the T = 6 output). Timing only the VAE (alternating 5 times): 342.2 → 334.2 ms. Stress test 78 of 80 match (2 are the known kind of divergence: frames 11–20 and 17–20 of a 5-frame chunk, max 0.0127 and 0.0089), no hangs. 40-step e2e (no sync): fast exp 39.982 / 39.888 s (VAE 0.3410 / 0.3324 s, same video), default 40.632 s (VAE 0.3344 s), no hangs | Default on (`WAN_VAE_C3D_UP=0` reverts; commit `05dcc613f76`) | VAE 0.341 → 0.333 s | −2.3% (VAE) | **39.888** (fast exp) | −0.008 | Bit-identical | **adopted** |
| — | 2026-10-13 | (VAE, #193) uint8 readback per chunk: after the clamp, the output of each cached chunk is converted to uint8 on device, the read starts with `.cpu(blocking=False)` and `record_event`, and after the next chunk's ops are enqueued the host waits for the previous chunk's read and arranges it into BTHWC (`WAN_VAE_CHUNK_D2H`). Bit-identical. Timing only the VAE (alternating 5 times): old 332.3, per chunk 332.1 ms, no difference. Breakdown: the waits for chunks 0–3 are almost 0 ms (each 5 ms reorder hides behind the device compute); the last chunk waits 55 ms and reorders in 4.4 ms. The roughly 5 ms of host work that gets hidden is cancelled by the added device work (about 45 post-processing ops per chunk, about 13 before; the read is also split into 5). A cProfile of one decode shows 1154 ttnn op calls × about 190 µs = 219 ms (about 30 ms on the Python side), which is shorter than the device time (about 300 ms) except for the first 1-frame chunk | `WAN_VAE_CHUNK_D2H` (reverted) | — | — | — | — | Bit-identical | rejected |
| — | 2026-10-13 | (VAE, #194) Larger pages for the host readback: between the clamp and the uint8 conversion, reshape the bf16 video (1,C,T,h,w) to (1,C,T,h×w) before the typecast (`ttnn.reshape` does not support uint8), and read each (c,t) as one 12480-byte page (currently 104-byte pages). On the host, each piece is viewed as (1,C,T,h,w) again and put in place (`WAN_VAE_D2H_FLAT`). Timing only the VAE (alternating 5 times): old 333.4, new 335.9 ms, so it is slower (the reshape costs more than the DMA saves). In one comparison, 31 pixels differed by 1 (taken as the known rare divergence, not investigated) | `WAN_VAE_D2H_FLAT` (reverted) | — | — | — | — | — | rejected |
| **195** | 2026-10-13 | (VAE) Keep the residual stream at latent resolution in row-major too. The decoder (with cache) no longer converts to tiles before the mid block. The residual blocks of the mid block and of the first up block (15×13 per chip, 384 channels) use the row-major path (norm directly into the neighbor pad buffer (#177), in-place neighbor pad, fused residual add (#189, #190)), and the a2a attention adds the residual in row-major (the non-a2a path used for the first 1-frame chunk still converts the input to tiles as before). Bit-identical (the T = 6 output, the 81-frame uint8 decode, stress test 80 of 80 match). Ops per 5-frame chunk 212 → 193, device time 73.15 → 72.68 ms. Timing only the VAE (alternating 5 times): 332.3 → 330.8 ms. 40-step e2e (no sync): fast exp 39.968 (the latent differed from #185: the known denoising race, video difference 7.78) / 39.975 s (VAE 0.3304 / 0.3302 s; the second run's video is the same); one default run hung in the denoising ring SDPA (on 2026-10-13, 3 of 12 fast-exp runs and 2 of 7 default runs). Also timed the zones of the top-level in-place neighbor pad (498 µs): the W-direction send takes about 485 µs (set by the packet count), and H runs in parallel at about 305–343 µs. Even if the receiver batched the W sends, H would become the floor and only about 150 µs per call would be saved, so this was not pursued | Default on (`WAN_VAE_RM_MID=0` reverts; commit `375f45d2e63`) | VAE 0.333 → 0.330 s | −0.9% (VAE) | 39.968 (fast exp) | −0.003 | Bit-identical | **adopted** |
| 196 | 2026-10-13 | (Pipeline, VAE) Optional on-device latent hand-off (`WAN_LAT_DEVICE=1`): `WanVAEDecoderAdapter.decode_tokens` takes the fp32 latent tokens, split along the DiT sequence, as they are. Still as pieces, it scales them per channel in fp32 (multiply, add) and converts them to bf16 (device fp32 multiply/add and typecast were checked to be bit-identical to torch on 2.1 M elements), then does a bf16 all-gather on the SP axis, unpatchify with a row-major reshape / permute, mesh_partition on the VAE H and W axes, pads the channels, and decodes. The pipeline (uint8) skips the host readback, unpatchify, scaling, and upload, and the sync between denoising and the VAE. The VAE input is bit-identical (`WAN_LAT_DEVICE_CHECK`), and when the latent is the same the video is the same too (3 of 4 check runs; the remaining 1 was the known denoising divergence). The hand-off ops take 5.3 ms (with sync). Timing only the VAE, including the same host readback as the pipeline (alternating 3 times): 338.3 → 331.1 ms. e2e without saving the latent: via the host 39.860 / 39.895 s (VAE 0.320 s), on device 39.876 / 39.951 / 39.864 / 39.949 s (a spread of ±50 ms, no visible difference). In e2e with a sync per step, "total − sum of steps" is 567.2 ms via the host and 535.9 ms on device (1 pair each). One on-device run (sync per step) hung in the VAE neighbor pad right after denoising (`minimal_default_writer`, `phase2_w_reader`), so the default stays off until this can be shown to be unrelated. Also, the run script no longer saves the latent by default (`HK_DUMP=1` saves it). The save was inside the timed VAE section (about 10 ms), so e2e numbers from here on are about 10 ms smaller than before for the same code | Optional, default off (`WAN_LAT_DEVICE=1`; commit `44e4b6deca9`) | — | — | — | — | Bit-identical | optional |
| — | 2026-10-13 | (Investigation, night-time VAE neighbor pad hang) Repeating the 81-frame decode 200 times in one process (2 resets per process) hangs in the VAE. In triage many chips looked stuck in neighbor pad (`phase2_w_reader` waiting on the semaphore of the W neighbor and the H barrier), but on a second look every case was one core of one chip (logical row 9, x = 2–4) stopped partway through the conv3d bias / untilize stage (once in the RMSNorm tilize), and neighbor pad was only caught up in it. All 16 VAE hangs from 2026-10-11 to 2026-10-12 (conv3d 15, RMSNorm 1) had this shape; they concentrated on a few specific chips, so a hardware-specific cause is possible. The first one came before the 2026-10-12 VAE changes (#174–#178). Row 9 is the row that first came into use for compute with Ethernet dispatch (#154). The current default config hung at iterations 118 and 57, without #192 and #195 at 39, without all of today's VAE changes (= the #185 state) at 3, and the on-device hand-off at 85 and 92. During the day, about 100 81-frame decodes and 440 iterations of the T = 6 stress test did not hang. The same night, the denoising ring SDPA hung with both fast exp and the default (e2e: 4 of 18 fast-exp runs, 3 of 9 default runs). This shows that today's VAE changes are not the cause, but the cause is not identified. The #185 state still includes the 2026-10-12 neighbor pad changes (#175, #177, #178), and it was not compared with configs from before them. Recounting all logs of the day (T = 6 counts as 0.4 of an 81-frame decode), the VAE hangs clustered in 2 time windows: 4 in about 49 decodes in the first (conv3d 3, permute 1), and 7 in about 397 in the second. The roughly 600 decodes between and before them had no hangs; at a constant rate (about 1%), that happens with only 0.2% probability. The rate changed during the day, but what changed it (the time of day, the single-chip sweep just before, how the test is built) is not identified. That day's e2e had 10 hangs in 56 runs (ring SDPA 9, VAE 1), and the ring SDPA hangs were spread out rather than clustered in time | — | — | — | — | — | — | investigation |
| — | 2026-10-13 | (Investigation, hang bisection 1) (1) Logs: denoising hangs (the stuck step is spread almost evenly over 0–39; 49 cases: 10, 14, 13, 12) were 0 in about 5,200 steps during the Tensix dispatch period (2026-10-05 to 2026-10-10, sync per step), and also 0 in 1,692 steps with sync on 2026-10-10 with Ethernet dispatch. On 2026-10-11, 5 in 4,124 steps with sync and 29 in 5,560 steps without sync; on 2026-10-12, 16 in 4,182 steps without sync. Post-denoising (VAE) hangs: 1 in 283 in the Tensix period (a 2-step check run), 0 in 49 with Ethernet dispatch on 2026-10-10, 10 in 357 on 2026-10-11 and 2026-10-12. (2) HEAD e2e (fast exp, no sync, 40 steps), alternating in blocks, 2 rounds: Ethernet dispatch (A), 32 runs: 1 ring SDPA hang (step 35) and 4 VAE row-9 conv3d hangs (1 of them in the warm-up run); Tensix (B), 38 runs: 0. VAE inside e2e (2 decodes per run): A 4 in 62, B 0 in 76. The ring SDPA rate of A is about 1 per 1,235 steps, lower than the about 1 per 230 steps without sync on 2026-10-11 and 2026-10-12 (the probability of 1 or fewer at that rate is 3%). At this rate, telling A and B apart for ring SDPA needs about 6,000 steps per config. (3) HEAD VAE (81-frame decode 200 times in one process), alternating: Ethernet (A, 8×10) 4 hangs in 568 (all with one core of row 9 stopped inside conv3d; cores (2,9), (4,9), (5,9), (2,9) of specific chips), Ethernet with only conv3d on 8×9 (C, `WAN_VAE_C3D_GRID_Y=9`, other ops on 8×10) 0 in 600, Tensix (B, 8×9) 0 in 1,200. At equal rates, the probability that all 4 fall in A is 6% against C and 0.3% against B and C (adding the decodes inside e2e: A 8 in 630, B 0 in 1,276, C 0 in 600, and the probability that all 8 fall in A is 0.002%). VAE hangs occur only when conv3d runs on the 8×10 grid (row 9 vs. the 80-core work split is not yet separated), so they could not happen before #154. Mismatches against the first output: A 0/568, C 2/600, B 5/1,200 | `WAN_VAE_C3D_GRID_Y` (uncommitted experimental switch) | — | — | — | — | — | investigation |
| **197** | 2026-10-13 | (VAE) On WH Galaxy, run conv3d on the first 9 rows of the grid by default. The hang bisection (row above) showed that the VAE hang (one core of row 9 of one chip stops inside conv3d) happens only when conv3d runs on the 8×10 grid of Ethernet dispatch, and never on 8×9, with either Ethernet dispatch or Tensix (8×10: 8 in 630; 8×9: 0 in 1,876). `_c3d_grid` checks the cluster type (`GALAXY`, `TG`) and limits the rows to 9; other systems keep the full grid. On 8×9, the blocking re-searched for 8×10 in #185 is slow (VAE alone 410 ms; 353 ms with the original table; 332 ms on 8×10), so `get_conv3d_config` uses the pre-#185 table for grids with fewer than 10 rows (this also applies to Tensix dispatch). The C_in blocking is the same, so the values are the same: the T = 6 VAE output is bit-identical to 8×10, and the fast-exp e2e video differs by 0.0 from the #185 reference run. e2e (no sync): fast exp 39.902 / 39.909 s (VAE 0.340 s), default 40.720 / 40.658 s (VAE 0.341 / 0.343 s); one more fast-exp run hung in the denoising ring SDPA at step 29 | Default on (`WAN_VAE_C3D_GRID_Y=0` gives the 8×10 grid; any other value limits it to that many rows; commit `60736588c4c`) | VAE 0.330 → 0.340–0.343 s | +3% (VAE) | 39.902 (fast exp) | −0.066 | Bit-identical | **adopted** |

![#166 (left) and #167 (right)](img/wan22_167_pv_sum_cmp.jpg)

![#249 (left) and #155 (right, Ethernet dispatch, 3 SP links)](img/wan22_155_eth_dispatch_3link_cmp.jpg)

![#249 (left) and #154 (right, Ethernet dispatch, default config)](img/wan22_154_eth_dispatch_cmp.jpg)

![#128 (top) and #130 (bottom, rejected, hazy)](img/wan22_130_exp_fast_cmp.jpg)

![#126 (top) and #128 (bottom)](img/wan22_128_qk_sb18_cmp.jpg)

![#123 (top) and #126 (bottom)](img/wan22_126_ln_post_fuse_hifi2_cmp.jpg)

![#123 (top) and #125 (bottom)](img/wan22_125_ln_post_fuse_cmp.jpg)

![#120 (left) and #123 (right)](img/wan22_123_ln_fpu_pre_cmp.jpg)

![#114 (left) and #117 (right)](img/wan22_117_sdpa_final_bf8_cmp.jpg)

![#112 (left) and #114 (right)](img/wan22_114_temb_fold_cmp.jpg)

![#109 (left) and #111 (right)](img/wan22_111_projout_tp_cmp.jpg)

### #1 details: FFN2 unfused

- **Motivation:** Item 1 of section 9.3. The fused MM+RS ran with default settings, with no config for the WH 8×9 grid (utilization 19%).
- **Change:** Added a branch to `WanTransformerBlock.forward` in `transformer_wan.py` that runs unfused when the environment variable `WAN_FFN2_UNFUSED=1` is set (9 lines). It is the same path that H3 uses when it cannot resolve a fused config.

Breakdown of the FFN2 part (per chip):

| | Fused (default config) | Unfused |
|---|---|---|
| Matmul | 5.18 ms (includes RS and addcmul) | 2.19 ms (72 cores, utilization 45%) |
| reduce-scatter | (included above) | 0.81 ms |
| addcmul | (included above) | 0.20 ms |
| **Total** | **5.18 ms** | **3.19 ms (−1.99 ms)** |

Whole pipeline (480p):

| Stage | #0 baseline | #1 unfused | Diff |
|---|---|---|---|
| Text encode | 0.09 s | 0.08 s | — |
| denoise | 108.93 s | 103.11 s | −5.82 s (−5.3%) |
| VAE decode | 1.36 s | 1.23 s | — |
| **Total** | **110.39 s** | **104.43 s** | **−5.96 s (−5.4%)** |

- The measured saving (−5.8 s) is a bit smaller than the one predicted from the block measurement (0.084 s per forward × 80 = about 6.7 s). This is probably because a single-block measurement does not include run-to-run variation or host-side dispatch time.
- The warning about the fused MM+RS default config (3,360 times per run in #0) dropped to 0.
- The video has the same composition and the same subject, with no artifacts. Fine details (wing pattern, positions) differ slightly. This is a bf16 rounding difference caused by a changed summation order (the same kind as TP4/SP8 vs TP8/SP4 in the H3 README).

![Frame comparison at 3 s (left: #0 baseline, right: #1 FFN2 unfused)](img/wan22_01_ffn2unfused_cmp3s.jpg)

**Next options:** Like the H3 follow-up PR (#57941), searching for a fused MM+RS config for the WH 8×9 grid and adding it back could be even faster than unfused (in H3, the whole ff2 went from 10.27 → 8.24 ms).

### #2 details: Matmul blocking for WH, and retuning the fused MM+RS

- **Motivation:** Item 4 of section 9.3. All main DiT matmuls (M=4096) ran with the generic config (8,8,8). The FFN2 fused MM+RS had no WH config (#1 worked around it by going unfused).
- **Search:** Added the 7 Wan 480p shapes to `sweep_mm_block_sizes.py` and searched with `wh_4x8_ring` (`MM_SWEEP_PROFILER_DUMP_EVERY=100000`, about 50 min total, 2,080 combinations). Combinations that do not fit in L1 are skipped automatically (there are many for the fused MM+RS).
- **Adopted configs** (added to the config table in `matmul.py`; times are op-level sweep values):

| Shape | Use | Generic config (8,8,8) | Adopted | Saving |
|---|---|---|---|---|
| 4096×5120×3840 | QKV (AGMM 8×8) | 2,317.6 µs (rank 8 of 355) | (8,8,10,(2,2)) 2,228.0 µs | −3.9% |
| 4096×5120×1280 | self output projection (with addcmul) | 1,202.2 µs (rank 4) | (8,8,5,(4,1)) 1,184.2 µs | −1.5% |
| (same) | cross Q and output projections (plain) | 999.2 µs (rank 3) | (same) 986.4 µs | −1.3% |
| 4096×5120×3456 | FFN1 + GELU (AGMM 8×8) | 2,096.4 µs (rank 4) | (8,10,8,(2,2)) 2,053.3 µs | −2.1% |
| 4096×3456×5120 | FFN2 matmul only (8×9) | K_block 8 is not a candidate. (8,9,8) gives 2,117 µs | (8,6,10,(2,2)) 1,881.1 µs | about −11% |
| 4096×3456×5120 | FFN2 fused MM+RS+addcmul | Default config (with warning) | (6,4,10,2,2) on an 8×7 grid, 2,757.3 µs (best on 8×6 is 2,786.7) | — |

- **Takeaways:** Almost all top configs use subblock (2,2) or (4,1), and the top 10 differ by only a few percent. Blocking does not give large savings (the same was true for H3 on WH; what helped there was the precision switch that drops fp32 accumulation).
- **The effect inside the block is even smaller:** In the warm state, matmul time includes the wait for the FSDP weight gather, so part of the saving is hidden.
- **2a vs 2b:** The fused MM+RS (2b) is 0.35 s faster over the whole pipeline than unfused + best blocking (2a) (repeat runs of the same setup differed by 0.13 s, so this counts as a real difference). 2b runs on the standard path with no switch, so `WAN_FFN2_UNFUSED` is no longer needed.
- **Caution:** The fused MM+RS on the 8×7 grid uses 1 reduce-scatter core per link (the same count that, in the H3 README, made the WH default config hang). It did not hang in the 162 sweep combinations, the block measurement, or one pipeline run (3,200 calls). Keep watching it in long runs.
- The remaining generic-config warnings are only for small matmuls: M=32 (timestep embedding), M=512 (text side), and M=6240 (VAE, etc.).

![Frame comparison at 3 s (left: #0 baseline, middle: #2a, right: #2b)](img/wan22_02_cmp3s.jpg)

### #3 details: ring SDPA chunk size

- **Motivation:** Item 3 of section 9.3 and the discussion in section 4.4. Wan 480p used (256, 256): 160 work units (16 chunks × 10 heads) are dealt to 63 cores in 3 rounds, so 15.3% of the slots are idle. k_chunk=256 is half of H3's (512), so fixed costs such as the online softmax correction are paid twice as often.
- **Method:** Added an override through the environment variable `WAN_SDPA_CHUNK="q,k"` to `attention_wan.py` and measured the block for each candidate in warm mode (about 1.5 min per candidate). Candidates were pre-filtered with H3's L1 formula (`2048·Sq·Sk + 67584·Sq + 32768·Sk + 116032 ≤ 1,499,136 B`, Sq=q/32, Sk=k/32).

| chunk (q, k) | ring SDPA | 1 block | Idle | Utilization (63-core basis) | Notes |
|---|---|---|---|---|---|
| **(256, 512)** | **13.06 ms** | **29.46 ms** | 15.3% | 40.8% | **Adopted** (same value as H3 15 s) |
| (352, 320) | 13.16 ms | 29.56 ms | 4.8% | 40.5% | Almost the same speed. 97.35 s for the whole pipeline |
| (256, 384) | 13.64 ms | 30.10 ms | 15.3% | 39.1% | |
| (192, 640) | 14.03 ms | 30.44 ms | 12.7% | 38.0% | |
| (352, 256) | 14.12 ms | 30.52 ms | 4.8% | 37.7% | Effect of less idle time only |
| (384, 256) | 14.55 ms | 30.85 ms | 12.7% | 36.6% | |
| (256, 256) | 14.69 ms | 31.04 ms | 15.3% | 36.3% | Original config |
| (352, 352) | 17.02 ms | 33.38 ms | 4.8% | — | k tile count 11 (prime) makes it much worse |
| (256, 544) | 18.71 ms | 34.93 ms | 15.3% | — | k tile count 17 (prime) makes it much worse |
| (128, 768) | 18.87 ms | 35.21 ms | 15.3% | 28.2% | q is too small |

- **Takeaways:**
  - **A larger k helps a lot.** With q=256, k 256 → 384 → 512 gives 14.69 → 13.64 → 13.06 ms. This is likely because there are fewer online softmax steps (fewer times the fixed cost is paid).
  - **Less idle time also helps** (q=352 with k=256 gives −0.57 ms).
  - **A prime k tile count is much worse** (11, 17). The guess is that the matmul subblocks inside the kernel can only be small (not verified). Choose k so that the tile count has many divisors (10, 12, 16).
  - No candidate failed by not fitting in L1. H3's formula worked as a guide here too.
- **Whole pipeline:** With (256, 512) as the default and no environment variable, 97.35 s (97.18 s with the environment variable set; the difference is within repeat-to-repeat variation).
- **Video:** No artifacts. But **the difference from #2b is visibly larger than for earlier changes** (deeper orange on the wings, a different shape for the flower in the back, etc.). The changed order and split of the softmax sums is amplified over 40 steps; both are valid videos for the same prompt. The block PCC is 99.9938%, actually higher than with the original config (99.9919%).

![Frame comparison at 3 s (left: #0 baseline, middle: #2b, right: #3)](img/wan22_03_cmp3s.jpg)

![#3 video at 0.5 s / 3 s / 4.9 s](img/wan22_03_seq.jpg)

### #4 details: Automatically deciding whether to use the fused MM+RS

- **Motivation:** A 720p check showed that FFN2 had fallen back to the default config again (the #2 config is for M=4,096). Instead of switching per resolution with an environment variable, add the same check as H3's PR #56886: "fuse only when the fused config can be resolved".
- **Change:** In the Ring branch of `transformer_wan.py`, if `resolves_fused_mmrs_config(M, ffn_dim / TP, dim, <device core grid>)` is False, run unfused (FFN1 → FFN2 matmul → reduce-scatter → addcmul). This function is a query that shares code with the actual config resolution (`_resolve_fused_mmrs_config`), so the check and the real path cannot drift apart. M is `padded_shape[-2]`, as in the fused path.
- **Check:** 480p (M=4,096) gives True (tuned fused path), 720p (M=9,472) gives False (unfused). With no environment variable, 480p takes 97.38 s (same as #3) and 720p takes 311.22 s (same as the unfused (c), 311.33 s). Both show 0 warnings about the fused MM+RS default config.
- On BH the rule engine returns a config, so it stays fused as before (not verified, because no BH environment was available).

### #5a details: FSDP weight prefetch (no effect)

- **Goal:** Finish the FSDP weight gather (about 2.9 ms per layer, running at the bandwidth limit) during SDPA (about 13 ms), so that the matmuls do not wait for weights.
- **Safety check:** The ring SDPA K/V rotation and the FSDP weight gather share the same semaphore sets in the SP direction (2 sets used alternately). Moving the output-projection weight gather before SDPA still uses at most 2 different sets at the same time, which stays within the alternating scheme. The weight buffers are also separate per shape. So a minimal version with only one prefetch was tried (no hang).
- **Result (chip 0, last run, ms):**

```
No prefetch:    … SDPA 3.514 → 16.579 | weight gather 16.627 → 17.207 | out proj 16.681 → 18.297 (1.62)
With prefetch:  … weight gather 3.502 → 3.855 | SDPA 3.872 → 16.930 | out proj 16.972 → 18.178 (1.21)
```

- The output projection no longer waits for weights and shrank from 1.62 → 1.21 ms, but **SDPA did not start until the weight gather finished**, so the order just swapped and the gain cancelled out (block 29.46 → 29.55 ms). A matmul can start while it partly overlaps the preceding weight gather, but SDPA does not overlap.
- **Conclusion:** With the current setup ("one sub-device over all cores"), the weight gather cannot be hidden behind SDPA. Hiding it needs splitting the cores into sub-devices (`tech_reports/SubDevices/`) and managing synchronization by hand, which is close to rebuilding the CCL management. The switch stays in the code (default off).
- **Alternative:** If memory allows, stop FSDP for some weights only (for example, the FFN) and keep them resident, which removes that part of the weight gather.

### #5b preparation: Free device memory, and the steps each of the two models handles (2026-10-01)

- **Memory** (logged in the pipeline with `WAN_MEM_LOG=1`, `ttnn.get_memory_view`, 12 GiB per chip):

| Point | Used | Free | Largest contiguous free |
|---|---|---|---|
| Right after pipeline build | 6.60 GiB | 5.37 GiB | 5.37 GiB |
| After step 1 | 7.53 GiB | 4.44 GiB | 4.26 GiB |
| End of generation | 7.73 GiB | 4.24 GiB | 4.11 GiB |

- The 6.6 GiB right after build is the DiT weights (2 models, FSDP, about 1.6 GiB), the text encoder (UMT5, estimated at about 2.7 GiB with TP=4), the VAE, and so on. What grows during the steps is buffers allocated up front for communication and the like (about 1.1 GiB).
- **Steps per model** (UniPC, flow_shift 12, 40 steps, boundary 875): the high-noise `transformer` handles **26 steps**, and the low-noise `transformer_2` handles 14 steps.
- **Idea of keeping the FFN weights resident:** For both models it needs about 4.6 GiB more and does not fit. For `transformer` only it needs about 2.3 GiB more and fits (about 1.9 GiB left), with an expected gain of about 2.3 s per video (about 2.4%). The converted-weight cache only tells "FSDP or not" apart per model, so the cache needs a new name and must be rebuilt (about 3.5 min, +29 GB). → Implemented in #5b.

### #5b details: Keeping the FFN weights resident (`transformer` side)

- **Change:** Added `ffn_fsdp` (default True) to `WanTransformerBlock` and `WanTransformer3DModel`. In `WanCheckpoint.build`, only the subfolders listed in the environment variable `WAN_FFN_RESIDENT` (`transformer`) get `ffn_fsdp=False`. For that model, `-ffnresident` is appended to the converted-weight cache name so that the old sharded cache is not read (building the new cache takes about 3.5 min).
- **Memory** (per chip): right after build 6.60 → 8.91 GiB, end of generation 7.73 → 10.04 GiB (+2.31 GiB, as expected). Free memory is 1.93 GiB (largest contiguous free 1.81 GiB).
- **Result:** At 480p, 95.41 s (the run that built the cache) and 95.24 s (the run that read the cache). −2.0 s (−2.1%) from 97.35 s for #3/#4. This almost matches the expected gain (about 2.3 s).
- **Video:** Almost identical to #3 (the computation is the same; only how the weights are held differs).
- **Not yet checked:** At 720p the activations are larger, so it is unclear whether it fits in memory. It is not the default; it is enabled through an environment variable.

![At 3 s (left: #3, right: #5b)](img/wan22_05b_cmp3s.jpg)

### #6 details: Dropping fp32 accumulation in the FFN (C)

- **Background:** The register that accumulates matmul results (DEST) holds 4 tiles with fp32 accumulation and 8 tiles with bf16. The current config table uses 4-tile subblocks ((2,2) or (4,1)), so dropping fp32 accumulation alone does not help (−0.25 ms, −0.9% in the block measurement).
- **Switch:** `WAN_MM_FP32_DEST=ffn|attn|all` turns off fp32 accumulation for the FFN (`ff_compute_kernel_config`) and for the attention projections (`mm_compute_kernel_config`). SDPA and RoPE are not affected.

| Condition (block, config table unchanged) | 1 block | PCC |
|---|---|---|
| Baseline | 29.36 ms | 99.9938% |
| No fp32 accumulation in the FFN only | 29.11 ms | 99.9910% |
| Attention projections only | 29.32 ms | 99.9908% |
| Both | 29.22 ms | 99.9895% |

- **Redoing the search:** Added `MM_SWEEP_FP32_DEST=0` (bf16 accumulation, subblocks up to 8 tiles) to the sweep tool and searched the 2 FFN shapes again.

| Shape | With fp32 accumulation (best) | **Without fp32 accumulation (best)** | Saving |
|---|---|---|---|
| FFN1 (AGMM 8×8, with GELU) | (8,10,8,(2,2)) 2,053.3 µs | **(8,4,14,(4,2)) 1,916.8 µs** | −6.6% |
| FFN2 (fused MM+RS, 8×7 grid) | (6,4,10,(2,2)) 2,757.3 µs | **(8,3,10,(4,2)) 2,307.7 µs** | −16.3% |

- **Integration:** A 4×2 subblock cannot be used with fp32 accumulation, so the config table is overridden at model build time with `register_matmul_configs` / `register_fused_mmrs_configs` only when `WAN_MM_FP32_DEST` is `ffn` or `all`.
- **Result (480p, combined with #5b):** 92.35 s (−2.9 s, −3.0% from 95.24 s for #5b). It is larger than the op-level estimate (about 1.9 s), probably because the FFN weights are resident: there is no weight-gather wait, so the matmul saving shows up in full. 0 warnings about the fused MM+RS default config.
- **Picture quality:** Same composition and subject, no artifacts. Small differences in the details of the stems and leaves.

![At 3 s (left: #5b, right: #6)](img/wan22_06_cmp3s.jpg)

![#6 video at 0.5 s / 3 s / 4.9 s](img/wan22_06_seq.jpg)

### #6b details: No fp32 accumulation for the attention projections either (optional, default off)

- **Search** (no fp32 accumulation, subblocks up to 8 tiles):

| Shape | With fp32 accumulation (best) | Without fp32 accumulation (best) | Saving |
|---|---|---|---|
| QKV (AGMM 8×8) | (8,8,10,(2,2)) 2,228.0 µs | (8,4,10,(4,2)) 2,103.7 µs | −124 µs |
| Sum of the output projection (with addcmul) + cross Q and output projections (plain ×2) | (8,8,5,(4,1)) 3,157.0 µs | (8,8,6,(4,2)) 3,073.9 µs | −83 µs |

- The two output projections have the same (M, K, N), so they share a config table entry. The config with the smallest total over 1 self + 2 cross calls was chosen.
- When `WAN_MM_FP32_DEST=attn|all`, `attention_wan.py` overrides the table with `register_matmul_configs`.
- **Result:** 91.70 s at 480p (−0.65 s, −0.7% from 92.35 s for #6).
- **Picture quality:** No artifacts, but the picture changes visibly compared with #5b and #6 (paler orange on the wings, a slightly smaller butterfly, a different layout of stems and leaves). Lowering the QKV precision changes K/V (the softmax inputs), and the difference likely builds up over 40 steps.
- **Decision:** The extra speed is small for how much the picture changes, and it has not been checked with quality metrics (CLIP / VBench) either, so it **stays as an optional switch, default off** (2026-10-01).

![At 3 s (left: #5b, middle: #6, right: #6b)](img/wan22_06b_cmp3s.jpg)

### #21 details: LoFi for the ring SDPA matmuls only (2026-10-05)

- **Background:** On October 1, making the whole ring SDPA LoFi caused a hang (#16). Later, PR #58223 landed in main. It added (1) an option to set the fidelity of only the two matmuls inside SDPA (`SDPAProgramConfig.matmul_math_fidelity`), and (2) re-recording of the matmul instruction replay when LoFi is used. The PR description says that a test of one combination "hangs with the old kernel".
- **Change:** `WAN_SDPA_MATMUL_FIDELITY=LoFi` passes `matmul_math_fidelity=LoFi` to the ring SDPA program config (max, exp and the correction stay HiFi2). Gemma4 (Blackhole) uses this combination in practice. There was no record of it being tried on Wormhole, so the op timeout was set to 120 s and testing started with the B=1 block test.
- **Result:** No hang. ring SDPA 13.00 → 12.04 ms (−7.4%), block 26.63 → 25.57 ms (−4.0%), PCC 99.9919 → 99.9772%. The saving is mainly in Phase 2 (P·V, mostly matmul). Phase 1 is bound by the pack thread and does not shrink (so the gain is about half of the estimate). Over 40 steps, 86.97 → **82.76 s** (−4.8%).
- **Picture quality:** No artifacts and good quality, but the content changes a lot (the butterfly wings go from orange to white/cream, the butterfly appears larger, and the composition differs). Lower QKᵀ precision affects the softmax, so the 40-step trajectory likely diverged a lot. Left: #20, right: #21 (at 1 s, 3 s, 5 s).
- **Decision (2026-10-05):** There is no visible quality loss, so it is a **candidate, optional, default on**, and work moves on to other optimizations. To disable it, set `WAN_SDPA_MATMUL_FIDELITY=none` (values such as `HiFi2` also work). Disable it when comparing with older videos. A quality evaluation on dozens of prompts has not been done yet.

**Hang summary (#16 and #21)**

| | October 1 (#16, hang) | October 5 (#21, success) |
|---|---|---|
| Machine | another Galaxy | the test Galaxy |
| Base | `a1146e45eef` (before PR #58223) | main `d1489f53` (after PR #58223) |
| Setting | Whole SDPA compute config set to LoFi (`WAN_SDPA_FIDELITY=LoFi`) | Only the matmuls (QKᵀ and P·V) LoFi, compute config HiFi2 |
| Result | Stuck on the very first ring SDPA call and stopped at the op timeout (600 s). Then an ETH heartbeat timeout followed | Normal from start to end. The chips were idle after the run |

- **Likely cause of the hang:** The old kernel reused the matmul instruction replay for P·V. At any HiFi level the recording is the same, so that is fine, but a LoFi recording differs per matrix shape. With LoFi, the small normalization matmul records a LoFi replay between two P·V setups, so P·V replays a recording for a different shape. The tile hand-off between the unpack, math and pack threads then no longer matches, and they deadlock. PR #58223 re-records the replay at every P·V setup under LoFi (`SDPA_COMPUTE_LOFI`), and the PR description also says that a test of one combination "hangs with the old kernel". ring SDPA does compute and the inter-chip K/V hand-off in one op, so when compute stops, the Ethernet cores freeze too, which likely made recovery hard.
- **What was checked, and what was not:** Checked: "with the latest kernel, LoFi for the matmuls only does not hang". The October 1 setting, "whole compute config LoFi", has not been tried again (matmul-only LoFi already gave enough gain, so there is little need). With the current default (CFG batching + LoFi SDPA matmuls), the CFG-batched block takes 49.59 ms (−4.1% from 51.72 ms for #20).
- **Lesson:** When trying lower-precision settings, start from a combination with a track record (here, "LoFi for the matmuls only", as used by Gemma4), and use a short op timeout.

![#20 (left) and #21 (right)](img/wan22_21_sdpa_mm_lofi_cmp.jpg)

### #23 details: LoFi matmuls (rejected, 2026-10-05)

- **Goal:** Make the matmuls, the largest part after SDPA in section 9.5 (19.1 ms per block), LoFi (one multiply pass). bf16 and bf8 both have a 7-bit mantissa, so the precision loss should be about the same as with bf8 + LoFi. As a first step, only the math fidelity was changed, with the weights kept in bf16.
- **Result:** 1 block 49.59 → 49.10 ms (−1.0%). FFN2 (fused MM+RS) shrank from 4.17 → 3.87 ms, but QKV (4.42 → 4.39) and FFN1 (4.62 → 4.55) barely changed. Precision dropped a lot: relative RMSE 1.8% → 5.1% (above the test threshold of 3.2%), and CCC (an agreement measure that also reacts to scale and mean shifts) fell to 99.86%, a much larger drop than PCC. LoFi does not use the low mantissa bits, so the values likely drift in one direction.
- **Takeaway:** The current matmuls are not bound by the compute of the matrix unit but by everything else (the compute/communication overlap inside AGMM, data reads, unpack, pack, etc.). To make the matmuls faster, change the data flow and the communication overlap, not the compute precision.
- **Not yet checked:** Whether the AGMM op really uses the given math fidelity for all of its matmuls. Comparing HiFi2 and LoFi with the sweep script, which measures a single matmul, would confirm it.

### #7 details: Building the softmax row sums inside DEST (failed)

- **Goal:** In the softmax, each tile of the probabilities P is written out for ③ P·V, then written out again one tile at a time for ④ the row sums, and accumulated in L1. The H3 analysis said that the "pack thread write-out" is heavy, so ④ was changed to "right after exp, add the tiles of the same row inside DEST with the SFPU, and write out only once per row".
- **Change:** Replaced ④ in `sub_exp_block_bcast_cols` (+49 lines). Added a function that runs the SFPU add from the pack thread, written the same way as `exp_packthread_tile`, and an instruction that makes the SFPU wait until the ③ write-out finishes (`TTI_STALLWAIT(STALL_SFPU, PACK)`). Kernels are compiled at runtime, so no C++ rebuild was needed.
- **Result:** It compiled and did not hang, but ring SDPA got worse, 13.06 → 16.79 ms (+28%), and precision dropped (PCC 99.9938 → 99.9779%, RMSE/σ 1.3 → 2.2%). **It was reverted**, and a measurement after the revert confirmed 13.062 ms again.
- **Why it got slower (estimate):**
  1. The packer's "accumulate into L1" is relatively cheap, because it happens inside the hardware write-out path. A two-operand SFPU add reads and writes DEST for every tile, so it costs about as much as exp. The number of adds (columns − 1 per row) is about the same as the number of exps, so the pack thread's SFPU work nearly doubled.
  2. Making the SFPU wait until the ③ write-out finished removed the overlap between the write-out and the SFPU.
  3. The precision drop happened because the SDPA DEST is bf16 (no fp32 accumulation), so the adds inside DEST are done in bf16.
- **Lesson:** The kernel was changed without checking how much of the "14 µs write-out" in the H3 breakdown was ④. **Before changing a kernel, measure the time per section (zone) inside the kernel on the target shape.** H3 measured this with a build that enabled the kernel's `profiling_enabled`.

### #10 details: Caching the cross-attention K/V

- **Idea:** The cross-attention K/V come only from the text (512 tokens), and the text stays the same for all 40 steps, so K/V have the same values at every step. Yet every step and every layer redid the K/V projection matmul (0.40 ms), its FSDP weight gather (0.40 ms), and the k normalization (0.03 ms).
- **(Idea C, considered and dropped)** Moving part of exp to the math thread was judged unsafe to implement: there is only one SFPU (and one set of SFPU registers) per core, and using it at the same time as the pack thread corrupts intermediate values.
- **Change:** Added a K/V cache per prompt tensor to `WanAttention`. The key is the prompt tensor itself (`id`). The conditional and unconditional prompts are different tensors, so they are kept apart naturally. It stores K after normalization and V after the head split. `WanTransformer3DModel.prepare_text_conditioning` (called whenever the pipeline prepares the prompts again) clears the cache of all blocks, so K/V from another prompt are never used. Enabled with the environment variable `WAN_CROSS_KV_CACHE=1`.
- **Block:** 27.59 → 26.71 ms (−0.88 ms, −3.2%, as expected). The op count went from 47 to 43. PCC 99.9919%, unchanged.
- **Whole pipeline (480p):** 92.35 → **89.79 s** (−2.56 s, −2.8%).
- **Memory:** The cache for the 2 models adds about 0.4 GiB. Free memory is 1.54 GiB (largest contiguous free 1.42 GiB).
- **Video:** The difference from #6 is max 44, mean 0.145 (on 0 to 255). Two #5b runs with exactly the same setup also differ by max 41, mean 0.113, so **the difference from the cache cannot be told apart from run-to-run noise**. Multi-chip WH runs are not bit-reproducible in the first place (probably because of the summation order in inter-chip communication, etc.).

### #11 details: trace (no effect)

- **Goal:** If there are sections where the host (Python) issuing ops one by one is slower than device execution, trace (record the op sequence once, then replay it on the device from the second time on) would remove that time.
- **Result:** 92.71 s (92.35 s without trace, same setup). **No saving.** Each of the roughly 40 ops in a block takes hundreds of µs to several ms, so the host always runs ahead of the device (device-bound). The cross-attention K/V cache (#10) branches on "compute only the first time", which does not work with trace, so it is turned off automatically when trace is on.

### #13 details: Time per section of one forward

- **Method:** Recording the whole pipeline with Tracy overflows the device-side buffer (about 125 zones per thread per launch), and data is lost. So `ttnn.synchronize_device` was inserted around each section and the host clock was used (`WAN_STEP_TIMING=1`, 6 steps). Because of the syncs, the total is a bit longer than in a normal run.
- **Breakdown of one forward (one transformer call):** timestep embedding 4.0 ms, patch embedding 0.35 ms, **40 blocks 1,085 to 1,160 ms**, output side (norm_out, proj_out, all_gather) 4.6 ms. CFG combination and the scheduler take about 4 ms per step.
- **Takeaway:** Everything outside the blocks is about 22 ms per step (2 forwards), about 0.9 s over 40 steps (about 1% of the total). **Only the inside of the block is worth optimizing.**

### Speeding up experiments: comparing with fewer steps (2026-10-01)

- About 72% of one experiment (40 steps) is generation itself. Setup (model loading, etc.) is about 32 s.
- Experiments that only compare speed run with **40 → 6 steps** (`WAN_PERF_STEPS=6`) and compare **the time per step between runs with the same step count**.
- Caution: 6 steps only run the high-noise `transformer` (`transformer_2` handles the last 14 of the 40 steps). Changes whose effect depends on the expert (for example, the #5b FFN residency applies to `transformer` only), adoption decisions, and quality checks still use 40 steps.
- The resolution is not lowered. Below 480p, SDPA (quadratic in sequence length) shrinks to about 1/16, and the FSDP weight gather and fixed costs dominate, so the bottleneck would be very different from 480p.

### #15 details: CFG batching (merging the conditional and unconditional forwards)

- **Idea:** CFG runs the transformer twice per step (with the prompt and with the negative prompt). The two runs differ only in the text; the latents and the timestep are the same. So the rows of both runs are stacked on each chip (4,096 rows → 8,192 rows), and the block runs only once. The expectation was that the FSDP weight gather and the matmul launches happen only once, and that the larger matrices are more efficient.
- **Implementation:** `WanTransformer3DModel.inner_step_cfg`. Row-wise work (norm, projections, FFN) stays at 8,192 rows. Only attention needs the two runs separated, so around the head split the tensor goes back and forth between `[1,1,2n,F]` and `[2,1,n,F]` (or `[1,2,n,F]`). The fused RMSNorm requires shape[0]=1 and `nlp_create_qkv_heads` requires shape[1]=1, so the layout is chosen per place. The cross-attention K/V are built per prompt, concatenated along the batch dimension, and cached per pair.
- **Block test:** Fixed `14b-480p-b2` so that it runs on the CFG path. The reference (torch) computes the two batches with shared RoPE and temb. PCC is 99.9939%, so it is correct.
- **15a result:** 77.11 ms per block, much slower than two separate runs (26.71 × 2 = 53.42 ms). ring SDPA takes 51.90 ms (**4×** the 13.06 ms at batch 1, although the work is only 2×). Everything except SDPA takes 25.21 ms, 2.1 ms less than 27.30 ms for two runs, so merging does help there.
- **Why 4×:** Lines 1046 to 1049 of `ring_joint_sdpa_program_factory.cpp` say: "the K/V store-and-forward chain can only be built per head, so it is disabled when B>1, and each core reads K/V on its own". In the chain, one core reads K/V from DRAM and passes it on to the next core in turn. Without it, many cores read the same K/V from DRAM again and again, and DRAM bandwidth becomes the bottleneck.
- **15b:** Only ring SDPA is sliced and called twice at batch 1 (`WAN_CFG_SDPA_SPLIT`, default on). 1 block **51.96 ms** (−2.7%), and ring SDPA is back to 13.06 ms × 2. In a 6-step e2e run, the time per step went 2,199 → **2,131 ms (−3.1%)**.
- **Where it shrank (1 block, compared with two runs):** FFN2 fused MM+RS 4.72 → 4.15 ms, FFN1 4.69 → 4.62 ms, cross-attention projections slightly. QKV (4.49 → 4.50) and to_out (2.19 → 2.19) are unchanged.
- **40 steps (480p):** 89.79 → **87.50 s** (−2.29 s, −2.6%), denoise 86.06 s. −20.7% from the 110.39 s baseline.
- **Video:** Same quality as #10, no artifacts. But the composition shifts slightly (leaf positions, butterfly size). The difference is max 255, mean 12.9, larger than the repeat noise for the same setup (mean 0.11). The matrix size changes the blocking (summation order), and that difference likely builds up over 40 steps. Left: #10, right: #15b (at 1 s, 3 s, 5 s).

![#10 (left) and #15b (right)](img/wan22_15b_cmp.jpg)

> **Caution (the composition changes):** With CFG batching on, the composition changes slightly compared with videos up to #10, even with the same prompt and the same random seed (the quality is the same).
> This is because the result is not bit-identical to the path that computes two forwards. The cause is that the 8,192-row matmuls use a different blocking (summation order).
> Set `WAN_CFG_BATCH=0` when comparing with older videos or checking reproducibility. 720p is not checked. The rows double, so activation memory also roughly doubles.
> 2026-10-01: set as "candidate, optional, default on".

- **15d (registering blocking for the M=8192 attention projections):** block 51.96 → 51.82 ms (−0.3%). QKV 4.50 → 4.44, to_out 2.19 → 2.07, cross projections 1.93 → 1.89 ms.
- **15d at 40 steps (default settings, no `WAN_CFG_BATCH`):** **86.90 s** (denoise 85.59 s, VAE 1.21 s). −2.89 s (−3.2%) from 89.79 s for #10, and **−21.3%** from the 110.39 s baseline. The video is equivalent to 15b. Generic-blocking warnings remain only for small matrices: the text encoder, the timestep embedding, the VAE, and proj_out (8192×5120×64).
- **Sweep of the M=8192 attention projections (with fp32 accumulation):** best QKV 4.398 ms (4.50 ms in the model), to_out 2.305 ms (2.19 ms in the model, where it is fused with addcmul), cross projections 1.874 ms (1.93 ms in the model). Registering all of them would save only about 0.2 ms (0.4%) per block, so this is low priority.

### #16 (in progress, 2026-10-01 18:00 onward): SDPA precision settings, and folding the CFG-batched self-attention into heads

- **Plan:** Run experiments at 480p. 720p comes later.
- **Re-estimate of idea B′ (computing the row sums with an FPU matmul):** Making the row sums with P × col_identity removes the pack thread's ④ (1.45 µs/row). But the math thread's work grows by about 25% of QKᵀ, and unpack also reads P again, so pack (9.6 µs) and math (about 9.5 µs) only end up about equal. The expected saving is small. The kernel change was postponed, and ideas that need only settings were measured first.
- **16a SDPA precision settings (B=1 block):** The plan was to measure, in order, math fidelity LoFi (`WAN_SDPA_FIDELITY`), bf8 inputs (`WAN_SDPA_DTYPE=bf8`), approximate exp (`WAN_SDPA_EXP_APPROX=1`), and LoFi + bf8.
  - Control (no change): 26.73 ms, ring SDPA 13.06 ms, PCC 99.9919% (reproduces #10).
  - **LoFi: hang** (started at 18:04; at 18:14:52, `TIMEOUT: device timeout, potential hang detected, the device is unrecoverable`). It stopped right after the ring SDPA kernel was compiled. The rest of the chain (bf8, approximate exp) and the queued 16b test were cancelled. Afterwards the chips stayed at aiclk 1000 MHz and about 37 W and did not go back to idle (500 MHz, about 18 W) (the hung kernel probably stayed on the cores).
  - At 18:20, a test that only reopens the device was tried once. It failed with `Timed out waiting for ETH heartbeat on device … ETH core e1-0 … Stuck at 0xabcdd641`. This is the same "needs a reset" symptom as in section 6.
  - **Likely cause (2026-10-05, checked in the code):** The ring SDPA in base `a1146e45eef`, where the hang happened, had no LoFi-specific handling. PR #58223, which landed in main the same day (October 1), re-records the P·V matmul instruction replay every time under LoFi (`SDPA_COMPUTE_LOFI`). According to its comment, the recording is the same at every HiFi level and can be reused, but a LoFi recording differs per matrix shape. In the old code, under LoFi, a recording for another shape (such as the small normalization matmul) was most likely replayed for P·V, the tile hand-off between threads stopped matching, and the threads deadlocked. Because compute stopped, the inter-chip K/V hand-off in the same op (the Ethernet cores) likely froze too, which led to the ETH heartbeat timeout. The ring SDPA LoFi tests exist only for Blackhole (`tests/nightly/blackhole/`). Not verified.
  - **Lesson:** On WH, ring SDPA hangs with LoFi (at least with chunk (256, 512) and the streaming kernel). Experiments that change math fidelity should have been tried first on one chip with a small shape.
- **16b Folding the CFG-batched self-attention into heads (`WAN_CFG_HEADFOLD`, default on):** In #15, extra ops appeared around SDPA: 3 transposes (0.54 ms), 6 slices (0.6 ms) and a concat (0.2 ms) (about 1.35 ms per block). Pass cond and uncond to ring SDPA not as "batch 2" but as "20 heads (10 × 2), B=1". Attention is independent per head, so the result is the same. With B=1, the K/V chain stays enabled. The norm output `[1,H,2n,E]` → `[1,2H,n,E]` needs only a reshape. → The result is added below.
  - **Result (2026-10-05, #22, the test Galaxy):** The glue ops disappeared as intended, saving about 1.1 ms. But ring SDPA processing 20 heads in one call took 26.06 ms, 2 ms slower than two calls of 10 heads (12.04 ms × 2 = 24.08 ms with LoFi matmuls), and the whole block got worse, 49.59 → 50.48 ms. The reason is unknown (possibly twice as much K/V circulates between chips in one call, which changes the compute/communication overlap, or the work assignment to cores changes). Rejected and removed from the code.

### Check at 720p (D, 2026-10-01)

Checked whether the optimizations made at 480p also help at 720p (1280×720, 81 frames, 40 steps). 720p has 9,472 rows per chip (75,600 tokens ÷ SP8, including padding), so the matrix shapes differ from 480p (4,096 rows).

| Condition (720p) | SDPA chunk | FFN2 | denoise | Total | Notes |
|---|---|---|---|---|---|
| (b) Effective baseline | (256, 256) | Fused (default config) | 345.72 s | 348.39 s | Roughly matches the 354 s in Wan2_2.md (as of March) |
| (a) Current code | (256, 512) | Fused (default config) | 320.99 s | 323.57 s | **−24.8 s (−7.1%)** from the SDPA change |
| **(c) Current code + FFN2 unfused** | (256, 512) | Unfused | **308.77 s** | **311.33 s** | A further **−12.2 s (−3.8%)**. **−10.6%** vs. the baseline |

- Much faster than the CI target (452 s).
- **The SDPA chunk tuning helps more at 720p** (480p −5.2%, 720p −7.1%), because the share of SDPA grows with resolution.
- **At 720p, FFN2 falls back to the default config again.** The fused MM+RS config registered in #2 is for M=4096, and there is no config for 720p (M=9,472) (3,360 warnings; 0 in (c)). The right fix is the same check as H3's PR #56886: "fuse only when the config can be resolved, otherwise run unfused". → Done in #4.
- At 720p, the 3 AGMM shapes and the FFN2 matmul hit existing config table entries (no generic-config warnings).
- Neither video has artifacts (figure below, at 3 s; left: (a), right: (c)).

![720p, at 3 s (left: (a) current code, right: (c) FFN2 unfused)](img/wan22_720p_cmp3s.jpg)

### Time per zone inside the SDPA kernel (2026-10-01)

- **Method:** In `sdpa_ring_v2` in `compute_streaming.hpp`, temporarily set the `profiling_enabled` passed to `sdpa_inner_loop_step` to `true` (1 line), and ran the block test once. From the device-side log (`profile_log_device.csv`, 734 MB), the zones of one chip and one core (slot 15, core (1,1)) were extracted. Only about 125 zones per thread per launch can be recorded, so this covers the first 1.5 steps. The kernel was reverted after the measurement. PCC 99.9919%, unchanged.
- **Structure of one step (Q 256 rows × K 512 columns):** Phase 1 (QKᵀ and softmax) is 4 Q row blocks × about 11.5 µs = about 46 µs (about 68%). Phase 2 (P·V) is 4 × about 5 to 6 µs = about 21 µs (about 32%). About 67 µs per step, almost the same as the 66 µs of H3 15 s.
- **Breakdown of one Phase 1 row block (about 11.6 µs):**

| Thread | Content | Busy time | Wait |
|---|---|---|---|
| unpack | QKᵀ inputs 4.5, subtraction inputs 2.0, max 0.8 | 7.6 µs (66%) | 4.0 µs |
| math | QKᵀ matmul 5.2, subtraction 2.5, max 0.2, etc. | 8.2 µs (71%) | 3.4 µs |
| **pack** | P write-out (③ + ④) 3.6, exp 2.9, QKᵀ result write-out 2.7, max 0.4 | **9.6 µs (83%)** | 2.0 µs |

- **Takeaways:** In Phase 1 the pack thread is the bottleneck (exp and the P write-out take 56% of a row). Meanwhile the FPU runs only 35% of the time (net QKᵀ 4.1 µs ÷ 11.6 µs). Phase 2 is almost all matmul, at about 80% of theoretical within the zone. The net FPU time for the whole step is 1,024 tiles × 32 cycles ≈ 32.8 µs, or 49%.
- **Why #7 failed:** #7 added SFPU adds to the pack thread, which was already the bottleneck.
- **Next steps:** Measure ④ alone, and consider computing the sums together with the P·V matmul (moving work to the FPU) or moving part of exp to the math thread.

**Measuring ③ and ④ separately** (temporarily added `P_PACK` and `ROWSUM_PACK` zones, reverted after the measurement). Breakdown of one pack-thread row block (about 12.3 µs; about 0.7 µs longer because of the added zones):

| Content | Time |
|---|---|
| exp (SFPU) | 2.88 µs |
| QKᵀ result write-out | 2.48 µs |
| ③ P write-out (for P·V) | 1.90 µs |
| **④ second write-out for the row sums** | **1.45 µs** |
| Other work around the P write-out (config switches, etc.) | 0.85 µs |
| Max, etc. | 0.4 µs |

- If ④ were removed completely, the upper bound is −1.45 µs per row, −5.8 µs per step (8.7%), −1.1 ms for SDPA, and about −2.6 s per 480p video. Once pack gets lighter, math (about 8.5 µs) becomes the next bottleneck, so a realistic gain is 1 to 2 s per video.
- Ideas: B′ (multiply P by `col_identity` to get the row sums on the FPU in Phase 1; about −1 µs/row), C (move part of exp to the math thread; about −0.7 µs/row).
- The main reason SDPA is slow is that the FPU runs only 49% of the whole step. The rest is pack work, subtraction, and waits between threads, and fixing one makes the next one the bottleneck. A large saving probably needs a structural kernel change, such as overlapping Phase 1 and Phase 2.

### Experiment and measurement methods (summary, 2026-10-02)

**Environment:** Another Galaxy (WH Galaxy 4×8, 32 chips). Before opening the device, every run checks that the chips are idle ("median power of the 32 chips < 40 W, aiclk ≤ 600 MHz") and aborts otherwise.

**1. Total time (the number used for final decisions)**
- The CI performance test itself: `test_performance_wan.py::test_pipeline_performance[wormhole_b0-t2v-resolution_480p-wh_4x8_sp1tp0]` (T2V-A14B, 832×480, 81 frames, 40 steps, with CFG).
- The `Total Pipeline` output (text encoder + denoise + VAE) was recorded as the "total", and `Denoising` as the denoise time.
- Repeat runs of the same setup differ by about 0.01 to 0.2 s, so differences below 0.5 s were treated as noise.
- About 2 to 2.5 min per run (about 32 s of setup + generation). The weights are read from a prebuilt cache (`TT_DIT_CACHE_DIR`).

**2. Device-side time of one block (to see what got faster)**
- Ran the block test `test_transformer_wan.py::test_wan_transformer_block[...14b-480p...ring_is_fsdp1]` with the Tracy device profiler (`python -m tracy -r -p -v -m pytest ...`).
- "Warm" measurement: `WAN_BLOCK_TEST_ITERS=3` runs the block 3 times in a row and measures "start of the first op → end of the last op" in the last run. This gives a state close to production, where the host runs ahead of the device.
- From the output `ops_perf_results_*.csv`, the time of each op was averaged over the 32 chips.
- The same test also checks PCC and relative RMSE against the torch (CPU, fp32) reference implementation (correctness check).

**3. Time per step and per section (fine-grained breakdown)**
- `WAN_STEP_TIMING=1`: syncs the device and measures with the host clock per step and per section of one forward (timestep embedding, patch embedding, blocks, output side).
- Recording the whole pipeline with Tracy overflows the device-side buffer (about 125 zones per thread per launch), so this method was used instead.

**4. Matmul blocking search**
- `models/tt_dit/utils/sweep_mm_block_sizes.py` (use cases: qkv / to_out / plain / ff1_gelu / ff2 / mmrs) measures every combination for the real shapes on hardware, and the best config was registered in the table.
- On WH Galaxy, `MM_SWEEP_PROFILER_DUMP_EVERY=100000` is required (with the default value the board stalls). For a search without fp32 accumulation, use `MM_SWEEP_FP32_DEST=0`.

**5. Zones inside the SDPA kernel**
- Temporarily enabled `profiling_enabled` in `compute_streaming.hpp` and extracted the zones of one core (the unpack / math / pack threads) from the device log. The kernel was reverted after the measurement.

**6. Picture quality checks**
- Compared the 40-step video with the previous version by pixel difference (max and mean), and looked at the frames at 1, 3 and 5 s side by side.
- The difference between repeats of the same setup (max 41, mean 0.11) was used as the noise level, because multi-chip WH runs are not bit-reproducible.
- Block PCC was only used to check that "the computation is not broken". Whether the video changes was checked on the video itself (see section 11).

**7. Running experiments faster**
- When comparing speed only, runs used `WAN_PERF_STEPS=6` (40 → 6 steps) and compared the time per step between runs with the same step count. Adoption decisions and quality checks used 40 steps.
- All switches are environment variables (`WAN_*`), so A/B comparisons used the same build.
- Each result was recorded in the experiment ledger (section 10).

### Zones inside the SDPA kernel (after LoFi, 2026-10-05, #26)

- **Method:** Same as on October 1. In `sdpa_ring_v2` in `compute_streaming.hpp`, temporarily set the `profiling_enabled` passed to `sdpa_inner_loop_step` to `true`, and ran the B=1 block test once. The zones of chip 15, core (1,1) were extracted for roughly the first 1.5 ring SDPA calls and compared. The kernel was reverted after the measurement.

| Part | Thread | HiFi2 (10/1) | LoFi matmuls (10/5) | Change |
|---|---|---|---|---|
| **Phase 1 (QKᵀ and softmax)** per call | math | 13.01 µs | 12.87 µs | About the same |
| | **pack** | **13.25 µs** | **13.14 µs** | About the same (**bottleneck**) |
| &nbsp;&nbsp;of which QKᵀ matmul | math | 2.21 µs/call | 1.94 µs/call | −12% |
| **Phase 2 (P·V)** per call | math | 10.53 µs | **8.41 µs** | **−20%** |
| &nbsp;&nbsp;of which P·V matmul | math | 5.15 µs/call | 3.78 µs/call | −27% |
| Total (about 1.5 calls) | | 121.5 µs | 112.2 µs | −7.7% (matches the −7.4% of the block measurement) |

- **Takeaways:** LoFi helped only Phase 2 (P·V). In Phase 1 the pack thread is busy almost 100% of the time, so faster matmuls do not change the time. Phase 1 is now about 60% of the total.
- **Breakdown of the Phase 1 pack-thread work (78.9 µs total):** writing the QKᵀ result from DEST to L1, 41.2 µs (52%, the largest); P write-out, 17.7 µs (22%); exp, 14.6 µs (18%).
- **The write-out format cannot be made lighter:** The QKᵀ result buffer (`cb_qk_im`) is already bf16 when fp32 accumulation is off (`ring_joint_sdpa_program_factory.cpp:1908`). bf8 would cut the data size, but it lowers the precision of the scores before exp, and the conversion may add pack work.
- **Conclusion:** The Phase 1 bottleneck comes from the kernel design (write the QKᵀ result to L1, take the max, then read it back for exp). Going further needs a structural kernel change that reduces write-outs (for example, keeping the QKᵀ result in DEST through the subtraction and exp).

### Confirming the Phase 1 bottleneck, and the outlook for a kernel rewrite (2026-10-05, #27, #28)

- **Correction to the zone measurement:** The zones used so far (`MaybeDeviceZoneScopedN`) include the time spent inside the zone waiting for the other threads. For example, the pack thread's "QKᵀ matmul + write-out" zone includes `tile_regs_wait()`, which waits for the math thread to hand over DEST. The three threads move at the same pace while they hand data to each other, so the zones of every thread are about the same length, and zone length alone does not show where the bottleneck is. The "pack is the bottleneck" claims of #26 and October 1 were based on numbers that included waits.
- **Test bed:** A single-chip SDPA (not ring) that uses the same in-kernel routine as ring SDPA (`sdpa_inner_loop_step`), measured on the same shape as one chip of ring (Q 4,096 × K 32,768, 10 heads, chunk (256, 512), LoFi matmuls). The device time is 12.995 ms, close to ring SDPA (12.04 ms), and PCC is 99.965%. It uses no inter-chip communication, so kernel-change experiments can be tried safely.
- **#27 Net work:** Temporarily added zones (`MM_WORK`, `MM_PACK_WORK`) to the matmul functions that cover only the part after DEST is acquired or received. The net work in Phase 1 (one Q row group, zone length about 14.5 µs) is **pack 12.2 µs (83%)** (QKᵀ write-out 6.94 + P write-out 2.77 + exp 2.15 + max 0.34), math 10.8 µs (75%) (QKᵀ matmul 8.93, including waits for inputs to arrive), and unpack 6.3 µs (59%). The net numbers confirmed that **the Phase 1 bottleneck is the pack thread, and more than half of it is the QKᵀ result write-out**. The QKᵀ write-out costs about 217 cycles per tile (2 KB) (about 9 bytes/cycle).
- **#28 blocked pack:** On Wormhole, writing out several tiles at once (blocked pack) is used only for 8 or more tiles per call (`MIN_BLOCKED_PACK_TILES = 8`; it is 4 on Blackhole). Our shape writes 4 tiles per call, so it writes out one tile at a time. Lowering the threshold to 4 barely changed SDPA: 12.995 → 12.897 ms (−0.75%). The slow write-out likely comes from the write-out itself (the DEST-to-L1 write speed), not from call overhead.
- **Outlook:** Going further needs fewer tiles written out. That means a design that does not write the QKᵀ result to L1 and finishes the row max, subtraction and exp inside DEST (applying the online softmax correction per matmul subblock). This is close to rebuilding the current kernel structure (fine-grained hand-offs between the three threads, semaphores, sharing with other ops). It would take days to weeks and has a high hang risk. This kernel is under active development upstream (PR #58223, #58032).

### Considering fundamental speedups (2026-10-05, after #29)

- **#29 bf8 intermediate buffer:** The first try (no format switch) looked faster, 12.95 → 12.35 ms, but PCC was nan: the computation was broken and the work likely ran empty. Adding a format switch before writing out the max and the sums still gave a nan PCC, and the time got worse, 13.13 ms. Writing out one tile at a given position and accumulating into L1 may not support block formats. Even a correct fix would likely not be faster, so this was dropped.
- **Estimate for a kernel rewrite (processing QKᵀ inside DEST):** Even if the QKᵀ write-out (about 57% of the Phase 1 pack work) were removed completely, the math thread (about 75% busy in Phase 1) would become the bottleneck. The upper bound is about −24% for Phase 1, about −14% for SDPA, and about −5 s per video (about 6%). DEST holds only 8 tiles at a time, so the online softmax correction would be needed per matmul subblock (4 tiles wide), and P·V would also be split per subblock. Output corrections would happen 4 times as often as now and eat much of the upper bound, so it is not even clear whether the net result would be faster. It is not worth weeks of work and the hang risk.
- **Reducing the computation itself (candidates):**

| Idea | What it is | Expected gain | Quality impact | Scope |
|---|---|---|---|---|
| Skip steps with a cache (TeaCache, FasterCache, PAB, etc.) | When the transformer output changes little between neighboring steps, skip the computation and reuse the previous result (residual) | 1.5 to 2× | Smallish; tunable with a threshold | Pipeline side only (host Python) |
| Distillation to fewer steps (few-step distillation LoRA) | Add a LoRA trained to give good quality with few steps, going from 40 → 4 to 8 steps | 5 to 10× | Medium. Need to obtain the LoRA and check its license | tt-metal has LoRA support |
| Kernel rewrite | As above | Up to about 6% (uncertain) | None | Weeks, hang risk |

### #30 details: Faster GELU_TANH in FFN1 (2026-10-05)

- **Observation:** FFN1 (AGMM + GELU) takes 4.62 ms inside the model, but the same shape and blocking measured alone with the sweep script takes 3.62 ms. The difference is the GELU type: the sweep uses `gelu_fast` (a 6-piece piecewise-linear approximation), and the model uses `GELU_TANH` (`gelu(approximate="tanh")`, which Wan was trained with). `gelu_fast` broke the video in earlier work on a P150, so it cannot be used.
- **Why it is slow:** The AGMM compute kernel (`ccl/all_gather_minimal_matmul_async/device/kernels/compute.cpp`) finishes the K-direction matmul for each output block (8×14 tiles), and then, in a separate loop (`add_bias_block`), goes tile by tile: "read back → add bias → GELU (SFPU) → write out". This post-processing runs on the math thread, and the next block's matmul stops meanwhile. On top of that, GELU_TANH computed tanh with heavy fp32-accurate code (`_sfpu_tanh_fp32_accurate_`, with exp and division corrections) even when DEST is bf16, and only rounded to bf16 at the end.
- **Change:** When DEST is bf16, compute tanh with the bf16 polynomial (`_sfpu_tanh_polynomial_`, the one the tanh op uses for bf16). The init (`gelu_tanh_init`) also loads the polynomial constants to match the DEST precision. With an fp32 DEST, nothing changes.
- **Result:** FFN1 4.62 → 4.12 ms, block 49.59 → 49.13 ms (−0.9%), PCC unchanged. At 40 steps, 82.76 → **81.84 s** (−25.9% from the baseline). Left: #21, right: #30 (at 1 s, 3 s, 5 s).
- **Remaining:** The 0.5 ms gap to the standalone best (3.62 ms) is the GELU (polynomial) computation, which runs in the post-processing loop without overlapping the matmul. Hiding it needs a kernel change that overlaps post-processing with the next block's matmul (for example, computing GELU on the pack thread; `gelu_tanh_tile_pack` already exists).

![#21 (left) and #30 (right)](img/wan22_30_gelu_poly_cmp.jpg)

### SDPA rewrite design: Freezing the max at the first chunk (2026-10-05, started)

- **Current design (FlashAttention flow):** For each K chunk, write QKᵀ (S) to L1 and recompute the row max. When the max changes, multiply the output and the sums of the previous chunks by `exp(previous max − current max)` to correct them before adding (called SALAD in the code: `sub_exp_first_col_blocks`, `salad_correct_fused`).
- **Proposed change:**
  1. Compute the first K chunk as now and determine the row max m₀.
  2. For the second and later chunks, fix m₀ as the reference. Fold the subtraction into the matmul: accumulate the product of a 1×N constant tile of "1" and a column tile of "−m₀" into the same DEST as QKᵀ (the inner dimension goes from 4 → 5 tiles, so the QKᵀ matmul grows by 25%). Compute exp inside DEST right after the matmul, and write out only P. Writing out S, the subtraction, and the max computation are no longer needed.
  3. The max does not change, so the correction factor is always 1, and the correction step (SALAD) goes away entirely. The output and the sums are simply accumulated into L1.
  4. Softmax gives the same result when the reference value is shifted, so the computation stays exact. If a later chunk has a score about 80 or more above m₀, exp overflows the bf16 range, so detect that case and fall back to the current design. Masked chunks (in Wan, only the last chunk, which has padding) use the current design.
- **Estimate:** Phase 1 loses the S write-out (6.9 µs), the subtraction (1.9 µs) and the max (about 1 µs), and the matmul grows by about +2.2 µs. Phase 2 loses the correction step (20 to 30% of the Phase 2 math). −20 to 30% for SDPA, about −5 to 8 s per video (7 to 10%).
- **Plan:** Change Phase 1 and Phase 2 of `sdpa_inner_loop_step` and the output hand-off (the switch between chunks). The new flow is a compile-time switch (e.g. `SDPA_FROZEN_MAX`), used only under Wan's conditions (no causal mask, unmasked chunks). Develop it step by step on the single-chip SDPA, checking correctness (PCC) and speed.

### #32 details: Prototype that removes the row max (2026-10-05)

- **Checking the score magnitude:** Wan's attention applies RMSNorm (qk-norm) to Q and K before the dot product, so the score magnitude has an upper bound set by the norm weights. Computed from the weights of 40 layers × 2 experts, the maximum is 28.8 in the typical case (spread evenly across heads) and 8,272 in the worst case (concentrated in one head). In the typical case, exp(scale·S) fits easily in the bf16 range (exponent up to 88).
- **Prototype:** Added a path to `sdpa_inner_loop_step` that computes exp(scale·S) without subtracting the row max (185 lines, `SDPA_RAW_EXP`). Phase 1 only does QKᵀ → exp inside DEST (pack thread) → write out P and the row sums. Phase 2 only adds P·V by accumulating into L1, with no max or correction step.
- **Result:** It ran without stalling, but PCC was 79% (even with a single K chunk). The cause: the SDPA exp (`exp_packthread_tile<approx=true>`) is an approximate version that assumes inputs ≤ 0. The current design always feeds "S − max" (≤ 0), so the problem never shows up. The idea of freezing the max at the first chunk has the same problem, because the inputs turn positive when later chunks have larger scores.
- **Speed upper bound:** A timing-only variant without the copy of the previous result gives 12.995 → 11.96 ms (−8.0%). The Phase 1 bottleneck moves from pack to math (the QKᵀ matmul, which was 75% busy in #27), so it shrinks no further.
- **To make it correct:** Using a guaranteed per-row upper bound on the scores (norm of the Q row × max norm over all K rows) as the reference keeps the inputs ≤ 0. But this needs a preprocessing step for the max norm over all K rows (from all chips, for ring SDPA), and op changes to pass it to SDPA (host C++ and the kernel that reads K). There is also a risk that exp becomes 0 if the bound is 87 or more above the real max (up to about 58 in the typical-case analysis). Including the preprocessing, the expected gain is −5 to 6% for SDPA, about −2 to 3 s per video.
- **What we learned:** Making SDPA faster needs a faster QKᵀ matmul itself (the hand-off between math and unpack).

### #33 details: Re-analyzing why ring SDPA is slow (2026-10-05)

- **Measurement:** Fixed the single-chip SDPA op so that it reads the matmul precision setting (added the same wiring as in ring to `sdpa_program_factory.cpp`), and measured the same net zones as in #27 with LoFi. The logs of #26 (ring, LoFi) and #27 (single chip, HiFi2) were also re-aggregated.
- **Correction:** The #26 and #27 numbers ("pack is 83% in Phase 1", "QKᵀ write-out 6.94 µs", etc.) included the first iteration in the average (more than twice as long, because it waits for the inputs to arrive). Counting only the steady state from the second iteration on, no thread reaches 100% (table below).

**Breaking down the 19% utilization (one ring SDPA call, 12.04 ms measured, 2.33 ms theoretical)**

| Factor | Efficiency | What it is |
|---|---|---|
| Cores used | 0.875 | 63 of 72 cores (1 column is for communication) |
| Work imbalance across cores | 0.85 | 160 Q chunks (10 heads × 16) are dealt to 63 cores, so 34 cores get 3 and 29 cores get 2. Everyone waits for the cores with 3 |
| Efficiency within one step | 0.27 | Measured about 61 µs vs. a theoretical matmul time of 16.4 µs for one step (Q 256 × K 512) |
| Product | **0.20** | Matches the measured 19%. Communication is hidden (the single-chip version, with no communication, is slower at 13.1 ms) |

**Flow of one step (about 61 µs, steady state, ring, LoFi)**

| Part | Time | What it is |
|---|---|---|
| Phase 1 | 4 × about 11 µs = about 44 µs | Per Q row group (2 tile rows): ① QKᵀ (math) → write S to L1 (pack), about 3.9 µs; ② read S back for the row max, about 0.6 µs; ③ subtraction (math) → exp (pack SFPU) → write out P and the row sums (pack), about 6.3 µs |
| Phase 2 | 4 × about 4 µs = about 17 µs | P·V (net matmul at about 50% of theoretical) and the correction of the previous result (SALAD) |

**Net work per thread per S tile (32×32) (Phase 1, single chip, LoFi, cycles)**

| Thread | Content | Total |
|---|---|---|
| math | QKᵀ 98 (theoretical 64), subtraction 114, max 6 | about 220 |
| pack | S write-out 68, exp 90 (theoretical 68), P write-out 107, max 10 | about 275 |
| unpack | QKᵀ inputs 146, subtraction 43, max 18 (time to issue instructions) | about 207 |
| Actual time | | about 350 (per S tile; about 476 including Phase 2. The theoretical matmul time is 128) |

- **Reason 1 (structure):** Within a row group, ① → ② → ③ run in sequence because of the data dependency (the max is known only once the whole row of S is ready) and the program order. There are only 2 DEST halves of buffering between the three threads, so pack mostly waits during ①, and math mostly waits during ③. Even if the math work (net about 64%) and the pack work (about 75%) overlapped perfectly, the bound would be about 41 µs/step (−33%).
- **Reason 2 (L1 round trips):** S goes "pack writes it out → read back for the max → read back for the subtraction", and P goes "write out → read back for P·V", so each S tile passes through L1 5 times. The subtraction taking 114 cycles per tile is also a wait for the S read-back.
- **Reason 3 (work imbalance):** The 160 chunks do not divide evenly over 63 cores, so 15% sits idle. #25's (352, 320) has less imbalance but is slower because K is shorter. The op can split K (`max_k_splits`), but only when each core has 1 chunk.
- **The matmul itself is not the main cause:** The net QKᵀ matmul runs at about 65% of theoretical and P·V at about 50%. The FPU is active for only about 27% of a step.

### #34 details: Re-measuring the no-max upper bound with LoFi, and per-core analysis (2026-10-06)

- **Op time (single-chip SDPA, on device, 3 runs):** original kernel LoFi 12.71 to 12.88 ms, HiFi2 12.89 to 12.97 ms; the #32 timing-only variant (no max or correction) LoFi 11.79 to 11.87 ms, HiFi2 11.96 to 12.06 ms. Even with LoFi, the difference stays at about −7%.
- **Time of one step (Q 256 × K 512) (core (1,1), steady state):** original kernel **59.6 µs**, timing-only variant **42.8 µs (−28%)**. Neither waits for data. One Phase 1 row group goes from about 11.3 → 7.4 µs. The compute alone shrinks as the paper estimate predicted (−25 to −43%).
- **Why the whole op does not shrink (aggregate over all cores):** For each core, measured the "time per step", the "wait for K/V to arrive", and the "finish time" (zones added temporarily and reverted after the measurement).

| | Original kernel | Timing-only variant |
|---|---|---|
| Per-step time on cores with no wait | 60 µs | 43 µs |
| Per-step time on cores with waits (12 on the single-chip test bed) | 67 µs (wait about 8 µs) | 72 µs (wait about 30 µs) |
| Finish time of cores with 3 chunks and no wait | 11.6 ms | **8.4 ms** |
| Op finish (set by the cores with waits) | 12.7 ms | 11.8 ms |

- **Production ring SDPA (block test, chip 0, LoFi, 11.99 ms):** About 62 µs per step on all cores. The op finish is set by the cores with 3 chunks (12.0 ms), so compute is currently the bottleneck. Cores with 2 chunks finish at 8.1 ms and then sit idle. Five cores ((8,5) and (1 to 4,7)) wait an average of 25 µs per step for K/V and stretch to 11.4 ms. They become the next bottleneck once compute gets faster.
- **What the waiting cores are (estimate):** The 5 cores sit at the boundary where the Q chunk count switches from "cores with 3" to "cores with 2", and they share head 6 with cores that have 3 chunks. K/V are relayed between cores per head (chain), so when cores with different paces are on the same chain, the faster ones likely have to wait for the slower ones.
- **What we learned:**
  1. The upper bound of the rewrite that goes as far as exp inside Dst is about −28% per step (LoFi). The "#32 plateaus at −8%" result was hidden by the precision-setting issue and by the cores waiting for K/V.
  2. To get that gain for the whole op, the cores waiting in the K/V relay (and the per-core work imbalance) also have to be fixed. If the imbalance and the waits are fixed and compute is also cut by 28%, ring SDPA would go from 12.0 ms → about 7 ms.
  3. Fixing only the work imbalance (spreading the 160 chunks evenly over 63 cores) leaves room for 12.0 → about 10.1 ms (−16%).

### #35 details: "The matmul gets slower" was a measurement artifact (2026-10-06)

- **Observation:** In the #34 timing-only variant (exp inside Dst), the math thread's QKᵀ matmul zone was long, about 200 cycles per S tile (theoretical 64).
- **Control experiment (single-chip SDPA, LoFi, median zone time of one matmul subblock (8 S tiles), cycles):** Instead of exp, the same amount of other work was put on the pack thread.

| Work placed on the pack thread | pack zone | math matmul zone | unpack matmul zone |
|---|---|---|---|
| None | 27 | 993 | 961 |
| 272 × `TTI_NOP` | 375 | 1,392 | 1,378 |
| 544 × `TTI_NOP` | 644 | 1,653 | 1,618 |
| 544 × `TTI_SFPNOP` | 644 | 1,647 | 1,581 |
| exp (SFPU) | 655 | 1,598 | 1,416 |
| **544 RISC-V `nop`s (no instructions issued to the coprocessor)** | **6,635** | **7,584** | **7,606** |

- **Cause:** The math `tile_regs_acquire` (waiting for a free Dst half) is an instruction that waits inside the coprocessor, not on the RISC-V side (`TTI_SEMWAIT`, `cmath_common.h:306`), so the RISC-V does not stall. As a result, a zone measured with the RISC-V-side clock includes **the time spent waiting for pack to free a Dst half** as "matmul time". Making the pack-side work longer made the matmul "look slower" by the same amount, whatever the type of work (SFPU, coprocessor NOP, RISC-V nop).
- **Correction:** At one point the conclusion was that "the FPU and SFPU share issue slots" or that "the instruction paths of the three threads are shared", but both were mistakes based on this apparent effect. This measurement does not show whether the FPU and SFPU can run at the same time.
- **Impact on earlier measurements:** The "net math work" of #27, #33, etc. (zones such as `MM_WORK`, measured from after `tile_regs_acquire`) may also include Dst waits. The pack zones may too, if the instructions that wait for math are on the coprocessor side. The reliable numbers are op-level and step-level times (wall-clock time). The −28% per step in #34 is wall-clock time, so it holds.
- **What we learned:** In the timing-only variant, Phase 1 is bound by the pack thread (exp, writing out P and the row sums), and math waits for pack. To measure net work, the RISC-V has to wait for the coprocessor to finish before and after the zone (e.g. `tensix_sync`).

### #36 details: Folding the two CFG passes into heads, and dealing SDPA evenly over 64 cores (2026-10-06)

- **Goal:** Remove the ring SDPA work imbalance found in #34. Today 160 Q chunks (10 heads × 16) are dealt to 63 cores, so 34 cores take 3 and 29 cores take 2, and the whole op waits for the cores with 3. Among the cores with 2, those that share a head with cores that have 3 are dragged to the pace of the 3-chunk cores through the K/V relay (chain) and wait (this is what the "waiting cores" of #34 are; see "Explainer: the K/V relay (chain) and ..." below for how it works).
- **Change (both are experimental switches):**
  1. `WAN_SDPA_GRID=8,8`: changes the ring SDPA cores from 7×9 = 63 to 8×8 = 64, and puts the ring communication cores in the last row instead of the last column (`ccl_core_grid_offset=(0, 8)`, `use_column_major_ccl=False`).
  2. `WAN_CFG_HEADFOLD=1`: folds the conditional and unconditional CFG passes along the head dimension into one ring SDPA call with 20 heads (same as #22). Attention is independent per head, so the computation is the same. The transpose/slice/concat glue also goes away.
  - Together, the work is 20 heads × 16 = 320 units, exactly 5 per core on 64 cores. The largest per-core work goes from 2 calls × 768 rows = 1,536 rows down to 1,280 rows.
- **Result (compared with the standard settings `WAN_FFN_RESIDENT=transformer WAN_CROSS_KV_CACHE=1 WAN_MM_FP32_DEST=ffn`):**

| | #30 | #36 |
|---|---|---|
| ring SDPA (1 block) | 12.04 × 2 = 24.08 ms | **22.39 ms** (1 call) |
| 1 block (CFG batching, warm) | 49.13 ms (54 ops) | **47.46 ms (−3.4%, 46 ops)** |
| 40 steps | 81.82 s (re-measured) | **77.96 s (−3.86 s, −4.7%)** |
| PCC | 99.9834% | 99.9834% |
| Video (pixel difference from #30) | — | max 34, mean 0.03 (smaller than the repeat noise for the same setup) |

- **Neither change helps on its own:** With 64 cores only (still 2 calls, chunk (256, 512)), the per-core maximum stays at 3, and on the second half of the mesh (the 16 chips 16 to 31), ring SDPA got slower, 12.0 → 14.7 ms (probably a communication delay from moving the communication cores to the last row; the first 16 chips stay at 12.1 ms, unchanged). Folding into heads only (63 cores) still leaves some cores with 6 units, since 320 ÷ 63 does not divide evenly (the #22 result). Combining both makes compute the bottleneck, and the slowdown on the second-half chips no longer shows.
- **chunk 128 is inefficient:** As another way to remove the imbalance, a Q chunk of 128 (320 units) gives 15.14 ms on 64 cores and 17.84 ms on 63 cores (both for one call). The imbalance goes away (6 → 5 units, −15%), but chunk 128 itself is about 1.5× slower for the same number of rows.
- **Record of a measurement mistake:** The first measurement was taken without the standard environment variables, so the baseline came out at 89.8 s, and #36 was almost wrongly judged as "no effect" (the FFN had fallen back to FP32 accumulation and an unfused FFN2). From now on, every run loads the standard environment variables, and the `env:` line in the log is checked.
- **Form enabled by default (commit `414d32c85d8`):** Folding into heads is on by default (`WAN_CFG_HEADFOLD=0` gives the original two-call form). The 64 cores are used **only for the head-folded ring SDPA** (a dedicated SDPA config, `ring_sdpa_program_config_fold`). Without folding, the original 7×9 grid stays (two calls on 64 cores are slower). The core count for the folded case can be changed with `WAN_CFG_HEADFOLD_GRID="x,y"`. The experimental `WAN_SDPA_GRID` was removed. Check: in the block test, ring SDPA takes 22.36 ms by default and 12.04 ms × 2 with `WAN_CFG_HEADFOLD=0`. At 40 steps (no switches, standard settings only), **77.93 s** was reproduced.
- Video: left: #30, right: #36 (at 1 s, 3 s, 5 s).

![#30 (left) and #36 (right)](img/wan22_36_fold_grid88_cmp.jpg)

### Explainer: the K/V relay (chain) and how uneven work makes cores wait (2026-10-06)

This section explains, step by step, the "chain" and the "2-chunk cores are held back by 3-chunk cores" that appear in #34 and #36.

**1. The work of one core**

- The unit of work in ring SDPA is "256 rows of Q for one head (a Q chunk)". One chip has 10 heads × 16 = 160 chunks.
- To compute one chunk, a core must read **all of K and V** for that head (this chip's part and the parts that arrive from other chips over the ring), 512 rows at a time (a K chunk), in order.
- The 16 chunks of one head are spread over several cores. For example, chunks 0-15 of head 0 go to cores 0-5, 3 chunks each (core 5 spans heads 0 and 1).

**2. What a chain (relay) is**

- Cores that work on the same head need **the same K and V**. If every core read them from DRAM on its own, the same data would be read many times and DRAM bandwidth would be wasted.
- So the cores of each head are lined up, and **only the first core reads K and V from DRAM; when it is done with them, it passes them to the next core over the NoC (the network between cores)**. The receiving core uses them and passes them on to the next core. This daisy chain is called a chain (a store-and-forward chain, a relay line). It works like a bucket brigade.

```
DRAM ──K,V──▶ [core A (first)] ──send──▶ [core B] ──send──▶ [core C] ──▶ …
                    use                     use                use
```

- A core forwards only "what the next core needs". The first core re-reads K and V once per chunk it owns, and forwards only as many of those passes as the next core has chunks (`next_core_q_chunks` in the code).

**3. How this relates to the ring iterations**

- In ring SDPA, K and V arrive in 8 parts (the ring iterations; one iteration = one chip's 4,096 rows = 8 K chunks).
- In each ring iteration, each core takes its Q chunks one at a time and computes each against the 8 K/V chunks that arrived, repeating this for as many chunks as it owns.

**4. When 3-chunk cores and 2-chunk cores share a chain**

- Spreading 160 chunks over 63 cores gives 3 chunks to cores 0-33 and 2 chunks to cores 34-62. Head 6 (chunks 96-111) spans cores 32 and 33 (3 each) and cores 34-38 (2 each), and the first core is core 32 (3 chunks).

```
Head 6 chain:  core32 (3, first) → core33 (3) → core34 (2) → … → core38 (2)

Time flow within a ring iteration (┃ = compute for one chunk):
core32  ┃1┃┃2┃┃3┃┃1┃┃2┃┃3┃ …   ← does not start reading the next iteration's K/V until it finishes chunk 3
core34  ┃1┃┃2┃░░░┃1┃┃2┃░░░ …   ← after its 2 chunks, waits until the next K/V arrives (░ = wait)
```

- Cores 34-38 finish their 2 chunks, but the next iteration's K/V only flows down after the first core, core 32, finishes its third chunk, so **they wait for one chunk's worth of time every iteration**. As a result, they have only 2 chunks of work but can only progress as fast as the 3-chunk cores. Measured (ring, chip 0): these 5 cores waited 25 µs per step on average and finished at 11.4 ms (the other 2-chunk cores at 8.1 ms, the 3-chunk cores at 12.0 ms).
- In the chains of the other heads, every core has 3 chunks (or every core has 2), so this kind of wait does not happen. Chains of heads with only 2-chunk cores finish at 8.1 ms and sit idle for the remaining ~4 ms.

**5. So the real issue is the imbalance, not the wait**

- Even if the 5 waiting cores did not wait, the op as a whole would not finish until the 3-chunk cores finish (12.0 ms). The wait is a symptom of the uneven work (3 vs. 2).
- #36 made it 320 work units (folded into 20 heads) on 64 cores, so every core has exactly 5. Since every core has the same work, no core waits inside a chain and no core finishes early and sits idle.

### #37 details: the "no max" upper bound in the real ring SDPA (2026-10-06)

- **Method:** The timing-only variant from #32 (no max and no correction, goes all the way to exp inside Dst; the result is not correct) was made to run inside ring SDPA too (only the number of buffers that save and reload partial results across ring iterations had to match; reverted after measuring). Measured with the CFG-batching block test in the #36 configuration (20 heads on 64 cores, 5 per core). It did not hang (PCC was nan, as expected).

| ring SDPA (1 block, chip 0) | current (#36) | timing-only variant |
|---|---|---|
| time | 22.29 ms | **21.50 ms (−3.6%)** |
| one step (median) | 62.5 µs | 55.7 µs (42.8 µs on the single-chip test bench) |
| wait for K/V arrival (median per step) | 1.6 µs | **13.7 µs** |
| core finish times | 19.98-22.29 ms | 15.23-21.50 ms |

- **Findings:** In the real ring SDPA, making the compute faster means K/V no longer arrive in time. Even a rewritten softmax is expected to shrink ring SDPA by only about −4%, so as things stand a rewrite is not worth much.
- **Reason (estimated):** The current op re-reads the head's K/V from the start for each Q chunk a core owns (5 of them) (read from DRAM by the per-head chain and relayed over the NoC). That works out to about 200 MB of K/V moved per ring iteration on one chip.
- **Next candidate:** The new experimental ring SDPA op (`exp_ring_joint_scaled_dot_product_attention`) gives one row of cores to one head, streams K/V once per row, and keeps the intermediate state in L1 ("head-serial passes", an L1 state FIFO, up to 3 heads per row). This greatly reduces K/V re-reads. Wan currently uses it only with tensor parallel 4 and sequence parallel 32, not on WH 4×8.

### #38 details: isolating why K/V arrive late (2026-10-06)

- **Method:** Temporary instrumentation was added to the ring SDPA reader (the data-movement processor that puts K/V into the compute-side buffers) to sum up time per kind of wait, recorded once as numbers at the end (`DeviceTimestampedData`; reverted after measuring). On the data-movement processors the RISC-V itself does the waiting, so the coprocessor-side wait problem from #35 does not arise. CFG-batching block test in the #36 configuration (20 heads, 64 cores), chip 0.

| reader time (per core, one ring SDPA call, ms) | current kernel (22.25 ms) | timing-only variant (21.43 ms) |
|---|---|---|
| wait for the ring signal (all-gather between chips) | 0.02 (max 0.10) | 0.01 (max 0.57) |
| wait for free space in the compute-side buffer (K / V) | 2.63 / 0.84 | 0.27 / 0.05 |
| wait for the relay from the previous core in the chain (K / V, median) | 3.42 / 3.49 | **5.26 / 5.00** (max 8.7 / 9.0) |
| read from DRAM (K / V, max over chain-head cores) | 6.96 / 7.01 | **7.10 / 7.13** |
| relay to the next core (K / V, median) | 1.55 / 1.60 | 1.48 / 1.59 |

- **Findings:**
  1. **The ring (the all-gather between chips) is not the bottleneck.** K/V from other chips have mostly arrived before they are needed (the wait is 0.1 ms or less per core).
  2. **The bottleneck is the reader of the chain-head core (the core that reads from DRAM).** For each K chunk, the head core does "read K from DRAM → relay to the next core → read V from DRAM → relay" in sequence on one processor. In the timing-only variant this alone takes about 17 ms per core (about 14 ms of DRAM reads + about 3 ms of relay), most of the op's 21.4 ms. DRAM reads per core run at about 6 GB/s.
  3. The downstream cores receive K/V at the head core's pace, so their relay wait grows (about 10 ms per core in the timing-only variant). With the current kernel too, even slightly faster compute hits the same wall.
- **Directions for improvement (same computation):**
  - Split the chain head's work (read K and V on different cores or processors, let cores in the middle of the chain also read from DRAM, etc.).
  - Make DRAM reads more efficient (amount read per request, NoC usage).
  - Reduce re-reads in the first place (a design that streams K/V once per row, like the exp op).

### #39 details: overlapping the K and V reads on the chain-head core (2026-10-06, rejected)

- **Goal:** In #38, the chain-head reader did "read K from DRAM → relay → read V from DRAM → relay" in sequence, taking about 54 µs per K chunk (reads of about 22 µs × 2 + relays of about 5 µs × 2). As the DRAM bandwidth tech report (`tech_reports/Saturating_DRAM_bandwidth`) says, issuing the next request before waiting for completion should overlap the reads and make them faster.
- **Change (for measurement; reverted after measuring):** Added a "do not wait for completion" flag to `fetch_block`. On the head core, right after issuing the K read requests, it reserves the V buffer, issues the V requests too, and then waits for both to complete. The V stage skips its read.
- **Results (ring SDPA, 1 block, average over 32 chips):**

| | no overlap | overlap |
|---|---|---|
| current kernel | 22.31 ms | 24.20 ms (PCC 99.9834%, unchanged) |
| timing-only variant (fast compute) | 21.49 ms | 24.27 ms |

- **Findings:** Overlapping the reads made it about 2 ms slower. The bottleneck seems to be DRAM and NoC congestion across the whole chip, not the read latency of a single core (20 chain-head cores read at the same time, and the ring all-gather's writes use the same DRAM). Delaying the K relay until the V read completes may also contribute.
- **Current K/V read volume:** K/V are re-read for each Q chunk a core owns (5), so one ring SDPA call reads about 1.6 GB (about 75 GB/s). That is about 1/4 of WH's theoretical DRAM bandwidth (about 288 GB/s), but every core reads from all banks (interleaved), which makes NoC congestion likely (as the tech report points out).
- **Next step:** The read volume itself must go down. With the new experimental ring SDPA op (K/V streamed once per row and used by all cores in the row), the read volume is expected to drop to 1/3-1/4.

### #40 details: all cores read either K or V from DRAM (2026-10-06, rejected)

- **Goal:** Split the chain-head core's work from #38 (reading and relaying K and V) to lighten its load. A reader-only change: stop relaying one of K or V through the chain and have each core read it from DRAM itself (reverted after measuring). K and V use the same chain (the same signalling), so if all cores stop relaying the same one at once, the hand-off stays consistent.
- **Results (ring SDPA, 1 block, average over 32 chips):**

| | current kernel | timing-only variant |
|---|---|---|
| current design (both K and V relayed by the chain) | 22.31 ms | 21.49 ms |
| all cores read only V from DRAM | 34.29 ms | 30.60 ms |
| all cores read only K from DRAM | 34.94 ms | 30.52 ms |

- **Findings:** More cores reading from DRAM makes it much slower (+50%). This is the same trend as the B=2 ring SDPA without the chain, which was about 4x slower (#15a). In terms of K/V volume, having all cores read one of them raises reads from about 1.6 GB to about 3.4 GB and takes 34 ms (about 100 GB/s). Even the current design (about 1.6 GB) reaches about 76 GB/s in the timing-only variant. **The wall is how fast the whole chip can read from DRAM (about 100 GB/s with this access pattern, about 1/3 of theoretical)**, and the chain relay actually helps by reducing the read volume. This also agrees with #39 (overlapping reads makes it slower).
- **Conclusion:** Changing how the work of a single core or the chain head is split does not make it faster. To speed up ring SDPA further, we need to either (1) reduce the K/V read volume itself (stream once per row like the exp op, or use larger Q chunks to cut re-reads), or (2) read DRAM in a less congested way (place K/V per DRAM bank and read them from nearby cores, etc., the tech report's method).

### #41 details: trying the experimental ring SDPA op on WH 4×8 (2026-10-06, did not work)

- **Goal:** #38-#40 showed that ring SDPA is capped by the volume of K/V read from DRAM (re-read per Q chunk). The experimental op `exp_ring_joint_scaled_dot_product_attention` (the "exp" in the name stands for experimental and has nothing to do with the exponential function; the compute kernel is the same `sdpa_inner_loop_step`, and the Wan settings are also `exp_approx_mode=False`, HiFi2) gives one row of cores to one head, streams K/V once per row, and keeps the intermediate state in L1, so the read volume was expected to drop to 1/3-1/4.
- **Method:** Added an experimental switch to Wan's attention to use this op on WH 4×8 as well (`WAN_EXP_RING_SDPA=1`, plus knobs for the chunks, the core range, the number of links and the workers per link; reverted after the trial).
- **Constraints hit (in order):**
  1. Exactly 2 links required (Wan on WH 4×8 uses 4) → worked around by passing 2 links to this op only.
  2. The number of Q chunks per head must be a multiple of the number of columns used for SDPA (7) → with 4,096 Q rows per chip, the Q chunk has to be 608-672 rows (4-15% padding waste, and about 1.4-2x the current compute per core).
  3. The number of rows used for SDPA must be even (WH has 9) → worked around with an 8×8 core range.
  4. Workers per link must be half the number of rows → worked around by setting it to 4.
  5. **L1 capacity:** No combination fit (2.02 MB with heads folded, Q 608, K 256; even the smallest unfolded case, Q 608, K 128, needs 1.54 MB; 1.39 MB is available).
- **Conclusion:** As built, this op does not fit Wan's shapes on WH 4×8 (4,096 Q rows per chip, 10 or 20 heads). Using it would need changes to the op (relax the chunks-per-head constraint, use less L1). Even if it ran, the chunk constraint raises compute per core by 1.4x or more, so it is doubtful that it would be faster once that is netted against the read savings.

### #42-#45 details: Q interleave to reduce K/V DRAM reads (2026-10-06)

**#42 Upper-bound estimate (timing-only):** A variant where each core reads K/V from DRAM only for its first Q chunk of a head and reuses the stale data left in the buffer for the rest (the result is not correct). DRAM reads drop to about 1/5. Ring SDPA (1 block) goes 22.31 → 20.23 ms with the current kernel, and 21.49 → **15.57 ms** combined with the timing-only (no max) variant.

**#43 Q interleave (correct implementation):** Today each core owns 5 consecutive chunks (e.g. chunks 0-4 of the same head), and the head of each per-head chain re-reads K/V once per chunk it owns. This was changed as follows (changes to the host-side op and the kernels, experimental switch `TT_RING_SDPA_Q_INTERLEAVE=1`).
- Core c owns the chunks with flat indices c, c+64, c+128, c+192, c+256 (one per pass p = 0-4). The kernel's index mapping `decompose_global_q_index` (`q_chunk_remapping.hpp`) reorders them as f′ = (f mod 5) × 64 + f ÷ 5.
- In pass p, the 16-core group g (cores 16g to 16g+15) computes the 16 chunks of head 4p+g, one each. The chain is fixed to the group's 16 cores (the same upstream and downstream in every pass) and works for any head ("any head" added to `chain_link.hpp`). The chain head reads K/V only once per pass. DRAM reads per ring iteration go from 100 to 20.
- Correctness: block PCC 99.9834% (unchanged).
- Result: chips in the right half of the mesh (0-15) went 22.31 → 19.96 ms, but the left half (16-31) got slower, 22.42 → 23.15 ms.

**#44 Why the left-half chips are slow:** Single-chip SDPA shows no difference (13.01 / 12.94 ms). Reader instrumentation showed that on the left-half chips the chain head's DRAM reads are about 2.5x slower, but only in the early iterations (0-4), while the ring all-gather is still running (iterations 5-7 run at the same speed). The all-gather data flow and the NoC route of the chain head's reads likely overlap and congest. Shorter chains with more reading cores made it worse (chain 8: 30.7 ms, chain 4: 25.4 ms). Reversing the chain direction made the relay go against NoC0 (which travels right and down) and made it much worse, 40.6 ms on all chips.

**#45 Move the reader to NoC1 and point the chain left:** Swapped the reader (K/V reads and relay) to NoC1 (travels left and up) and the writer to NoC0 (`TT_RING_SDPA_READER_NOC1=1`), and reversed the chain order to match NoC1's direction (`TT_RING_SDPA_Q_INTERLEAVE_ORDER=rev`).

| ring SDPA (1 block) | chips 0-15 | chips 16-31 | max |
|---|---|---|---|
| #36 | 22.31 | 22.42 | 22.76 |
| Q interleave (NoC0, rightward) | 19.96 | 23.15 | 23.41 |
| **Q interleave + NoC1, leftward (chain 16)** | **20.66** | **20.68** | **20.84** |
| same, chain 8 | 22.56 | 22.36 | 22.80 |
| same, chain 4 | 31.85 | 32.00 | 32.21 |
| NoC1, rightward (direction mismatch) | 29.99 | 29.99 | 30.19 |
| NoC1, no interleave | 51.03 | 51.18 | 51.58 |
| timing-only variant (no max) + Q interleave + NoC1, leftward | 17.21 | 17.32 | 17.54 |

- **40 steps: 77.93 → 75.49 s (−2.4 s).** The video content is almost the same, with slightly shifted positions (mean pixel difference vs. #36: 4.15). This is likely small numerical differences, from the change in which core computes each chunk and in what order, growing over the steps. Left: #36, right: #45.
- **Findings:** Aligning the chain's relay direction with the NoC's direction is decisive. The fewer cores read from DRAM, the faster (chain 16 is best). In this configuration, if a rewritten softmax made the compute faster, ring SDPA could shrink by about another −16% (→ 17.5 ms).
- Change: 3 files in the host op and the kernels (switched by environment variables, off by default).

![#36 (left) and #45 (right)](img/wan22_45_qil_cmp.jpg)

### #46 details: bounded exp mode, subtracting a per-row upper bound (2026-10-06, rejected)

- **Goal:** In the #45 configuration, the timing-only variant without the max and the correction (SALAD) runs ring SDPA in 17.5 ms (20.8 ms today). Achieve this with a correct computation.
- **Approach:** Subtract a guaranteed per-row upper bound r_i = scale·‖q_i‖·max_j‖k_j‖ + a margin of 1.0 before computing exp (by the Cauchy-Schwarz inequality, the exp input is then always −1 or less).
  - The subtraction is done by adding a fifth product to the QKᵀ matmul: "bound column tile × constant tile (first row is 1, 128 for padded keys)". The extra matmul is almost free (checked with the timing-only variant, 17.2 → 17.3 ms).
  - ‖q_i‖ and the bound are computed in compute per Q chunk (per ring iteration) (q² → row sums via a matmul with a constant tile → multiply by max K² and take the square root → add the margin and flip the sign). Max K² (the maximum over all chips) is computed in ttnn on the model side and passed in through the attention sink input. The writer builds the constant tile. exp uses the variant that clamps its input to −88.5 or above (`ClampToNegative`), which makes padded keys 0.
  - Changes to the host op, kernels and model (about 400 lines; switches `TT_RING_SDPA_BOUNDED=1`, `WAN_SDPA_BOUNDED=1`).
- **Result:** It ran without hanging. At first PCC was 3% (the read and write positions of `cb_qkt_im`, the matmul scratch area, had drifted apart; fixed by reserving and releasing in units of 16 tiles per row so that it wraps exactly once per Q chunk). Even after the fix, **PCC was 95.8%**, which does not pass. Ring SDPA took 21.4 ms (slower than today's 20.8 ms, because of the bound computation and copying the previous result).
- **Why the precision falls short (estimate):** In the bounded approach, for the keys that attention concentrates on (elements with large P), the exp input becomes "−(bound − actual max)". This gap widens with the angle between Q and K and is around 100-300 in units before scaling. Adding S (hundreds) and the bound inside a BF16 Dst rounds the result to the BF16 step at that magnitude (1-2), which is 0.1-0.2 after scaling and a 10-20% error in P. With the current approach (subtracting the actual row max), S − max for the concentrated elements is close to 0 and the error is only about 0.02. Doing the subtraction in FP32 would require building a per-row subtraction into exp (SFPU), which roughly triples exp's instruction count and slows the pack thread, which is the bottleneck.
- **Conclusion:** With a BF16 Dst, approaches that do not subtract the max lack precision. Work on the compute side of ring SDPA stops here (#45's 20.8 ms is the current best).

### #47 details: the matmul (AGMM) is bound by data supply, not compute (2026-10-06)

- **Method:** Built a timing-only variant of the AGMM compute kernel that does not call the matmul (`matmul_blocks`) (inputs are received and discarded, the output is not correct; reverted after measuring). CFG-batching block test in the #45 configuration.

| matmul (AGMM) | normal | compute skipped | N (output columns) |
|---|---|---|---|
| QKV projection | 4.45 ms | 4.17 ms | 3,840 |
| FFN1 (incl. GELU) | 4.12 ms | 3.94 ms | 3,456 |
| self-attn output projection | 2.03 ms | 1.86 ms | 1,280 |
| cross Q projection, output projection | 1.89 ms × 2 | 1.74 ms × 2 | 1,280 |

- **Findings:** Skipping the compute saves only about 6%. AGMM is bound by data supply: the TP all-gather of the input, weight reads, output writes and so on. Time is nearly proportional to the number of output columns N (about 0.83 ms per 1,000 of N), and the all-gather of the input (in0) is the same size for every projection, so it does not seem to be the main cause. Next: measure the waits on the reader and writer side to find which data flow is the bottleneck.

### #48-#49 details: BFP8 weights and activations (2026-10-06)

- **Method:** Built experimental switches that change only the tensor format to BFP8, without changing the compute settings (fidelity, Dst precision) (`set_weights_only_cast` in `quant_config.py`; environment variables added to the performance test and the block test).
  - Weights (#48): the weights and biases of the layers listed in `WAN_WEIGHTS_BF8` are converted to BFP8 at load time. The self-attn and cross-attn output projections are excluded, because the AGMM with fused addition requires the added tensor and the weights to have the same tile size (`ternary_a_tile_size == in1_tile_size`).
  - Activations (#49): `WAN_SDPA_BF8=1` makes the ring SDPA inputs Q, K, V BFP8 (Q and K by having the normalization kernel output BFP8, V by a typecast). The K/V volume sent around the ring also drops to nearly half.
- **Results (#45 configuration, 480p, 40 steps, `WAN_STEP_TIMING=1`):**

| configuration | steady-state step (high-noise / low-noise) | denoise | total | video pixel difference (vs. #45) |
|---|---|---|---|---|
| #45 (BF16) | ~1,830 ms | 73.89 s | 75.49 s | — |
| #48 BFP8 weights | 1,769 / 1,795 ms | 72.54 s | 73.86 s | 18.9 (same composition) |
| BFP8 SDPA inputs only | 1,748 / 1,786 ms | 76.47 s* | 77.71 s* | 27.9 (composition changes) |
| #49 BFP8 weights + SDPA inputs | **1,686 / 1,713 ms** | **69.46 s** | **70.74 s** | 28.4 (composition changes) |

  \* The ring SDPA kernel was recompiled in the first measured step (7.6 s, a side effect of the experimental switch). Without that, the total is about 71.9 s.
- **Block (#48, Tracy):** 45.61 → 43.73 ms. QKV 4.45 → 4.01, FFN1 4.12 → 3.73, FFN2 4.15 → 3.75 ms, and the gathers also drop by about 0.5 ms. It gets faster by the amount of weight reads saved, consistent with #47's "AGMM is bound by data supply".
- **Quality:** Visual picture quality is kept in all cases (no blur, noise or color breakdown). With BFP8 weights the butterfly has the same position and orientation and only details differ. With BFP8 SDPA inputs, as in #21, the butterfly's position and orientation change (a normal video with a different composition).
- **Decision:** The precision loss is not bad, so keep these as optional, composition-changing optimizations (off by default). Do not push BFP8 further (e.g. AGMM input activations) for now; go back to analyzing AGMM data supply (continuing #47).
- Comparison image: `img/wan22_49_bf8_cmp.png` (left #45, right #49; frames 16, 48, 80).

### #50-#52 details: fully resident weights and splitting the text encoder (2026-10-06)

- **Background:** Within one block, the FSDP all-gather (weights split over the 8 SP devices are gathered just before use) takes about 2.7 ms for the attention weights and about 1.2 ms for the FFN weights (Tracy, CFG-batching block test). The high-noise FFN is already resident (#5b), so what remains is the attention weights and the low-noise side.
- **#50:** Built a switch, `WAN_FULL_RESIDENT=<subfolder>`, that builds and loads that transformer without FSDP (with its cache in a separate location).
  - Fully resident in BF16 ran out of DRAM during loading (12 banks × 1.07 GB per chip, nearly full).
  - So `WAN_WEIGHTS_BF8_LOAD=1` was added to convert the #48 layers' weights to BFP8 at the point of conversion from torch (a BF16 copy is never placed on the device; separate cache too). The high-noise weights went from about 7.0 GB to about 4.2 GB per chip and fit.
  - High-noise steady-state step: 1,769 → 1,732 ms. Total 73.86 → 71.61 s.
  - The composition changed slightly from #48 (butterfly position and orientation), because the rounding of the weights to BFP8 moved from a device typecast to a host-side conversion (the host rounds from FP32, so it is actually closer to the original values).
- **DRAM usage (`WAN_MEM_LOG=1`, #50):** 9.35 GiB after init, 11.18 GiB during steps (0.79 GiB free). The text encoder (UMT5-XXL, 5.7B) uses about 2.65 GiB per chip at TP=4.
- **#51:** Splitting the text encoder TP=8 over the 8-device axis (the SP axis) frees about 1.3 GiB (`WAN_ENC_ON_SP_AXIS=1`; changes to the pipeline and performance test settings).
  - At first the video had nothing to do with the prompt (a hand gripping a red ball). The encoder splits the prompt batch over "the axis not used for TP", but that axis was being derived from the DiT's TP axis. Fixed by deriving it from the encoder's own TP axis.
  - The freed space makes the low-noise FFN resident too (BFP8). Low-noise steady-state step 1,797 → 1,769 ms, total 71.34 s. Text encoding time is unchanged at 0.08 s.
  - Making the low-noise side fully resident needs about 2.4 GB more and does not fit (estimate).
- **#52:** Adding BFP8 ring SDPA inputs (the #49 switch) to #51 gives 68.05 s (steady-state step 1,647 / 1,685 ms). The composition is in the same family as #49, and picture quality is kept.
- Comparison image: `img/wan22_52_all_cmp.png` (left #45, right #52).

### #53-#58 details: analyzing what bounds AGMM (fused matmul and all-gather) (2026-10-06)

**AGMM recap:** On 8×8 cores, the output's M (rows) is split along x and N (columns) along y. For each output subblock it owns (M block × N block), each core accumulates over K, one K block at a time.
- in0 (activations) is read by the first core of each column (the injector) from DRAM (including what arrived from other chips by the all-gather) and passed down the chain to the 8 cores in that column.
- in1 (weights) is read from DRAM by the first core of each row and passed along the chain to the 8 cores in that row.
- Each chain hop goes "wait until the next core is ready to receive → write → send the arrival signal", and a core forwards only after it has received everything.

**#53 Measurement (wait-time accumulation added to the DM kernels; CFG-batching block test, device 0):**

| matmul | measured | compute only (timing-only variant where DM hands data over instantly) | DM only (timing-only variant that skips the matmul) |
|---|---|---|---|
| QKV (BF16 weights) | 4.47 ms | 3.80 ms | 4.17 ms (#47) |
| QKV (BFP8 weights) | 4.02 ms | 3.73 ms | 3.22 ms |
| FFN1 (BFP8 weights, incl. GELU) | 3.74 ms | 3.51 ms | 2.90 ms |
| output projection, cross Q (BF16) | 1.84-1.95 ms | 1.42-1.57 ms | — |

- With BF16 weights, DM and compute weigh about the same and wait on each other. With BFP8 weights, DM gets lighter and **compute becomes the bottleneck**.
- DM breakdown (QKV, BF16): in1 receiving cores spend 2.6 ms of 4.4 ms waiting for data from upstream. in0 receiving cores: 1.7 ms waiting for data, 1.1 ms waiting for CB space, 0.8 ms forwarding to the next core (about 6 µs per hop per block).
- Total DRAM traffic (in0 re-read once per N block, in1 once per M block) corresponds to an effective 70-90 GB/s.

**#54 Blockings that cut re-reads:** QKV at (8, 5, 15) and (8, 8, 15), and FFN1 at (16, 4, 14), reduce DRAM re-reads, but both got slower (QKV 4.45 ms, FFN1 3.91 ms). Small K blocks mean more FP32 partial-result accumulation into L1, and N block 15 forces a 4×1 subblock, which slows compute. (8, 10, 15) with BF16 weights does not fit in L1 (1,513 KB > 1,499 KB).

**#55 Triple-buffered CBs:** Making the in0 and in1 CBs 3 deep with `TT_AGMM_CB_DEPTH` (a host op switch) sped FFN1 up from 3.74 to 3.58 ms but slowed the output projection from 1.95 to 2.05 ms, so the block as a whole barely changed.

**Single-chip compute efficiency (`minimal_matmul`, same work per core as QKV):**

| setting | time | per tile product |
|---|---|---|
| HiFi2, FP32 Dst, 2×2, BF16 weights | 4.60 ms | 59.9 ns |
| same, BFP8 weights | 4.22 ms | 55.0 ns |
| HiFi4, FP32, 2×2, BFP8 | 5.92 ms | 77.1 ns |
| LoFi, FP32, 2×2, BFP8 | 3.40 ms | 44.3 ns |
| HiFi2, **BF16 Dst, 4×2**, BFP8 | 3.53 ms | 45.9 ns |
| LoFi, BF16 Dst, 4×2, BFP8 | 2.82 ms | 36.7 ns |

- One fidelity level (one multiply pass) accounts for only about 11 ns. The remaining 25-33 ns are fixed costs outside the math (reading tiles, hand-offs per pass, etc.), which shrink with a 4×2 subblock in a BF16 Dst, where Dst holds 8 tiles.
- Against the theoretical compute rate (32 cycles per tile at HiFi2), the current efficiency is about 65%.

**#56-#58 (e2e, 40 steps):**
- Only switching the projections to LoFi (#56) improves the steady-state step just 1,732 → 1,711 ms (−1.2%). Lighter compute makes DM (especially in0 of the output projection) the bottleneck.
- Moving the attention projections to a BF16 Dst (#6b) and QKV to a 4×2 subblock (#57) gives 70.05 s.
- Stacking LoFi and BFP8 ring SDPA inputs on #57 (#58) gives **66.88 s** (steady-state step 1,594 / 1,637 ms). The composition changes, but picture quality is kept (`img/wan22_58_fast_cmp.png`, left #52, right #58).
- Note: After a measurement patch, the profiler exited abnormally; the device clocks did not come down and the Ethernet heartbeat stopped. `tt-smi -glx_reset` recovered it.

### #59-#61 details: BFP8 activations, and fixing bugs in BFP8 SDPA (2026-10-06)

- **#59:** The #53 analysis showed that AGMM spends much of its time handing over in0 (activations) and reading tiles (unpack), so in0 was made BFP8 (`WAN_AGMM_IN_BF8=1`).
  - First the effect was checked with a typecast right before AGMM; then the LayerNorm outputs (norm1, norm2, norm3, which all feed only AGMM) were made BFP8 directly, removing the typecast. Only the output projection's input (the SDPA output) still needs a typecast.
  - If not specified, the AGMM output dtype follows the input. So the output was pinned to BF16 (when the QKV output became BFP8, the RMSNorm after it stalled).
  - With in0 in BFP8, not only does in0's DRAM and chain traffic halve, the compute side's tile reads also get lighter (QKV reached 3.25 ms, faster than the 3.73 ms "compute-only lower bound").

| matmul (BFP8 weights) | in0 BF16 | in0 BFP8 |
|---|---|---|
| QKV | 4.02 ms | 3.25 ms |
| output projection (+ residual) | 1.95 ms | 1.47 ms |
| cross Q | 1.84 ms | 1.23 ms |
| cross output | 1.88 ms | 1.28 ms |
| FFN1 | 3.74 ms | 3.55 ms |

  - e2e (on top of #51): 71.34 → 67.64 s (steady-state step 1,732 → 1,633 ms). Block PCC 99.9824%. The composition changes, but picture quality is kept.
- **#60:** Making the FFN1 output (the input of FFN2's matmul + reduce-scatter) BFP8 as well takes FFN2 from 3.71 to 3.36 ms and FFN1 from 3.55 to 3.41 ms (less to write out). e2e 66.89 s.
- **Bug found 1 (BFP8 SDPA):** The ring SDPA output dtype follows Q. So in #49, #52 and #58 the SDPA output was BFP8, and so was the output of the output-projection + residual-add op, meaning **even the residual stream was BFP8**. Fixed so the output dtype matches the residual. After the fix, the #52 equivalent runs 68.43 s (68.05 s before), with a small video difference (pixel difference 4.1).
- **Bug found 2 (program cache):** With BFP8 SDPA inputs, the empty joint input was typecast into a new tensor every time, so the ring SDPA program was rebuilt on every call (the first measured step was about 0.7 s longer). Fixed by reusing the typecast tensor. One rebuild still remains in the first step (about 0.35 s, cause not identified).
- **#61 (everything combined):** 62.28 s (steady-state step 1,498 / 1,538 ms, −43.6%). Comparison image `img/wan22_61_fastest_cmp.png` (left #58, right #61).

### #66-#68 details: fixing how the experimental switches are applied (2026-10-06)

- **How it was found:** In the #61 configuration, only the first measured step was about 0.35-0.95 s longer. Investigated with temporary hooks that log ops that missed the cache (`TT_LOG_CACHE_MISS`, at the host op entry) and that log the components of the ring SDPA hash.
  - The hash of the ring SDPA input tensors differed between warm-up and measurement. Printing the tensor specs showed **Q, K, V were BF16 in warm-up and BFP8 in measurement**.
  - The performance test's warm-up (2 steps) runs while the pipeline is being built, and the test's own "Running warmup iteration" does nothing without tracing (0.00 s). Meanwhile, the experimental switches (LoFi and BFP8 SDPA inputs via `set_weights_only_cast`) were applied after the pipeline was built.
  - As a result, (1) the first measured step built new programs, and (2) LoFi took effect on only some of the matmuls (#56's "LoFi only gives −1.2%" was wrong).
- **Fix:** LoFi and BFP8 SDPA inputs are now applied when the transformer is built (the same place as the BFP8 weight loading). In the #61 configuration: 62.28 → 59.33 s (steady-state step 1,498 → 1,434 ms, first step 1,461 ms). The BFP8-SDPA-inputs-only configuration (#52 equivalent) has the same steady-state step and video, and is faster only by the first step's overhead (67.81 s).
- **Caution (hang):** Once LoFi took full effect, the configuration with LoFi on attention projections that still use an FP32 Dst (#56 equivalent, `WAN_MM_FP32_DEST=ffn`) hung twice in a row in the second warm-up step (low-noise side) (recovered with `tt-smi -glx_reset`). Use LoFi only together with a BF16 Dst (#57).
- **#67:** To make the output projection weights BFP8 too, the fused-add constraint of AGMM (residual and weights must have the same tile size and format) can now be lifted with an environment variable. 58.88 s.
- **#68:** With LoFi taking full effect, the #62 blockings (QKV (16,10,10), FFN1 (16,8,14), output projection (16,8,5)) also help in e2e. **58.14 s** (steady-state step 1,409 / 1,439 ms, −47.3% from the baseline). Comparison image `img/wan22_68_fastest_cmp.png` (left #67, right #68).

### #83-#85 details: the GELU epilogue of FFN1 (2026-10-07)

- **Isolation:** Removing GELU in a timing-only run takes FFN1 (AGMM + GELU) from 2.36 to 1.81 ms. GELU accounted for about 0.55 ms per block (about 0.9 s over 40 steps). That is about 1.3 µs per tile, close to the SFPU compute time itself.
- **What did not help (#83):** Fusing the epilogue loop into the last K block overlaps GELU with the matmul of only 1 of 20 K blocks, so there was no effect.
- **What helped (#84):** Evaluated one row at a time, the tanh polynomial inside GELU_TANH waits for the previous SFPMAD result every time. WH's tanh op already had a variant that alternates between 2 rows to fill that wait, so GELU_TANH uses it too. The order of operations is the same, so results do not change (PCC and video identical). Combined with #31's "GELU on the pack thread", FFN1 went from 2.36 to 2.01 ms.
- **What remains:** The gap to the 1.81 ms without GELU is about 0.2 ms. The gap between the matmul itself (1.81 ms) and theoretical (1.11 ms) is larger. Reducing the unpacker's load with full sync did not make it faster (#85), so the unpacker is not the bottleneck.

### #86-#89 details: confirming what bounds FFN2, ring SDPA and the matmuls (2026-10-07)

- **FFN2 (#86, #91):** Changing the number of RS workers, the number of links, the buffers or the group size does not make it faster. The reader looked busy all the time, but in #91, stopping the reader's reads, the writer's sends or the compute kernel's math, one at a time, did not change the time. The bottleneck is the structure itself, where the matmul and RS are tightly synchronized through windows and inter-chip signals; partial changes do not shrink it.
- **Ring SDPA (#87):** Moving exp to the math thread hangs. That exhausts the easy ways to lighten the pack thread's load in Phase 1 (QKᵀ and softmax). What remains is a rewrite that processes the QKᵀ result inside DEST (earlier estimate: at most −5 s, several weeks).
- **Matmuls (#89):** Comparing LoFi and HiFi2 shows MVMUL runs at the theoretical rate, and the fixed part outside it (per-call setup and unpack) is about 30% for QKV. Changing the subblock shape (4×2, 8×1, and with full sync 8×2, 2×7) does not help. For QKV and the output projection the data-transfer lower bound is above the compute lower bound; only FFN1 is compute-bound.
- **Outside the blocks (measured together with #89):** The non-block part of a step is about 15 ms (preprocessing 5, postprocessing 7.5, solver 2). About 0.6 s over 40 steps.

### #90 details: assessing a ring SDPA rewrite in advance (2026-10-08)

- **Goal:** Earlier measurements (2026-10-05) showed Phase 1 bound by the pack thread, with more than half of that spent writing out the QKᵀ result. Processing QKᵀ inside DEST was expected to shrink it a lot.
- **Measured in the current configuration:** Removing the write-out entirely saves only −0.61 ms (3%). Adding one more QKᵀ matmul, on the other hand, costs +1.91 ms. After LoFi, BFP8, chunk (128, 1024), Q interleave and multicast, the bottleneck had moved from the pack write-out to the matmul (math thread) and exp.
- **What SDPA still spends time on:** The matmuls (QKᵀ and P·V, 5.2 ms theoretical) run at the theoretical rate. exp is already fast at about 80 cycles per tile. Thinning out the max reduce (about 1.5 ms) requires adding a margin to keep the approximate exp's assumption (input ≤ 0), which lowers picture quality (#82). Switching to the exact exp costs +3 ms. With the current kernel design, this is close to the practical lower bound.

### #92 details: improving the VAE decode (2026-10-08)

- **Breakdown:** The decode's ~0.93 s is almost all device compute (decoder body 926 ms). Measuring the ops of one chunk (3 frames in time) with Tracy (excluding weight-preparation ops): 3D convolution 41.1 ms (44%), neighbor pad (halo exchange) with adjacent chips 14.4 ms, normalization 9.4 ms, all-gather 7.8 ms, layout conversion 7 ms, attention core 4.1 ms.
- **Convolution blocking:** The table key is (H and W split counts, input and output channels, kernel, T, H, W). The shapes for WH 4×8, 480p, 3 frames at a time (#75) were not in the table, and the log showed "fallback" and "NONE" warnings. Added the WH 4×8 / 480p / 3-frame layer set to the existing brute-force sweep script (`bruteforce_conv3d_sweep.py`) and swept 12 layers on one chip (up to 500 candidates per layer, about 25 minutes in total). The 2 temporal layers had a hard-coded 1×1×1 blocking and were 41x and 16x slower.
- **Attention split:** The mid-block attention (spatial attention within each frame, in latent space) splits frames (3 per chunk) across chips. Splitting over all 32 chips pads with 29 zero frames and gathers the result back along both H and W. Splitting over only the 8 chips of the W axis needs a gather along W only (the compute is the same, 1 frame per chip).
- **What remains:** Device time per chunk is about 76 ms, but about 109 ms in the pipeline (765 ms ÷ 7). The ~30 ms difference looks like gaps in op dispatch from the host (about 220 ops per chunk). The neighbor pad (14 ms) copies the whole activation into a new padded buffer every time.

![#126 (left) and #92 (right)](img/wan22_92_vae_cmp.jpg)

### #150 details: relaying ring SDPA K/V between chips (2026-10-10, parked)

**Goal:** Ring SDPA reserves cores dedicated to the K/V all-gather (the 8 cores of the last row), so it runs on 64 cores, each owning 10 of the 640 Q chunks
(20 heads × 32). Letting the writers of the SDPA cores do the all-gather would make all 72 cores usable, at 9 chunks per core
(ring SDPA time is proportional to the number of chunks per core).

**Design:**
- Of the 8 writer cores in the last row, one forward and one backward core per link (4 links) act as relays. The relays of link l handle heads l, l+4, ….
- A forward relay sends its own chip's chunks, then forwards the chunks arriving from the backward neighbor to the forward neighbor (backward relays do the reverse). Each hop goes through DRAM.
  Multi-hop fabric multicast either did not reach 4 hops away or deadlocked the ring.
- A packet is 4 consecutive pages of the same DRAM bank (4,352 B). The flow is DRAM → double-buffered L1 staging → fabric → the same page of the neighbor chip's gathered buffer.
- Fabric atomic incs bump the next relay's arrival counter after each head is sent, and, after each chunk is sent, the per-chunk counters of the neighbor chip's K/V injectors (the cores that read from DRAM, 3 cores).
  Counters are never decremented; each core derives its thresholds from its own call count.
- The same gathered buffer is reused every call, so a relay does not send until its counterpart on the destination chip (opposite direction, same link) signals that it has entered the current call (a start handshake).
- The relay makes progress while the writer waits on CBs (hooks inserted into the wait loops of `cb_reserve_back` and `cb_wait_front` in `dataflow_api.h`) and while it waits for signals from compute.

**Speed:** Ring SDPA 15.39 → 13.89 ms. But the op right after it grows from 0.50 to 1.33 ms (the next CCL op absorbs the skew in SDPA finish times across chips).
Block 32.97 → 31.68 ms. 40 steps: 45.00 → 43.48 s, and 43.88 → 42.57 s with fast exp (#138). Three 40-step e2e runs
matched the base latent hashes at every step.

**Problem: results occasionally change.** In the block test that runs 2000 blocks in a row, feeding in the previous output (#148), about 30% of runs ended with a different final value (base was identical in all of 20+ runs). What was checked:
- Repeating SDPA alone 40,000 times with alternating inputs never diverged. It happens only inside the block.
- A test that runs the block twice on the same input and compares intermediate stage outputs caught the divergence 3 times. Twice it was a 1-2 ULP (BF8) divergence in the SDPA output itself
  (once across all Q chunks of head 0, once in a single element), both in the outputs of non-relay cores. The final contents of the destination gathered buffer were the same both times.
  In the remaining case, only the test's copy of the SDPA output diverged, by a lot.
- No problem was found in the relay protocol (start handshake, counters, reuse of packet headers, flushing fabric sends), the receiving router's flush
  (the WH implementation waits for the data write ack before incrementing the counter), the NOC state at writer exit (write acks and read responses match the software counts),
  or CB / L1 buffer overlaps (cached programs are checked every time too).
- It also happens on the 8×8 grid (10 per core, no zones). A variant that finishes all relaying before the writer's own work (drain-first) gives the same value in all 6 runs,
  but ring SDPA takes 16.79 ms, slower than base.
- Most likely the timing created by the relay exposes a race elsewhere in SDPA, but its location has not been found.
- A variant that runs the relay not in the writer but in the reader of cores that only wait for multicast (row 7) (`TT_RING_SDPA_RELAY_READER=1`) gave the same value all 4 times.
  However, the reader is on NOC1, its DRAM reads are slow, and a long relay step delays its own multicast receive. Ring SDPA stays around base (15.39 ms): 16.31 ms with 6 staging
  slots, 15.43 with 3, 15.29 ms with 1.
- Narrowing where the writer's relay is hooked: with only the `cb_reserve_back` wait, ring SDPA is 14.25 ms and the block 31.48 ms, faster than with all hooks (31.68 ms),
  but 1 of 6 runs gave a different value (with only the `cb_wait_front` wait it is slow, 15.61 ms).
- Measuring stack usage with the watcher: the smallest free BRISC stack is 532 bytes, in AGMM's in0 kernel, and the SDPA writer with the relay has more headroom than that
  (not a stack overflow).
- Summary: interleaving the relay with the writer's own work inside the SDPA writer occasionally changes the values. Outside the writer (drain-first, reader) the values stay the same, but it is slow.
- Both captured divergences were in the first-pass head of a multicast group (the K/V the injector reads first, right after waiting on the counter). So a variant where the injector waits about 20 µs after the remote counter before reading was tried (ring SDPA 13.89 → 14.08 ms), but 1 of 10 runs gave a different value. The receiving router waits for the data write ack before incrementing the counter, so it is not a race where the counter is read before the data either.
- With a host-side sync of all chips inserted just before SDPA, 1 of 8 runs differed; just after SDPA, 1 of 16. The frequency drops but does not go to zero, so the race is inside the SDPA call. But SDPA alone does not show it even over 40,000 runs, so it needs the combination with the block's other ops (state left in L1, the NOC, the routers). The investigation stopped here.

**Decision:** It saves about 1.4 s (3%) per run, but it is not deterministic, so it is not adopted.

**Operational note:** In one case, right after a test process was killed partway through, the next few runs diverged (hardware state such as NOC counters persists across processes).
Before a determinism test, if any process was killed partway through, reinitialize with `tt-smi -glx_reset`.

### #154 details: an 8×10 compute grid with Ethernet dispatch (2026-10-10, optional)

- **Motivation:** Both the matmuls and ring SDPA have been tuned to the point where the core count sets the time (bound by unpack and tile hand-offs). WH Galaxy has 8×10 = 80 Tensix cores, but the standard configuration reserves all 8 cores of the 10th row for dispatch (receiving and distributing commands), so ops can use only 8×9 = 72 cores. The core descriptor (`wormhole_b0_80_arch_eth_dispatch.yaml`) has a Galaxy configuration that puts dispatch on Ethernet cores, with an 8×10 compute grid.
- **Problem:** Enabling it as is stops with "No more available dispatch cores". On Galaxy, all 16 Ethernet links per chip are connected, 4 directions × 4 links (a torus), and the fabric uses all of them, so no Ethernet core is free.
- **How:** On each edge of the SP axis (the ring of 8 chips, used only by ring SDPA), take 2 of the 4 links out of the fabric at both ends. tt_cluster has a mechanism to drop unstable UBB links (ones that retrain often); putting the links in that same set (`frequent_retrain_cores_`) makes the fabric place no router on them, and their Ethernet cores become "free" dispatch candidates (`HK_DISABLE_ETH_CHANS="chip:channel,…"`). The list of links to drop depends on the cabling, so it is generated with `gen_disable_list.py` (the K highest-numbered channels connecting neighboring chips in a row of the logical mesh). With 4 free cores per chip, the standard 1-CQ Ethernet dispatch, which puts prefetch, dispatch and dispatch_s on separate cores, runs.
- **After dropping links:** The dispatch firmware replaces the base Ethernet firmware, so the dropped links stay down after the run until `tt-smi -glx_reset`. In the next run, physical topology discovery sees 2 links per SP-axis edge, and the standard mesh graph descriptor (4 links) cannot be mapped. Passing a 2-channel (RELAXED) descriptor (`galaxy_torus_xy_ch2.textproto`) via `TT_MESH_GRAPH_DESC_PATH` opens the mesh in both the 4-link and 2-link states. The number of fabric routing planes is set per axis from the minimum physical channel count, so the TP axis keeps all 4 links. Returning to the standard configuration needs a reset.
- **Using 8×10:** (1) Ring SDPA computes on 8×9 = 72 cores, with the all-gather workers (2 links × 2 directions) on the 10th row. The zone split that divides the 640 Q chunks into 64 cores × 9 + 8 cores × 8 is the non-relay part of the #150 diff. 15.74 → 15.01 ms (14.17 ms with 3 SP-axis links). (2) The QKV and FFN1 matmuls (transposed grid, N split over rows) on 8×9 workers: QKV goes from 15 to 14 N tiles per core (block 16,8,14, 1.97 → 1.78 ms), FFN1 from 14 to 12 (16,8,12, 2.01 → 1.75 ms). In e2e, L1 is smaller because the kernel config region is enlarged by 128 KB (#109), so QKV with K block 10 does not fit. (3) Output-projection-type matmuls (N 1280 = 40 tiles) stay at 5 tiles per core even with 9 rows, so they keep the 8×9 blocking (`WAN_MM_GRID=8,9` makes the matmul config see an 8×9 grid). FFN2 (#142) does not get faster even on the full 8×10: 1.95 → 1.98-2.02 ms. Cross-attn SDPA on 80 cores splits 320 units evenly at 4 each: 0.37 → 0.31 ms.
- **Result:** In the block test without FSDP, one block goes 26.47 → 25.19 ms. 40-step e2e: 45.00 → **43.01 s** (steady-state step about 1,055 ms); with fast exp (#138): 43.88 → **42.14 s**. Same composition. Block test PCC 99.9692 → 99.9694%.
- **Determinism:** In the 2000-block chained block test, 6 of 7 runs had the same hash. The one that differed was the first run right after a Tensix-dispatch run; in other conditions too, "only the first run right after switching differs" was seen (when switching the dispatch mode, reset or do one warm-up run). The variant without the wider matmuls gives the same hash as the Tensix-dispatch reference (72-core SDPA does not change results).
- **Keeping 3 SP-axis links (rejected):** Dropping only 1 link per edge leaves 2 free cores per chip. Adding a 1-CQ configuration without dispatch_s (prefetch → dispatch on 2 cores, `HK_ETH_NO_DISPATCH_S=1`) makes it run: ring SDPA 14.17 ms, 40 steps 41.01 s (fast exp). But the chained block test gave a different value every time, and once it hung. Isolation: the no-dispatch_s configuration matched 3/3 with 2 links (not the cause). 3 SP-axis links also differ in 1 of 8 runs with Tensix dispatch, and it does not happen with 2 or 4 links. It is likely a timing-dependent race when the ring SDPA all-gather runs on 3 links (the DRAM bank ownership divides 12 banks evenly by 3; the per-link worker sync (the master waits for everyone before signalling SDPA) and the atomic notification sent with a flush after the data look correct; cause not identified). Fixing it would save about another 1.1 s per run.
- **Fix for 3 SP-axis links (#155):** The 3-link race was in the WH-specific send scheme of the ring SDPA all-gather. On WH, each head's 512 pages are split into a contiguous range per link (171, 171, 170 pages for 3 links) and sent in scatter packets of 2 pages each. Blackhole splits the work by DRAM bank (link L owns banks L, L+3, …) and sends 4 consecutive pages of the same bank in one packet. Using the latter on WH too (`TT_RING_AG_BANK_OWNED_WH=1`) gave the reference hash in all 6 runs (which part of the WH scheme breaks with odd-sized ranges is not identified). It was not fixed by having SDPA wait for everything before reading, alternating the K/V buffers per call, dropping split transfers, or changing which channels are dropped. The current `env.sh` default is this configuration (1 link dropped per edge, no dispatch_s, per-bank sends): 40 steps 42.68 s (41.85 s with fast exp). Adding the QKV and FFN1 matmuls that use the 10th row makes 2 of 4 chained block tests differ, so they are left out (the same even with #133's relay deferral turned off).
- **Matmul write-out race (#156):** The race when QKV and FFN1 use the 10th row was because the function that writes AGMM output row by row (`write_block_sync_granular`) freed the output CB right after issuing the NOC writes. Compute writes the next block into the freed space, so rows not yet read out could be overwritten. It did not show on the 8×8 grid, but appeared once the grid changed and the timing shifted. Fixed by waiting for completion per row before freeing (the same function in minimal_matmul was already fixed), and the `env.sh` default now widens QKV and FFN1 too. 40 steps 41.87 s (41.02 s with fast exp).
- **e2e determinism (#157):** Even when the chained block test matches, agreement between 40-step e2e runs had to be checked separately. The #156 configuration diverged from step 32. Only the e2e test enlarges the kernel config region (#109), and combined with Ethernet dispatch without dispatch_s, results change. Without the enlargement, all 3 runs matched at every step, and it costs only 3.6 ms per step, so the region is not enlarged when dispatch_s is off. The current `env.sh` default gives 40 steps in 42.00 s (41.19 s with fast exp).
- **Wrong bank-owned check (#162):** The bank-owned send scheme added in #155 to avoid the 3-SP-link race had never been enabled, because of a bug in the environment variable check (the 6/6 match in #155 was luck). On 2026-10-11, the default chained block test started matching only about 40% of the time; isolating it with k2 (2 SP links), 2 links for the SP CCLs only, and Tensix dispatch narrowed it to the 3-link SP CCLs, where the bug was found. After the fix: 5/5 match, e2e 41.81 s (41.01 s with fast exp).
- **Right after a reset:** The first run right after a reset can fail (link training is not finished in time and an SP edge appears to have one link fewer, or a timeout waiting on the prefetch queue). Running again works.
- **Caution:** The `HK_DISABLE_ETH_CHANS` list is for the test Galaxy's cabling. On other Galaxies, regenerate it with `gen_disable_list.py`.

### How the experiments were run (making the experiments themselves fast and reliable, 2026-10-07)

How much an optimization effort achieves depends on how many ideas can be tried, and how reliably. This section collects what was done to improve how experiments are run, separately from the individual optimizations.

**1. Measure in stages (narrow down with cheap measurements, confirm with expensive ones)**

| stage | time per run | use |
|---|---|---|
| single-chip micro-benchmark | tens of seconds | isolates compute-unit efficiency and the effect of dtype, subblock and fidelity alone; it does not use the fabric, so hangs matter less |
| block test (1 block × 3 runs, Tracy) | about 2 min | per-op time and PCC; blocking searches are done here |
| short e2e (`WAN_PERF_STEPS` = 2-6) | about 2-3 min | per-section time (`WAN_FWD_TIMING`), building caches, checking for hangs |
| e2e (40 steps) | about 4-6 min | adoption decisions; reports steady-state step, first step, total, video pixel differences (against both the reference version and #45) and comparison images in one run |

- Block test and e2e results sometimes disagree (#62: a blocking that helped in the block test did not help in e2e). A disagreement is itself a clue (it led to finding the #66 bug).

**2. Rebuilding the weight cache: 45 min → 5 min (`TT_DIT_FAST_CACHE_SAVE`, section 11.8)**
- Every change to the weight format or residency meant rebuilding a cache of about 16 GB each. Writing it out read every parameter back from the device (about 12.6 MB/s).
- Writing out the host-side tensors created during conversion directly cut this from 21 minutes to 10 seconds.
- After confirming that a cache built the new way gives the same video as before (pixel difference 0.0), the read-back verification was turned off for experiments.
- This made it cheap to try combinations of BFP8 layers and residency (#67 and #71-#73 came after this).

**3. Add your own measurement tools (see what the existing profiler cannot)**
- **DM kernel wait times:** Add wall-clock accumulation to the kernel and write it out with `DeviceTimestampedData` (ring SDPA reader, AGMM in0 and in1). TRISC zones are not used, because they also include coprocessor-side waits.
- **Timing-only variants that measure lower bounds:** "skip the matmul" (DM-only lower bound), "DM hands data over instantly" (compute-only lower bound), "read K/V only once", and so on. They tell compute-bound from supply-bound with numbers, not guesses (#47, #53). For FFN2, a timing-only variant with the whole output in BFP8 (the residual also becomes BFP8, so precision is ignored) went from 3.10 to 2.43 ms, which showed that reduce-scatter volume matters and led to the real implementation that makes only the partial sums BFP8 (#76).
- **When Tracy post-processing fails:** Compute op times from the kernel zones in the raw device log (`profile_log_device.csv`). FW zones are not used, because they include the wait for the next op.
- **Why the program cache misses:** Temporary hooks that log ops that missed the cache and the components of the hash pinned down that Q, K, V had different dtypes in warm-up and measurement (#66).
- Always remove investigation instrumentation before committing.

**4. Experimental switches are environment variables and do not change the default behavior**
- Every optimization can be turned on and off with an environment variable (`WAN_*`, `TT_*`). Combinations can be compared without rebuilding.
- Mind when a switch is applied. The performance test's warm-up runs while the pipeline is built, so changing settings afterwards rebuilds programs in the first measured step and only affects some of the matmuls (#66). Apply settings when the model is built.
- Record composition-changing changes (BFP8, LoFi, etc.) separately from those that keep the composition, and tell them apart by the video's mean pixel difference (about 0.1 between repeats of the same setting, about 20-50 when the composition changes).

**5. Expect hangs and device faults, and recover quickly**
- Use a short op timeout for experimental settings (`TT_METAL_OPERATION_TIMEOUT_SECONDS=120`). A hang then stops quickly and you can move on to the reset.
- **Hang watchdog (2026-10-07):** The run script has a watchdog. When a device timeout appears in the log, it kills the process without waiting for the shutdown (2 × 120 s of waiting) and automatically runs `tt-smi -glx_reset` and the idle-state check. Optionally, it records call stacks of all cores with `tt-triage` before the reset (this is how the cause of the #78 hang was found). The cost of one hang dropped from about 10 minutes to about 3.
- Recovery steps: kill leftover processes → `tt-smi -glx_reset` → check with `tt-smi -s` that all 32 chips are idle (about 500 MHz, about 17 W) → confirm operation with a short e2e. If DRAM retraining fails, reset again.
- After an abnormal exit, the clocks can stay high (1,000 MHz). A pre-run gate (power, clock) detects this and stops the run.

[↑ Back to contents](#contents)

## 11. Q&A notes (2026-09-30)

Answers to questions that came up during the work, collected for later reference. Numbers come from the same measurements as sections 9 and 10.

### 11.1 Is video generation bound by compute or communication?

- Unlike LLM decode (one token at a time, bound by weight reads), video diffusion **re-prefills the whole video every step (32,760 tokens at 480p)**. There is no KV cache either. The compute per weight byte is large, so it is basically **compute-bound**.
- Communication within a layer: the TP all-gather (AGMM) and reduce-scatter (MM+RS), the SP K/V ring rotation (Ring SDPA), and the FSDP weight gather. Measurements on H3 showed communication time at 1/5-1/10 of compute time, mostly hidden behind compute.
- However, the "hiding machinery" eats into compute efficiency (fused ops dedicate cores to communication). Compute utilization stays around 30% because of kernel efficiency, small ops, and the small amount of work per chip.

### 11.2 The K/V ring needs data from distant chips; is that a problem?

- In Ring Attention, each chip exchanges K/V **only with its neighbors**, repeating this 7 times (about half that if going both ways) to cycle through all chips. It never talks to distant chips directly, so distance does not matter even with mesh links (Ulysses, common on GPUs, uses all-to-all communication and so suits NVLink with all-to-all connectivity).
- The "compute ÷ communication" ratio per received block is proportional to the number of local Q rows (sequence length ÷ SP). There is about 12x headroom for H3 15 s and about 3x for Wan 480p (estimates). With shorter sequences, larger SP or faster SDPA, communication starts to show.

### 11.3 Does splitting the sequence (SP) break the dependencies between frames?

- Inside the DiT, frames are processed **at the same time**, not in order, and with bidirectional full attention every token sees every token. The only sequential dependency is along the steps (the 40 steps run in order).
- Ring Attention uses online softmax and gives **mathematically the same result** as without splitting. The only difference is bf16 rounding from the order of additions.
- What truly depends on order in time is the VAE (causal in time). That is why the VAE is split by height and width, not by time.

### 11.4 Where are SDPA's intermediate results kept?

- At each ring stage, "the output and LSE (log-sum-exp) so far" are read from DRAM into L1, combined with the contribution of the new K/V, and written back to DRAM (confirmed in `ring_joint_sdpa.cpp` and `ring_joint_writer.cpp`). The full token × token score matrix is never stored anywhere.
- L1 (about 1.43 MiB per core) holds only the pieces being computed (chunks of Q, K, V and scores, the partial output, and per-row max and sum). **The maximum chunk size is set by L1 capacity**.

### 11.5 What is the fused path?

See section 9.4. It combines "matmul → communication → epilogue" in one kernel and sends tiles as soon as they are ready. Compute and communication overlap, at the cost of dedicating cores to communication. Settings must be searched for each grid shape; without a setting, defaults are used and it is slow.

### 11.6 Idle cores in SDPA, and other bottlenecks

- SDPA work is split into work units of "1 Q chunk × 1 head" and spread over 63 cores. If the unit count does not divide evenly by the core count, some cores wait in the last round (H3 5 s: 210 units → 4 rounds, 16.7% idle; Wan 480p original settings: 160 units → 3 rounds, 15.3%).
- Besides idling, the bottleneck is **softmax processing**, half of each step inside the kernel (measured on H3: of a 66 µs step, the matmuls take 33 µs; exp 14 µs, writing the probabilities out twice 14 µs, max subtraction 10 µs, hand-offs between threads about 11 µs). DRAM and links are only a few percent utilized.
- Online softmax is already used. It solved the memory problem, but the per-step softmax processing remains as the bottleneck.
- The middle ground between regular and online softmax is the **k_chunk size**: the larger it is, within what fits in L1, the closer to regular softmax (#3 went from k=256 to 512 for SDPA −11%). Keeping scores in DRAM is too much data and would take longer than all of SDPA (about 2.7 GB per layer at Wan 480p). Other kernel-side tricks exist, such as "apply the correction only when needed".

### 11.7 How complex can processing on a Tensix core be?

- One core = 5 small RISC-Vs (2 for data movement, 3 to drive compute), an FPU (32×32 tile matmul), an SFPU (elementwise ops such as exp and GELU), Unpacker/Packer, about 1.5 MB of L1, and 2 NoCs.
- Heavy compute is FPU/SFPU tile operations. The RISC-Vs can run arbitrary C++ control code, but they are small scalar CPUs and not suited to number crunching. One op runs as 3 kernels, reader, compute and writer, in parallel, passing data through ring buffers in L1.
- References: METALIUM_GUIDE.md (https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md ), the TT-NN op list (https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api.html ), the list of tile operations on Tensix (https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/compute/compute.html ), the instruction set (https://github.com/tenstorrent/tt-isa-documentation ), and the repo's `tech_reports/` (`matrix_engine/`, `GEMM_FLOPS/`, `FlashAttention/`).

### 11.8 What limits experiment turnaround

| experiment | time per run | where the time goes |
|---|---|---|
| matmul blocking sweep (1 shape) | about 10 min | kernel compilation at about 1.5 s per combination (CPU); the device is busy only about 0.1% of the time |
| Tracy measurement of 1 block | about 1.5-2 min | the test itself + about 30 s of CSV conversion |
| full pipeline (480p) | about 2.3 min | about 100 s of generation + about 30 s for loading the weight cache, etc. |

- Compilation does not use the device, but the sweep tool runs "compile → run" sequentially in one process, so the device sits waiting. Only one process can open the device, so simple parallelization is not possible.
- Kernels are cached on disk, so with model-side changes only, there is almost no compilation from the second run on.
- For now: narrow the sweep candidates, do the screening with 1-block measurements, and use the full pipeline only for confirmation.

**Addendum (2026-10-07): faster weight cache rebuilds**
- Every change to the weight format (which layers are BFP8) or residency meant rebuilding the transformer cache (about 16 GB each, 855 parameters × 32 chips). Each took about 2 minutes to convert (torch → device) and **about 21 minutes** to write out (about 12.6 MB/s). The write-out read each parameter back from the device.
- `TT_DIT_FAST_CACHE_SAVE=1` (changes to `tt_dit`'s Parameter and cache, commit `6bfa255142e`): keep the host-side tensors created during conversion, send them to the device, and use them directly for the write-out as well. **Write-out: 21 minutes → 10 seconds**.
- Verification (`TT_DIT_CACHE_VERIFY=1`) still takes about 7.5 minutes per cache even when comparing against the host-side copy (Python converts each shard back to torch to compare). A cache built the new way was confirmed twice to give exactly the same video as one built the old way (pixel difference 0.0), so verification is off for experimental caches.
- Result: one e2e run (2 steps) that builds 2 caches went from **about 45 minutes to 4 minutes 50 seconds**.
- An incremental C++ rebuild takes 0.5-1.5 minutes (the earlier "5-10 minutes" was an overestimate).

[↑ Back to contents](#contents)

## 12. Operational lessons

- **Wrap pytest (`python -m tracy -r -p -v -m pytest …`) with `TT_METAL_OPERATION_TIMEOUT_SECONDS` and the shell's `timeout`.**
- **`hf download` can exit with code 1 even on success** (a CLI bug). Check the file sizes against the values on the Hub.
- **With `HF_HUB_OFFLINE=1`, Wan cannot be loaded** (`from_pretrained(trust_remote_code=True)` queries the Hub API).
- **Matmul blocking sweeps need `MM_SWEEP_PROFILER_DUMP_EVERY=100000`** (on WH Galaxy, running with the default stalls and needs a reset; noted in the H3 README). Errors for combinations that do not fit in L1 (`beyond max L1`, `clash with L1`) are expected, and the sweep continues.
- **Measure block performance in a warm state.** A single run has host-side waits between ops, which hides the overlap of communication and compute (a conclusion about FSDP had to be corrected because of this).

[↑ Back to contents](#contents)

## 13. MiniMax-H3

MiniMax-H3 speed-up work is tracked separately.

[↑ Back to contents](#contents)

## Sources

- PR #56886 Enable MiniMax-H3 on Wormhole Galaxy (merged) — description, diff (`pipeline_minimax_h3.py`, `utils/matmul.py`, `mmrs_config.py`, `transformer_block_minimax_h3.py`, `packing.py`, `attention_minimax_h3.py`, `weights_minimax_h3.py`, `utils/cache.py`, `sweep_mm_block_sizes.py`, `MiniMaxH3.md`)
- PR #57941 MiniMax-H3: Wormhole Galaxy optimizations (Open, Draft) — `minimax_h3_wormhole/README.md`, `minimax_h3_blackhole/README.md`, `minimax_h3_blackhole/agmm_fused_vs_unfused.md`
- PR #57029 (closed without a description)
- main: `models/tt_dit/models/MiniMaxH3.md` (BH performance table), `models/tt_dit/models/Wan2_2.md` (Wan performance table), `pipelines/wan/pipeline_wan.py`, `tests/models/wan2_2/test_performance_wan.py`
- Hugging Face: the `MiniMaxAI/MiniMax-H3` model card, the `Wan-AI/Wan2.2-T2V-A14B` model card (Computational Efficiency table, Wan-Bench 2.0 figure), and each repo's creation date (createdAt from the API)
