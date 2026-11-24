// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "recurrent_decode_step_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "recurrent_decode_step.hpp"

namespace ttnn::operations::experimental::kda::recurrent_decode_step::detail {

void bind_recurrent_decode_step(nb::module_& mod) {
    ttnn::bind_function<"recurrent_decode_step", "ttnn.experimental.kda.">(
        mod,
        R"doc(
            One KDA decode step for P (user, head) pairs: S <- S * decay (per key channel),
            S <- S + k^T ((v - k S) beta), o = q S, on each pair's [128 x 128] state (kept in bf16, the update in
            fp32).

            Args:
                * :attr:`state`: [.., 128, 128] BFLOAT16 TILE in DRAM, P states, updated in place.
                * :attr:`q`, :attr:`k`, :attr:`v`, :attr:`decay`: [.., P, 128] FLOAT32 TILE in DRAM, one row per pair.
                * :attr:`beta`: [.., P, 1] FLOAT32 TILE in DRAM.

            Keyword Args:
                * :attr:`form`: the plain step's form (bits; the values the same): 128 a pair's value blocks in a
                  per-core rotated order (the DRAM banks spread), 256 the grid's top rows read on NoC 1, 1024 the
                  units over the cores by grid row speed (the top 5 rows on NoC 1 with 256).
                * :attr:`output`: o written into it (else a new tensor; the chunked form: DRAM or L1 interleaved).
                * :attr:`saved`, :attr:`keep`, :attr:`heads`, :attr:`tokens`: the chunked verify form (a verify step
                  of `tokens` = 2 .. 8 tokens a pair, q .. beta holding them all: token k's P rows from k P on):
                  `state` holds the start state of the last step and `saved` its tokens' rows (FLOAT32 TILE, a
                  unit's slot of 1536 tokens + 64 bytes in its own whole tiles, kda.chunk_saved_host); a user's keep
                  code c (keep [1, B] UINT32 ROW_MAJOR in DRAM, B = P / heads) redoes the first c + 1 of them (c =
                  tokens - 1: all) before this step's tokens, all as one chunk.  It writes its
                  start state (after the redone tokens) over `state` and its tokens' rows to `saved`.  o has tokens P
                  rows.  The values are those of the plain steps up to rounding (not bit for bit).
                * :attr:`pair_major`: (the chunked form) q .. beta by pair, token k of pair p at row p tokens + k
                  (decode_prepare's pair_major), rather than row p + k pairs.
                * :attr:`source`, :attr:`saved_source`: (the chunked form, out of place) the last step's start state
                  and rows read from these (default `state` / `saved`); this step's still go to `state` / `saved`
                  (a ping-pong of two state sets: the step can be run again from the same inputs).

            Returns:
                o [.., P, 128] FLOAT32 TILE.
        )doc",
        &ttnn::experimental::kda::recurrent_decode_step,
        nb::arg("state").noconvert(),
        nb::arg("q").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("v").noconvert(),
        nb::arg("decay").noconvert(),
        nb::arg("beta").noconvert(),
        nb::kw_only(),
        nb::arg("source") = nb::none(),
        nb::arg("keep") = nb::none(),
        nb::arg("heads") = 0,
        nb::arg("output") = nb::none(),
        nb::arg("form") = 0,
        nb::arg("tokens") = 0,
        nb::arg("saved") = nb::none(),
        nb::arg("pair_major") = false,
        nb::arg("saved_source") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::recurrent_decode_step::detail
