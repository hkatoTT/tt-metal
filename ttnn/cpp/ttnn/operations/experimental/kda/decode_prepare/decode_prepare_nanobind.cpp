// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "decode_prepare_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "decode_prepare.hpp"

namespace ttnn::operations::experimental::kda::decode_prepare::detail {

void bind_decode_prepare(nb::module_& mod) {
    ttnn::bind_function<"decode_prepare", "ttnn.experimental.kda.">(
        mod,
        R"doc(
            KDA decode before the delta rule: the width-4 conv and SiLU (rounded to bf16), the L2 norms of q and k (q
            scaled), the decay exp(lower sigmoid((f + dt_bias) rate)) and beta; the conv history shifted in place.

            Returns:
                (q, k, v, decay [1, 1, B H, 128], beta [1, 1, B H, 1]) FLOAT32 TILE, one row per (user, head) pair.

            Keyword Args (a verify step):
                * :attr:`users`: the call's B users (default proj's rows; with more tokens token k's rows k B ..); the
                  history has B rows.
                * :attr:`keep`, :attr:`saved`: (2 .. 8 tokens) a user's keep code c [1, B] (UINT32 ROW_MAJOR) below
                  the saved rows n (tokens - 1; saved is [n, B, 3 H 128]) reads its history back by n - c in (saved 0
                  .. saved n - 1, history0, history1, history2); saved k gets row k + 1 of (the read history, the
                  tokens' inputs).
                * :attr:`tokens`: tokens a user (1 .. 8), token k's proj and fg rows k B .. (k + 1) B (its conv after
                  the earlier tokens'); the outputs have tokens B H rows, token 0's pairs first; the history moves by
                  `tokens`.
                * :attr:`pair_major`: the outputs by pair instead, its tokens in turn (row p tokens + k).
                * :attr:`l1_outputs`: the outputs L1 interleaved (for a delta rule reading them next).
                * :attr:`split_channels`: q / k / v's units over 2 cores each by channel halves (the same values; off
                  where the cores are too few).
                * :attr:`history0_out` .. :attr:`history2_out`, :attr:`saved_out`: (with keep and saved; all four
                  or none) the new history and checkpoint rows written there, the inputs left as they were (a
                  ping-pong of two state sets: the call can be run again from the same inputs).
                * :attr:`row_split`: the plain step's call (one token, no keep): the top 6 grid rows' reads on NoC 1
                  and the units over the cores in proportion to their grid row's speed (the same values).
        )doc",
        &ttnn::experimental::kda::decode_prepare,
        nb::arg("proj").noconvert(),
        nb::arg("fg").noconvert(),
        nb::arg("history0").noconvert(),
        nb::arg("history1").noconvert(),
        nb::arg("history2").noconvert(),
        nb::arg("taps").noconvert(),
        nb::arg("gate_consts").noconvert(),
        nb::arg("heads"),
        nb::arg("lower"),
        nb::arg("q_scale"),
        nb::kw_only(),
        nb::arg("users") = 0,
        nb::arg("keep") = nb::none(),
        nb::arg("saved") = nb::none(),
        nb::arg("tokens") = 1,
        nb::arg("pair_major") = false,
        nb::arg("l1_outputs") = false,
        nb::arg("split_channels") = false,
        nb::arg("row_split") = false,
        nb::arg("history0_out") = nb::none(),
        nb::arg("history1_out") = nb::none(),
        nb::arg("history2_out") = nb::none(),
        nb::arg("saved_out") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::decode_prepare::detail
