// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The fused form (GLM53_KDA_D2=1, ttnn/kda_d2.py): the KDA verify step's reader: this core's decode_prepare item
// (reader_decode_prepare.hpp, args from 0), then its recurrent_decode_step chunked-form units (d2_chunk_reader.hpp =
// the tree's reader with CB / CT / RT offsets D2_CB / D2_CT / D2_RT).  The chunk reader's first DRAM reads (the last
// step's saved slot and the first batch's state) and its setup (keep codes, the constant tiles' zeros, beta / MZ rows)
// go out as soon as the prepare reader is done (its reads landed), under the prepare's compute and writes; before its
// first read of the prepare outputs (q / k / v / decay / beta rows, L1) it waits for semaphore D2_SEM to reach its
// runtime arg D2_READY_RT: one increment from each prepare item of each head its pairs touch (7 a head), sent after
// that item's writes landed (d2_writer.cpp).
#if defined(D2_DM_OS)  // (the data mover at -Os: a smaller binary; the same values)
#pragma GCC optimize("Os")
#endif
#include "ttnn/cpp/ttnn/operations/experimental/kda/decode_prepare/device/kernels/dataflow/reader_decode_prepare.hpp"

#ifndef D2_SEM
#define D2_SEM 2
#endif

FORCE_INLINE void d2_wait_ready(bool first) {
    if (first) {
        DeviceZoneScopedN("D2-WAIT");
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(D2_SEM)), get_arg_val<uint32_t>(D2_READY_RT));
    }
}
#define D2_ROWS_HOOK(first) d2_wait_ready(first)

#include "d2_chunk_reader.hpp"

void kernel_main() {
#if D2_PREP
    kda_prepare_reader::run<0, 0>();
#endif
    d2_chunk_reader();
}
