// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "mp_common.hpp"

// The prefill MoE's readers on both NoCs (GU5_NOCS / DNCB_NOCS 2): the writer (BRISC, NoC 1) hands NoC 1 over to its
// core's reader (NCRISC) for DRAM reads, as decode_matmul's control kernel does.  Dedicated NoC mode: both RISCs use
// NoC 1's NIU and share its counters, so the writer reads nothing on NoC 1 after this (writes and semaphores only);
// its command buffers' transaction ids set to 0 and NoC 1's outstanding counts of ids 1 .. 15 cleared (earlier
// programs can leave them non-zero) before S_NOC1 tells the reader, whose NoC 1 reads carry id 1 and are waited for
// by that id only.
inline void hand_over_noc1() {
    using namespace moe_prefill;
    noc_async_read_set_trid(0, noc_index);
    noc_async_write_set_trid(0, noc_index);
    ncrisc_noc_set_transaction_id(noc_index, write_reg_cmd_buf, 0);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK & ~1u, noc_index);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(S_NOC1)), 1);
}
