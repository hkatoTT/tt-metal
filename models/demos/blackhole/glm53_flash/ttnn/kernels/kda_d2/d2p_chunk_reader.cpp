// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The read-ahead form (GLM53_KDA_D2=2): recurrent_decode_step's chunked-form reader (d2_chunk_reader.hpp: the tree's,
// args from 0) with every batch's state already in CB_S's memory (an L1 buffer the prepare program read it into:
// d2p_prep_reader.cpp), so the reader only pushes it (D2_PREFETCH_ALL with D2_STATE_RESIDENT: CB_S holds every unit of
// this core).
#define D2_ROWS_HOOK(first)
#include "d2_chunk_reader.hpp"

void kernel_main() { d2_chunk_reader(); }
