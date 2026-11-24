// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The fused form (GLM53_KDA_D2=1): the KDA verify step's compute: this core's decode_prepare item (decode_prepare.hpp:
// its startup, CT 0 - 2, RT 0 - 3), then its chunked-form recurrence units (d2_chunk_compute.hpp: CBs + D2_CB, RT from
// D2_RT) without a second startup (the same fp32 formats; a second one would reset the math / pack sync while the pack
// may still work).
// (D2_UNPACK_OS: the unpacker thread at -Os, the math and pack threads as built (a smaller binary for the kernel
// config ring; the same instructions issued in the same order, the same values))
#if defined(D2_UNPACK_OS) && defined(TRISC_UNPACK)
#pragma GCC optimize("Os")
#endif
#include "ttnn/cpp/ttnn/operations/experimental/kda/decode_prepare/device/kernels/compute/decode_prepare.hpp"
#define D2_NO_STARTUP 1
#include "d2_chunk_compute.hpp"

void kernel_main() {
    kda_prepare_compute::run();
    if (get_arg_val<uint32_t>(D2_RT + 1) > 0) {  // (a core without chunk units: the chunk compute would wait for the
        d2_chunk_compute();                       // constants its reader makes with the first unit only)
    }
}
