// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The read-ahead form (GLM53_KDA_D2=2): decode_prepare's reader (reader_decode_prepare.hpp, args from 0), then, its own
// reads landed, this core's recurrence units' state (the next program's, recurrent_decode_step's chunked form: start,
// count from runtime arg D2P_RT) read into an L1 buffer of this core (the next program's CB_S, at its base, batch after
// batch) under the prepare's compute and writes (DRAM is nearly idle there).  CT args from D2P_CT: the state's
// accessor.
#include "ttnn/cpp/ttnn/operations/experimental/kda/decode_prepare/device/kernels/dataflow/reader_decode_prepare.hpp"

void kernel_main() {
    kda_prepare_reader::run<0, 0>();
    const uint32_t state_addr = get_arg_val<uint32_t>(D2P_RT + 0);
    const uint32_t start = get_arg_val<uint32_t>(D2P_RT + 1);
    const uint32_t count = get_arg_val<uint32_t>(D2P_RT + 2);
    const uint32_t buf = get_arg_val<uint32_t>(D2P_RT + 3);
    constexpr auto state_args = TensorAccessorArgs<D2P_CT>();
    const auto s_state = TensorAccessor(state_args, state_addr);
    constexpr uint32_t N = 4, S_TILE = 2048;  // (bf16 state tiles)
    DeviceZoneScopedN("D2P-PREFETCH");
    for (uint32_t u = start, t = 0; u < start + count;) {
        const uint32_t p = u / N, j0 = u % N;
        const uint32_t nb = N - j0 < start + count - u ? N - j0 : start + count - u;
        for (uint32_t b = 0; b < nb; ++b) {
            for (uint32_t i = 0; i < N; ++i) {  // (the chunk reader's order and places: tile t + b N + i)
                noc_async_read(s_state.get_noc_addr(p * N * N + i * N + j0 + b), buf + (t + b * N + i) * S_TILE, S_TILE);
            }
        }
        t += nb * N;
        u += nb;
    }
    noc_async_read_barrier();
}
