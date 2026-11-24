// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One KDA decode step (tt/kda.py, kda_recurrent) per work unit: a (user, head) pair's value block j, its state's
// column of tiles S_ij (i = 0..3; rows: key channels, columns: value channels).  The value blocks of a pair are
// independent; a core's units are consecutive, and the pair's q / k / decay / beta tiles are held across them:
//   DS_i = S_ij * decay_i (per key row)     exact fp32: S and the decay column in Dest, SFPU multiply
//   r = sum_i k_i DS_i                       matmul (SrcA / SrcB operands, as the composite op's k S)
//   delta = (v_j - r) * beta                 exact fp32 on row 0
//   S'_ij = DS_i + k_i^T delta               exact fp32: k column and delta row broadcast into Dest, SFPU multiply-add
//   o_j = sum_i q_i S'_ij                    matmul
// The state is kept in bf16: its tiles unpacked through SrcA into the fp32 Dest (exact); S' packed fp32 to CB_SNEW
// (o's matmul input) and bf16 to CB_SOUT (the packer rounds), so the next step starts from the bf16 state.  The state
// never passes through SrcA / SrcB as a matmul operand.

#include <cstdint>
#include "api/debug/kernel_profiler.hpp"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/rows_sfpu.hpp"

namespace kda_step_compute {
// the circular buffers (recurrent_decode_step_program_factory.cpp)
constexpr uint32_t CB_S = 0;       // state in (bf16)
constexpr uint32_t CB_ECOL = 1;    // decay columns (exact, broadcast)
constexpr uint32_t CB_KCOL = 2;    // k columns (exact, broadcast)
constexpr uint32_t CB_KROW = 3;    // k rows (matmul)
constexpr uint32_t CB_QROW = 4;    // q rows (matmul)
constexpr uint32_t CB_VROW = 5;    // v rows (exact)
constexpr uint32_t CB_BROW = 6;    // beta in row 0 (exact)
constexpr uint32_t CB_DS_X = 16;   // decayed state (exact)
constexpr uint32_t CB_DS_M = 17;   // decayed state (matmul), CB_DS_X's memory
constexpr uint32_t CB_DELTA = 19;  // delta rows (exact, broadcast)
constexpr uint32_t CB_SNEW = 20;   // new state (fp32: o's matmul input)
constexpr uint32_t CB_SOUT = 21;   // new state out (bf16)
constexpr uint32_t CB_O = 22;      // q S' out

constexpr uint32_t N = 4;  // tiles per 128

// Dest[at] <- sum_i rows[i] @ col[i]: a [1 x 128] row (row 0 of 4 tiles) times a [128 x 32] column of tiles.
FORCE_INLINE void row_times_column(uint32_t rows, uint32_t col, uint32_t at = 0) {
    matmul_init(rows, col);
    for (uint32_t i = 0; i < N; ++i) {
        matmul_tiles(rows, col, i, i, at);
    }
}

FORCE_INLINE void pack_four(uint32_t cb) {
    for (uint32_t i = 0; i < N; ++i) {
        pack_tile<true>(i, cb, i);
    }
}

// One update of a unit: its state column from CB_S (popped), the pair's rows from the row CBs (held by the caller),
// S' to CB_SNEW (fp32, o's input) and CB_SOUT (bf16, the writer's), o to CB_O.
FORCE_INLINE void update() {
    CircularBuffer s(CB_S), vrow(CB_VROW);
    CircularBuffer ds_x(CB_DS_X), ds_m(CB_DS_M), delta(CB_DELTA), snew(CB_SNEW), sout(CB_SOUT), o(CB_O);
    // DS_i = S_ij * decay_i: Dest 0-3 the state tiles, Dest 4-7 the decay columns.  CB_DS_X and CB_DS_M share
    // their memory: one pack serves both.
    s.wait_front(N);
    {
    DeviceZoneScopedN("C-DS");
    ds_x.reserve_back(N);
    ds_m.reserve_back(N);
    tile_regs_acquire();
    unary_bcast_init<BroadcastType::COL>(CB_ECOL);
    for (uint32_t i = 0; i < N; ++i) {
        unary_bcast<BroadcastType::COL>(CB_ECOL, i, N + i);
    }
    reconfig_data_format_srca(CB_KROW, CB_S);  // (the bf16 state tiles; back to fp32 below)
    copy_init(CB_S);
    for (uint32_t i = 0; i < N; ++i) {
        copy_tile(CB_S, i, i);
    }
    reconfig_data_format_srca(CB_S, CB_KROW);
    mul_binary_tile_init();
    for (uint32_t i = 0; i < N; ++i) {
        mul_binary_tile(i, N + i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_four(CB_DS_X);
    tile_regs_release();
    ds_x.push_back(N);
    ds_m.push_back(N);
    s.pop_front(N);

    // r = sum_i k_i DS_i (Dest 0), then delta = (v_j - r) * beta on row 0
    {
    DeviceZoneScopedN("C-R");
    ds_m.wait_front(N);
    vrow.wait_front(1);
    delta.reserve_back(1);
    tile_regs_acquire();
    row_times_column(CB_KROW, CB_DS_M);
    copy_init(CB_VROW);
    copy_tile(CB_VROW, 0, 1);
    sub_binary_tile_init();
    kda_rows::sub<1>(1, 0, 0);  // row 0 only (rows_sfpu.hpp): the delta row
    copy_init(CB_BROW);
    copy_tile(CB_BROW, 0, 1);
    mul_binary_tile_init();
    kda_rows::mul<1>(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_DELTA);
    tile_regs_release();
    delta.push_back(1);
    ds_m.pop_front(N);
    vrow.pop_front(1);

    {
    DeviceZoneScopedN("C-SN");
    // S'_ij = DS_i + k_i^T delta: Dest 0 the delta row broadcast down, Dest 1-4 the k columns times it, plus the
    // decayed state (Dest 0, 5-7).
    delta.wait_front(1);
    ds_x.wait_front(N);
    snew.reserve_back(N);
    sout.reserve_back(N);
    tile_regs_acquire();
    unary_bcast_init<BroadcastType::ROW>(CB_DELTA);
    unary_bcast<BroadcastType::ROW>(CB_DELTA, 0, 0);
    unary_bcast_init<BroadcastType::COL>(CB_KCOL);
    for (uint32_t i = 0; i < N; ++i) {
        unary_bcast<BroadcastType::COL>(CB_KCOL, i, 1 + i);
    }
    mul_binary_tile_init();
    for (uint32_t i = 0; i < N; ++i) {
        mul_binary_tile(0, 1 + i, 1 + i);
    }
    copy_init(CB_DS_X);
    copy_tile(CB_DS_X, 0, 0);
    for (uint32_t i = 1; i < N; ++i) {
        copy_tile(CB_DS_X, i, N + i);
    }
    add_binary_tile_init();
    add_binary_tile(1, 0, 1);
    for (uint32_t i = 1; i < N; ++i) {
        add_binary_tile(1 + i, N + i, 1 + i);
    }
    tile_regs_commit();
    tile_regs_wait();
    // S' fp32 for o's matmul, then bf16 (the packer rounds) for the state
    for (uint32_t i = 0; i < N; ++i) {
        pack_tile<true>(1 + i, CB_SNEW, i);
    }
    pack_reconfig_data_format(CB_SNEW, CB_SOUT);
    for (uint32_t i = 0; i < N; ++i) {
        pack_tile<true>(1 + i, CB_SOUT, i);
    }
    pack_reconfig_data_format(CB_SOUT, CB_O);  // (the fp32 outputs after this)
    tile_regs_release();
    snew.push_back(N);
    sout.push_back(N);
    delta.pop_front(1);
    ds_x.pop_front(N);

    {
    DeviceZoneScopedN("C-O");
    // o_j = sum_i q_i S'_ij
    snew.wait_front(N);
    o.reserve_back(1);
    tile_regs_acquire();
    row_times_column(CB_QROW, CB_SNEW);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_O);
    tile_regs_release();
    o.push_back(1);
    snew.pop_front(N);
    }}}}  // the zones' blocks
}

FORCE_INLINE void run() {
    compute_kernel_hw_startup(CB_KROW, CB_DS_M, CB_DELTA);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);

    CircularBuffer ecol(CB_ECOL), kcol(CB_KCOL), krow(CB_KROW), qrow(CB_QROW), brow(CB_BROW);

    for (uint32_t u = start; u < start + count; ++u) {
        DeviceZoneScopedN("C-UNIT");  // zones to the end of the unit (a block each): the phases' times are differences
        const bool first = u == start || u % N == 0;
        const bool last = u + 1 == start + count || u % N == N - 1;
        if (first) {
            ecol.wait_front(N);
            krow.wait_front(N);
            brow.wait_front(1);
            kcol.wait_front(N);
            qrow.wait_front(N);
        }
        update();
        if (last) {
            ecol.pop_front(N);
            krow.pop_front(N);
            brow.pop_front(1);
            kcol.pop_front(N);
            qrow.pop_front(N);
        }
    }
}

}  // namespace kda_step_compute

void kernel_main() { kda_step_compute::run(); }
