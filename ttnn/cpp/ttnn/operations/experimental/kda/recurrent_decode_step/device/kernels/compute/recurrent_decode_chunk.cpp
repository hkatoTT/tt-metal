// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// recurrent_decode_step's chunked verify form (`saved`): a core's units (pair p, value block j;
// consecutive) a pair at a time, the pair's chunk of 2T rows in one tile's rows 0-15 (q at rows 16 + row): the last
// step's r tokens redone from the saved rows at rows T - r .. T - 1 (r = 1 .. T; the rows before them do nothing:
// decay 1, beta 0) from S0 (the state: the start of the last step), this step's T at rows T .. 2T - 1.  With
// G_p = prod_{i <= p} decay_i
// (per key channel), K+ = K G, K- = K / G, Q+ = Q G:
//   M = [K+ ; Q+] K-^T;  N = -strictly_lower(M) beta (rows);  F = lower(M) rows 16.. + I (rows 16..)
//   Tinv = (I + N)(I + N^2)(I + N^4)(I + N^8) (= (I + A)^-1, A strictly lower, N = -A);  Tb = Tinv beta (columns)
//   C = [-Tb K+ ; Q+] (E = I_bot - Tb times [K+ ; Q+])
// and per value block:
//   X1 = C S0 + [Tb ; 0] V           (rows 0-15 U, rows 16-31 Q+ S0)
//   O' = F X1                        (rows 16 + position: the tokens' o)
//   state after t = G_t (S0 + Kd_t^T X1)   (G_t: every column the channels' G at row t, exact in fp32; Kd_t^T:
//                                           K-^T with the rows after t zeroed; t = T - 1: this step's start, written
//                                           over the state, from which the next step redoes)
// The decay of S0 is applied in fp32 on the SFPU (the matrix engine truncates fp32 to tf32); the rest goes through
// the matrix engine.  The rows do not depend on r: the reader alone reads the keep codes.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/transpose.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"
#ifdef TRISC_MATH
#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"

namespace ckernel::sfpu {
// The column-wise (top to bottom) cumulative product of a tile in Dest, in two 16-row parts (rows 0-15, rows 16-31):
// calculate_cumsum's pattern (SFPTRANSP puts each of 4 rows in an LREG, the replayed chain multiplies them in order,
// LREG7 carrying to the next 4 rows) with SFPMUL for SFPADD (exact to fp32 rounding).
inline void calculate_kda_cumprod() {
#define KDA_CP_GROUP(R0, R1, R2, R3, OFF, REPLAY)                          \
    TTI_SFPLOAD(R0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, (OFF));       \
    TTI_SFPLOAD(R1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, (OFF) + 2);   \
    TTI_SFPLOAD(R2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, (OFF) + 16);  \
    TTI_SFPLOAD(R3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, (OFF) + 18);  \
    TTI_SFPTRANSP(0, 0, 0, 0);                                            \
    lltt::replay(REPLAY, 8);                                              \
    TTI_SFPTRANSP(0, 0, 0, 0);                                            \
    TTI_SFPSTORE(R0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, (OFF));      \
    TTI_SFPSTORE(R1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, (OFF) + 2);  \
    TTI_SFPSTORE(R2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, (OFF) + 16); \
    TTI_SFPSTORE(R3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, (OFF) + 18);
#define KDA_CP_ONES        \
    TTI_SFPMOV(0, 10, 4, 0); \
    TTI_SFPMOV(0, 10, 5, 0); \
    TTI_SFPMOV(0, 10, 6, 0); \
    TTI_SFPMOV(0, 10, 7, 0);
    KDA_CP_ONES
    KDA_CP_GROUP(0, 1, 2, 3, 0, 0)
    KDA_CP_GROUP(4, 5, 6, 7, 4, 8)
    KDA_CP_GROUP(0, 1, 2, 3, 8, 0)
    KDA_CP_GROUP(4, 5, 6, 7, 12, 8)
    KDA_CP_ONES
    KDA_CP_GROUP(0, 1, 2, 3, 32, 0)
    KDA_CP_GROUP(4, 5, 6, 7, 36, 8)
    KDA_CP_GROUP(0, 1, 2, 3, 40, 0)
    KDA_CP_GROUP(4, 5, 6, 7, 44, 8)
#undef KDA_CP_GROUP
#undef KDA_CP_ONES
}

template <bool APPROX>
inline void kda_cumprod_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    load_replay_buf(0, 16, [] {
        TTI_SFPMUL(7, 0, 9, 0, 0);  // L0 = L7 * L0 + 0
        TTI_SFPNOP;
        TTI_SFPMUL(0, 1, 9, 1, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(1, 2, 9, 2, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(2, 3, 9, 3, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(3, 4, 9, 4, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(4, 5, 9, 5, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(5, 6, 9, 6, 0);
        TTI_SFPNOP;
        TTI_SFPMUL(6, 7, 9, 7, 0);
        TTI_SFPNOP;
    });
}

// Rows 16-31 (faces 2, 3) of a tile in Dest set to 0 (the others kept)
inline void calculate_kda_zero_bottom() {
    TTI_SFPMOV(0, 9, 0, 0);  // L0 = 0 (LCONST_0)
#define KDA_Z4(A)                                                    \
    TTI_SFPSTORE(0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, (A));     \
    TTI_SFPSTORE(0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, (A) + 2); \
    TTI_SFPSTORE(0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, (A) + 4); \
    TTI_SFPSTORE(0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, (A) + 6);
    KDA_Z4(32)
    KDA_Z4(40)
    KDA_Z4(48)
    KDA_Z4(56)
#undef KDA_Z4
}
}  // namespace ckernel::sfpu
#endif

namespace {
constexpr uint32_t T = KDA_CHUNK_TOKENS;
// the powers N^2 .. of the inverse's product: 2T positions, N^(2^(POWERS + 1)) = 0
constexpr uint32_t POWERS = T <= 2 ? 1 : T <= 4 ? 2 : 3;
constexpr uint32_t N = 4;  // tiles per 128

// reader -> compute
constexpr uint32_t CB_KQ = 0;      // K rows 0-15, Q rows 16 + position (exact copies)
constexpr uint32_t CB_AL = 1;      // decay rows 0-15, again in rows 16-31 (exact copies)
constexpr uint32_t CB_BT = 2;      // beta: column 0 (rows), row 0 (rows)
constexpr uint32_t CB_MZ = 3;      // row 0: 1 on rows < K (slot 0's Kd)
constexpr uint32_t CB_V = 4;       // v rows of the batch's blocks (matmul)
constexpr uint32_t CB_S = 5;       // S0 tiles of the batch's blocks (bf16)
constexpr uint32_t CB_CONST = 6;   // -strictly lower 16, lower 16 at rows 16.., I, I at rows / columns 16.. (the
constexpr uint32_t CB_CONSTM = 7;  // reader's; CB_CONSTM: the same memory, matmul operands)
// compute's own
constexpr uint32_t CB_G = 8;      // G (copies, row broadcasts)
constexpr uint32_t CB_KQP = 9;    // [K+ ; Q+] (matmul)
constexpr uint32_t CB_KM = 10;    // K- (matmul, transposes)
constexpr uint32_t CB_M = 11;     // M (copies)
constexpr uint32_t CB_N = 12;     // N (matmul)
constexpr uint32_t CB_P = 13;     // N^2, N^4, N^8 (matmul)
constexpr uint32_t CB_Y = 14;     // the inverse's partial products (copies; CB_YM: the same memory, matmul)
constexpr uint32_t CB_YM = 15;
constexpr uint32_t CB_F = 16;     // F (matmul)
constexpr uint32_t CB_TB = 17;    // Tb (matmul)
constexpr uint32_t CB_E = 18;     // E (matmul)
constexpr uint32_t CB_C = 19;     // C (matmul)
constexpr uint32_t CB_KZ = 21;    // K-^T masked to the redone positions (matmul)
constexpr uint32_t CB_GC = 22;    // G_t columns at the last redone row (copies)
constexpr uint32_t CB_X1 = 23;    // X1 (matmul)
// compute -> writer
constexpr uint32_t CB_O = 24;   // O' (fp32)
constexpr uint32_t CB_SO = 25;  // the state's 4 tiles, then slot 0's 4 (the state's dtype)

constexpr uint32_t C_MASKA = 0, C_MASKF = 1, C_I = 2, C_IBOT = 3;

// (pack_n, copy_n, bcast_row_at: noclone too, one copy each in place of the compiler's constant-propagated clone per
// call site: about 3 KB less of the three TRISCs' binaries, so that the kernel config of this op and decode_prepare's
// fit the dispatcher's ring together)
__attribute__((noinline, noclone)) void pack_n(uint32_t d0, uint32_t n, uint32_t cb) {
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile<true>(d0 + i, cb, i);
    }
}

__attribute__((noinline, noclone)) void copy_n(uint32_t cb, uint32_t t0, uint32_t n, uint32_t d0) {
    copy_init(cb);
    for (uint32_t i = 0; i < n; ++i) {
        copy_tile(cb, t0 + i, d0 + i);
    }
}

// Row t of tile `tile` of cb (fp32) broadcast down Dest[dst]: the unpacker's read pointer moved t rows (64 B in faces
// 0 and 1) for the call
__attribute__((noinline, noclone)) void bcast_row_at(uint32_t cb, uint32_t tile, uint32_t t, uint32_t dst) {
    UNPACK((get_local_cb_interface(cb).fifo_rd_ptr += t * 4));
    unary_bcast<BroadcastType::ROW>(cb, tile, dst);
    UNPACK((get_local_cb_interface(cb).fifo_rd_ptr -= t * 4));
}

FORCE_INLINE void begin() { tile_regs_acquire(); }
FORCE_INLINE void end_pack() {
    tile_regs_commit();
    tile_regs_wait();
}

// The pair's shared rounds (P1-P9)
void pair_rounds() {
    CircularBuffer kq(CB_KQ), al(CB_AL), bt(CB_BT), mz(CB_MZ);
    CircularBuffer g(CB_G), kqp(CB_KQP), km(CB_KM), m(CB_M), n(CB_N), p(CB_P), y(CB_Y), ym(CB_YM), f(CB_F),
        tb(CB_TB), e(CB_E), c(CB_C), kz(CB_KZ), gc(CB_GC);

    {
    DeviceZoneScopedN("C-P1");
    // P1: G = the cumulative products of the decay rows (rows 0-15, and their copy in rows 16-31)
    al.wait_front(N);
    g.reserve_back(N);
    begin();
    copy_n(CB_AL, 0, N, 0);
    MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::cumsum>(ckernel::sfpu::kda_cumprod_init<false>)));
    for (uint32_t i = 0; i < N; ++i) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::calculate_kda_cumprod, i, VectorMode::RC_custom)));
    }
    end_pack();
    pack_n(0, N, CB_G);
    tile_regs_release();
    g.push_back(N);
    al.pop_front(N);

    }
    {
    DeviceZoneScopedN("C-P2");
    // P2: [K+ ; Q+] = [K ; Q] G, then K- = K / G (rows 16-31 zero)
    kq.wait_front(N);
    g.wait_front(N);
    kqp.reserve_back(N);
    begin();
    copy_n(CB_G, 0, N, 0);
    copy_n(CB_KQ, 0, N, N);
    mul_binary_tile_init();
    for (uint32_t i = 0; i < N; ++i) {
        mul_binary_tile(N + i, i, N + i);
    }
    end_pack();
    pack_n(N, N, CB_KQP);
    tile_regs_release();
    kqp.push_back(N);
    km.reserve_back(N);
    begin();
    copy_n(CB_G, 0, N, 0);
    recip_tile_init();
    for (uint32_t i = 0; i < N; ++i) {
        recip_tile(i);
    }
    copy_n(CB_KQ, 0, N, N);
    mul_binary_tile_init();
    for (uint32_t i = 0; i < N; ++i) {
        mul_binary_tile(i, N + i, i);
    }
    MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
    for (uint32_t i = 0; i < N; ++i) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::calculate_kda_zero_bottom, i, VectorMode::RC_custom)));
    }
    end_pack();
    pack_n(0, N, CB_KM);
    tile_regs_release();
    km.push_back(N);
    kq.pop_front(N);

    }
    {
    DeviceZoneScopedN("C-P34");
    // P3: M = [K+ ; Q+] K-^T
    kqp.wait_front(N);
    km.wait_front(N);
    m.reserve_back(1);
    begin();
    matmul_init(CB_KQP, CB_KM, 1);
    for (uint32_t i = 0; i < N; ++i) {
        matmul_tiles(CB_KQP, CB_KM, i, i, 0);
    }
    end_pack();
    pack_tile(0, CB_M);
    tile_regs_release();
    m.push_back(1);

    // P4: N = -strictly_lower(M) beta (rows); F = lower(M) at rows 16.. + I at rows 16..
    m.wait_front(1);
    bt.wait_front(1);
    n.reserve_back(1);
    f.reserve_back(1);
    begin();
    copy_init(CB_M);
    copy_tile(CB_M, 0, 0);
    copy_tile(CB_M, 0, 2);
    copy_init(CB_CONST);
    copy_tile(CB_CONST, C_MASKA, 1);
    copy_tile(CB_CONST, C_MASKF, 3);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    mul_binary_tile(2, 3, 2);
    unary_bcast_init<BroadcastType::COL>(CB_BT);
    unary_bcast<BroadcastType::COL>(CB_BT, 0, 1);
    copy_init(CB_CONST);
    copy_tile(CB_CONST, C_IBOT, 3);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    add_binary_tile_init();
    add_binary_tile(2, 3, 2);
    end_pack();
    pack_tile(0, CB_N);
    pack_tile(2, CB_F);
    tile_regs_release();
    n.push_back(1);
    f.push_back(1);
    m.pop_front(1);

    }
    {
    DeviceZoneScopedN("C-P5");
    // P5: Tinv = (I + N)(I + N^2)(I + N^4)(I + N^8)
    n.wait_front(1);
    p.reserve_back(POWERS);
    begin();
    matmul_init(CB_N, CB_N, 0);
    matmul_tiles(CB_N, CB_N, 0, 0, 0);
    end_pack();
    pack_tile(0, CB_P);
    tile_regs_release();
    p.push_back(1);
    for (uint32_t k = 1; k < POWERS; ++k) {
        p.wait_front(k);
        begin();
        matmul_init(CB_P, CB_P, 0);
        matmul_tiles(CB_P, CB_P, k - 1, k - 1, 0);
        end_pack();
        pack_tile(0, CB_P);
        tile_regs_release();
        p.push_back(1);
    }
    y.reserve_back(1);
    ym.reserve_back(1);
    begin();
    copy_init(CB_CONST);
    copy_tile(CB_CONST, C_I, 0);
    matmul_init(CB_N, CB_CONSTM, 0);
    matmul_tiles(CB_N, CB_CONSTM, 0, C_I, 0);  // I + N
    end_pack();
    pack_tile(0, CB_Y);
    tile_regs_release();
    y.push_back(1);
    ym.push_back(1);
    n.pop_front(1);
    p.wait_front(POWERS);
    for (uint32_t k = 0; k < POWERS; ++k) {
        y.wait_front(1);
        ym.wait_front(1);
        y.reserve_back(1);
        ym.reserve_back(1);
        begin();
        copy_init(CB_Y);
        copy_tile(CB_Y, 0, 0);
        matmul_init(CB_P, CB_YM, 0);
        matmul_tiles(CB_P, CB_YM, k, 0, 0);  // Y + P_k Y
        end_pack();
        pack_tile(0, CB_Y);
        tile_regs_release();
        y.push_back(1);
        ym.push_back(1);
        y.pop_front(1);
        ym.pop_front(1);
    }
    p.pop_front(POWERS);

    }
    {
    DeviceZoneScopedN("C-P67");
    // P6: Tb = Tinv beta (columns); E = I_bot - Tb
    y.wait_front(1);
    tb.reserve_back(1);
    e.reserve_back(1);
    begin();
    copy_init(CB_Y);
    copy_tile(CB_Y, 0, 0);
    unary_bcast_init<BroadcastType::ROW>(CB_BT);
    unary_bcast<BroadcastType::ROW>(CB_BT, 0, 1);
    copy_init(CB_CONST);
    copy_tile(CB_CONST, C_IBOT, 2);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    sub_binary_tile_init();
    sub_binary_tile(2, 0, 2);
    end_pack();
    pack_tile(0, CB_TB);
    pack_tile(2, CB_E);
    tile_regs_release();
    tb.push_back(1);
    e.push_back(1);
    y.pop_front(1);
    ym.pop_front(1);
    bt.pop_front(1);

    // P7: C = E [K+ ; Q+]
    e.wait_front(1);
    c.reserve_back(N);
    begin();
    matmul_init(CB_E, CB_KQP, 0);
    for (uint32_t i = 0; i < N; ++i) {
        matmul_tiles(CB_E, CB_KQP, 0, i, i);
    }
    end_pack();
    pack_n(0, N, CB_C);
    tile_regs_release();
    c.push_back(N);
    e.pop_front(1);
    kqp.pop_front(N);

    }
    {
    DeviceZoneScopedN("C-P8");
    // P8: K-^T (through SrcA: the matmuls truncate to tf32 anyway) masked to the redone positions
    mz.wait_front(1);
    kz.reserve_back(N);
    begin();
    transpose_init(CB_KM);
    for (uint32_t i = 0; i < N; ++i) {
        transpose_tile(CB_KM, i, i);
    }
    unary_bcast_init<BroadcastType::ROW>(CB_MZ);
    for (uint32_t i = 0; i < N; ++i) {
        unary_bcast<BroadcastType::ROW>(CB_MZ, 0, N + i);
    }
    mul_binary_tile_init();
    for (uint32_t i = 0; i < N; ++i) {
        mul_binary_tile(N + i, i, N + i);
    }
    end_pack();
    pack_n(N, N, CB_KZ);
    tile_regs_release();
    kz.push_back(N);
    km.pop_front(N);
    mz.pop_front(1);

    }
    {
    DeviceZoneScopedN("C-P9");
    // P9: the G_t columns (every column G at row t, rows the channels; exact) at t = T - 1, the last redone row (the
    // unpacker's read pointer moved to the row)
    gc.reserve_back(N);
    begin();
    unary_bcast_init<BroadcastType::ROW>(CB_G);
    for (uint32_t i = 0; i < N; ++i) {
        bcast_row_at(CB_G, i, T - 1, i);
    }
    transpose_dest_init<true>();
    for (uint32_t i = 0; i < N; ++i) {
        transpose_dest<true>(i);
    }
    end_pack();
    pack_n(0, N, CB_GC);
    tile_regs_release();
    gc.push_back(N);
    g.pop_front(N);
    }
}

// A value block's rounds: X1, O', the step's start state
void block_rounds() {
    CircularBuffer v(CB_V), s(CB_S), x1(CB_X1), o(CB_O), so(CB_SO);
    v.wait_front(1);
    s.wait_front(N);
    // X1 = C S0 + Tb V
    x1.reserve_back(1);
    begin();
    reconfig_data_format(CB_S, CB_C);
    matmul_init(CB_C, CB_S, 0);
    for (uint32_t i = 0; i < N; ++i) {
        matmul_tiles(CB_C, CB_S, i, i, 0);
    }
    reconfig_data_format(CB_V, CB_TB);
    matmul_init(CB_TB, CB_V, 0);
    matmul_tiles(CB_TB, CB_V, 0, 0, 0);
    end_pack();
    pack_tile(0, CB_X1);
    tile_regs_release();
    x1.push_back(1);
    v.pop_front(1);
    // O' = F X1
    x1.wait_front(1);
    o.reserve_back(1);
    begin();
    matmul_init(CB_F, CB_X1, 0);
    matmul_tiles(CB_F, CB_X1, 0, 0, 0);
    end_pack();
    pack_tile(0, CB_O);
    tile_regs_release();
    o.push_back(1);
    // the state after t = G_t (S0 + Kd_t^T X1) at t = T - 1 (Kd: K-^T masked to the redone rows)
    {
        const uint32_t kd = CB_KZ;
        so.reserve_back(N);
        begin();
        reconfig_data_format_srca(CB_X1, CB_S);
        copy_init(CB_S);
        for (uint32_t i = 0; i < N; ++i) {
            copy_tile(CB_S, i, i);
        }
        reconfig_data_format_srca(CB_S, CB_X1);
        matmul_init(kd, CB_X1, 0);
        for (uint32_t i = 0; i < N; ++i) {
            matmul_tiles(kd, CB_X1, i, 0, i);
        }
        copy_n(CB_GC, 0, N, N);
        mul_binary_tile_init();
        for (uint32_t i = 0; i < N; ++i) {
            mul_binary_tile(i, N + i, i);
        }
        end_pack();
        pack_reconfig_data_format(CB_X1, CB_SO);  // (the bf16 state tiles; back to fp32 after)
        pack_n(0, N, CB_SO);
        pack_reconfig_data_format(CB_SO, CB_X1);
        tile_regs_release();
        so.push_back(N);
    }
    x1.pop_front(1);
    s.pop_front(N);
}
}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_KQP, CB_KM, CB_G);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    CircularBuffer consts(CB_CONST), constm(CB_CONSTM), bt(CB_BT);
    CircularBuffer f(CB_F), tb(CB_TB), c(CB_C), kz(CB_KZ), gc(CB_GC);
    consts.wait_front(4);
    constm.wait_front(4);
    for (uint32_t u = start; u < start + count;) {
        const uint32_t j0 = u % N;
        const uint32_t nb = N - j0 < start + count - u ? N - j0 : start + count - u;
        bt.wait_front(1);
        pair_rounds();
        f.wait_front(1);
        tb.wait_front(1);
        c.wait_front(N);
        kz.wait_front(N);
        gc.wait_front(N);
        {
            DeviceZoneScopedN("C-BLK");
            for (uint32_t b = 0; b < nb; ++b) {
                block_rounds();
            }
        }
        f.pop_front(1);
        tb.pop_front(1);
        c.pop_front(N);
        kz.pop_front(N);
        gc.pop_front(N);
        u += nb;
    }
}
