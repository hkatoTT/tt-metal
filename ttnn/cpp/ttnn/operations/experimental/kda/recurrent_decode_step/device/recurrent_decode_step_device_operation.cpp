// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "recurrent_decode_step_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::experimental::prim {

namespace {
constexpr uint32_t DIM = 128;  // key and value head dim

void check_device_tile(const Tensor& t, const char* name, tt::tt_metal::DataType dtype, bool l1_ok = false) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE && t.is_allocated(), "{} must be an allocated device tensor", name);
    TT_FATAL(
        t.dtype() == dtype, "{} must be {}", name, dtype == tt::tt_metal::DataType::FLOAT32 ? "FLOAT32" : "BFLOAT16");
    TT_FATAL(t.layout() == tt::tt_metal::Layout::TILE, "{} must be TILE layout", name);
    TT_FATAL(!t.memory_config().is_sharded(), "{} must be interleaved", name);
    TT_FATAL(
        t.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM ||
            (l1_ok && t.memory_config().buffer_type() == tt::tt_metal::BufferType::L1),
        "{} must be in DRAM{} (rows are read and written in 64-byte pieces)",
        name,
        l1_ok ? " or L1" : "");
}

void check_device_f32_tile(const Tensor& t, const char* name, bool l1_ok = false) {
    check_device_tile(t, name, tt::tt_metal::DataType::FLOAT32, l1_ok);
}

// a state tensor: BFLOAT16, like the state
void check_state_tile(const Tensor& t, const Tensor& state, const char* name) {
    check_device_tile(t, name, tt::tt_metal::DataType::BFLOAT16);
    TT_FATAL(t.padded_shape() == state.padded_shape(), "{} must have the state's shape", name);
}
}  // namespace

void RecurrentDecodeStepDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& state = tensor_args.state;
    check_state_tile(state, state, "state");
    TT_FATAL(
        state.logical_shape()[-1] == DIM && state.logical_shape()[-2] == DIM,
        "state must be [.., {}, {}], got {}",
        DIM,
        DIM,
        state.logical_shape());
    TT_FATAL(state.physical_volume() == args.pairs * DIM * DIM, "state must hold one state per pair");
    const bool chunk = tensor_args.saved.has_value();
    const uint32_t rows_in = args.tokens * args.pairs;  // (the chunked form: the tokens' rows)
    const bool l1_ok = chunk;  // (the chunked form: q .. beta and o may be L1 interleaved)
    for (const auto* t : {&tensor_args.q, &tensor_args.k, &tensor_args.v, &tensor_args.decay}) {
        check_device_f32_tile(*t, "q / k / v / decay", l1_ok);
        TT_FATAL(t->logical_shape()[-1] == DIM, "q / k / v / decay must be [.., P, {}]", DIM);
        TT_FATAL(t->logical_shape()[-2] == rows_in, "q / k / v / decay must have one row per pair");
        TT_FATAL(t->logical_volume() == rows_in * DIM, "q / k / v / decay must be [P, {}] (leading dims 1)", DIM);
    }
    const auto& beta = tensor_args.beta;
    check_device_f32_tile(beta, "beta", l1_ok);
    TT_FATAL(beta.logical_shape()[-2] == rows_in, "beta must have one row per pair");
    TT_FATAL(
        beta.padded_shape()[-1] == tt::constants::TILE_WIDTH, "beta must be one tile wide (column 0 is used)");
    TT_FATAL(
        beta.physical_volume() == tt::round_up(rows_in, tt::constants::TILE_HEIGHT) * tt::constants::TILE_WIDTH,
        "beta must be [P, 1] (leading dims 1)");
    TT_FATAL(state.device() == tensor_args.q.device(), "all tensors must be on the same device");
    if (tensor_args.output.has_value()) {
        check_device_f32_tile(tensor_args.output.value(), "output", l1_ok);
        TT_FATAL(
            tensor_args.output->logical_shape()[-2] == rows_in && tensor_args.output->logical_shape()[-1] == DIM,
            "output must be [.., {}, {}]",
            rows_in,
            DIM);
    }
    if (!chunk) {  // the plain step
        TT_FATAL(args.tokens == 1, "the plain step: one token a pair (more: the chunked verify form, `saved`)");
        TT_FATAL(
            !tensor_args.source.has_value() && !tensor_args.keep.has_value() && !tensor_args.saved_source.has_value() &&
                !args.pair_major,
            "source, keep, saved_source, pair_major: the chunked verify form (saved) only");
        TT_FATAL(
            (args.form & ~(128u | 256u | 1024u)) == 0,
            "form: bits 128, 256, 1024 (the plain step's units by grid row speed)");
        return;
    }
    // the chunked verify form
    TT_FATAL(tensor_args.keep.has_value(), "the chunked verify form takes keep with saved");
    if (tensor_args.source.has_value()) {  // (out of place: the last step's start state from it)
        check_state_tile(tensor_args.source.value(), state, "source");
    }
    TT_FATAL(args.tokens >= 2 && args.tokens <= 8, "the chunked form: 2-8 tokens");
    TT_FATAL(args.form == 0, "the chunked verify form: no form bits");
    const auto& f = tensor_args.keep.value();
    TT_FATAL(f.storage_type() == StorageType::DEVICE && f.is_allocated(), "keep must be an allocated device tensor");
    TT_FATAL(f.dtype() == tt::tt_metal::DataType::UINT32, "keep: UINT32");
    TT_FATAL(f.layout() == tt::tt_metal::Layout::ROW_MAJOR, "keep must be ROW_MAJOR");
    TT_FATAL(
        !f.memory_config().is_sharded() && f.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
        "keep must be DRAM interleaved");
    TT_FATAL(args.heads > 0 && args.pairs % args.heads == 0, "keep needs the heads a user, dividing the pairs");
    TT_FATAL(f.logical_volume() == args.pairs / args.heads, "keep must be [1, B]");
    const auto& sv = tensor_args.saved.value();
    TT_FATAL(sv.storage_type() == StorageType::DEVICE && sv.is_allocated(), "saved must be an allocated device tensor");
    TT_FATAL(sv.dtype() == tt::tt_metal::DataType::FLOAT32 && sv.layout() == tt::tt_metal::Layout::TILE,
             "saved must be FLOAT32 TILE");
    TT_FATAL(sv.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM && !sv.memory_config().is_sharded(),
             "saved must be DRAM interleaved");
    const uint32_t slot_tiles = (1536 * args.tokens + 64 + 4095) / 4096;  // a unit's slot: whole fp32 tiles
    TT_FATAL(sv.physical_volume() == uint64_t{args.pairs} * 4 * slot_tiles * 1024,
             "saved must hold {} fp32 tiles a unit (pairs x 4 units; got {})", slot_tiles, sv.logical_shape());
    if (tensor_args.saved_source.has_value()) {  // (out of place: the last step's rows from it)
        const auto& ss = tensor_args.saved_source.value();
        TT_FATAL(ss.storage_type() == StorageType::DEVICE && ss.is_allocated() &&
                     ss.dtype() == sv.dtype() && ss.layout() == sv.layout() &&
                     ss.padded_shape() == sv.padded_shape() && ss.memory_config() == sv.memory_config(),
                 "saved_source must be like saved");
    }
}

RecurrentDecodeStepDeviceOperation::spec_return_value_t RecurrentDecodeStepDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output->tensor_spec();
    }
    auto shape = tensor_args.q.logical_shape();
    if (tensor_args.saved.has_value()) {
        shape = ttnn::Shape({1, 1, args.tokens * args.pairs, DIM});
    }
    return tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tt::tt_metal::MemoryConfig{
                tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
}

RecurrentDecodeStepDeviceOperation::tensor_return_value_t RecurrentDecodeStepDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output.value();
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.q.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor kda_recurrent_decode_step(
    const Tensor& state,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& decay,
    const Tensor& beta,
    const std::optional<Tensor>& source,
    const std::optional<Tensor>& keep,
    uint32_t heads,
    const std::optional<Tensor>& output,
    uint32_t form,
    uint32_t tokens,
    const std::optional<Tensor>& saved,
    bool pair_major,
    const std::optional<Tensor>& saved_source) {
    using OperationType = ttnn::experimental::prim::RecurrentDecodeStepDeviceOperation;
    // the chunked verify form (saved): `tokens` tokens a pair, q .. beta holding them all (token k's P rows from k P on,
    // or with pair_major pair p's T rows from p T on)
    const uint32_t t = saved.has_value() ? tokens : 1;
    TT_FATAL(!saved.has_value() || (tokens >= 2 && tokens <= 8), "the chunked verify form: tokens 2 .. 8");
    TT_FATAL(saved.has_value() || tokens <= 1, "tokens: the chunked verify form (saved) only");
    const uint32_t pairs = q.logical_shape()[-2] / t;
    auto attrs = OperationType::operation_attributes_t{pairs, keep.has_value() ? heads : 0, t, form, pair_major};
    auto targs = OperationType::tensor_args_t{state, q, k, v, decay, beta, source, keep, output, saved, saved_source};
    return ttnn::device_operation::launch<OperationType>(attrs, targs);
}

}  // namespace ttnn::prim
