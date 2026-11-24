// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bfp_utils.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/bfloat2.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::bfp_utils {

namespace nb = nanobind;

// Pack float32 data into BFP tiles, return raw uint32 packed data as numpy array.
// Input accepts any array_api-compatible source (numpy, torch, jax, etc.)
template <typename PackFn>
static nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>> pack_impl(
    const PackFn& pack_fn,
    const nb::ndarray<nb::array_api, const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>& input,
    bool row_major_input,
    bool is_exp_a) {
    ttsl::Span<const float> data_span(input.data(), input.size());
    auto packed = pack_fn(data_span, row_major_input, is_exp_a, std::nullopt);

    auto* result = new uint32_t[packed.size()];
    std::copy(packed.begin(), packed.end(), result);

    nb::capsule owner(result, [](void* p) noexcept { delete[] static_cast<uint32_t*>(p); });
    size_t shape[] = {packed.size()};
    return nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>>(result, 1, shape, std::move(owner));
}

// Unpack raw uint32 BFP tile data back to float32 numpy array.
// Input accepts any array_api-compatible source (numpy, torch, jax, etc.)
template <typename UnpackFn>
static nb::ndarray<nb::numpy, float, nb::ndim<1>> unpack_impl(
    const UnpackFn& unpack_fn,
    const nb::ndarray<nb::array_api, const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>& input,
    bool row_major_output,
    bool is_exp_a) {
    ttsl::Span<const uint32_t> data_span(input.data(), input.size());
    auto unpacked = unpack_fn(data_span, row_major_output, is_exp_a, std::nullopt);

    auto* result = new float[unpacked.size()];
    std::copy(unpacked.begin(), unpacked.end(), result);

    nb::capsule owner(result, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    size_t shape[] = {unpacked.size()};
    return nb::ndarray<nb::numpy, float, nb::ndim<1>>(result, 1, shape, std::move(owner));
}

void py_module(nb::module_& mod) {
    mod.def(
        "pack_bfp8",
        [](const nb::ndarray<nb::array_api, const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>& input,
           bool row_major_input,
           bool is_exp_a) { return pack_impl(pack_as_bfp8_tiles<float>, input, row_major_input, is_exp_a); },
        nb::arg("input"),
        nb::arg("row_major_input") = false,
        nb::arg("is_exp_a") = false,
        R"doc(Pack float32 data into BFP8 tile format. Returns raw uint32 packed data.)doc");

    mod.def(
        "pack_bfp4",
        [](const nb::ndarray<nb::array_api, const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>& input,
           bool row_major_input,
           bool is_exp_a) { return pack_impl(pack_as_bfp4_tiles<float>, input, row_major_input, is_exp_a); },
        nb::arg("input"),
        nb::arg("row_major_input") = false,
        nb::arg("is_exp_a") = false,
        R"doc(Pack float32 data into BFP4 tile format. Returns raw uint32 packed data.)doc");

    mod.def(
        "pack_bfp2",
        [](const nb::ndarray<nb::array_api, const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>& input,
           bool row_major_input,
           bool is_exp_a) {
            return pack_impl(tt::tt_metal::pack_as_bfp2_tiles<float>, input, row_major_input, is_exp_a);
        },
        nb::arg("input"),
        nb::arg("row_major_input") = false,
        nb::arg("is_exp_a") = false,
        R"doc(Pack float32 data into BFP2 tile format. Returns raw uint32 packed data.)doc");

    mod.def(
        "unpack_bfp8",
        [](const nb::ndarray<nb::array_api, const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>& input,
           bool row_major_output,
           bool is_exp_a) { return unpack_impl(unpack_bfp8_tiles_into_float_vec, input, row_major_output, is_exp_a); },
        nb::arg("input"),
        nb::arg("row_major_output") = false,
        nb::arg("is_exp_a") = false,
        R"doc(Unpack raw BFP8 tile data back to float32.)doc");

    mod.def(
        "unpack_bfp4",
        [](const nb::ndarray<nb::array_api, const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>& input,
           bool row_major_output,
           bool is_exp_a) { return unpack_impl(unpack_bfp4_tiles_into_float_vec, input, row_major_output, is_exp_a); },
        nb::arg("input"),
        nb::arg("row_major_output") = false,
        nb::arg("is_exp_a") = false,
        R"doc(Unpack raw BFP4 tile data back to float32.)doc");

    mod.def(
        "unpack_bfp2",
        [](const nb::ndarray<nb::array_api, const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>& input,
           bool row_major_output,
           bool is_exp_a) {
            return unpack_impl(tt::tt_metal::unpack_bfp2_tiles_into_float_vec, input, row_major_output, is_exp_a);
        },
        nb::arg("input"),
        nb::arg("row_major_output") = false,
        nb::arg("is_exp_a") = false,
        R"doc(Unpack raw BFP2 tile data back to float32.)doc");

    mod.def(
        "pack_transposed_slices",
        [](const std::vector<nb::ndarray<nb::ndim<2>, nb::device::cpu>>& slices,
           int dim,
           tt::tt_metal::DataType dtype) -> ttnn::Tensor {
            TT_FATAL(!slices.empty(), "pack_transposed_slices: no slices");
            TT_FATAL(
                dtype == tt::tt_metal::DataType::BFLOAT8_B || dtype == tt::tt_metal::DataType::BFLOAT4_B,
                "pack_transposed_slices: BFLOAT8_B or BFLOAT4_B, not {}",
                dtype);
            const size_t rows = slices[0].shape(0);
            const size_t cols = slices[0].shape(1);
            std::vector<const uint16_t*> bases;
            std::vector<size_t> strides;
            bases.reserve(slices.size());
            strides.reserve(slices.size());
            for (const auto& s : slices) {
                TT_FATAL(
                    s.dtype().code == static_cast<uint8_t>(nb::dlpack::dtype_code::Bfloat) && s.dtype().bits == 16 &&
                        s.dtype().lanes == 1,
                    "pack_transposed_slices: bfloat16 slices only");
                TT_FATAL(
                    s.shape(0) == rows && s.shape(1) == cols && s.stride(1) == 1 && s.stride(0) >= 0 &&
                        static_cast<size_t>(s.stride(0)) >= cols,
                    "pack_transposed_slices: each slice [{}, {}] with unit column stride and rows at least that wide",
                    rows,
                    cols);
                bases.push_back(static_cast<const uint16_t*>(s.data()));
                strides.push_back(static_cast<size_t>(s.stride(0)));
            }
            std::vector<uint32_t> packed;
            {
                nb::gil_scoped_release release;  // (the slices stay alive in `slices`)
                packed = dtype == tt::tt_metal::DataType::BFLOAT8_B
                             ? pack_transposed_slices_as_bfp8_tiles(bases, strides, rows, cols, dim)
                             : pack_transposed_slices_as_bfp4_tiles(bases, strides, rows, cols, dim);
            }
            const uint32_t out_rows = dim == 1 ? cols : slices.size() * cols;
            const uint32_t out_cols = dim == 1 ? slices.size() * rows : rows;
            // the spec from_torch gives a [1, 1, out_rows, out_cols] tensor of this dtype in tiles (default tile and
            // memory config), as the one-pass row-major packing builds it (host_tensor_factory.cpp)
            const auto shape = tt::tt_metal::Shape({1, 1, out_rows, out_cols});
            const tt::tt_metal::TensorSpec requested(
                shape,
                tt::tt_metal::TensorLayout(
                    dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), tt::tt_metal::MemoryConfig{}));
            auto spec = tt::tt_metal::TensorSpec(
                requested.logical_shape(),
                tt::tt_metal::TensorLayout(
                    dtype,
                    tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE, requested.tile()),
                    requested.memory_config(),
                    requested.tensor_layout().get_alignment()));
            TT_FATAL(
                packed.size() * sizeof(uint32_t) == spec.compute_packed_buffer_size_bytes(),
                "pack_transposed_slices: packed {} bytes, the spec {}",
                packed.size() * sizeof(uint32_t),
                spec.compute_packed_buffer_size_bytes());
            return ttnn::Tensor(
                tt::tt_metal::HostTensor::from_buffer(tt::tt_metal::HostBuffer(std::move(packed)), std::move(spec)));
        },
        nb::arg("slices"),
        nb::arg("dim"),
        nb::arg("dtype"),
        R"doc(The concatenation of transposed bfloat16 slices ([rows, cols] each, row-major with any row stride) as a host
        tensor [1, 1, ..] of block-float tiles, packed in one pass: the bytes from_torch(torch.cat([s.T for s in
        slices], dim)[None, None], dtype, TILE) gives.  dim 1: [cols, n rows]; dim 0: [n cols, rows].)doc");

    mod.def("get_l1_alignment", &tt::tt_metal::hal::get_l1_alignment, R"doc(Get L1 memory alignment in bytes.)doc");
    mod.def(
        "get_dram_alignment", &tt::tt_metal::hal::get_dram_alignment, R"doc(Get DRAM memory alignment in bytes.)doc");

    mod.def(
        "compute_shard_page_mapping",
        [](std::vector<uint32_t> tensor_shape_vec,
           std::vector<uint32_t> shard_shape_vec,
           uint32_t page_height,
           uint32_t page_width,
           const tt::tt_metal::CoreRangeSet& core_range_set,
           tt::tt_metal::ShardOrientation shard_orientation,
           tt::tt_metal::ShardDistributionStrategy distribution_strategy)
            -> std::vector<std::pair<tt::tt_metal::CoreCoord, std::vector<uint32_t>>> {
            auto spec = tt::tt_metal::BufferDistributionSpec::from_shard_spec(
                tt::tt_metal::Shape(tensor_shape_vec),
                tt::tt_metal::Shape(shard_shape_vec),
                tt::tt_metal::Shape2D(page_height, page_width),
                core_range_set,
                shard_orientation,
                distribution_strategy);
            auto mapping = spec.compute_page_mapping();
            std::vector<std::pair<tt::tt_metal::CoreCoord, std::vector<uint32_t>>> result;
            result.reserve(mapping.all_cores.size());
            for (size_t i = 0; i < mapping.all_cores.size(); i++) {
                std::vector<uint32_t> pages;
                pages.reserve(mapping.core_host_page_indices[i].size());
                for (auto p : mapping.core_host_page_indices[i]) {
                    if (p != tt::tt_metal::UncompressedBufferPageMapping::PADDING) {
                        pages.push_back(p);
                    }
                }
                result.emplace_back(mapping.all_cores[i], std::move(pages));
            }
            return result;
        },
        nb::arg("tensor_shape"),
        nb::arg("shard_shape"),
        nb::arg("page_height"),
        nb::arg("page_width"),
        nb::arg("core_range_set"),
        nb::arg("shard_orientation"),
        nb::arg("distribution_strategy") = tt::tt_metal::ShardDistributionStrategy::ROUND_ROBIN_1D,
        R"doc(Compute shard-to-core page mapping using BufferDistributionSpec.

Returns list of (CoreCoord, page_indices) tuples, one per core.
Each page_index is a host page ID (for TILE_LAYOUT, page = tile).
Use ROUND_ROBIN_1D for HEIGHT/WIDTH sharding, GRID_2D for BLOCK sharding.
)doc");
}

}  // namespace ttnn::bfp_utils
