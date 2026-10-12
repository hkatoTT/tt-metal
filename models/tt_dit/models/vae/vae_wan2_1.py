# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from diffusers.models import AutoencoderKLWan
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole

from ...layers.linear import Linear
from ...layers.module import Module, ModuleList, Parameter
from ...layers.normalization import RMSNorm
from ...parallel.config import VaeHWParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache
from ...utils.conv3d import (
    ConvDims,
    _ntuple,
    aligned_channels,
    compute_decoder_dims,
    compute_encoder_dims,
    conv3d_blocking_hash,
    conv_pad_height,
    conv_pad_in_channels,
    conv_pad_width,
    count_convs,
    get_conv3d_config,
)
from ...utils.substate import pop_substate, rename_substate
from ...utils.tensor import (
    fast_device_to_host,
    float_to_uint8,
    float_to_unit_range,
    local_device_to_torch,
    typed_tensor,
    typed_tensor_2dshard,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

CACHE_T = 2


# Keep the residual stream of the residual blocks without a 1x1 shortcut row major (RMSNorm and the residual add
# read row major directly, bit-identical): drops the untilize after norm1 and the tilize before the residual add of
# every such block, and the layout round trip after the upsamplers. WAN_VAE_RM_RESID=0 restores the tiled stream.
_RM_RESID = os.environ.get("WAN_VAE_RM_RESID", "1") == "1"
# Causal 3x3x3 convs keep their temporal cache already padded: the last 2 frames of the neighbor-pad output
# (halos included) are the next chunk's front frames, which neighbor pad copies in place of zeros (bit-identical).
# Drops the cache concat (a copy of every chunk input) and the unpadded cache slices. WAN_VAE_NP_CACHE=0 restores
# the concat path.
_NP_CACHE = os.environ.get("WAN_VAE_NP_CACHE", "1") == "1"


# The residual add of the row-major residual blocks is done inside conv2 (conv3d residual_tensor: the writer reads
# the residual rows, compute adds them to the bf16 conv result before the untilize; bit-identical). Only for conv2
# configs with one C_in block and full-block output (see WanCausalConv3d.can_fuse_residual). WAN_VAE_C3D_RESID=0
# restores the separate add.
def _c3d_resid() -> bool:
    return os.environ.get("WAN_VAE_C3D_RESID", "1") == "1"


# Also the conv2 with several C_in blocks (fp32 partials, rounded to bf16 before the add).
def _c3d_resid_all() -> bool:
    return os.environ.get("WAN_VAE_C3D_RESID_ALL", "1") == "1"


# EXPERIMENT (hkato): a row-major RMSNorm in front of a padded-cache conv writes straight into the interior of the
# neighbor-pad buffer, and neighbor pad then fills only halos, borders and front frames (no interior copy).
_NP_NORM = os.environ.get("WAN_VAE_NP_NORM", "1") != "0"
# Mid-block attention: devices holding the same frame no longer all compute every query of it. With the frames
# split over the W axis, each H device computes only its own H rows (queries partitioned over the H axis, keys and
# values whole), so the output is already H-sharded (no H re-partition, a 4x smaller W gather); a single frame
# (no T split) is computed by each device for its own H x W shard only. Bit-identical. WAN_VAE_ATTN_QSPLIT=0
# restores the replicated queries.
_ATTN_QSPLIT = os.environ.get("WAN_VAE_ATTN_QSPLIT", "1") == "1"
# With frames split over the W axis, move each frame to its device with an all-to-all over W and gather it over H,
# and send the output back with an all-to-all (instead of gathering all frames everywhere): same values.
_ATTN_A2A = os.environ.get("WAN_VAE_ATTN_A2A", "1") == "1"


def _get_w_mask(cache, x_BTHWC, logical_w, parallel_config, mesh_device, dtype):
    """
    Return a cached mask that zeros width-padding columns beyond logical_w.
    """
    sharded_w = x_BTHWC.shape[3]
    key = (sharded_w, logical_w)
    if key not in cache:
        padded_w = sharded_w * parallel_config.width_parallel.factor
        mask = torch.ones(1, 1, 1, padded_w, 1)
        mask[:, :, :, logical_w:, :] = 0.0
        cache[key] = typed_tensor(
            mask,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_axis=parallel_config.width_parallel.mesh_axis,
            shard_dim=3,
            dtype=dtype,
        )
    return cache[key]


def conv3d_to_linear_weight(state):
    weight = state["weight"]
    out_c, in_c, kt, kh, kw = weight.shape
    assert kt == kh == kw == 1
    weight = weight.reshape(out_c, in_c)
    padded_out_c = aligned_channels(out_c)
    padded_in_c = aligned_channels(in_c)
    weight = torch.nn.functional.pad(weight, (0, padded_in_c - in_c, 0, padded_out_c - out_c))
    bias = state["bias"]
    bias = torch.nn.functional.pad(bias, (0, padded_out_c - out_c))
    state["weight"] = weight
    state["bias"] = bias
    return state


class WanAttentionBlock(Module):
    def __init__(
        self,
        *,
        dim: int,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
        sdpa_t_fracture_w_only: bool = False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.sdpa_t_fracture_w_only = (
            sdpa_t_fracture_w_only or os.environ.get("WAN_VAE_SDPA_W_ONLY") == "1"
        )  # EXPERIMENT (hkato)

        self.norm = RMSNorm(
            embedding_dim=dim,
            norm_eps=1e-12,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.to_qkv = Linear(
            in_features=dim,
            out_features=dim * 3,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.proj = Linear(
            in_features=dim,
            out_features=dim,
            mesh_device=mesh_device,
            dtype=dtype,
        )

        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=32,
            k_chunk_size=256,
            exp_approx_mode=False,  # NOTE: False is more correct
        )
        self.hifi4_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        def permute_conv2d_weights(weight):
            out_c, in_c, kh, kw = weight.shape
            assert kh == kw == 1
            weight = weight.permute(0, 2, 3, 1).reshape(out_c, in_c)
            return weight

        if "to_qkv.weight" in state:
            state["to_qkv.weight"] = permute_conv2d_weights(state["to_qkv.weight"])

        if "proj.weight" in state:
            state["proj.weight"] = permute_conv2d_weights(state["proj.weight"])

        if "norm.gamma" in state:
            state["norm.weight"] = state.pop("norm.gamma").squeeze()

    def _a2a(self, x: ttnn.Tensor, *, in_dim: int, out_dim: int, mesh_axis: int) -> ttnn.Tensor:
        return ttnn.experimental.all_to_all_async_generic(
            x,
            in_dim=in_dim,
            out_dim=out_dim,
            num_links=self.ccl_manager.links_for_axis(mesh_axis),
            topology=self.ccl_manager.topology,
            cluster_axis=mesh_axis,
        )

    def _forward_a2a(self, x_BTHWC: ttnn.Tensor, logical_h: int, logical_w: int) -> ttnn.Tensor | None:
        """The qsplit path (one frame per W device, queries split over H) with all-to-all moves; None if it does
        not apply."""
        B, T, h, w, C = x_BTHWC.shape
        pc = self.parallel_config
        hf, wf = pc.height_parallel.factor, pc.width_parallel.factor
        h_axis, w_axis = pc.height_parallel.mesh_axis, pc.width_parallel.mesh_axis
        H, W, BT = h * hf, w * wf, B * T
        if not (
            _ATTN_QSPLIT
            and self.sdpa_t_fracture_w_only
            and T > 1
            and hf > 1
            and wf > 1
            and BT <= wf
            and H <= logical_h
            and (logical_w == 0 or W <= logical_w)
        ):
            return None
        residual_BTHWC = x_BTHWC
        x = ttnn.reshape(ttnn.to_layout(x_BTHWC, ttnn.ROW_MAJOR_LAYOUT), (BT, 1, h * w, C))
        if wf > BT:
            x = ttnn.pad(x, [(0, wf - BT), (0, 0), (0, 0), (0, 0)], value=0.0)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        # frame f -> W device f: (1, wf, h * w, C), the W blocks of this device's frame
        x = self._a2a(x, in_dim=1, out_dim=0, mesh_axis=w_axis)
        x = self.ccl_manager.all_gather_persistent_buffer(x, dim=0, mesh_axis=h_axis)  # (hf, wf, h * w, C)
        x = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (hf, wf, h, w * C))
        x = ttnn.permute(x, (0, 2, 1, 3))  # raster order: (hf, h, wf, w * C)
        x_TNC = ttnn.to_layout(ttnn.reshape(x, (1, H * W, C)), ttnn.TILE_LAYOUT)

        x_TNC = self.norm(x_TNC, compute_kernel_config=self.hifi4_compute_kernel_config)
        default_block_size = (2, 2, 2) if x_TNC.dtype == ttnn.float32 else (8, 8, 8)
        x_TND = self.to_qkv(
            x_TNC, compute_kernel_config=self.mm_compute_kernel_config, default_block_size=default_block_size
        )
        q_THNC, k_THNC, v_THNC = ttnn.transformer.split_query_key_value_and_split_heads(
            x_TND, num_heads=1, transpose_key=False
        )
        q_rm = ttnn.mesh_partition(ttnn.to_layout(q_THNC, ttnn.ROW_MAJOR_LAYOUT), dim=2, cluster_axis=h_axis)
        q_THNC = ttnn.to_layout(q_rm, ttnn.TILE_LAYOUT)
        out_THNC = ttnn.transformer.scaled_dot_product_attention(
            ttnn.typecast(q_THNC, ttnn.bfloat16) if q_THNC.dtype != ttnn.bfloat16 else q_THNC,
            ttnn.typecast(k_THNC, ttnn.bfloat16) if k_THNC.dtype != ttnn.bfloat16 else k_THNC,
            ttnn.typecast(v_THNC, ttnn.bfloat16) if v_THNC.dtype != ttnn.bfloat16 else v_THNC,
            is_causal=False,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        out_THNC = ttnn.typecast(out_THNC, q_THNC.dtype) if out_THNC.dtype != q_THNC.dtype else out_THNC
        out_TND = self.proj(
            ttnn.transformer.concatenate_heads(out_THNC),
            compute_kernel_config=self.mm_compute_kernel_config,
            default_block_size=default_block_size,
        )
        # (1, h * W, C): this device's frame, its H rows -> W blocks, then back to the frame owners
        o = ttnn.reshape(ttnn.to_layout(out_TND, ttnn.ROW_MAJOR_LAYOUT), (1, h, wf, w * C))
        o = ttnn.permute(o, (0, 2, 1, 3))  # (1, wf, h, w * C)
        o = ttnn.to_layout(ttnn.reshape(o, (1, wf, h * w, C)), ttnn.TILE_LAYOUT)
        o = self._a2a(o, in_dim=0, out_dim=1, mesh_axis=w_axis)  # (wf, 1, h * w, C): every frame of this W block
        if wf > BT:
            o = o[:BT, :, :, :]
        o = ttnn.reshape(ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT), (B, T, h, w, C))
        return ttnn.add(ttnn.to_layout(o, ttnn.TILE_LAYOUT), residual_BTHWC)

    def forward(self, x_BTHWC: ttnn.Tensor, logical_h: int, logical_w: int = 0) -> ttnn.Tensor:
        """
        x_BTHWC: (B, T, H, W, C) fractured on H and W, TILE layout

        returns: (B, T, H, W, C) fractured on H and W, TILE layout
        """
        assert len(x_BTHWC.shape) == 5
        assert x_BTHWC.layout == ttnn.TILE_LAYOUT
        residual_BTHWC = x_BTHWC
        if _ATTN_A2A:
            out_a2a = self._forward_a2a(x_BTHWC, logical_h, logical_w)
            if out_a2a is not None:
                return out_a2a

        # Gather height and width for replicated attention
        if self.parallel_config.height_parallel.factor > 1:
            x_BTHWC = self.ccl_manager.all_gather_persistent_buffer(
                x_BTHWC, dim=2, mesh_axis=self.parallel_config.height_parallel.mesh_axis
            )
        if self.parallel_config.width_parallel.factor > 1:
            x_BTHWC = self.ccl_manager.all_gather_persistent_buffer(
                x_BTHWC, dim=3, mesh_axis=self.parallel_config.width_parallel.mesh_axis
            )

        x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.ROW_MAJOR_LAYOUT)

        padded_h = x_BTHWC.shape[2]
        padded_w = x_BTHWC.shape[3]
        # Use > (not !=) to only slice when there are EXCESS padding (decoder upsample
        # amplifies mesh_partition padding). The encoder's 1::2 downsampling creates a deficit
        # (shape * factor < logical) which must NOT trigger slicing.
        if padded_h > logical_h:
            x_BTHWC = x_BTHWC[:, :, :logical_h, :, :]
        if logical_w > 0 and padded_w > logical_w:
            x_BTHWC = x_BTHWC[:, :, :, :logical_w, :]
        B, T, H, W, C = x_BTHWC.shape
        x_TNC = ttnn.reshape(x_BTHWC, (B * T, H * W, C))

        # Split T (batch) across devices to reduce redundant SDPA compute.
        # SDPA batch elements are independent, so they can be trivially partitioned.
        # When sdpa_t_fracture_w_only=True, fracture only on the W axis (fewer devices,
        # but avoids conflicts with H-axis sharding on some mesh/shape combinations).
        # Default (False) fractures across all devices (H x W axes) for maximum parallelism.
        h_axis = self.parallel_config.height_parallel.mesh_axis
        h_devices = self.parallel_config.height_parallel.factor
        w_axis = self.parallel_config.width_parallel.mesh_axis
        w_devices = self.parallel_config.width_parallel.factor
        w_only = self.sdpa_t_fracture_w_only
        fracture_devices = w_devices if w_only else h_devices * w_devices
        BT = B * T
        split_t = fracture_devices > 1 and T > 1
        if split_t:
            padded_BT = ((BT + fracture_devices - 1) // fracture_devices) * fracture_devices
            if padded_BT > BT:
                x_TNC = ttnn.pad(x_TNC, [(0, padded_BT - BT), (0, 0), (0, 0)], value=0.0)
            if w_only:
                x_TNC = ttnn.mesh_partition(x_TNC, dim=0, cluster_axis=w_axis)
            else:
                x_TNC = ttnn.mesh_partition(x_TNC, dim=0)

        no_spatial_pad = padded_h == H and not (logical_w > 0 and padded_w > logical_w)
        qsplit = _ATTN_QSPLIT and split_t and w_only and h_devices > 1 and no_spatial_pad and H % h_devices == 0
        # Without a T split (one frame) every device held the whole frame and computed all its queries: each now
        # computes only the queries of its own H x W shard, so the output is already the local shard.
        qlocal = (
            _ATTN_QSPLIT
            and not split_t
            and (h_devices > 1 or w_devices > 1)
            and no_spatial_pad
            and H % h_devices == 0
            and W % w_devices == 0
        )

        x_TNC = ttnn.to_layout(x_TNC, ttnn.TILE_LAYOUT)
        x_TNC = self.norm(x_TNC, compute_kernel_config=self.hifi4_compute_kernel_config)
        default_block_size = (2, 2, 2) if x_TNC.dtype == ttnn.float32 else (8, 8, 8)
        x_TND = self.to_qkv(
            x_TNC, compute_kernel_config=self.mm_compute_kernel_config, default_block_size=default_block_size
        )
        q_THNC, k_THNC, v_THNC = ttnn.transformer.split_query_key_value_and_split_heads(
            x_TND, num_heads=1, transpose_key=False
        )
        if qsplit:
            # rows [h * H / h_devices, (h + 1) * H / h_devices) of the frame, all W columns (row-major split: the
            # per-device token count H * W / h_devices need not be a tile multiple)
            q_rm = ttnn.to_layout(q_THNC, ttnn.ROW_MAJOR_LAYOUT)
            q_rm = ttnn.mesh_partition(q_rm, dim=2, cluster_axis=h_axis)
            q_THNC = ttnn.to_layout(q_rm, ttnn.TILE_LAYOUT)
        elif qlocal:
            q_rm = ttnn.reshape(ttnn.to_layout(q_THNC, ttnn.ROW_MAJOR_LAYOUT), (BT, H, W, C))
            if h_devices > 1:
                q_rm = ttnn.mesh_partition(q_rm, dim=1, cluster_axis=h_axis)
            if w_devices > 1:
                q_rm = ttnn.mesh_partition(q_rm, dim=2, cluster_axis=w_axis)
            q_THNC = ttnn.to_layout(
                ttnn.reshape(q_rm, (BT, 1, (H // h_devices) * (W // w_devices), C)), ttnn.TILE_LAYOUT
            )
        out_THNC = ttnn.transformer.scaled_dot_product_attention(
            ttnn.typecast(q_THNC, ttnn.bfloat16) if q_THNC.dtype != ttnn.bfloat16 else q_THNC,
            ttnn.typecast(k_THNC, ttnn.bfloat16) if k_THNC.dtype != ttnn.bfloat16 else k_THNC,
            ttnn.typecast(v_THNC, ttnn.bfloat16) if v_THNC.dtype != ttnn.bfloat16 else v_THNC,
            is_causal=False,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        out_THNC = ttnn.typecast(out_THNC, q_THNC.dtype) if out_THNC.dtype != q_THNC.dtype else out_THNC
        out_TNC = ttnn.transformer.concatenate_heads(out_THNC)
        out_TND = self.proj(
            out_TNC, compute_kernel_config=self.mm_compute_kernel_config, default_block_size=default_block_size
        )

        # Gather T back before layout conversion (all-gather requires TILE)
        if split_t:
            out_TND = self.ccl_manager.all_gather_persistent_buffer(out_TND, dim=0, mesh_axis=w_axis)
            if not w_only:
                out_TND = self.ccl_manager.all_gather_persistent_buffer(out_TND, dim=0, mesh_axis=h_axis)
            if padded_BT > BT:
                out_TND = out_TND[:BT, :, :]

        out_TND = ttnn.to_layout(out_TND, ttnn.ROW_MAJOR_LAYOUT)

        if qsplit or qlocal:
            out_BTHWC = ttnn.reshape(out_TND, (B, T, H // h_devices, W if qsplit else W // w_devices, C))
            if qsplit and self.parallel_config.width_parallel.factor > 1:
                out_BTHWC = ttnn.mesh_partition(out_BTHWC, dim=3, cluster_axis=w_axis)
            out_BTHWC = ttnn.to_layout(out_BTHWC, ttnn.TILE_LAYOUT)
            return ttnn.add(out_BTHWC, residual_BTHWC)

        needs_h_repad = padded_h > logical_h
        needs_w_repad = logical_w > 0 and padded_w > logical_w

        if needs_h_repad or needs_w_repad:
            # Re-pad spatial dims to match the gathered (pre-slice) shape so
            # mesh_partition can evenly split H and W back to devices.
            # Use 4D form (BT, H, W, C) so H and W pads are independent.
            out_THWC = ttnn.reshape(out_TND, (B * T, H, W, C))
            if needs_w_repad:
                out_THWC = ttnn.pad(out_THWC, [(0, 0), (0, 0), (0, padded_w - W), (0, 0)], value=0.0)
            if needs_h_repad:
                out_THWC = ttnn.pad(out_THWC, [(0, 0), (0, padded_h - H), (0, 0), (0, 0)], value=0.0)
            H_final = out_THWC.shape[1]
            W_final = out_THWC.shape[2]
            out_BTHWC = ttnn.reshape(out_THWC, (B, T, H_final, W_final, C))
        else:
            out_BTHWC = ttnn.reshape(out_TND, (B, T, H, W, C))

        # Scatter height and width
        if self.parallel_config.height_parallel.factor > 1:
            out_BTHWC = ttnn.mesh_partition(
                out_BTHWC, dim=2, cluster_axis=self.parallel_config.height_parallel.mesh_axis
            )
        if self.parallel_config.width_parallel.factor > 1:
            out_BTHWC = ttnn.mesh_partition(
                out_BTHWC, dim=3, cluster_axis=self.parallel_config.width_parallel.mesh_axis
            )

        out_BTHWC = ttnn.to_layout(out_BTHWC, ttnn.TILE_LAYOUT)
        return ttnn.add(out_BTHWC, residual_BTHWC)


class WanCausalConv3d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int = 1,
        padding: Sequence[int] | int = 0,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
        conv_dims: ConvDims = ConvDims(),
    ) -> None:
        super().__init__()

        self.unpadded_in_channels = in_channels
        self.in_channels = aligned_channels(in_channels)
        if self.in_channels != self.unpadded_in_channels:
            logger.warning(f"Padding in_channels from {self.unpadded_in_channels} to {self.in_channels}")
        self.out_channels = out_channels

        self.kernel_size = _ntuple(kernel_size, 3)
        self.stride = _ntuple(stride, 3)
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.dtype = dtype

        padding = _ntuple(padding, 3)
        external_padding = list(padding)
        internal_padding = list(padding)
        # t padding is handled explicitly and depends on the cache.
        external_padding[0] = 2 * padding[0]
        internal_padding[0] = 0
        # HW padding may be handled by the halo CCL if the model is parallelized
        if self.parallel_config.height_parallel.factor > 1:
            external_padding[1] = padding[1]
            internal_padding[1] = 0
        if self.parallel_config.width_parallel.factor > 1:
            external_padding[2] = padding[2]
            internal_padding[2] = 0
        self.external_padding = tuple(external_padding)
        self.internal_padding = tuple(internal_padding)

        self.conv_config = get_conv3d_config(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            dtype,
            grid_size=self.mesh_device.compute_with_storage_grid_size(),
            h_factor=self.parallel_config.height_parallel.factor,
            w_factor=self.parallel_config.width_parallel.factor,
            T=conv_dims.T,
            H=conv_dims.H,
            W=conv_dims.W,
        )
        logger.debug(f"Loaded conv_config: {self.conv_config}")

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4
            if dtype == ttnn.float32
            else ttnn.MathFidelity.HiFi2,  # Do not use HiFi3/4 with bf16 data type and fp32_dest_acc on WH due to accuracy issues.
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        d = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.in_channels
        self.weight = Parameter(
            total_shape=[d, self.out_channels],
            device=mesh_device,
            pad_value=0,
            dtype=dtype,
        )
        self.bias = Parameter(total_shape=[1, self.out_channels], device=mesh_device, pad_value=0, dtype=dtype)

        self._w_mask_cache: dict[tuple, ttnn.Tensor] = {}
        self.np_cache = (
            _NP_CACHE
            and self.external_padding[0] == CACHE_T
            and self.external_padding[1] > 0
            and self.parallel_config.height_parallel.factor > 1
            and self.external_padding[2] > 0
            and self.parallel_config.width_parallel.factor > 1
        )

    def forward_np_cache(
        self,
        x_BTHWC: ttnn.Tensor,
        logical_h: int,
        padded_cache_BTHWC: ttnn.Tensor | None,
        logical_w: int = 0,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Causal conv with a padded temporal cache (see _NP_CACHE). Returns (output, next padded cache)."""
        hf = self.parallel_config.height_parallel.factor
        wf = self.parallel_config.width_parallel.factor
        assert x_BTHWC.shape[2] * hf <= logical_h, "padded cache path has no row mask"
        assert logical_w == 0 or x_BTHWC.shape[3] * wf <= logical_w, "padded cache path has no column mask"
        h_axis = self.parallel_config.height_parallel.mesh_axis
        w_axis = self.parallel_config.width_parallel.mesh_axis
        padded = self.ccl_manager.neighbor_pad_persistent_buffer(
            x_BTHWC,
            dims=[2, 3],
            pad_left=[self.external_padding[1], self.external_padding[2]],
            pad_right=[self.external_padding[1], self.external_padding[2]],
            padding_mode="zeros",
            axes=[h_axis, w_axis],
            neighbor_sems=[
                self.ccl_manager.get_np_ping_pong_semaphore(h_axis),
                self.ccl_manager.get_np_ping_pong_semaphore(w_axis),
            ],
            num_links=[
                get_neighbor_pad_num_links(self.ccl_manager, x_BTHWC, 2, h_axis),
                get_neighbor_pad_num_links(self.ccl_manager, x_BTHWC, 3, w_axis),
            ],
            logical_h=0,
            t_front_pad=CACHE_T,
            front_tensor=padded_cache_BTHWC,
        )
        return self._conv_padded(padded)

    def forward_np_cache_norm(
        self,
        x_BTHWC: ttnn.Tensor,
        norm,
        norm_compute_kernel_config,
        logical_h: int,
        padded_cache_BTHWC: ttnn.Tensor | None,
        logical_w: int = 0,
        residual: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """norm(x) followed by forward_np_cache, the row-major norm output written straight into the interior of
        the padded neighbor-pad buffer (see _NP_NORM). `residual` is added to the conv output (_c3d_resid)."""
        hf = self.parallel_config.height_parallel.factor
        wf = self.parallel_config.width_parallel.factor
        assert x_BTHWC.layout == ttnn.ROW_MAJOR_LAYOUT
        assert x_BTHWC.shape[2] * hf <= logical_h, "padded cache path has no row mask"
        assert logical_w == 0 or x_BTHWC.shape[3] * wf <= logical_w, "padded cache path has no column mask"
        h_axis = self.parallel_config.height_parallel.mesh_axis
        w_axis = self.parallel_config.width_parallel.mesh_axis
        ph, pw = self.external_padding[1], self.external_padding[2]
        buf = self.ccl_manager.get_np_ping_pong_buffer(
            x_BTHWC.shape, [2, 3], [ph, pw], [ph, pw], dtype=x_BTHWC.get_dtype(), t_front_pad=CACHE_T
        )
        norm(x_BTHWC, compute_kernel_config=norm_compute_kernel_config, output_tensor=buf, output_pad=(CACHE_T, ph, pw))
        padded = self.ccl_manager.neighbor_pad_in_place(
            buf,
            dims=[2, 3],
            pad_left=[ph, pw],
            pad_right=[ph, pw],
            axes=[h_axis, w_axis],
            neighbor_sems=[
                self.ccl_manager.get_np_ping_pong_semaphore(h_axis),
                self.ccl_manager.get_np_ping_pong_semaphore(w_axis),
            ],
            num_links=[
                get_neighbor_pad_num_links(self.ccl_manager, x_BTHWC, 2, h_axis),
                get_neighbor_pad_num_links(self.ccl_manager, x_BTHWC, 3, w_axis),
            ],
            t_front_pad=CACHE_T,
            front_tensor=padded_cache_BTHWC,
        )
        return self._conv_padded(padded, residual)

    def can_fuse_residual(self) -> bool:
        """conv3d residual_tensor support: bf16, full-block output, tile-aligned patch blocks."""
        cfg = self.conv_config
        return (
            self.dtype == ttnn.bfloat16
            and (_c3d_resid_all() or cfg.C_in_block == self.in_channels)
            and cfg.C_out_block > 32
            and (cfg.T_out_block * cfg.H_out_block * cfg.W_out_block) % 32 == 0
        )

    def _conv_padded(self, padded: ttnn.Tensor, residual: ttnn.Tensor | None = None) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        next_cache_BTHWC = padded[:, -CACHE_T:, :, :, :]
        out_BTHWC = ttnn.experimental.conv3d(
            input_tensor=padded,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data,
            device=self.mesh_device,
            config=self.conv_config,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.internal_padding,
            padding_mode="zeros",
            dtype=self.dtype,
            compute_kernel_config=self.compute_kernel_config,
            residual_tensor=residual,
        )
        return out_BTHWC, next_cache_BTHWC

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            weight_tt = ttnn.from_torch(state["weight"], dtype=self.dtype, pad_value=0)
            prepared = ttnn.experimental.prepare_conv3d_weights(
                weight_tensor=weight_tt, C_in_block=self.conv_config.C_in_block, device=self.mesh_device
            )
            state["weight"] = local_device_to_torch(prepared)
        if "bias" in state:
            state["bias"] = state["bias"].reshape(1, -1)

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        logical_h: int,
        cache_x_BTHWC: ttnn.Tensor | None = None,
        logical_w: int = 0,
    ) -> ttnn.Tensor:
        """
        x_BTHWC: (B, T, H, W, C) fractured on H and W, ROW_MAJOR layout
        cache_x_BTHWC: (B, T1, H, W, C) fractured on H and W

        returns: (B, T, H, W, C) fractured on H and W, ROW_MAJOR layout
        """
        assert x_BTHWC.layout == ttnn.ROW_MAJOR_LAYOUT, f"WanCausalConv3d expects ROW_MAJOR input, got {x_BTHWC.layout}"
        # NOTE: T padding is handled explicitly and depends on the cache
        t_front_padding = self.external_padding[0]
        if cache_x_BTHWC is not None and t_front_padding > 0:
            # concat on T
            x_BTHWC = ttnn.concat([cache_x_BTHWC, x_BTHWC], dim=1)
            t_front_padding -= cache_x_BTHWC.shape[1]
        # Halo exchange (height and/or width padding)
        # Masking (zero rows at/beyond logical_h) is fused into neighbor_pad via logical_h parameter,
        # eliminating a separate mul-mask op. The local-copy kernel writes MEM_ZEROS_BASE for masked rows
        # while still draining the CB, so the 2-phase pipeline (H then W) correctly propagates zeros.
        # This is valid for linear H topology: the last device (which has padded rows) is is_last_device
        # and never sends those rows via H fabric; phase-2 W exchange reads from the already-zeroed output.
        # NOTE: logical_w is not yet fused into neighbor_pad (no C++ kernel support). Pre-conv mul-mask
        # workaround zeros width-padding columns before the halo exchange.
        h_pad_needed = self.external_padding[1] > 0 and self.parallel_config.height_parallel.factor > 1
        w_pad_needed = self.external_padding[2] > 0 and self.parallel_config.width_parallel.factor > 1

        # Width pre-conv masking: zero padding columns so neighbor_pad doesn't propagate non-zero padding.
        if (
            logical_w > 0
            and self.parallel_config.width_parallel.factor > 1
            and x_BTHWC.shape[3] * self.parallel_config.width_parallel.factor > logical_w
        ):
            x_BTHWC = ttnn.mul(
                x_BTHWC,
                _get_w_mask(self._w_mask_cache, x_BTHWC, logical_w, self.parallel_config, self.mesh_device, self.dtype),
            )

        # T-front causal zero padding: fuse into neighbor_pad when h_pad_needed (avoids a
        # separate reshape+pad+reshape and an intermediate tensor allocation).
        # Fall back to standalone ttnn.pad when there is no H halo exchange to piggyback on.
        fuse_t_front_pad = t_front_padding > 0 and h_pad_needed
        if t_front_padding > 0 and not fuse_t_front_pad:
            B, T, H, W, C = x_BTHWC.shape
            x_BTNC = ttnn.reshape(x_BTHWC, (B, T, H * W, C))
            x_BTNC = ttnn.pad(x_BTNC, [(0, 0), (t_front_padding, 0), (0, 0), (0, 0)], value=0.0)
            x_BTHWC = ttnn.reshape(x_BTNC, (B, T + t_front_padding, H, W, C))

        if h_pad_needed or w_pad_needed:
            dims, pad_left, pad_right = [], [], []
            axes, neighbor_sems, links = [], [], []
            if h_pad_needed:
                dims.append(2)
                pad_left.append(self.external_padding[1])
                pad_right.append(self.external_padding[1])
                axes.append(self.parallel_config.height_parallel.mesh_axis)
                neighbor_sems.append(
                    self.ccl_manager.get_np_ping_pong_semaphore(self.parallel_config.height_parallel.mesh_axis)
                )
                links.append(
                    get_neighbor_pad_num_links(
                        self.ccl_manager, x_BTHWC, 2, self.parallel_config.height_parallel.mesh_axis
                    )
                )
            if w_pad_needed:
                dims.append(3)
                pad_left.append(self.external_padding[2])
                pad_right.append(self.external_padding[2])
                axes.append(self.parallel_config.width_parallel.mesh_axis)
                neighbor_sems.append(
                    self.ccl_manager.get_np_ping_pong_semaphore(self.parallel_config.width_parallel.mesh_axis)
                )
                links.append(
                    get_neighbor_pad_num_links(
                        self.ccl_manager, x_BTHWC, 3, self.parallel_config.width_parallel.mesh_axis
                    )
                )

            fused_logical_h = (
                logical_h
                if h_pad_needed and x_BTHWC.shape[2] * self.parallel_config.height_parallel.factor > logical_h
                else 0
            )
            x_BTHWC = self.ccl_manager.neighbor_pad_persistent_buffer(
                x_BTHWC,
                dims=dims,
                pad_left=pad_left,
                pad_right=pad_right,
                padding_mode="zeros",
                axes=axes,
                neighbor_sems=neighbor_sems,
                num_links=links,
                logical_h=fused_logical_h,
                t_front_pad=t_front_padding if fuse_t_front_pad else 0,
            )

        x_BTHWC = ttnn.experimental.conv3d(
            input_tensor=x_BTHWC,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data,
            device=self.mesh_device,
            config=self.conv_config,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.internal_padding,
            padding_mode="zeros",
            dtype=self.dtype,
            compute_kernel_config=self.compute_kernel_config,
        )

        return x_BTHWC


class WanResidualBlock(Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
        conv_dims: ConvDims = ConvDims(),
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mesh_device = mesh_device

        self.norm1 = RMSNorm(
            embedding_dim=in_dim,
            norm_eps=1e-12,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
            dtype=dtype,
            fused_activation=ttnn.UnaryOpType.SILU,
        )
        self.conv1 = WanCausalConv3d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
            conv_dims=conv_dims,
        )
        self.norm2 = RMSNorm(
            embedding_dim=out_dim,
            norm_eps=1e-12,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
            dtype=dtype,
            fused_activation=ttnn.UnaryOpType.SILU,
        )
        self.conv2 = WanCausalConv3d(
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
            conv_dims=conv_dims,
        )

        if in_dim != out_dim:
            self.conv_shortcut = Linear(
                in_features=in_dim,
                out_features=out_dim,
                mesh_device=mesh_device,
                dtype=dtype,
            )
        else:
            self.conv_shortcut = None

        self.matmul_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4
            if (is_blackhole() or dtype == ttnn.float32)
            else ttnn.MathFidelity.HiFi2,  # Do not use HiFi3/4 with fp32_dest_acc on WH due to accuracy issues.
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.norm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "norm1.gamma" in state:
            state["norm1.weight"] = state.pop("norm1.gamma").squeeze()

        if "norm2.gamma" in state:
            state["norm2.weight"] = state.pop("norm2.gamma").squeeze()

        def conv_1d_to_matmul_weight(weight):
            out_c, in_c, kt, kh, kw = weight.shape
            assert kt == kh == kw == 1
            weight = weight.reshape(out_c, in_c)
            return weight

        if "conv_shortcut.weight" in state:
            state["conv_shortcut.weight"] = conv_1d_to_matmul_weight(state["conv_shortcut.weight"])

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        logical_h: int,
        feat_cache: list[ttnn.Tensor] | None = None,
        feat_idx: list[int] = [0],
        logical_w: int = 0,
    ) -> ttnn.Tensor:
        row_major = x_BTHWC.layout == ttnn.ROW_MAJOR_LAYOUT
        if row_major and self.conv_shortcut is not None:
            x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)
            row_major = False
        assert (
            row_major or x_BTHWC.layout == ttnn.TILE_LAYOUT
        ), f"WanResidualBlock expects TILE or ROW_MAJOR input, got {x_BTHWC.layout}"
        h_tile_BTHWC = (
            self.conv_shortcut(x_BTHWC, compute_kernel_config=self.matmul_compute_kernel_config)
            if self.conv_shortcut is not None
            else x_BTHWC
        )
        fuse_norm1 = _NP_NORM and row_major and feat_cache is not None and self.conv1.np_cache
        if not fuse_norm1:
            x_norm_silu_tile_BTHWC = self.norm1(x_BTHWC, compute_kernel_config=self.norm_compute_kernel_config)
            x_BTHWC = (
                x_norm_silu_tile_BTHWC if row_major else ttnn.to_layout(x_norm_silu_tile_BTHWC, ttnn.ROW_MAJOR_LAYOUT)
            )

        # Cached conv
        if fuse_norm1:
            idx = feat_idx[0]
            x_conv_BTHWC, feat_cache[idx] = self.conv1.forward_np_cache_norm(
                x_BTHWC, self.norm1, self.norm_compute_kernel_config, logical_h, feat_cache[idx], logical_w=logical_w
            )
            feat_idx[0] += 1
        elif feat_cache is not None and self.conv1.np_cache:
            idx = feat_idx[0]
            x_conv_BTHWC, feat_cache[idx] = self.conv1.forward_np_cache(
                x_BTHWC, logical_h, feat_cache[idx], logical_w=logical_w
            )
            feat_idx[0] += 1
        elif feat_cache is not None:
            # Prepare to cache the current activation for future use
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                # Current activation is too short, so append the cached activation as well
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)

            x_conv_BTHWC = self.conv1(x_BTHWC, logical_h, feat_cache[idx], logical_w=logical_w)
            # NOTE: Should deallocate feat_cache[idx] after it's reassigned
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_conv_BTHWC = self.conv1(x_BTHWC, logical_h, logical_w=logical_w)

        fuse_norm2 = _NP_NORM and feat_cache is not None and self.conv2.np_cache
        fuse_resid = fuse_norm2 and row_major and _c3d_resid() and self.conv2.can_fuse_residual()
        if not fuse_norm2:
            x_BTHWC = self.norm2(x_conv_BTHWC, compute_kernel_config=self.norm_compute_kernel_config)

        # Cached conv
        if fuse_norm2:
            idx = feat_idx[0]
            x_conv_BTHWC, feat_cache[idx] = self.conv2.forward_np_cache_norm(
                x_conv_BTHWC,
                self.norm2,
                self.norm_compute_kernel_config,
                logical_h,
                feat_cache[idx],
                logical_w=logical_w,
                residual=h_tile_BTHWC if fuse_resid else None,
            )
            feat_idx[0] += 1
            if fuse_resid:
                return x_conv_BTHWC
        elif feat_cache is not None and self.conv2.np_cache:
            idx = feat_idx[0]
            x_conv_BTHWC, feat_cache[idx] = self.conv2.forward_np_cache(
                x_BTHWC, logical_h, feat_cache[idx], logical_w=logical_w
            )
            feat_idx[0] += 1
        elif feat_cache is not None:
            # Prepare to cache the current activation for future use
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                # Current activation is too short, so append the cached activation as well
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)

            x_conv_BTHWC = self.conv2(x_BTHWC, logical_h, feat_cache[idx], logical_w=logical_w)
            # NOTE: Should deallocate feat_cache[idx] after it's reassigned
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_conv_BTHWC = self.conv2(x_BTHWC, logical_h, logical_w=logical_w)

        # Add residual
        if row_major:
            return ttnn.add(h_tile_BTHWC, x_conv_BTHWC)
        x_tile_BTHWC = ttnn.to_layout(x_conv_BTHWC, ttnn.TILE_LAYOUT)
        x_tile_BTHWC = ttnn.add(h_tile_BTHWC, x_tile_BTHWC)
        return x_tile_BTHWC


class WanMidBlock(Module):
    def __init__(
        self,
        *,
        dim: int,
        num_layers: int = 1,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
        sdpa_t_fracture_w_only: bool = False,
        conv_dims: ConvDims = ConvDims(),
    ) -> None:
        super().__init__()

        self.dim = dim
        self.mesh_device = mesh_device
        resnets = ModuleList()
        attentions = ModuleList()

        resnets.append(
            WanResidualBlock(
                in_dim=dim,
                out_dim=dim,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                dtype=dtype,
                conv_dims=conv_dims,
            )
        )

        for _ in range(num_layers):
            attentions.append(
                WanAttentionBlock(
                    dim=dim,
                    mesh_device=mesh_device,
                    ccl_manager=ccl_manager,
                    parallel_config=parallel_config,
                    dtype=dtype,
                    sdpa_t_fracture_w_only=sdpa_t_fracture_w_only,
                )
            )
            resnets.append(
                WanResidualBlock(
                    in_dim=dim,
                    out_dim=dim,
                    mesh_device=mesh_device,
                    ccl_manager=ccl_manager,
                    parallel_config=parallel_config,
                    dtype=dtype,
                    conv_dims=conv_dims,
                )
            )

        self.resnets = resnets
        self.attentions = attentions

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        logical_h: int,
        feat_cache: list[ttnn.Tensor] | None = None,
        feat_idx: list[int] = [0],
        logical_w: int = 0,
    ) -> ttnn.Tensor:
        assert x_BTHWC.layout == ttnn.TILE_LAYOUT, f"WanMidBlock expects TILE input, got {x_BTHWC.layout}"
        x_res_BTHWC = self.resnets[0](x_BTHWC, logical_h, feat_cache, feat_idx, logical_w=logical_w)
        x_BTHWC = x_res_BTHWC
        for i in range(len(self.attentions)):
            x_attn_BTHWC = self.attentions[i](x_BTHWC, logical_h, logical_w=logical_w)
            x_BTHWC = self.resnets[i + 1](x_attn_BTHWC, logical_h, feat_cache, feat_idx, logical_w=logical_w)
        return x_BTHWC


class WanConv2d(Module):
    """
    A conv2d implemented with conv3d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int = 1,
        padding: Sequence[int] | int = 0,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
        conv_dims: ConvDims = ConvDims(),
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = _ntuple(kernel_size, 3)
        self.stride = _ntuple(stride, 3)
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.dtype = dtype

        padding = _ntuple(padding, 3)
        external_padding = list(padding)
        internal_padding = list(padding)
        # t padding is handled explicitly and depends on the cache.
        external_padding[0] = 2 * padding[0]
        internal_padding[0] = 0
        # HW padding may be handled by the halo CCL if the model is parallelized
        if self.parallel_config.height_parallel.factor > 1:
            external_padding[1] = padding[1]
            internal_padding[1] = 0
        if self.parallel_config.width_parallel.factor > 1:
            external_padding[2] = padding[2]
            internal_padding[2] = 0
        self.external_padding = tuple(external_padding)
        self.internal_padding = tuple(internal_padding)

        self.conv_config = get_conv3d_config(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            dtype,
            grid_size=self.mesh_device.compute_with_storage_grid_size(),
            h_factor=self.parallel_config.height_parallel.factor,
            w_factor=self.parallel_config.width_parallel.factor,
            T=conv_dims.T,
            H=conv_dims.H,
            W=conv_dims.W,
        )
        logger.debug(f"Loaded conv_config: {self.conv_config}")

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4
            if dtype == ttnn.float32
            else ttnn.MathFidelity.HiFi2,  # Do not use HiFi3/4 with bf16 data type and fp32_dest_acc on WH due to accuracy issues.
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        d = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.in_channels
        self.weight = Parameter(
            total_shape=[d, self.out_channels],
            device=mesh_device,
            pad_value=0,
            dtype=dtype,
        )
        self.bias = Parameter(
            total_shape=[1, self.out_channels],
            device=mesh_device,
            pad_value=0,
            dtype=dtype,
        )

        self._w_mask_cache: dict[tuple, ttnn.Tensor] = {}

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            weight_tt = ttnn.from_torch(state["weight"].unsqueeze(2), dtype=self.dtype, pad_value=0)
            prepared = ttnn.experimental.prepare_conv3d_weights(
                weight_tensor=weight_tt, C_in_block=self.conv_config.C_in_block, device=self.mesh_device
            )
            state["weight"] = local_device_to_torch(prepared)
        if "bias" in state:
            state["bias"] = state["bias"].reshape(1, -1)

    def forward(self, x_BTHWC: ttnn.Tensor, logical_h: int, logical_w: int = 0) -> ttnn.Tensor:
        assert x_BTHWC.layout == ttnn.ROW_MAJOR_LAYOUT, f"WanConv2d expects ROW_MAJOR input, got {x_BTHWC.layout}"

        # Halo exchange (height and/or width padding)
        # Masking fused into neighbor_pad via logical_h (see WanCausalConv3d.forward for details).
        # NOTE: logical_w is not yet fused into neighbor_pad. Pre-conv mul-mask workaround zeros width-padding columns.
        h_pad_needed = self.external_padding[1] > 0 and self.parallel_config.height_parallel.factor > 1
        w_pad_needed = self.external_padding[2] > 0 and self.parallel_config.width_parallel.factor > 1

        # Width pre-conv masking: zero padding columns so neighbor_pad doesn't propagate non-zero padding.
        if (
            logical_w > 0
            and self.parallel_config.width_parallel.factor > 1
            and x_BTHWC.shape[3] * self.parallel_config.width_parallel.factor > logical_w
        ):
            x_BTHWC = ttnn.mul(
                x_BTHWC,
                _get_w_mask(self._w_mask_cache, x_BTHWC, logical_w, self.parallel_config, self.mesh_device, self.dtype),
            )

        if h_pad_needed or w_pad_needed:
            dims, pad_left, pad_right = [], [], []
            axes, neighbor_sems, links = [], [], []
            if h_pad_needed:
                dims.append(2)
                pad_left.append(self.external_padding[1])
                pad_right.append(self.external_padding[1])
                axes.append(self.parallel_config.height_parallel.mesh_axis)
                neighbor_sems.append(
                    self.ccl_manager.get_np_ping_pong_semaphore(self.parallel_config.height_parallel.mesh_axis)
                )
                links.append(
                    get_neighbor_pad_num_links(
                        self.ccl_manager, x_BTHWC, 2, self.parallel_config.height_parallel.mesh_axis
                    )
                )
            if w_pad_needed:
                dims.append(3)
                pad_left.append(self.external_padding[2])
                pad_right.append(self.external_padding[2])
                axes.append(self.parallel_config.width_parallel.mesh_axis)
                neighbor_sems.append(
                    self.ccl_manager.get_np_ping_pong_semaphore(self.parallel_config.width_parallel.mesh_axis)
                )
                links.append(
                    get_neighbor_pad_num_links(
                        self.ccl_manager, x_BTHWC, 3, self.parallel_config.width_parallel.mesh_axis
                    )
                )

            fused_logical_h = (
                logical_h
                if h_pad_needed and x_BTHWC.shape[2] * self.parallel_config.height_parallel.factor > logical_h
                else 0
            )
            x_BTHWC = self.ccl_manager.neighbor_pad_persistent_buffer(
                x_BTHWC,
                dims=dims,
                pad_left=pad_left,
                pad_right=pad_right,
                padding_mode="zeros",
                axes=axes,
                neighbor_sems=neighbor_sems,
                num_links=links,
                logical_h=fused_logical_h,
            )

        x_BTHWC = ttnn.experimental.conv3d(
            input_tensor=x_BTHWC,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data,
            device=self.mesh_device,
            config=self.conv_config,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.internal_padding,
            padding_mode="zeros",
            dtype=self.dtype,
            compute_kernel_config=self.compute_kernel_config,
        )

        return x_BTHWC


class WanResample(Module):
    def __init__(
        self,
        *,
        dim: int,
        mode: str,
        resample_out_dim: int | None = None,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
        tconv_dims: ConvDims = ConvDims(),
        spatial_dims: ConvDims = ConvDims(),
    ) -> None:
        super().__init__()

        self.dim = dim
        self.mode = mode
        self.mesh_device = mesh_device
        resample_out_dim = resample_out_dim or (dim // 2 if "upsample" in mode else dim)

        assert mode in ["upsample2d", "upsample3d", "downsample2d", "downsample3d"]

        # Spatial conv operates after upsample (at upsampled resolution)
        self.conv = WanConv2d(
            in_channels=dim,
            out_channels=resample_out_dim,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
            conv_dims=spatial_dims,
        )

        self.is_upsample = "upsample" in mode
        self.is_3d = "3d" in mode

        if self.is_3d:
            # Time conv operates before spatial upsample (at current resolution)
            self.time_conv = WanCausalConv3d(
                in_channels=dim,
                out_channels=dim * 2 if self.is_upsample else dim,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0) if self.is_upsample else (0, 0, 0),
                stride=(1, 1, 1) if self.is_upsample else (2, 1, 1),
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                dtype=dtype,
                conv_dims=tconv_dims,
            )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "resample.1", "conv")

    def forward(
        self,
        x_BTHWC,
        logical_h,
        feat_cache: list[ttnn.Tensor] | None = None,
        feat_idx: list[int] = [0],
        logical_w: int = 0,
    ) -> tuple[ttnn.Tensor, int, int]:
        assert x_BTHWC.layout == ttnn.ROW_MAJOR_LAYOUT, f"WanResample expects ROW_MAJOR input, got {x_BTHWC.layout}"
        B, T, H, W, C = x_BTHWC.shape
        if self.is_3d and self.is_upsample:  # upsample3d
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_idx[0] += 1
                    if T > 2:
                        # Frame 0 passes through; frames 1+ get time_conv + temporal doubling
                        x_first = x_BTHWC[:, :1, :, :, :]
                        x_rest = x_BTHWC[:, 1:, :, :, :]
                        x_time_rest = self.time_conv(x_rest, logical_h, logical_w=logical_w)
                        T_rest = x_time_rest.shape[1]
                        x_BTHW2C = ttnn.reshape(x_time_rest, (B, T_rest, H, W, 2, C))
                        x_BT2HWC = ttnn.permute(x_BTHW2C, (0, 1, 4, 2, 3, 5))
                        x_rest_doubled = ttnn.reshape(x_BT2HWC, (B, T_rest * 2, H, W, C))
                        x_BTHWC = ttnn.concat([x_first, x_rest_doubled], dim=1)
                        # Cache last CACHE_T frames from x_rest for the next chunk.
                        # Zero-pad the front if fewer than CACHE_T frames available.
                        cache_start = max(x_rest.shape[1] - CACHE_T, 0)
                        cache_x = x_rest[:, cache_start:, :, :, :]
                        if cache_x.shape[1] < CACHE_T:
                            pad_t = CACHE_T - cache_x.shape[1]
                            cache_x_BNC = ttnn.reshape(cache_x, (B, cache_x.shape[1], H * W, C))
                            cache_x_BNC = ttnn.pad(cache_x_BNC, [(0, 0), (pad_t, 0), (0, 0), (0, 0)], value=0.0)
                            cache_x = ttnn.reshape(cache_x_BNC, (B, CACHE_T, H, W, C))
                        feat_cache[idx] = cache_x
                    else:
                        feat_cache[idx] = "Rep"
                else:
                    t_start = x_BTHWC.shape[1] - CACHE_T
                    cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
                    is_rep = isinstance(feat_cache[idx], str) and feat_cache[idx] == "Rep"
                    assert not (
                        isinstance(feat_cache[idx], str) and not is_rep
                    ), "If feat_cache[idx] is a string, it must be 'Rep'"
                    if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None and not is_rep:
                        cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)

                    if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None and is_rep:
                        # When feat_cache[idx] is "Rep", we need to pad the cache_x_BTHWC with zeros
                        # Padding only works on the lowest 3 dims
                        cache_x_B1NC = ttnn.reshape(cache_x_BTHWC, (B, 1, H * W, C))
                        cache_x_BTNC = ttnn.pad(cache_x_B1NC, [(0, 0), (1, 0), (0, 0), (0, 0)], value=0.0)
                        cache_x_BTHWC = ttnn.reshape(cache_x_BTNC, (B, 2, H, W, C))

                    if is_rep:
                        x_time_BTHWU = self.time_conv(x_BTHWC, logical_h, logical_w=logical_w)
                    else:
                        x_time_BTHWU = self.time_conv(x_BTHWC, logical_h, feat_cache[idx], logical_w=logical_w)
                    x_BTHWU = x_time_BTHWU
                    feat_cache[idx] = cache_x_BTHWC
                    feat_idx[0] += 1

                    T1 = x_BTHWU.shape[1]
                    x_BTHW2C = ttnn.reshape(x_BTHWU, (B, T1, H, W, 2, C))
                    x_BT2HWC = ttnn.permute(x_BTHW2C, (0, 1, 4, 2, 3, 5))
                    x_BTHWC = ttnn.reshape(x_BT2HWC, (B, T1 * 2, H, W, C))
            else:
                # NOTE: This else section was written with the sole objective of
                # of getting our model to match the reference  model (i.e. huggingface's
                # WanResample module)'s behavior in
                # test_vae_wan2_1.py::test_wan_resample when feat_cache is None
                # and for T = 1, 2, 4.
                #
                # It's very possible that the overall algorithm  may be incorrect for other edge
                # cases not tested. It may also be we deliberately break compatibility against reference
                # models for performance purposes. This needs to be audited.
                #
                # Claude was not able to reference this piece of code to any pat of the WanResample
                # code from hugging face's models, but the results pass for the test cases.
                #
                # TODO: Does the comment below 1) make sense? 2) convey the correct intention?
                # I don't know. Why should we try to replicate the cached path for feat_cache = None?
                # Replicate cached path's "Rep" boundary behavior:
                # - Frame 0: output directly (no time_conv), like when feat_cache is None → "Rep"
                # - Frames 1..T-1: time_conv with zero-padded boundary
                # This gives contexts: frame 1 = [0,0,x_1], frame 2 = [0,x_1,x_2], frame t≥3 = [x_{t-2},x_{t-1},x_t]
                # which matches the cached path exactly.
                x_first = x_BTHWC[:, :1, :, :, :]  # frame 0: identity (T=1)
                if T > 2:
                    x_rest = x_BTHWC[:, 1:, :, :, :]  # frames 1..T-1
                    x_time_rest = self.time_conv(
                        x_rest, logical_h, logical_w=logical_w
                    )  # zero-padded: context [0,0,x_1], [0,x_1,x_2], ...
                    T_rest = x_time_rest.shape[1]  # = T-1
                    x_BTHW2C = ttnn.reshape(x_time_rest, (B, T_rest, H, W, 2, C))
                    x_BT2HWC = ttnn.permute(x_BTHW2C, (0, 1, 4, 2, 3, 5))
                    x_rest_doubled = ttnn.reshape(x_BT2HWC, (B, T_rest * 2, H, W, C))  # 2*(T-1) frames
                    x_BTHWC = ttnn.concat([x_first, x_rest_doubled], dim=1)  # 1 + 2*(T-1) = 2*T-1 frames
                # If T=1: x_BTHWC = x_first (unchanged), no temporal doubling, same as cached "Rep" path

        if self.is_upsample:
            T2 = x_BTHWC.shape[1]
            x_NHWC = ttnn.reshape(x_BTHWC, (B * T2, H, W, C))
            x_upsamped_NHWC = ttnn.upsample(x_NHWC, scale_factor=2)
            logical_h *= 2
            if logical_w > 0:
                logical_w *= 2
            H2, W2 = x_upsamped_NHWC.shape[1], x_upsamped_NHWC.shape[2]
            x_BTHWC = ttnn.reshape(x_upsamped_NHWC, (B, T2, H2, W2, C))
            x_conv_BTHWC = self.conv(x_BTHWC, logical_h, logical_w=logical_w)
        else:
            x_conv_BTHWC = self.conv(x_BTHWC, logical_h, logical_w=logical_w)
            x_conv_BTHWC = x_conv_BTHWC[
                :, :, 1::2, 1::2, :
            ]  # 2x2 strided convolution output to support patched conv, with only right and bottom padding
            logical_h //= 2
            if logical_w > 0:
                logical_w //= 2

        # Handle downsample3d
        if self.is_3d and not self.is_upsample:
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = ttnn.clone(x_conv_BTHWC)
                    feat_idx[0] += 1
                else:
                    cache_x_BTHWC = ttnn.clone(x_conv_BTHWC[:, -1:, :, :, :])
                    x_conv_BTHWC = self.time_conv(
                        ttnn.concat([feat_cache[idx][:, -1:, :, :, :], x_conv_BTHWC], dim=1),
                        logical_h,
                        logical_w=logical_w,
                    )
                    feat_cache[idx] = cache_x_BTHWC
                    feat_idx[0] += 1
            else:
                # Full-T mode: the whole clip is encoded in one pass, so there is no feat_cache
                # to carry temporal context across chunks. Reproduce the cached path's arithmetic
                # in a single call: frame 0 passes through untouched (the cached path only stores
                # it and returns x unchanged on the first iteration), and frames 1..T-1 go through
                # the strided temporal conv with frame 0 supplying the real context frame -- the
                # same tensor the cached path would have prepended from feat_cache.
                #   cached : time_conv(concat([cache_last, chunk]))  -> len(chunk) // 2 per chunk
                #   full-T : time_conv(concat([frame0,     rest ]))  -> (T - 1) // 2 in one go
                # Without this branch the temporal conv never ran and latent T stayed at pixel T.
                T_conv = x_conv_BTHWC.shape[1]
                if T_conv > 2:
                    x_first_BTHWC = x_conv_BTHWC[:, :1, :, :, :]
                    x_time_BTHWC = self.time_conv(x_conv_BTHWC, logical_h, logical_w=logical_w)
                    x_conv_BTHWC = ttnn.concat([x_first_BTHWC, x_time_BTHWC], dim=1)

        return x_conv_BTHWC, logical_h, logical_w


class WanUpBlock(Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        upsample_mode: str | None = None,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
        res_dims: ConvDims = ConvDims(),
        tconv_dims: ConvDims = ConvDims(),
        spatial_dims: ConvDims = ConvDims(),
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_res_blocks = num_res_blocks
        self.upsample_mode = upsample_mode
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        assert upsample_mode in ["upsample2d", "upsample3d"] or upsample_mode is None

        resnets = ModuleList()
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                WanResidualBlock(
                    in_dim=current_dim,
                    out_dim=out_dim,
                    mesh_device=mesh_device,
                    ccl_manager=ccl_manager,
                    parallel_config=parallel_config,
                    dtype=dtype,
                    conv_dims=res_dims,
                )
            )
            current_dim = out_dim
        self.resnets = resnets

        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = WanResample(
                dim=out_dim,
                mode=upsample_mode,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                dtype=dtype,
                tconv_dims=tconv_dims,
                spatial_dims=spatial_dims,
            )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "upsamplers.0", "upsamplers")

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        logical_h: int,
        feat_cache: list[ttnn.Tensor] | None = None,
        feat_idx: list[int] = [0],
        logical_w: int = 0,
    ) -> tuple[ttnn.Tensor, int, int]:
        for resnet in self.resnets:
            x_res_BTHWC = resnet(x_BTHWC, logical_h, feat_cache, feat_idx, logical_w=logical_w)
            x_BTHWC = x_res_BTHWC
        if self.upsamplers is not None:
            if x_BTHWC.layout != ttnn.ROW_MAJOR_LAYOUT:
                x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.ROW_MAJOR_LAYOUT)
            x_upsampled_BTHWC, logical_h, logical_w = self.upsamplers(
                x_BTHWC,
                logical_h,
                feat_cache,
                feat_idx,
                logical_w=logical_w,
            )
            x_BTHWC = x_upsampled_BTHWC if _RM_RESID else ttnn.to_layout(x_upsampled_BTHWC, ttnn.TILE_LAYOUT)
        return x_BTHWC, logical_h, logical_w


class WanDecoder3d(Module):
    def __init__(
        self,
        *,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales=(),
        temperal_upsample: Sequence[bool] = (False, True, True),
        out_channels: int = 3,
        is_residual: bool = False,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
        sdpa_t_fracture_w_only: bool = False,
        height: int = 0,
        width: int = 0,
        t_chunk_size: int | None = None,
        cached: bool = False,
    ) -> None:
        super().__init__()

        assert not is_residual, "is_residual is not supported"
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        # channel counts per stage: [latent, up0, up1, up2, up3]
        channel_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        h_factor = parallel_config.height_parallel.factor
        w_factor = parallel_config.width_parallel.factor
        stage_hw, stage_t = compute_decoder_dims(
            height,
            width,
            h_factor,
            w_factor,
            t_chunk_size,
            temperal_upsample=temperal_upsample,
            num_stages=len(dim_mult) - 1,
            cached=cached,
        )

        # init block
        lat_h, lat_w = stage_hw[0]
        lat_dims = ConvDims(stage_t[0].T_res, lat_h, lat_w)
        self.conv_in = WanCausalConv3d(
            z_dim,
            channel_dims[0],
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
            conv_dims=lat_dims,
        )

        # middle blocks
        self.mid_block = WanMidBlock(
            dim=channel_dims[0],
            num_layers=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
            sdpa_t_fracture_w_only=sdpa_t_fracture_w_only,
            conv_dims=lat_dims,
        )

        # upsample blocks
        self.up_blocks = ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(channel_dims[:-1], channel_dims[1:])):
            # residual (+attention) blocks
            if i > 0 and not is_residual:
                # wan vae 2.1
                in_dim = in_dim // 2

            # determine if we need upsampling
            up_flag = i != len(dim_mult) - 1
            # determine upsampling mode, if not upsampling, set to None
            upsample_mode = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"

            stage_h, stage_w = stage_hw[i]
            next_h, next_w = stage_hw[i + 1] if up_flag else (0, 0)
            T_res, T_tconv, T_spatial = stage_t[i]

            # Create and add the upsampling block
            # NOTE: Different codepath if is_residual. Not implemented yet.
            up_block = WanUpBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                upsample_mode=upsample_mode,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                dtype=dtype,
                res_dims=ConvDims(T_res, stage_h, stage_w),
                tconv_dims=ConvDims(T_tconv, stage_h, stage_w),
                spatial_dims=ConvDims(T_spatial, next_h, next_w),
            )
            self.up_blocks.append(up_block)

        # output blocks
        full_h, full_w = stage_hw[-1]
        self.norm_out = RMSNorm(
            embedding_dim=out_dim,
            norm_eps=1e-12,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
            dtype=dtype,
            fused_activation=ttnn.UnaryOpType.SILU,
        )
        self.conv_out = WanCausalConv3d(
            out_dim,
            out_channels,
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
            conv_dims=ConvDims(stage_t[-1].T_res, full_h, full_w),
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "norm_out.gamma" in state:
            state["norm_out.weight"] = state.pop("norm_out.gamma").squeeze()

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        logical_h: int,
        feat_cache: list[ttnn.Tensor] | None = None,
        feat_idx: list[int] = [0],
        first_chunk: bool = False,
        logical_w: int = 0,
    ) -> tuple[ttnn.Tensor, int, int]:
        # NOTE: first_chunk is not used. It would be needed for WanResidualUpBlock.
        ## conv1
        if feat_cache is not None and self.conv_in.np_cache:
            idx = feat_idx[0]
            x_BTHWC, feat_cache[idx] = self.conv_in.forward_np_cache(
                x_BTHWC, logical_h, feat_cache[idx], logical_w=logical_w
            )
            feat_idx[0] += 1
        elif feat_cache is not None:
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                # Current activation is too short, so append the cached activation as well
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)
            x_BTHWC = self.conv_in(x_BTHWC, logical_h, feat_cache[idx], logical_w=logical_w)
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_BTHWC = self.conv_in(x_BTHWC, logical_h, logical_w=logical_w)

        x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)

        ## middle
        x_BTHWC = self.mid_block(x_BTHWC, logical_h, feat_cache, feat_idx, logical_w=logical_w)

        ## upsamples
        for _, up_block in enumerate(self.up_blocks):
            x_BTHWC, logical_h, logical_w = up_block(x_BTHWC, logical_h, feat_cache, feat_idx, logical_w=logical_w)

        ## head
        fuse_norm_out = (
            _NP_NORM and feat_cache is not None and self.conv_out.np_cache and x_BTHWC.layout == ttnn.ROW_MAJOR_LAYOUT
        )
        if not fuse_norm_out:
            x_norm_tile_BTHWC = self.norm_out(x_BTHWC)
            if x_norm_tile_BTHWC.layout != ttnn.ROW_MAJOR_LAYOUT:
                x_norm_tile_BTHWC = ttnn.to_layout(x_norm_tile_BTHWC, ttnn.ROW_MAJOR_LAYOUT)
            x_BTHWC = x_norm_tile_BTHWC

        if fuse_norm_out:
            idx = feat_idx[0]
            x_BTHWC, feat_cache[idx] = self.conv_out.forward_np_cache_norm(
                x_BTHWC, self.norm_out, None, logical_h, feat_cache[idx], logical_w=logical_w
            )
            feat_idx[0] += 1
        elif feat_cache is not None and self.conv_out.np_cache:
            idx = feat_idx[0]
            x_BTHWC, feat_cache[idx] = self.conv_out.forward_np_cache(
                x_BTHWC, logical_h, feat_cache[idx], logical_w=logical_w
            )
            feat_idx[0] += 1
        elif feat_cache is not None:
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                # Current activation is too short, so append the cached activation as well
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)
            x_BTHWC = self.conv_out(x_BTHWC, logical_h, feat_cache[idx], logical_w=logical_w)
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_BTHWC = self.conv_out(x_BTHWC, logical_h, logical_w=logical_w)
        return x_BTHWC, logical_h, logical_w


class WanDecoder(Module):
    def __init__(
        self,
        *,
        base_dim: int = 96,
        decoder_base_dim: int | None = None,
        z_dim: int = 16,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales=(),
        temperal_downsample: Sequence[bool] = (False, True, True),
        latents_mean: Sequence[float] = (
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ),
        latents_std: Sequence[float] = (
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ),
        is_residual: bool = False,
        in_channels: int = 3,
        out_channels: int = 3,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
        sdpa_t_fracture_w_only: bool = False,
        height: int = 0,
        width: int = 0,
        t_chunk_size: int | None = None,
        cached: bool = False,
    ) -> None:
        super().__init__()

        assert not is_residual, "is_residual is not supported"
        self.z_dim = z_dim
        self.temperal_upsample = temperal_downsample[::-1]
        self.out_channels = out_channels
        decoder_base_dim = decoder_base_dim or base_dim

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.dtype = dtype

        # Linear for post_quant_conv
        self.post_quant_conv = Linear(
            in_features=aligned_channels(z_dim),
            out_features=aligned_channels(z_dim),
            mesh_device=mesh_device,
            dtype=dtype,
        )

        self.decoder = WanDecoder3d(
            dim=decoder_base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_upsample=self.temperal_upsample,
            out_channels=out_channels,
            is_residual=is_residual,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
            sdpa_t_fracture_w_only=sdpa_t_fracture_w_only,
            height=height,
            width=width,
            t_chunk_size=t_chunk_size,
            cached=cached,
        )

        self.cached_conv_count = count_convs(self.decoder)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "post_quant_conv.weight" in state and "post_quant_conv.bias" in state:
            state_sub = conv3d_to_linear_weight(pop_substate(state, "post_quant_conv"))
            state["post_quant_conv.weight"] = state_sub["weight"]
            state["post_quant_conv.bias"] = state_sub["bias"]

        pop_substate(state, "encoder")
        pop_substate(state, "quant_conv")

    def clear_cache(self):
        self._conv_idx = [0]
        self._feat_cache = [None] * self.cached_conv_count

    def prepare_input(self, latents):
        tt_latents_BTHWC = latents.permute(0, 2, 3, 4, 1)
        tt_latents_BTHWC = conv_pad_in_channels(tt_latents_BTHWC)
        tt_latents_BTHWC, logical_h = conv_pad_height(tt_latents_BTHWC, self.parallel_config.height_parallel.factor)
        tt_latents_BTHWC, logical_w = conv_pad_width(tt_latents_BTHWC, self.parallel_config.width_parallel.factor)
        return tt_latents_BTHWC, logical_h, logical_w

    def forward(
        self,
        z_BTHWC: ttnn.Tensor,
        logical_h: int,
        t_chunk_size: int | None = 1,
        logical_w: int = 0,
    ) -> tuple[ttnn.Tensor, int, int]:
        """
        t_chunk_size controls how the T dimension is processed:
            None  – full-T single pass, no caching (fastest, most memory)
            1     – one frame at a time with caching (slowest, least memory)
            N > 1 – N frames at a time with caching between chunks
        """
        assert t_chunk_size is None or t_chunk_size >= 1, f"t_chunk_size must be None or >= 1, got {t_chunk_size}"
        B, T, H, W, C = z_BTHWC.shape
        if logical_w == 0:
            logical_w = W

        self.clear_cache()
        z_tile_BTHWC = ttnn.to_layout(z_BTHWC, ttnn.TILE_LAYOUT)
        x_tile_BTHWC = self.post_quant_conv(z_tile_BTHWC)
        x_BTHWC = ttnn.to_layout(x_tile_BTHWC, ttnn.ROW_MAJOR_LAYOUT)

        if t_chunk_size is None or t_chunk_size >= T:
            # No-cache full-T single-pass mode
            out_BTHWC, new_logical_h, new_logical_w = self.decoder(
                x_BTHWC,
                logical_h,
                feat_cache=None,
                feat_idx=None,
                logical_w=logical_w,
            )
            output_BCTHW = ttnn.permute(out_BTHWC, (0, 4, 1, 2, 3))
        else:
            output_BCTHW = None
            # Process frame 0 on its own first, then the remaining frames in groups of
            # t_chunk_size. This mirrors the reference decode loop (which feeds frame 0
            # alone with first_chunk=True before the rest) and the WanEncoder chunking.
            # It keeps every upsample3d "first chunk" (cache-empty) invocation at T=1, so
            # the temporal-doubling boundary matches frame-by-frame decoding for any
            # t_chunk_size. Feeding a T>1 first chunk would instead hit the "Rep" branch
            # and skip doubling the non-boundary frames, dropping output frames.
            chunk_bounds = [(0, 1)] + [(s, min(s + t_chunk_size, T)) for s in range(1, T, t_chunk_size)]
            import time as _hk_time  # EXPERIMENT (hkato): WAN_VAE_TIMING=1 per-chunk timing

            _hk_t = os.environ.get("WAN_VAE_TIMING") == "1"
            _hk_ct = []
            for chunk_idx, (t_start, t_end) in enumerate(chunk_bounds):
                if _hk_t:
                    ttnn.synchronize_device(self.mesh_device)
                    _hk_c0 = _hk_time.perf_counter()
                self._conv_idx = [0]
                out_BTHWC, new_logical_h, new_logical_w = self.decoder(
                    x_BTHWC[:, t_start:t_end, :, :, :],
                    logical_h,
                    feat_cache=self._feat_cache,
                    feat_idx=self._conv_idx,
                    logical_w=logical_w,
                )
                if _hk_t:
                    ttnn.synchronize_device(self.mesh_device)
                    _hk_c1 = _hk_time.perf_counter()
                out_BCTHW = ttnn.permute(out_BTHWC, (0, 4, 1, 2, 3))
                if output_BCTHW is None:
                    output_BCTHW = out_BCTHW
                else:
                    output_BCTHW = ttnn.concat([output_BCTHW, out_BCTHW], dim=2)
                if _hk_t:
                    ttnn.synchronize_device(self.mesh_device)
                    _hk_ct.append(
                        "%d:%.1f+%.1f"
                        % (t_end - t_start, (_hk_c1 - _hk_c0) * 1e3, (_hk_time.perf_counter() - _hk_c1) * 1e3)
                    )
            if _hk_t:
                print("WAN_VAE_CHUNKS " + " ".join(_hk_ct), flush=True)
            self.clear_cache()

        output_tile_BCTHW = ttnn.to_layout(output_BCTHW, ttnn.TILE_LAYOUT)
        output_BCTHW = ttnn.clamp(output_tile_BCTHW, min=-1.0, max=1.0)
        output_BCTHW = ttnn.to_layout(output_BCTHW, ttnn.ROW_MAJOR_LAYOUT)
        return (output_BCTHW, new_logical_h, new_logical_w)


class WanEncoder3D(Module):
    def __init__(
        self,
        in_channels: int = 3,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        is_residual: bool = False,  # wan 2.2 vae use a residual downblock
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
        dtype: ttnn.DataType = ttnn.bfloat16,
        height: int = 0,
        width: int = 0,
        encoder_t_chunk_size: int | None = None,
    ) -> None:
        super().__init__()

        assert not is_residual, "is_residual is not supported"
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        num_stages = len(dim_mult) - 1
        h_factor = parallel_config.height_parallel.factor
        w_factor = parallel_config.width_parallel.factor
        stage_hw, stage_t = compute_encoder_dims(
            height,
            width,
            h_factor,
            w_factor,
            encoder_t_chunk_size,
            temperal_downsample=temperal_downsample,
            num_stages=num_stages,
        )

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        full_h, full_w = stage_hw[0]
        self.conv_in = WanCausalConv3d(
            in_channels,
            dims[0],
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
            conv_dims=ConvDims(stage_t[0].T_res, full_h, full_w),
        )

        # downsample blocks.
        self.down_blocks = ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            stage_h, stage_w = stage_hw[i]
            res_dims = ConvDims(stage_t[i].T_res, stage_h, stage_w)

            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    WanResidualBlock(
                        in_dim=in_dim,
                        out_dim=out_dim,
                        mesh_device=mesh_device,
                        ccl_manager=ccl_manager,
                        parallel_config=parallel_config,
                        dtype=dtype,
                        conv_dims=res_dims,
                    )
                )
                if scale in attn_scales:
                    self.down_blocks.append(
                        WanAttentionBlock(
                            dim=out_dim,
                            mesh_device=mesh_device,
                            ccl_manager=ccl_manager,
                            parallel_config=parallel_config,
                            dtype=dtype,
                        )
                    )
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                next_h, next_w = stage_hw[i + 1]
                self.down_blocks.append(
                    WanResample(
                        dim=out_dim,
                        mode=mode,
                        mesh_device=mesh_device,
                        ccl_manager=ccl_manager,
                        parallel_config=parallel_config,
                        dtype=dtype,
                        tconv_dims=ConvDims(stage_t[i].T_tconv, next_h, next_w),
                        spatial_dims=ConvDims(stage_t[i].T_spatial, stage_h, stage_w),
                    )
                )
                scale /= 2.0

        # middle blocks
        lat_h, lat_w = stage_hw[-1]
        lat_dims = ConvDims(stage_t[-1].T_res, lat_h, lat_w)
        self.mid_block = WanMidBlock(
            dim=out_dim,
            num_layers=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
            conv_dims=lat_dims,
        )

        # output blocks
        self.norm_out = RMSNorm(
            embedding_dim=out_dim,
            norm_eps=1e-12,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
            dtype=dtype,
            fused_activation=ttnn.UnaryOpType.SILU,
        )
        self.conv_out = WanCausalConv3d(
            out_dim,
            z_dim,
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
            conv_dims=lat_dims,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "norm_out.gamma" in state:
            state["norm_out.weight"] = state.pop("norm_out.gamma").squeeze()

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        logical_h: int,
        feat_cache: list[ttnn.Tensor] | None = None,
        feat_idx: list[int] = [0],
        logical_w: int = 0,
    ) -> tuple[ttnn.Tensor, int, int]:
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                # Current activation is too short, so append the cached activation as well
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)
            x_BTHWC = self.conv_in(x_BTHWC, logical_h, feat_cache[idx], logical_w=logical_w)
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_BTHWC = self.conv_in(x_BTHWC, logical_h, logical_w=logical_w)

        x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)

        ## downsamples
        for _, down_block in enumerate(self.down_blocks):
            if isinstance(down_block, WanResample):
                x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.ROW_MAJOR_LAYOUT)
                x_BTHWC, logical_h, logical_w = down_block(
                    x_BTHWC,
                    logical_h,
                    feat_cache,
                    feat_idx,
                    logical_w=logical_w,
                )
                x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)
            elif isinstance(down_block, WanResidualBlock):
                x_BTHWC = down_block(x_BTHWC, logical_h, feat_cache, feat_idx, logical_w=logical_w)
            elif isinstance(down_block, WanAttentionBlock):
                x_BTHWC = down_block(x_BTHWC, logical_h, logical_w=logical_w)
            else:
                raise ValueError(f"Unsupported downblock type: {type(down_block)}")

        ## middle
        x_BTHWC = self.mid_block(x_BTHWC, logical_h, feat_cache, feat_idx, logical_w=logical_w)

        ## head
        x_silu_tile_BTHWC = self.norm_out(x_BTHWC)
        x_BTHWC = ttnn.to_layout(x_silu_tile_BTHWC, ttnn.ROW_MAJOR_LAYOUT)

        if feat_cache is not None:
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                # Current activation is too short, so append the cached activation as well
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)
            x_BTHWC = self.conv_out(x_BTHWC, logical_h, feat_cache[idx], logical_w=logical_w)
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_BTHWC = self.conv_out(x_BTHWC, logical_h, logical_w=logical_w)
        return x_BTHWC, logical_h, logical_w


class WanEncoder(Module):
    def __init__(
        self,
        base_dim=96,
        in_channels: int = 3,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        is_residual: bool = False,
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
        dtype: ttnn.DataType = ttnn.bfloat16,
        height: int = 0,
        width: int = 0,
        encoder_t_chunk_size: int | None = None,
        latents_mean: Sequence[float] | None = None,
        latents_std: Sequence[float] | None = None,
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.out_channels = z_dim * 2  # Mean and logvar

        if (latents_mean is None) != (latents_std is None):
            raise ValueError("latents_mean and latents_std must be given together")
        if latents_mean is not None:
            if len(latents_mean) != z_dim or len(latents_std) != z_dim:
                raise ValueError(
                    f"latents_mean/latents_std must have z_dim={z_dim} entries, "
                    f"got {len(latents_mean)}/{len(latents_std)}"
                )
            self._latents_mean = torch.tensor(latents_mean, dtype=torch.float32)
            self._latents_inv_std = 1.0 / torch.tensor(latents_std, dtype=torch.float32)
        else:
            self._latents_mean = None
            self._latents_inv_std = None

        self.encoder = WanEncoder3D(
            in_channels=in_channels,
            dim=base_dim,
            z_dim=self.out_channels,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            is_residual=is_residual,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
            height=height,
            width=width,
            encoder_t_chunk_size=encoder_t_chunk_size,
        )
        # Linear for quant_conv
        self.quant_conv = Linear(
            in_features=aligned_channels(self.out_channels),
            out_features=aligned_channels(self.out_channels),
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.cached_conv_count = count_convs(self.encoder)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "quant_conv.weight" in state and "quant_conv.bias" in state:
            state_sub = conv3d_to_linear_weight(pop_substate(state, "quant_conv"))
            weight, bias = state_sub["weight"], state_sub["bias"]
            if self._latents_mean is not None:
                weight, bias = self._fold_latent_norm(weight, bias)
            state["quant_conv.weight"] = weight
            state["quant_conv.bias"] = bias

        pop_substate(state, "decoder")
        pop_substate(state, "post_quant_conv")

    def _fold_latent_norm(self, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold ``(z - latents_mean) / latents_std`` into quant_conv.

        Both are per-output-channel affines on the same axis, so the composition is another
        affine and the encoder can emit normalized latents directly. Only the first ``z_dim``
        output rows are touched: the rest carry the logvar, which forward discards.

        ``weight`` is (padded_out, padded_in) here, before Linear transposes it.
        """
        scale = torch.ones(weight.shape[0], dtype=weight.dtype)
        shift = torch.zeros(bias.shape[0], dtype=bias.dtype)
        scale[: self.z_dim] = self._latents_inv_std.to(weight.dtype)
        shift[: self.z_dim] = (-self._latents_mean * self._latents_inv_std).to(bias.dtype)
        return weight * scale[:, None], bias * scale + shift

    def clear_cache(self):
        self._conv_idx = [0]
        self._feat_cache = [None] * self.cached_conv_count

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        logical_h: int,
        encoder_t_chunk_size: int | None = 4,
        logical_w: int = 0,
    ) -> tuple[ttnn.Tensor, int, int]:
        """
        encoder_t_chunk_size controls how the T dimension is processed:
            chunk_size => None  - single pass over all T frames, no feat_cache (fastest, most
                                  memory). Temporal downsamples take their context frame
                                  straight from frame 0 instead of the cache, which is
                                  numerically equivalent to the chunked path.
            chunk_size => N     - process frame 0 alone, then cache it, then process in chunks consisting
                                  of (N - 1) frames from the input plus 1 from the cache; trailing
                                  (T-1) % N frames are dropped. This behavior is intended to match
                                  the reference  WAN encoder behavior.
        """
        assert (
            encoder_t_chunk_size is None or encoder_t_chunk_size >= 4
        ), f"encoder_t_chunk_size must be None or >= 4, got {encoder_t_chunk_size}"
        B, T, H, W, C = x_BTHWC.shape
        if logical_w == 0:
            logical_w = W
        logger.info(f"WanEncoder.forward: T={T}, encoder_t_chunk_size={encoder_t_chunk_size}")

        if encoder_t_chunk_size is None:
            output_BTHWC, new_logical_h, new_logical_w = self.encoder(
                x_BTHWC,
                logical_h,
                feat_cache=None,
                feat_idx=None,
                logical_w=logical_w,
            )
        else:
            self.clear_cache()
            output_BTHWC = None

            # Frame 0 alone (required by downsample3d cache initialization)
            self._conv_idx = [0]
            out_BTHWC, new_logical_h, new_logical_w = self.encoder(
                x_BTHWC[:, :1, :, :, :],
                logical_h,
                feat_cache=self._feat_cache,
                feat_idx=self._conv_idx,
                logical_w=logical_w,
            )
            output_BTHWC = out_BTHWC

            # The intent here was to match the reference _encode loop in diffusers.AutoencoderKLWan.: frame 0 is
            # processed alone and cached (above), then iterate over groups of encoder_t_chunk_size (with 1 frame
            # in the chunk coming from the cache each time). Any trailing (T - 1) % chunk frames are dropped, because the
            # stride-2 downsample3d time_conv cannot form a valid temporal window from a partial tail.
            # NOTE: This change was proposed by Claude, so feel free to double check the validity. Also
            # if we no longer intend to match exactly the reference behavior we should revisit this algorithm.
            num_chunks = (T - 1) // encoder_t_chunk_size
            dropped = (T - 1) - num_chunks * encoder_t_chunk_size
            if dropped:
                logger.warning(
                    f"WanEncoder.forward: T={T} is not equal to 1 + k * encoder_t_chunk_size, where k is an integer representing a whole number of chunks (encoder_t_chunk_size={encoder_t_chunk_size})); "
                    f"dropping (T - 1) % encoder_t_chunk_size = {dropped} trailing frame(s)"
                )

            for i in range(num_chunks):
                self._conv_idx = [0]
                t_start = 1 + i * encoder_t_chunk_size
                t_end = t_start + encoder_t_chunk_size
                out_BTHWC, new_logical_h, new_logical_w = self.encoder(
                    x_BTHWC[:, t_start:t_end, :, :, :],
                    logical_h,
                    feat_cache=self._feat_cache,
                    feat_idx=self._conv_idx,
                    logical_w=logical_w,
                )
                output_BTHWC = ttnn.concat([output_BTHWC, out_BTHWC], dim=1)

            self.clear_cache()

        output_tile_BTHWC = ttnn.to_layout(output_BTHWC, ttnn.TILE_LAYOUT)
        output_tile_BTHWC = self.quant_conv(output_tile_BTHWC)
        output_BTHWC = ttnn.to_layout(output_tile_BTHWC, ttnn.ROW_MAJOR_LAYOUT)
        # Permute to channel second expected by torch
        output_BCTHW = ttnn.permute(output_BTHWC, (0, 4, 1, 2, 3))
        # Trim padding on output channels. Already normalized by latents_mean/std when
        # those were given to __init__, since quant_conv absorbed them.
        output_BCTHW = output_BCTHW[:, : self.z_dim, :, :, :]  # Get the mean
        return (output_BCTHW, new_logical_h, new_logical_w)


def get_neighbor_pad_num_links(ccl_manager, input_tensor, dim, mesh_axis=None):
    """
    Neighbor pad can use no more links than the product of the upper dimensions of the input tensor.
    """
    upper_dims = 1
    for i in range(dim):
        upper_dims *= input_tensor.shape[i]
    links = ccl_manager.links_for_axis(mesh_axis) if mesh_axis is not None else ccl_manager.num_links
    return min(upper_dims, links)


class WanVAEDecoderAdapter:
    """Torch-in (BCTHW), torch-out (output_type-dependent) VAE decoder for the Wan VAE.

    Applies per-channel mean/std denormalize, runs the TT-NN decoder, and transfers the result to
    host with output_type-aware pre-fn / permute. Weights are loaded via ``cache.load_model``;
    ``reload_weights``/``deallocate_weights`` are idempotent and intended to be driven by the
    caller (e.g. via ``register_coresident_exclusions``).
    """

    def __init__(
        self,
        *,
        checkpoint_name: str,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        height: int,
        width: int,
        num_frames: int,
        vae_t_chunk_size: int | None,
        vae_dtype: ttnn.DataType = ttnn.bfloat16,
        sdpa_t_fracture_w_only: bool = False,
    ) -> None:
        self.device = ccl_manager.mesh_device
        self._parallel_config = parallel_config
        self._ccl_manager = ccl_manager
        self._checkpoint_name = checkpoint_name
        self._t_chunk_size = vae_t_chunk_size

        self._torch_vae = AutoencoderKLWan.from_pretrained(checkpoint_name, subfolder="vae", trust_remote_code=True)

        full_latent_T = (num_frames - 1) // 4 + 1
        decoder_t_chunk_size = full_latent_T if vae_t_chunk_size is None else vae_t_chunk_size

        self._decoder = WanDecoder(
            base_dim=self._torch_vae.config.base_dim,
            z_dim=self._torch_vae.config.z_dim,
            dim_mult=self._torch_vae.config.dim_mult,
            num_res_blocks=self._torch_vae.config.num_res_blocks,
            attn_scales=self._torch_vae.config.attn_scales,
            temperal_downsample=self._torch_vae.config.temperal_downsample,
            out_channels=self._torch_vae.config.out_channels,
            is_residual=self._torch_vae.config.is_residual,
            mesh_device=self.device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=vae_dtype,
            sdpa_t_fracture_w_only=sdpa_t_fracture_w_only,
            height=height,
            width=width,
            t_chunk_size=decoder_t_chunk_size,
            cached=(vae_t_chunk_size is not None),
        )

        self._latents_mean = torch.tensor(self._torch_vae.config.latents_mean, dtype=self._torch_vae.dtype).view(
            1, self._torch_vae.config.z_dim, 1, 1, 1
        )
        self._latents_std = torch.tensor(self._torch_vae.config.latents_std, dtype=self._torch_vae.dtype).view(
            1, self._torch_vae.config.z_dim, 1, 1, 1
        )

    @property
    def config(self):
        return self._torch_vae.config

    @property
    def dtype(self):
        return self._torch_vae.dtype

    @property
    def decoder(self) -> WanDecoder:
        return self._decoder

    def torch_state_dict(self) -> dict[str, torch.Tensor]:
        return self._torch_vae.state_dict()

    def is_loaded(self) -> bool:
        return self._decoder.is_loaded()

    def deallocate_weights(self) -> None:
        self._decoder.deallocate_weights()

    def reload_weights(self) -> None:
        blocking_key = conv3d_blocking_hash(self._decoder)
        subfolder = f"vae_{blocking_key}" if blocking_key else "vae"
        cache.load_model(
            self._decoder,
            model_name=os.path.basename(self._checkpoint_name),
            subfolder=subfolder,
            parallel_config=self._parallel_config,
            mesh_shape=tuple(self.device.shape),
            mesh_device=self.device,
            get_torch_state_dict=lambda: self._torch_vae.state_dict(),
        )

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, *, output_type: str) -> torch.Tensor:
        import time as _hk_time  # EXPERIMENT (hkato): WAN_VAE_TIMING=1 section timing

        _hk_t = os.environ.get("WAN_VAE_TIMING") == "1"
        _hk_m = []

        def _hk_mark(name):
            if _hk_t:
                ttnn.synchronize_device(self.device)
                _hk_m.append((name, _hk_time.perf_counter()))

        _hk_mark("start")
        latents = latents.to(self._torch_vae.dtype)
        latents = latents * self._latents_std + self._latents_mean
        _hk_mark("scale")

        tt_latents_BTHWC, logical_h, logical_w = self._decoder.prepare_input(latents)
        if self._decoder.dtype == ttnn.bfloat16 and os.environ.get("WAN_VAE_IN_BF16", "1") != "0":
            # ttnn.from_torch converts float32 -> bfloat16 slowly on the host (same rounding as torch).
            tt_latents_BTHWC = tt_latents_BTHWC.to(torch.bfloat16).contiguous()
        _hk_mark("prep")
        shard_mapping = {
            self._parallel_config.height_parallel.mesh_axis: 2,
            self._parallel_config.width_parallel.mesh_axis: 3,
        }
        mesh_shape = tuple(self.device.shape)
        hk_in_mode = os.environ.get("WAN_VAE_IN_SHARDS", "2")
        hk_blocks = None
        if hk_in_mode == "2" and tt_latents_BTHWC.dtype == torch.bfloat16 and tt_latents_BTHWC.shape[0] == 1:
            # [1, T, H, W, C] -> [m0, m1, T, h, w, C] blocks in row-major device order, one contiguous copy.
            _, T_, H, W, C_ = tt_latents_BTHWC.shape
            h_ax = self._parallel_config.height_parallel.mesh_axis
            hf, wf = mesh_shape[h_ax], mesh_shape[1 - h_ax]
            hk_b = tt_latents_BTHWC.reshape(T_, hf, H // hf, wf, W // wf, C_).permute(1, 3, 0, 2, 4, 5)
            if h_ax == 1:
                hk_b = hk_b.transpose(0, 1)
            hk_blocks = hk_b.reshape(mesh_shape[0] * mesh_shape[1], T_, H // hf, W // wf, C_).contiguous()
            hk_host = ttnn.from_torch(
                hk_blocks,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0),
            )
        elif hk_in_mode == "1" and tt_latents_BTHWC.dtype == torch.bfloat16:
            # Per-device shards from torch views: ShardTensor2dMesh spends ~8 ms on the host for this 8 MB tensor.
            H, W = tt_latents_BTHWC.shape[2], tt_latents_BTHWC.shape[3]
            h_ax = self._parallel_config.height_parallel.mesh_axis
            w_ax = self._parallel_config.width_parallel.mesh_axis
            hs, ws = H // mesh_shape[h_ax], W // mesh_shape[w_ax]
            hk_shards = []
            for i0 in range(mesh_shape[0]):
                for i1 in range(mesh_shape[1]):
                    hi, wi = (i0, i1)[h_ax], (i0, i1)[w_ax]
                    hk_shards.append(
                        ttnn.from_torch(
                            tt_latents_BTHWC[:, :, hi * hs : (hi + 1) * hs, wi * ws : (wi + 1) * ws, :].contiguous(),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                        )
                    )
            hk_host = ttnn.from_host_shards(hk_shards, ttnn.MeshShape(*mesh_shape))
        else:
            hk_host = typed_tensor_2dshard(
                tt_latents_BTHWC,
                self.device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                shard_mapping=shard_mapping,
                dtype=self._decoder.dtype,
                on_host=True,
            )
        _hk_mark("shard")
        hk_dev = ttnn.to_device(hk_host, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if hk_blocks is not None:
            hk_dev = ttnn.reshape(hk_dev, (1,) + tuple(hk_blocks.shape[1:]))
        if os.environ.get("WAN_VAE_IN_CHECK") == "1":  # compare with the ShardTensor2dMesh upload
            hk_ref = typed_tensor_2dshard(
                tt_latents_BTHWC,
                self.device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                shard_mapping=shard_mapping,
                dtype=self._decoder.dtype,
            )
            hk_eq = all(
                torch.equal(ttnn.to_torch(a), ttnn.to_torch(b))
                for a, b in zip(ttnn.get_device_tensors(hk_dev), ttnn.get_device_tensors(hk_ref))
            )
            print("WAN_VAE_IN_CHECK equal=%s" % hk_eq, flush=True)
        tt_latents_BTHWC = hk_dev

        _hk_mark("input")
        self.reload_weights()
        _hk_mark("reload")
        _hk_prof = None
        if os.environ.get("WAN_VAE_PROFILE") == "1" and getattr(self, "_hk_calls", 0) >= 1:  # EXPERIMENT (hkato)
            import cProfile

            _hk_prof = cProfile.Profile()
            _hk_prof.enable()
        self._hk_calls = getattr(self, "_hk_calls", 0) + 1
        tt_video_BCTHW, new_logical_h, new_logical_w = self._decoder(
            tt_latents_BTHWC, logical_h, t_chunk_size=self._t_chunk_size, logical_w=logical_w
        )
        if _hk_prof is not None:
            _hk_prof.disable()
            import pstats

            pstats.Stats(_hk_prof).sort_stats("tottime").print_stats(30)
        _hk_mark("decoder")

        concat_dims = [None, None]
        concat_dims[self._parallel_config.height_parallel.mesh_axis] = 3
        concat_dims[self._parallel_config.width_parallel.mesh_axis] = 4
        d2h_permute = (0, 2, 3, 4, 1) if output_type in ("np", "uint8") else None

        if output_type == "uint8":
            pre_fn = float_to_uint8
        elif output_type == "np":
            pre_fn = float_to_unit_range
        else:
            pre_fn = None

        video_torch = fast_device_to_host(
            tt_video_BCTHW,
            self.device,
            concat_dims,
            ccl_manager=self._ccl_manager,
            pre_transfer_fn=pre_fn,
            permute=d2h_permute,
        )

        _hk_mark("d2h")
        if _hk_t:
            print(
                "WAN_VAE_TIMING "
                + " ".join("%s=%.1f" % (b[0], (b[1] - a[1]) * 1e3) for a, b in zip(_hk_m, _hk_m[1:]))
                + " ms",
                flush=True,
            )
        if d2h_permute is not None:
            # Output is (B, T, H, W, C) — trim height and width.
            return video_torch[:, :, :new_logical_h, :new_logical_w, :]
        # Output is (B, C, T, H, W) — trim height and width.
        return video_torch[:, :, :, :new_logical_h, :new_logical_w]
