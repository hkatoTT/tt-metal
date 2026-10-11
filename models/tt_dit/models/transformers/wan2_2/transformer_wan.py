# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from diffusers import WanTransformer3DModel as TorchWanTransformer3DModel
from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed as TorchWanRotaryPosEmbed
from loguru import logger

if not os.environ.get("TT_DIT_DEBUG"):
    logger.disable(__name__)


import ttnn

from ....layers.embeddings import WanPatchEmbed, WanTimeTextImageEmbedding
from ....layers.feedforward import ParallelFeedForward
from ....layers.linear import Linear
from ....layers.module import Module, ModuleList, Parameter
from ....layers.normalization import DistributedLayerNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils import cache
from ....utils.matmul import (
    FusedMMRSConfig,
    get_matmul_config,
    get_matmul_core_grid,
    mm_grid,
    register_fused_mmrs_configs,
    register_matmul_configs,
    resolves_fused_mmrs_config,
)
from ....utils.mochi import get_rot_transformation_mat
from ....utils.padding import pad_vision_seq_parallel
from ....utils.substate import pop_substate, rename_substate
from ....utils.tensor import bf16_tensor, float32_tensor, from_torch, unflatten
from ....utils.tracing import traced_function
from .attention_wan import WanAttention, hk_xattn_compact_layout


class WanTransformerBlock(Module):
    def __init__(
        self,
        *,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attention_norm: bool = True,
        eps: float = 1e-6,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
        ffn_fsdp: bool = True,
        sdpa_chunk_size_overrides: dict | None = None,
        lora_enabled: bool = False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.cross_attention_norm = cross_attention_norm
        self.eps = eps

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        fsdp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        self.norm1 = DistributedLayerNorm(
            dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.attn1 = WanAttention(
            dim=dim,
            num_heads=num_heads,
            eps=eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            is_self=True,
            sdpa_chunk_size_overrides=sdpa_chunk_size_overrides,
            lora_enabled=lora_enabled,
        )

        self.attn2 = WanAttention(
            dim=dim,
            num_heads=num_heads,
            eps=eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            is_self=False,
            sdpa_chunk_size_overrides=sdpa_chunk_size_overrides,
            lora_enabled=lora_enabled,
        )

        self.norm2 = (
            DistributedLayerNorm(
                dim,
                norm_eps=eps,
                norm_elementwise_affine=True,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
            )
            if cross_attention_norm
            else None
        )

        self.ffn = ParallelFeedForward(
            dim,
            inner_dim=ffn_dim,
            activation_fn="gelu_tanh",
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            # ffn_fsdp=False keeps the FFN weights resident (TP-sharded only) to skip their per-layer gathers
            fsdp_mesh_axis=fsdp_mesh_axis if ffn_fsdp else None,
            lora_enabled=lora_enabled,
        )

        self.norm3 = DistributedLayerNorm(
            dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.scale_shift_table = Parameter(
            total_shape=[1, 1, 6, dim],
            mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
            device=mesh_device,
            dtype=ttnn.float32,
        )

        # WAN_MM_FP32_DEST=ffn|all: accumulate the FFN matmuls in bf16 dest instead of fp32
        ffn_fp32_dest = os.environ.get("WAN_MM_FP32_DEST", "") not in ("ffn", "all")
        self.ff_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=ffn_fp32_dest,
            packer_l1_acc=True,
        )
        if not ffn_fp32_dest:
            # bf16 dest holds 8 tiles instead of 4, so the FFN blockings were re-swept with 4x2 subblocks
            # (MM_SWEEP_FP32_DEST=0, WH 4x8, 480p). They are invalid with fp32 dest, hence registered only here.
            register_matmul_configs(
                {
                    "8x8": {
                        (4096, 5120, 3456): (8, 4, 14, (4, 2)),  # ff1 1916.8 us (fp32 dest: 2053.3)
                        (8192, 5120, 3456): (8, 5, 14, (4, 2)),  # ff1, CFG-batched rows: 3622.6 us
                    }
                }
            )
            register_fused_mmrs_configs(
                {
                    ttnn.CoreCoord(8, 9): {
                        # ff2 fused MM+RS 2307.7 us (fp32 dest: 2757.3)
                        (4096, 3456, 5120): FusedMMRSConfig(ttnn.CoreCoord(8, 7), 8, 3, 10, 4, 2, None, 1),
                        # ff2, CFG-batched rows: 4129.9 us
                        (8192, 3456, 5120): FusedMMRSConfig(ttnn.CoreCoord(8, 7), 8, 3, 10, 4, 2, None, 1),
                    }
                }
            )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "ffn.net.0.proj", "ffn.ff1")
        rename_substate(state, "ffn.net.2", "ffn.ff2")

        if "scale_shift_table" in state:
            state["scale_shift_table"] = state["scale_shift_table"].unsqueeze(0)

    def forward(
        self,
        spatial_1BND: ttnn.Tensor,
        prompt_1BLP: ttnn.Tensor,
        temb_1BTD: ttnn.Tensor,
        N: int,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
        cross_attn_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        spatial_1BND: fractured N on SP, fractured D on TP
        prompt_1BLP: replicated on SP, replicated D on TP
        temb_1BTD: replicated on SP, fractured D on TP
        N: logical sequence length of the spatial input
        rope_cos_BANH: fractured N on SP, A (num_heads) on TP
        rope_sin_BANH: fractured N on SP, A (num_heads) on TP
        trans_mat: replicated on SP, replicated D on TP

        Outputs:
        spatial_1BND: fractured N on SP, fractured D on TP
        """

        if isinstance(temb_1BTD, (tuple, list)):
            # EXPERIMENT (hkato): WAN_TEMB_PRESLICE=1 -- the timestep embedding arrives pre-chunked (sliced once per
            # step for all blocks); the constant table is chunked once per block. Same fp32 adds, no per-block
            # untilize/slice/tilize chains.
            hk_tab = getattr(self, "_hk_table_chunks", None)
            if hk_tab is None or hk_tab[0] is not self.scale_shift_table.data:
                hk_tab = (self.scale_shift_table.data, ttnn.chunk(self.scale_shift_table.data, 6, dim=2))
                self._hk_table_chunks = hk_tab
            hk_fold = os.environ.get("WAN_TEMB_FOLD") == "1"
            if hk_fold:  # EXPERIMENT (hkato): "1 +" folded into the cached table, gates added straight to bf16
                if len(hk_tab) == 2:
                    t = hk_tab[1]
                    hk_tab = (hk_tab[0], t, (t[0], t[1] + 1.0, t[2], t[3], t[4] + 1.0, t[5]))
                    self._hk_table_chunks = hk_tab
                tf, tm = hk_tab[2], temb_1BTD
                shift_msa_1B1D = tf[0] + tm[0]
                scale_msa_1B1D = tf[1] + tm[1]  # already 1 + scale
                gate_msa_1B1D = ttnn.add(tf[2], tm[2], dtype=ttnn.bfloat16)
                c_shift_msa_1B1D = tf[3] + tm[3]
                c_scale_msa_1B1D = tf[4] + tm[4]  # already 1 + scale
                c_gate_msa_1B1D = ttnn.add(tf[5], tm[5], dtype=ttnn.bfloat16)
            else:
                shift_msa_1B1D, scale_msa_1B1D, gate_msa_1B1D, c_shift_msa_1B1D, c_scale_msa_1B1D, c_gate_msa_1B1D = (
                    tab + tmb for tab, tmb in zip(hk_tab[1], temb_1BTD)
                )
        else:
            hk_fold = False
            assert temb_1BTD.shape[2] == 6, "wan2.2 14b expects 6 chunks in timestep embedding"

            shifted_temb_1BTD = self.scale_shift_table.data + temb_1BTD
            (
                shift_msa_1B1D,
                scale_msa_1B1D,
                gate_msa_1B1D,
                c_shift_msa_1B1D,
                c_scale_msa_1B1D,
                c_gate_msa_1B1D,
            ) = ttnn.chunk(shifted_temb_1BTD, 6, dim=2)

        # NOTE: workaround - addcmul (fused and unfused) is less accurate with fp32 gate input
        if not hk_fold:
            gate_msa_1B1D = ttnn.typecast(gate_msa_1B1D, dtype=ttnn.bfloat16)
            c_gate_msa_1B1D = ttnn.typecast(c_gate_msa_1B1D, dtype=ttnn.bfloat16)

        norm_dtype = ttnn.bfloat8_b if os.environ.get("WAN_AGMM_IN_BF8") == "1" else None  # EXPERIMENT
        spatial_normed_1BND = self.norm1(
            spatial_1BND,
            dynamic_weight=(scale_msa_1B1D if hk_fold else 1.0 + scale_msa_1B1D),
            dynamic_bias=shift_msa_1B1D,
            dtype=norm_dtype,
        )

        # Self attention on spatial with fused residual addcmul
        # Fuses: spatial_1BND = spatial_1BND + to_out(attn_output) * gate_msa_1B1D
        spatial_1BND = self.attn1(
            spatial_1BND=spatial_normed_1BND,
            N=N,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trans_mat=trans_mat,
            addcmul_residual=spatial_1BND,
            addcmul_gate=gate_msa_1B1D,
        )

        # Cross attention on prompt
        spatial_normed_1BND = self.norm2(spatial_1BND, dtype=norm_dtype)

        if os.environ.get("WAN_XATTN_FUSED_ADD") == "1":  # EXPERIMENT: residual add fused into cross to_out
            if getattr(self, "_ones_gate", None) is None or tuple(self._ones_gate.shape) != tuple(gate_msa_1B1D.shape):
                self._ones_gate = ttnn.ones_like(gate_msa_1B1D)
            spatial_1BND = self.attn2(
                spatial_1BND=spatial_normed_1BND,
                N=N,
                prompt_1BLP=prompt_1BLP,
                cross_attn_mask=cross_attn_mask,
                addcmul_residual=spatial_1BND,
                addcmul_gate=self._ones_gate,
            )
        else:
            attn_output_1BND = self.attn2(
                spatial_1BND=spatial_normed_1BND,
                N=N,
                prompt_1BLP=prompt_1BLP,
                cross_attn_mask=cross_attn_mask,
            )
            spatial_1BND = spatial_1BND + attn_output_1BND

        # Feed Forward
        spatial_normed_1BND = self.norm3(
            spatial_1BND,
            dynamic_weight=(c_scale_msa_1B1D if hk_fold else 1.0 + c_scale_msa_1B1D),
            dynamic_bias=c_shift_msa_1B1D,
            dtype=norm_dtype,
        )

        if self.ccl_manager.topology == ttnn.Topology.Linear:
            if self.parallel_config.tensor_parallel.factor > 1:
                spatial_normed_1BND = self.ccl_manager.all_gather_persistent_buffer(
                    spatial_normed_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
                )

            spatial_ff_1BND = self.ffn(spatial_normed_1BND, compute_kernel_config=self.ff_compute_kernel_config)
            # spatial_1BND = spatial_1BND + spatial_ff_1BND * c_gate_msa_1B1D
            # NOTE: higher precision compute config in addcmul may be needed for correctness
            spatial_1BND = ttnn.addcmul(spatial_1BND, spatial_ff_1BND, c_gate_msa_1B1D)
        elif os.environ.get("WAN_FFN2_UNFUSED", "0") == "1" or not resolves_fused_mmrs_config(
            spatial_normed_1BND.padded_shape[-2],
            self.ffn_dim // self.parallel_config.tensor_parallel.factor,
            self.dim,
            mm_grid(self.mesh_device),
        ):
            # Unfused ff1 (AGMM) -> ff2 (matmul + reduce-scatter) -> addcmul when no measured or rule-derived
            # fused MM+RS blocking resolves for this (M, K, N) on the device grid: the fused op's default
            # fallback is slower than not fusing (Wan 480p on WH: 5.18 ms vs 3.19 ms). Same gate as MiniMax-H3.
            # WAN_FFN2_UNFUSED=1 forces this path.
            spatial_ff_1BND = self.ffn(
                spatial_normed_1BND,
                compute_kernel_config=self.ff_compute_kernel_config,
                parallel_config=self.parallel_config,
            )
            spatial_1BND = ttnn.addcmul(spatial_1BND, spatial_ff_1BND, c_gate_msa_1B1D)
        else:
            # Fused FFN + addcmul at the RS final write step (Phase 2).
            # Both 'a' (spatial_1BND) and 'b' (c_gate_msa_1B1D) are already at [D/tp] size
            # on each TP device — no AllGather or scatter matmul needed.
            spatial_1BND = self.ffn.forward_fused_addcmul(
                spatial_normed_1BND,
                spatial_1BND,
                c_gate_msa_1B1D,
                scalar=1.0,
                compute_kernel_config=self.ff_compute_kernel_config,
                parallel_config=self.parallel_config,
            )

        return spatial_1BND


class WanTransformer3DModel(Module):
    def __init__(
        self,
        *,
        patch_size: tuple = (1, 2, 2),
        num_heads: int = 40,
        dim: int = 5120,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        rope_max_seq_len: int = 1024,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = True,
        ffn_fsdp: bool = True,
        model_type: str = "t2v",
        output_dtype: ttnn.DataType = ttnn.float32,
        lora_enabled: bool = False,
    ) -> None:
        super().__init__()

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.is_fsdp = is_fsdp
        self.lora_enabled = lora_enabled
        self.fsdp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis if is_fsdp else None
        self.model_type = model_type
        self.cached_rope_features = {}
        self.output_dtype = output_dtype

        assert model_type in ["t2v", "i2v"], "model_type must be either t2v or i2v"
        if model_type == "i2v":
            in_channels = 36
        else:
            assert in_channels == 16, "in_channels must be 16 for t2v"

        self.patch_size = patch_size
        self.dim = dim

        self.out_channels = out_channels

        # NOTE: Fallback
        self.rope = TorchWanRotaryPosEmbed(
            dim // num_heads,
            patch_size,
            rope_max_seq_len,
        )

        self.patch_embedding = WanPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=dim,
            mesh_device=mesh_device,
            tp_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=dim,
            time_freq_dim=freq_dim,
            time_proj_dim=dim * 6,
            text_embed_dim=text_dim,
            mesh_device=self.mesh_device,
            tp_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.blocks = ModuleList(
            WanTransformerBlock(
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                cross_attention_norm=cross_attn_norm,
                eps=eps,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                is_fsdp=is_fsdp,
                ffn_fsdp=ffn_fsdp,
                lora_enabled=lora_enabled,
            )
            for i in range(num_layers)
        )

        self.norm_out = DistributedLayerNorm(
            dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.proj_out = Linear(
            dim,
            patch_size[0] * patch_size[1] * patch_size[2] * self.out_channels,
            bias=True,
            mesh_device=mesh_device,
        )

        self.scale_shift_table = Parameter(
            total_shape=[1, 2, dim],
            device=mesh_device,
            mesh_axes=[None, None, parallel_config.tensor_parallel.mesh_axis],
            dtype=ttnn.float32,
        )

        self.hifi4_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def save(self, directory: str | Path, /, *, prefix: str = "") -> None:
        super().save(directory, prefix=prefix)

        directory = Path(directory)

        # Torch fallbacks
        torch.save(self.rope.state_dict(), directory / f"{prefix}rope.pt")

    def load(self, directory: str | Path, /, *, prefix: str = "") -> None:
        super().load(directory, prefix=prefix)

        directory = Path(directory)

        # Torch fallbacks
        self.rope.load_state_dict(torch.load(directory / f"{prefix}rope.pt"))

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Torch fallbacks
        self.rope.load_state_dict(pop_substate(state, "rope"))

    def get_rope_features(self, hidden_states):
        if tuple(hidden_states.shape) not in self.cached_rope_features:
            rope_features = self.prepare_rope_features(hidden_states)
            self.cached_rope_features[tuple(hidden_states.shape)] = rope_features
        return self.cached_rope_features[tuple(hidden_states.shape)]

    def prepare_rope_features(self, hidden_states):
        """
        Given video input, compute RoPE features.
        Return tensors on device.
        """
        logger.info(f"Preparing rope features for shape {hidden_states.shape}")
        rope_cos, rope_sin = self.rope(hidden_states)

        # Convert to TT tensors with proper sharding
        rope_cos_1HND = rope_cos.permute(0, 2, 1, 3)
        rope_sin_1HND = rope_sin.permute(0, 2, 1, 3)

        rope_cos_1HND = pad_vision_seq_parallel(
            rope_cos_1HND, num_devices=self.parallel_config.sequence_parallel.factor
        )
        rope_sin_1HND = pad_vision_seq_parallel(
            rope_sin_1HND, num_devices=self.parallel_config.sequence_parallel.factor
        )

        trans_mat = get_rot_transformation_mat()

        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        tt_rope_cos_1HND = from_torch(
            rope_cos_1HND,
            device=self.mesh_device,
            dtype=ttnn.float32,
            mesh_axes=[..., sp_axis, None],
        )
        tt_rope_sin_1HND = from_torch(
            rope_sin_1HND,
            device=self.mesh_device,
            dtype=ttnn.float32,
            mesh_axes=[..., sp_axis, None],
        )
        tt_trans_mat = bf16_tensor(trans_mat, device=self.mesh_device)

        logger.info(f"TT rope cos shape: {tt_rope_cos_1HND.shape}")
        logger.info(f"TT rope sin shape: {tt_rope_sin_1HND.shape}")
        logger.info(f"TT trans mat shape: {tt_trans_mat.shape}")

        return tt_rope_cos_1HND, tt_rope_sin_1HND, tt_trans_mat

    def set_cross_kv_cache_enabled(self, enabled: bool) -> None:
        for block in self.blocks:
            block.attn2.set_cross_kv_cache_enabled(enabled)

    def prepare_text_conditioning(self, encoder_hidden_states, hk_max_len=None):
        # new text conditioning: drop cross-attention K/V cached for previous prompt tensors
        hk_set = os.environ.get("WAN_XATTN_COMPACT") == "1" and hk_max_len is not None  # EXPERIMENT (hkato)
        hk_layout = hk_xattn_compact_layout(hk_max_len) if hk_set else None
        for block in self.blocks:
            block.attn2.clear_cross_kv_cache()
            if hk_set:
                block.attn2._hk_xcompact = hk_layout
        if hk_set:
            logger.info(f"cross-attention text compaction for {hk_max_len} tokens: {hk_layout}")
        tt_prompt_1BLP = self.condition_embedder.forward_text(encoder_hidden_states)

        logger.info(f"TT prompt shape: {tt_prompt_1BLP.shape}")
        return tt_prompt_1BLP

    def prepare_timestep_conditioning(self, timestep):
        tt_temb_11BD, tt_timestep_proj_1BTD = self.condition_embedder.forward_timestep(timestep, timestep_seq_len=None)
        tt_timestep_proj_1BTD = unflatten(ttnn.squeeze(tt_timestep_proj_1BTD, -2), -1, (6, -1))
        logger.info(f"TT temb shape: {tt_temb_11BD.shape}")
        logger.info(f"TT timestep proj shape: {tt_timestep_proj_1BTD.shape}")
        return tt_temb_11BD, tt_timestep_proj_1BTD

    def hk_prefill_temb(self, timesteps: list[float]) -> None:
        """EXPERIMENT (hkato, WAN_TEMB_BATCH=1): the timestep embeddings of the whole run in one forward.

        The embedding only depends on t; one step's MLP (fp32, M = 1) costs ~4 ms, mostly weight reads. Placing the
        timesteps on the row axis makes one M = len(timesteps) pass, and each step takes its row.
        """
        # a fixed row count (repeating the last t), so the short warm-up run compiles every program of the real run
        rows = list(timesteps) + [timesteps[-1]] * max(
            0, int(os.environ.get("WAN_TEMB_BATCH_ROWS", "40")) - len(timesteps)
        )
        t_all = float32_tensor(torch.tensor(rows, dtype=torch.float32).reshape(1, 1, -1, 1), device=self.mesh_device)
        temb_all, proj_all = self.condition_embedder.forward_timestep(t_all, timestep_seq_len=None)
        cache = {}
        for i, v in enumerate(rows):
            temb = ttnn.slice(temb_all, [0, 0, i, 0], [1, 1, i + 1, temb_all.shape[3]])
            proj = ttnn.slice(proj_all, [0, 0, i, 0], [1, 1, i + 1, proj_all.shape[3]])
            proj = unflatten(ttnn.squeeze(proj, -2), -1, (6, -1))
            cache.setdefault(float(v), (temb, proj))
        self._hk_temb_cache = cache

    def prepare_conditioning(self, timestep, encoder_hidden_states):
        """
        Given inputs, execute the combined timestep and text embedding.
        Return tensors on device.
        """
        tt_temb_11BD, tt_timestep_proj_1BTD = self.prepare_timestep_conditioning(timestep)
        tt_prompt_1BLP = self.prepare_text_conditioning(encoder_hidden_states)

        logger.info(f"TT temb shape: {tt_temb_11BD.shape}")
        logger.info(f"TT timestep proj shape: {tt_timestep_proj_1BTD.shape}")
        logger.info(f"TT prompt shape: {tt_prompt_1BLP.shape}")
        return tt_temb_11BD, tt_timestep_proj_1BTD, tt_prompt_1BLP

    def preprocess_spatial_input_host(self, spatial):
        B, C, F, H, W = spatial.shape
        logger.info(f"Preprocessing spatial input with shape {spatial.shape}")
        assert B == 1, "Batch size must be 1"
        pF, pH, pW = self.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
        N = patch_F * patch_H * patch_W

        # Patchify video input
        spatial = spatial.reshape(B, C, patch_F, pF, patch_H, pH, patch_W, pW)
        spatial = spatial.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(1, B, N, pF * pH * pW * C)
        logger.info(f"spatial input after patchifying: {spatial.shape}")

        spatial = pad_vision_seq_parallel(spatial, num_devices=self.parallel_config.sequence_parallel.factor)
        logger.info(f"spatial input after padding: {spatial.shape}")

        return spatial, N

    def preprocess_spatial_input(self, spatial):
        spatial, N = self.preprocess_spatial_input_host(spatial)
        spatial = bf16_tensor(
            spatial, device=self.mesh_device, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis, shard_dim=-2
        )
        logger.info(f"TT spatial shape: {spatial.shape}")
        return spatial, N

    def postprocess_spatial_output_host(self, spatial_1BND, F, H, W, N):
        """
        This is the reverse of preprocess_spatial_input
        Input is of shape: 1 B (patch_F patch_H patch_W) (pF pH pW C)
        returns shape: B C F H W
        """
        assert len(spatial_1BND.shape) == 4
        assert spatial_1BND.shape[0] == 1
        B = spatial_1BND.shape[1]
        pF, pH, pW = self.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
        logger.info(f"Postprocessing spatial output with shape {spatial_1BND.shape}")
        spatial_BND = spatial_1BND.squeeze(0)

        spatial_BND = spatial_BND[:, :N]  # Slice out sequence-parallel padding tokens
        logger.info(f"Spatial output after slicing: {spatial_BND.shape}")

        spatial_patches = spatial_BND.reshape(B, patch_F, patch_H, patch_W, pF, pH, pW, self.out_channels)
        logger.info(f"Spatial output after reshaping: {spatial_patches.shape}")

        spatial_BCFHW = spatial_patches.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, self.out_channels, F, H, W)
        logger.info(f"Spatial output after permuting: {spatial_BCFHW.shape}")
        return spatial_BCFHW

    def postprocess_spatial_output(self, spatial_1BND, F, H, W, N):
        """
        This is the reverse of preprocess_spatial_input
        Input is of shape: 1 B (patch_F patch_H patch_W) (pF pH pW C)
        returns shape: B C F H W
        """
        assert len(spatial_1BND.shape) == 4
        assert spatial_1BND.shape[0] == 1
        # Gather sequence-parallel output
        spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
            spatial_1BND, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
        )

        spatial_1BND = ttnn.to_torch(ttnn.get_device_tensors(spatial_1BND)[0])

        spatial_BCFHW = self.postprocess_spatial_output_host(spatial_1BND, F, H, W, N)
        return spatial_BCFHW

    def forward(
        self,
        spatial: torch.Tensor,
        prompt: ttnn.Tensor,
        timestep: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Inputs are all torch tensors
            y is an optional argument for image-to-video generation.
            We assume that preprocessing has already been done for y and y has the same shape as spatial.
        Output is torch tensor
        """

        if self.model_type == "i2v":
            assert y is not None, "y must be provided for image-to-video generation"

        B, C, F, H, W = spatial.shape
        pF, pH, pW = self.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
        N = patch_F * patch_H * patch_W

        rope_cos_1HND, rope_sin_1HND, trans_mat = self.get_rope_features(spatial)

        temb_11BD, timestep_proj_1BTD, prompt_1BLP = self.prepare_conditioning(timestep, prompt)

        # Concatenate spatial and y along the channel dimension
        if self.model_type == "i2v":
            print(spatial.shape, y.shape)
            spatial = torch.cat([spatial, y], dim=1)

        spatial_1BNI, N = self.preprocess_spatial_input(spatial)

        spatial_1BND = self.patch_embedding(spatial_1BNI)

        for idx, block in enumerate(self.blocks):
            spatial_1BND = block(
                spatial_1BND=spatial_1BND,
                prompt_1BLP=prompt_1BLP,
                temb_1BTD=timestep_proj_1BTD,
                N=N,
                rope_cos=rope_cos_1HND,
                rope_sin=rope_sin_1HND,
                trans_mat=trans_mat,
            )

        scale_shift_1BSD = self.scale_shift_table.data + temb_11BD
        shift_11BD, scale_11BD = ttnn.chunk(scale_shift_1BSD, 2, -2)

        spatial_norm_1BND = self.norm_out(
            spatial_1BND, dynamic_weight=(1 + scale_11BD), dynamic_bias=shift_11BD, dtype=ttnn.float32
        )

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_norm_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_norm_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        proj_out_1BNI = self.proj_out(
            spatial_norm_1BND, compute_kernel_config=self.hifi4_compute_kernel_config, dtype=ttnn.float32
        )

        spatial_out = self.postprocess_spatial_output(proj_out_1BNI, F, H, W, N)

        return spatial_out

    def _synced_time(self) -> float:
        ttnn.synchronize_device(self.mesh_device)
        return time.perf_counter()

    def inner_step(
        self, spatial_1BNI, prompt_1BLP, rope_cos_1HND, rope_sin_1HND, trans_mat, N, timestep, gather_output=True
    ):
        """
        Reduced forward function which assumes outer loop has cached certain inputs that are step independent:
            - prompt_1BLP
            - rope_cos_1HND
            - rope_sin_1HND
            - trans_mat
            - N

        Spatial input is a tensor with layout `1 B (patch_F patch_H patch_W) (pF pH pW C)`.
        Spatial output is an fp32 ttnn.Tensor on device with same layout.
        """
        # WAN_STEP_TIMING=1: synced wall time of each section of one forward (adds device syncs)
        marks = [self._synced_time()] if os.environ.get("WAN_STEP_TIMING", "0") == "1" else None
        temb_11BD, timestep_proj_1BTD = self.prepare_timestep_conditioning(timestep)
        if marks:
            marks.append(self._synced_time())

        spatial_1BND = self.patch_embedding(spatial_1BNI)
        if marks:
            marks.append(self._synced_time())

        for block in self.blocks:
            spatial_1BND = block(
                spatial_1BND=spatial_1BND,
                prompt_1BLP=prompt_1BLP,
                temb_1BTD=timestep_proj_1BTD,
                N=N,
                rope_cos=rope_cos_1HND,
                rope_sin=rope_sin_1HND,
                trans_mat=trans_mat,
            )
        if marks:
            marks.append(self._synced_time())
        scale_shift_1BSD = self.scale_shift_table.data + temb_11BD
        shift_11BD, scale_11BD = ttnn.chunk(scale_shift_1BSD, 2, -2)

        spatial_norm_1BND = self.norm_out(
            spatial_1BND, dynamic_weight=(1 + scale_11BD), dynamic_bias=shift_11BD, dtype=ttnn.float32
        )

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_norm_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_norm_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        spatial_1BNI = self.proj_out(
            spatial_norm_1BND, compute_kernel_config=self.hifi4_compute_kernel_config, dtype=self.output_dtype
        )

        # Gather fp32 spatial output across sequence parallel devices (remains on device)
        if gather_output:
            spatial_1BNI = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BNI, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
            )

        if marks:
            marks.append(self._synced_time())
            temb_ms, patch_ms, blocks_ms, post_ms = ((b - a) * 1e3 for a, b in zip(marks, marks[1:]))
            # print: logger.info from this module did not reach the perf-test log
            print(
                f"WAN_TIMING inner temb={temb_ms:.2f} patch={patch_ms:.2f} blocks={blocks_ms:.2f} "
                f"post={post_ms:.2f} ms",
                flush=True,
            )

        return spatial_1BNI

    def _hk_proj_out_tp(self, spatial_norm_1BND):
        """EXPERIMENT (hkato): proj_out with K split over the TP axis + all-reduce (reduce-scatter + all-gather over rows)
        of the fp32 partials; the bias is added once after the reduction."""
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        if getattr(self, "_hk_proj_out_w_tp", None) is None:
            w = ttnn.to_torch(ttnn.get_device_tensors(self.proj_out.weight.data)[0])
            dims = [None, None]
            dims[tp_axis] = -2
            self._hk_proj_out_w_tp = ttnn.from_torch(
                w,
                dtype=self.proj_out.weight.data.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=tuple(dims)
                ),
            )
        w_tp = self._hk_proj_out_w_tp
        M, K, N = spatial_norm_1BND.padded_shape[-2], spatial_norm_1BND.padded_shape[-1], w_tp.padded_shape[-1]
        core_grid = get_matmul_core_grid(self.mesh_device)
        part = ttnn.experimental.minimal_matmul(
            input_tensor=spatial_norm_1BND,
            weight_tensor=w_tp,
            bias_tensor=None,
            config=get_matmul_config(M, K, N, core_grid, None),
            compute_kernel_config=self.hifi4_compute_kernel_config,
            dtype=ttnn.float32,
        )
        part = self.ccl_manager.reduce_scatter(part, dim=2, mesh_axis=tp_axis)
        part = self.ccl_manager.all_gather(part, dim=2, mesh_axis=tp_axis, use_hyperparams=False)
        return ttnn.add(part, self.proj_out.bias.data, dtype=self.output_dtype)

    def inner_step_cfg(
        self,
        spatial_1BNI,
        prompt_1BLP,
        negative_prompt_1BLP,
        rope_cos_1HND,
        rope_sin_1HND,
        trans_mat,
        N,
        timestep,
        gather_output=True,
    ):
        """inner_step for the cond and uncond passes of classifier-free guidance at once.

        Both passes share the latent and the timestep and differ only in the text, so the per-device spatial
        rows of the two passes are stacked into one [1, 1, 2n, D] tensor: every row-wise op (norms, projections,
        FFN) runs once on 2n rows, with one FSDP weight gather instead of two. WanAttention splits the passes
        again where needed (blocks see `_cfg_batch == 2`). Returns (cond, uncond), each laid out like inner_step's.
        """
        hk_marks = [self._synced_time()] if os.environ.get("WAN_FWD_TIMING") == "1" else None
        hk_fine = os.environ.get("HK_FWD_FINE") == "1" and hk_marks is not None
        hk_cache = getattr(self, "_hk_temb_cache", None)
        hk_t = getattr(self, "_hk_cur_t", None)
        if hk_cache is not None and hk_t in hk_cache:  # EXPERIMENT (hkato): WAN_TEMB_BATCH (hk_prefill_temb)
            temb_11BD, timestep_proj_1BTD = hk_cache[hk_t]
            if os.environ.get("HK_TEMB_CHECK") == "1":
                ref_temb, ref_proj = self.prepare_timestep_conditioning(timestep)
                for name, a, b in (("temb", temb_11BD, ref_temb), ("proj", timestep_proj_1BTD, ref_proj)):
                    ta = ttnn.to_torch(ttnn.get_device_tensors(a)[0]).float()
                    tb = ttnn.to_torch(ttnn.get_device_tensors(b)[0]).float()
                    print(
                        f"HK_TEMB_CHECK {name} {tuple(ta.shape)} {tuple(tb.shape)} maxdiff {float((ta - tb).abs().max()):.3e}",
                        flush=True,
                    )
        else:
            temb_11BD, timestep_proj_1BTD = self.prepare_timestep_conditioning(timestep)
        if os.environ.get("WAN_TEMB_PRESLICE") == "1":  # EXPERIMENT (hkato): chunk once per step for all blocks
            timestep_proj_1BTD = tuple(ttnn.chunk(timestep_proj_1BTD, 6, dim=2))
        if hk_fine:
            hk_t1 = self._synced_time()
        spatial_1BND = self.patch_embedding(spatial_1BNI)
        if hk_fine:
            hk_t2 = self._synced_time()
        spatial_1BND = ttnn.concat([spatial_1BND, spatial_1BND], dim=2)
        if hk_marks:
            hk_marks.append(self._synced_time())
            if hk_fine:
                print(
                    f"WAN_FWD_FINE temb={(hk_t1 - hk_marks[0]) * 1e3:.2f} patch={(hk_t2 - hk_t1) * 1e3:.2f} "
                    f"concat={(hk_marks[-1] - hk_t2) * 1e3:.2f} ms",
                    flush=True,
                )

        for block in self.blocks:
            block.attn1._cfg_batch = 2
            block.attn2._cfg_batch = 2
        try:
            hk_flush = int(os.environ.get("WAN_PROF_FLUSH_EVERY", "0"))  # EXPERIMENT (hkato): avoid profiler overflow
            if hk_flush > 0:
                ttnn.synchronize_device(self.mesh_device)
                ttnn.ReadDeviceProfiler(self.mesh_device)
            for hk_i, block in enumerate(self.blocks):
                spatial_1BND = block(
                    spatial_1BND=spatial_1BND,
                    prompt_1BLP=(prompt_1BLP, negative_prompt_1BLP),
                    temb_1BTD=timestep_proj_1BTD,
                    N=N,
                    rope_cos=rope_cos_1HND,
                    rope_sin=rope_sin_1HND,
                    trans_mat=trans_mat,
                )
                if hk_flush > 0 and (hk_i + 1) % hk_flush == 0:
                    ttnn.synchronize_device(self.mesh_device)
                    ttnn.ReadDeviceProfiler(self.mesh_device)
        finally:
            for block in self.blocks:
                block.attn1._cfg_batch = 1
                block.attn2._cfg_batch = 1
        if hk_marks:
            hk_enq = (time.perf_counter() - hk_marks[-1]) * 1e3  # EXPERIMENT (hkato): host enqueue time, no sync
            hk_marks.append(self._synced_time())
            print(f"WAN_FWD host_enqueue_blocks={hk_enq:.2f} ms", flush=True)

        scale_shift_1BSD = self.scale_shift_table.data + temb_11BD
        shift_11BD, scale_11BD = ttnn.chunk(scale_shift_1BSD, 2, -2)
        spatial_norm_1BND = self.norm_out(
            spatial_1BND, dynamic_weight=(1 + scale_11BD), dynamic_bias=shift_11BD, dtype=ttnn.float32
        )
        if hk_marks and os.environ.get("HK_FWD_FINE") == "1":
            hk_t3 = self._synced_time()
            print(f"WAN_FWD_FINE norm_out={(hk_t3 - hk_marks[-1]) * 1e3:.2f} ms", flush=True)
        if os.environ.get("WAN_PROJ_OUT_TP") == "1" and self.parallel_config.tensor_parallel.factor > 1:
            spatial_1BNI = self._hk_proj_out_tp(spatial_norm_1BND)
        else:
            if self.parallel_config.tensor_parallel.factor > 1:
                spatial_norm_1BND = self.ccl_manager.all_gather_persistent_buffer(
                    spatial_norm_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
                )
            spatial_1BNI = self.proj_out(
                spatial_norm_1BND, compute_kernel_config=self.hifi4_compute_kernel_config, dtype=self.output_dtype
            )

        n = spatial_1BNI.shape[2] // 2
        c = spatial_1BNI.shape[3]
        outs = []
        for lo in (0, n):
            half = ttnn.slice(spatial_1BNI, [0, 0, lo, 0], [1, 1, lo + n, c])
            if gather_output:
                half = self.ccl_manager.all_gather_persistent_buffer(
                    half, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
                )
            outs.append(half)
        if hk_marks:
            hk_marks.append(self._synced_time())
            pre, blocks, post = ((b - a) * 1e3 for a, b in zip(hk_marks, hk_marks[1:]))
            print(f"WAN_FWD pre={pre:.2f} blocks={blocks:.2f} post={post:.2f} ms", flush=True)
        return outs[0], outs[1]

    # Prep run is False because we warmup the entire pipeline first. Remove if this is not desired.
    @traced_function(device=lambda self: self.mesh_device, clone_prep_inputs=False, prep_run=False)
    def combined_step(
        self,
        do_classifier_free_guidance: bool,
        spatial_1BNI: ttnn.Tensor,
        prompt_1BLP: ttnn.Tensor,
        negative_prompt_1BLP: ttnn.Tensor,
        N: int,
        rope_cos_1HND: ttnn.Tensor,
        rope_sin_1HND: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
        timestep: ttnn.Tensor,
        guidance_scale: ttnn.Tensor,
        *,
        gather_output: bool = True,
    ) -> ttnn.Tensor:
        # CFG batching (WAN_CFG_BATCH, on by default; WAN_CFG_BATCH=0 runs cond / uncond as two forwards).
        # Not bit-identical to the two-forward path: the 8192-row matmuls use other blockings (another
        # accumulation order), and over 40 steps this shifts the composition a little at equal quality.
        # Use WAN_CFG_BATCH=0 when comparing against videos generated without it.
        if do_classifier_free_guidance and os.environ.get("WAN_CFG_BATCH", "1") == "1":
            cond, uncond = self.inner_step_cfg(
                spatial_1BNI,
                prompt_1BLP,
                negative_prompt_1BLP,
                rope_cos_1HND,
                rope_sin_1HND,
                trans_mat,
                N,
                timestep,
                gather_output=gather_output,
            )
            return ttnn.lerp(uncond, cond, guidance_scale)

        cond = self.inner_step(
            spatial_1BNI,
            prompt_1BLP,
            rope_cos_1HND,
            rope_sin_1HND,
            trans_mat,
            N,
            timestep,
            gather_output=gather_output,
        )
        if not do_classifier_free_guidance:
            return cond

        uncond = self.inner_step(
            spatial_1BNI,
            negative_prompt_1BLP,
            rope_cos_1HND,
            rope_sin_1HND,
            trans_mat,
            N,
            timestep,
            gather_output=gather_output,
        )

        combined = ttnn.lerp(uncond, cond, guidance_scale)

        return combined


class WanCheckpoint:
    """A Wan transformer-subfolder checkpoint: fetches weights and builds loaded transformers."""

    def __init__(self, name: str, subfolder: str) -> None:
        self._name = name
        self._subfolder = subfolder
        torch_transformer = TorchWanTransformer3DModel.from_pretrained(
            name,
            subfolder=subfolder,
            trust_remote_code=True,
        )
        torch_transformer.eval()
        self._config = torch_transformer.config
        self._state_dict = torch_transformer.state_dict()

    @property
    def subfolder(self) -> str:
        return self._subfolder

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._state_dict)

    @staticmethod
    def _bf8_load_names() -> list[str]:
        if os.environ.get("WAN_WEIGHTS_BF8_LOAD") != "1" or not os.environ.get("WAN_WEIGHTS_BF8"):
            return []
        return os.environ["WAN_WEIGHTS_BF8"].split(",")

    def _cache_suffix(self, is_fsdp: bool) -> str:
        suffix = "-ffnresident" if self._ffn_resident(is_fsdp) else ""
        if self._bf8_load_names():
            suffix += "-bf8w-" + "-".join(self._bf8_load_names())
        return suffix

    def _full_resident(self, is_fsdp: bool) -> bool:
        # EXPERIMENT: WAN_FULL_RESIDENT=<subfolder>[,<subfolder>] drops FSDP for that checkpoint entirely.
        return is_fsdp and self._subfolder in os.environ.get("WAN_FULL_RESIDENT", "").split(",")

    def _ffn_resident(self, is_fsdp: bool) -> bool:
        # WAN_FFN_RESIDENT=<subfolder>[,<subfolder>] keeps that checkpoint's FFN weights resident (no FSDP on
        # ff1/ff2) while the rest of the model stays FSDP. Costs ~2.3 GiB/device for a 14B transformer.
        return is_fsdp and self._subfolder in os.environ.get("WAN_FFN_RESIDENT", "").split(",")

    def build(
        self,
        *,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool,
        model_type: str,
        lora_enabled: bool = False,
    ) -> WanTransformer3DModel:
        """Construct a ``WanTransformer3DModel`` for this checkpoint (weights NOT loaded).

        Loading is deferred so the caller can manage the lifecycle (deallocate / reload).
        """
        c = self._config
        ffn_resident = self._ffn_resident(is_fsdp)
        if self._full_resident(is_fsdp):
            is_fsdp = False
        model = WanTransformer3DModel(
            patch_size=c.patch_size,
            num_heads=c.num_attention_heads,
            dim=c.num_attention_heads * c.attention_head_dim,
            in_channels=c.in_channels,
            out_channels=c.out_channels,
            text_dim=c.text_dim,
            freq_dim=c.freq_dim,
            ffn_dim=c.ffn_dim,
            cross_attn_norm=c.cross_attn_norm,
            eps=c.eps,
            rope_max_seq_len=c.rope_max_seq_len,
            mesh_device=ccl_manager.mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            ffn_fsdp=not ffn_resident,
            model_type=model_type,
            lora_enabled=lora_enabled,
        )
        if os.environ.get("WAN_SDPA_BF8") == "1" or os.environ.get("WAN_PROJ_LOFI"):
            from models.tt_dit.pipelines.wan.quant_config import set_block_proj_fidelity

            for block in model.blocks:
                if os.environ.get("WAN_SDPA_BF8") == "1":
                    block.attn1._sdpa_input_dtype = ttnn.bfloat8_b
                if os.environ.get("WAN_PROJ_LOFI"):
                    set_block_proj_fidelity(block, os.environ["WAN_PROJ_LOFI"].split(","))
        if self._bf8_load_names():
            from models.tt_dit.pipelines.wan.quant_config import _WEIGHT_ONLY_LAYERS

            for block in model.blocks:
                for name in self._bf8_load_names():
                    linear = _WEIGHT_ONLY_LAYERS[name](block)
                    linear.weight.dtype = ttnn.bfloat8_b
                    if getattr(linear, "bias", None) is not None:
                        linear.bias.dtype = ttnn.bfloat8_b
        return model

    def load(
        self,
        model: WanTransformer3DModel,
        *,
        mesh_device: ttnn.MeshDevice,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool,
    ) -> None:
        """Load (or reload) weights for a previously-built transformer."""
        cache_suffix = self._cache_suffix(is_fsdp)
        if self._full_resident(is_fsdp):
            is_fsdp = False
        cache.load_model(
            tt_model=model,
            get_torch_state_dict=lambda: self._state_dict,
            # resident FFN weights are cached with a different sharding, so they get their own cache entry
            model_name=os.path.basename(self._name) + cache_suffix,
            subfolder=self._subfolder,
            parallel_config=parallel_config,
            mesh_shape=tuple(mesh_device.shape),
            mesh_device=mesh_device,
            is_fsdp=is_fsdp,
        )
