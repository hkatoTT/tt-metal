# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import torch

import ttnn
from models.common.utility_functions import is_blackhole

from ....layers.linear import ColParallelLinear, LoRAColParallelLinear
from ....layers.module import Module
from ....layers.normalization import DistributedRMSNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils.matmul import get_matmul_config, register_matmul_configs
from ....utils.substate import pop_substate, rename_substate
from ....utils.tensor import bf16_tensor


class WanAttention(Module):
    # Map from (is_blackhole, sp_factor, tp_factor) -> (q_chunk_size, k_chunk_size)
    sdpa_chunk_size_map = {
        (False, 2, 4): (256, 256),
        (False, 8, 4): (256, 512),  # WH Galaxy 4x8: 480p ring SDPA 14.69 -> 13.06 ms (swept 2026-09-30)
        (True, 2, 2): (128, 512),
        (True, 8, 4): (288, 512),
        (True, 32, 4): (224, 512),
    }
    default_sdpa_chunk_size = (256, 256)

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
        is_self: bool = True,
        sdpa_chunk_size_overrides: dict | None = None,
        lora_enabled: bool = False,
    ) -> None:
        super().__init__()

        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.is_self = is_self
        # WAN_CROSS_KV_CACHE=1: cross-attention K/V depend only on the prompt, which is the same tensor for every
        # denoising step, so keep the post-norm K and head-split V per prompt tensor. Cleared through
        # clear_cross_kv_cache() whenever the transformer prepares new text conditioning.
        self._cross_kv_cache = {}
        self._cross_kv_cache_enabled = os.environ.get("WAN_CROSS_KV_CACHE", "0") == "1"

        self.n_local_heads = self.num_heads // self.parallel_config.tensor_parallel.factor

        fsdp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        rms_kwargs = {
            "embedding_dim": dim,
            "norm_eps": eps,
            "norm_elementwise_affine": True,
            "bias": False,
            "mesh_device": mesh_device,
            "mesh_axis": parallel_config.tensor_parallel.mesh_axis,
            "ccl_manager": ccl_manager,
        }

        self.norm_q = DistributedRMSNorm(**rms_kwargs)
        self.norm_k = DistributedRMSNorm(**rms_kwargs)

        col_parallel_kwargs = {
            "bias": True,
            "mesh_device": mesh_device,
            "mesh_axis": parallel_config.tensor_parallel.mesh_axis,
            "fsdp_mesh_axis": fsdp_mesh_axis,
            "ccl_manager": ccl_manager,
        }
        ColCls = LoRAColParallelLinear if lora_enabled else ColParallelLinear

        if is_self:
            # Fused QKV for self-attention: single matmul split into 3 outputs
            self.to_qkv = ColCls(
                dim,
                3 * dim,
                chunks=3,
                **col_parallel_kwargs,
            )
        else:
            # Cross-attention: Q from spatial, K/V from prompt
            self.to_q = ColCls(dim, dim, **col_parallel_kwargs)
            # Fused KV: single matmul split into 2 outputs
            self.to_kv = ColCls(
                dim,
                2 * dim,
                chunks=2,
                **col_parallel_kwargs,
            )

        self.to_out = ColCls(
            dim,
            dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.dummy_joint_input = bf16_tensor(torch.zeros((1, self.n_local_heads, 0, self.head_dim)), device=mesh_device)

        full_grid = self.mesh_device.compute_with_storage_grid_size()
        # EXPERIMENT (hkato): WAN_XSDPA_CHUNK="q,k" / WAN_XSDPA_EXP_APPROX=1 for the plain (cross-attention) SDPA
        hk_xq, hk_xk = (int(v) for v in os.environ.get("WAN_XSDPA_CHUNK", "256,256").split(","))
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=full_grid,
            q_chunk_size=hk_xq,
            k_chunk_size=hk_xk,
            exp_approx_mode=os.environ.get("WAN_XSDPA_EXP_APPROX") == "1",  # NOTE: False is more correct
        )

        self.sdpa_worker_grid = (full_grid.x - 1, full_grid.y)  # Reserve last column for CCL
        # The CFG head fold (see forward) runs one 2H-head ring SDPA. On WH 4x8 480p its 2 x 10 x 16 = 320 Q chunks
        # split evenly over 8x8 workers (5 per core) with the ring all-gather workers on the last row, while 7x9
        # leaves some cores one chunk more: ring SDPA 24.08 -> 22.39 ms per block, 40 steps 81.82 -> 77.96 s.
        # Only for the fold: two 10-head calls on 8x8 are slower than on 7x9. WAN_CFG_HEADFOLD_GRID="x,y" overrides.
        fold_grid = os.environ.get("WAN_CFG_HEADFOLD_GRID")
        self.fold_sdpa_worker_grid = (
            tuple(int(v) for v in fold_grid.split(",")) if fold_grid else (full_grid.x, full_grid.y - 1)
        )
        self.fold_ccl_row_major = self.fold_sdpa_worker_grid[1] < full_grid.y
        chunk_lookup = {**self.sdpa_chunk_size_map, **(sdpa_chunk_size_overrides or {})}
        ring_sdpa_chunk_size = chunk_lookup.get(
            (
                is_blackhole(),
                self.parallel_config.sequence_parallel.factor,
                self.parallel_config.tensor_parallel.factor,
            ),
            self.default_sdpa_chunk_size,
        )
        # WAN_SDPA_CHUNK="q,k" overrides the ring SDPA (q_chunk, k_chunk), e.g. for chunk sweeps.
        if os.environ.get("WAN_SDPA_CHUNK"):
            ring_sdpa_chunk_size = tuple(int(v) for v in os.environ["WAN_SDPA_CHUNK"].split(","))

        # Ring SDPA QK^T and P.V matmuls at LoFi by default (max / exp / rescale keep the compute config's HiFi2).
        # WH 4x8 480p: ring SDPA 13.00 -> 12.04 ms, 40 steps 86.97 -> 82.76 s. The video keeps its quality but its
        # content changes noticeably (QK^T precision feeds the softmax). WAN_SDPA_MATMUL_FIDELITY=none restores the
        # compute config's fidelity; other values (e.g. HiFi2) pick that fidelity. Validated only with the streaming
        # kernel from #58223 on; on the older kernel a LoFi compute config hung the ring SDPA on WH 4x8.
        sdpa_mm_fidelity = os.environ.get("WAN_SDPA_MATMUL_FIDELITY", "LoFi")
        sdpa_mm_fidelity = None if sdpa_mm_fidelity.lower() in ("", "none", "off") else sdpa_mm_fidelity
        ring_sdpa_config_args = dict(
            q_chunk_size=ring_sdpa_chunk_size[0],
            k_chunk_size=ring_sdpa_chunk_size[1],
            exp_approx_mode=os.environ.get("WAN_SDPA_EXP_APPROX") == "1",  # EXPERIMENT; False is more correct
            matmul_math_fidelity=getattr(ttnn.MathFidelity, sdpa_mm_fidelity) if sdpa_mm_fidelity else None,
        )
        self.ring_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.sdpa_worker_grid, **ring_sdpa_config_args
        )
        self.ring_sdpa_program_config_fold = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.fold_sdpa_worker_grid, **ring_sdpa_config_args
        )

        self.use_exp_ring_sdpa = (
            self.parallel_config.tensor_parallel.factor == 4 and self.parallel_config.sequence_parallel.factor == 32
        )

        if self.use_exp_ring_sdpa:
            self.exp_ring_sdpa_program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=full_grid,
                q_chunk_size=ring_sdpa_chunk_size[0],
                k_chunk_size=ring_sdpa_chunk_size[1],
                exp_approx_mode=False,
            )

        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,  # NOTE: Set to True if there's a correctness issue
        )

        self.rope_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

        # WAN_MM_FP32_DEST=attn|all: accumulate the attention projections in bf16 dest instead of fp32.
        # Off by default: 480p gains only 0.7% and the generated video visibly changes.
        mm_fp32_dest = os.environ.get("WAN_MM_FP32_DEST", "") not in ("attn", "all")
        self.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=mm_fp32_dest,
            packer_l1_acc=True,
        )
        if not mm_fp32_dest:
            # attention projection blockings re-swept with 4x2 subblocks (bf16 dest, WH 4x8, 480p)
            register_matmul_configs(
                {
                    "8x8": {
                        (4096, 5120, 3840): (8, 4, 10, (4, 2)),  # to_qkv 2103.7 us (fp32 dest: 2228.0)
                        (4096, 5120, 1280): (8, 8, 6, (4, 2)),  # to_out 1159.9 / cross to_q, to_out 957.0 us
                    }
                }
            )

    def clear_cross_kv_cache(self) -> None:
        self._cross_kv_cache.clear()

    def set_cross_kv_cache_enabled(self, enabled: bool) -> None:
        # Must be off under trace: a trace captured on a cache hit has no K/V ops, so a replay after the prompt
        # buffer is overwritten in place would keep using the previous prompt's K/V.
        self._cross_kv_cache_enabled = enabled
        if not enabled:
            self._cross_kv_cache.clear()

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "to_out.0", "to_out")

        def _interleave_heads(tensors: list[torch.Tensor]):
            """Interleave weight/bias tensors so column-parallel fracturing assigns correct heads per device.

            Each tensor has out-dim = num_heads * head_dim.  We interleave them on the heads
            axis so that after TP column-sharding each device gets the right heads from every tensor.

            Args:
                tensors: list of tensors in [out, in] PyTorch convention (or [out, 1] for bias).
            Returns:
                Merged tensor in [out, in] PyTorch convention.
            """
            n_dev = self.parallel_config.tensor_parallel.factor
            # Transpose to [in, out]
            tensors = [t.T for t in tensors]
            # Reshape out dim to [in, n_dev, n_local_heads, head_dim]
            tensors = [t.reshape(t.shape[0], n_dev, self.n_local_heads, self.head_dim) for t in tensors]
            # Concatenate on the heads dim so each device shard gets its own heads from every tensor
            merged = torch.cat(tensors, dim=2)  # [in, n_dev, len(tensors)*n_local_heads, head_dim]
            merged = merged.reshape(merged.shape[0], len(tensors) * self.num_heads * self.head_dim)
            # Transpose back to [out, in] PyTorch convention
            return merged.T

        if self.is_self:
            # Merge separate to_q, to_k, to_v weights into fused to_qkv
            q_state = pop_substate(state, "to_q")
            k_state = pop_substate(state, "to_k")
            v_state = pop_substate(state, "to_v")

            state["to_qkv.weight"] = _interleave_heads([q_state["weight"], k_state["weight"], v_state["weight"]])
            if "bias" in q_state:
                bias = _interleave_heads(
                    [q_state["bias"].unsqueeze(-1), k_state["bias"].unsqueeze(-1), v_state["bias"].unsqueeze(-1)]
                )
                state["to_qkv.bias"] = bias.squeeze(-1)
        else:
            # Merge separate to_k, to_v weights into fused to_kv
            k_state = pop_substate(state, "to_k")
            v_state = pop_substate(state, "to_v")

            state["to_kv.weight"] = _interleave_heads([k_state["weight"], v_state["weight"]])
            if "bias" in k_state:
                bias = _interleave_heads([k_state["bias"].unsqueeze(-1), v_state["bias"].unsqueeze(-1)])
                state["to_kv.bias"] = bias.squeeze(-1)

    def _to_out_fused_addcmul(
        self,
        x: ttnn.Tensor,
        addcmul_residual: ttnn.Tensor,
        addcmul_gate: ttnn.Tensor,
        compute_kernel_config=None,
        parallel_config=None,
        dtype=None,
    ) -> ttnn.Tensor:
        """Fused to_out projection + addcmul: output = residual + (matmul(x, W) + bias) * gate."""
        to_out = self.to_out
        if dtype is None and x.get_dtype() == ttnn.bfloat8_b:
            # a bf8 activation (e.g. ring SDPA run on bf8 q/k/v) must not make the residual stream bf8
            dtype = addcmul_residual.get_dtype()

        # Reads to_out.weight.data directly, so runtime-mode LoRA (which keeps
        # the delta in self._runtime_A/B) would silently no-op. Fuse mode is
        # fine — the delta lives in weight.data.
        if getattr(to_out, "lora_mode", None) == "runtime" and getattr(to_out, "is_lora_active", False):
            raise RuntimeError(
                "runtime LoRA mode is incompatible with WanAttention._to_out_fused_addcmul; "
                "construct to_out with lora_mode='fuse'"
            )

        # Handle FSDP weight gathering (mirrors ColParallelLinear.forward)
        if to_out.fsdp_mesh_axis is not None and to_out.mesh_device.shape[to_out.fsdp_mesh_axis] > 1:
            unsqueezed_weight = ttnn.unsqueeze_to_4D(to_out.weight.data)
            weight = self.ccl_manager.all_gather_persistent_buffer(
                unsqueezed_weight, dim=2, mesh_axis=to_out.fsdp_mesh_axis
            )
            weight = ttnn.reshape(weight, (weight.shape[-2], weight.shape[-1]))
        else:
            weight = to_out.weight.data

        if parallel_config is not None and parallel_config.tensor_parallel.factor > 1:
            M, K, N = x.padded_shape[-2], weight.padded_shape[-2], weight.padded_shape[-1]
            full_grid = self.mesh_device.compute_with_storage_grid_size()
            core_grid = ttnn.CoreCoord(full_grid.x, full_grid.y - 1)
            matmul_config = get_matmul_config(M, K, N, core_grid)

            if os.environ.get("WAN_AGMM_IN_BF8") == "1":  # EXPERIMENT: bf8 activation input, bf16 output
                dtype = dtype or ttnn.bfloat16
                if x.get_dtype() == ttnn.bfloat16:
                    x = ttnn.typecast(x, ttnn.bfloat8_b)
            ag_persistent_buffer = self.ccl_manager.get_ag_ping_pong_buffer(
                x.shape, 3, parallel_config.tensor_parallel.mesh_axis, dtype=x.get_dtype()
            )
            ag_global_semaphores = self.ccl_manager.get_ag_ping_pong_semaphore(
                parallel_config.tensor_parallel.mesh_axis
            )
            output = ttnn.experimental.all_gather_minimal_matmul_async(
                input_tensor=x,
                weight_tensor=weight,
                bias_tensor=to_out.bias.data if to_out.bias is not None else None,
                config=matmul_config,
                compute_kernel_config=compute_kernel_config or to_out.compute_config,
                persistent_output_buffer=ag_persistent_buffer,
                multi_device_global_semaphore=ag_global_semaphores,
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=parallel_config.tensor_parallel.mesh_axis,
                barrier_semaphore=None,
                force_transpose=True,
                num_workers_per_link=full_grid.x // self.ccl_manager.num_links,
                num_buffers_per_channel=48 if not is_blackhole() else 24,
                scalar=1.0,
                addcmul_input_tensor1=addcmul_residual,
                addcmul_input_tensor2=addcmul_gate,
                dtype=dtype,
            )[0]
        else:
            M, K, N_out = x.padded_shape[-2], x.padded_shape[-1], weight.padded_shape[-1]
            core_grid = self.mesh_device.compute_with_storage_grid_size()
            matmul_config = get_matmul_config(M, K, N_out, core_grid)

            output = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                x,
                weight,
                1.0,  # scalar
                addcmul_residual,
                addcmul_gate,
                bias_tensor=to_out.bias.data if to_out.bias is not None else None,
                config=matmul_config,
                compute_kernel_config=compute_kernel_config or to_out.compute_config,
                dtype=dtype,
            )
        return output

    def _hk_concat_out(self, spatial_BHNE, prompt_1BLP, cfg_batch):
        # concat_heads of the attention output (factored out so TT_RING_SDPA_OUT_CONCAT can skip it)
        if (
            prompt_1BLP is None
            and os.environ.get("WAN_SDPA_OUT_CAST_FIRST") == "1"
            and os.environ.get("WAN_AGMM_IN_BF8") == "1"
            and spatial_BHNE.dtype == ttnn.bfloat16
        ):
            # EXPERIMENT (hkato): cast the self-attention output to bf8 before concat_heads (half the data to move);
            # the to_out AGMM would cast it right after concat_heads anyway
            spatial_BHNE = ttnn.typecast(spatial_BHNE, ttnn.bfloat8_b)
        spatial_1BND = ttnn.transformer.concatenate_heads(spatial_BHNE)
        spatial_1BND = ttnn.unsqueeze(spatial_1BND, 0)
        if cfg_batch == 2:  # [1, 2, n, D] -> [1, 1, 2n, D], back to the stacked rows
            spatial_1BND = ttnn.reshape(
                spatial_1BND, (1, 1, spatial_1BND.shape[1] * spatial_1BND.shape[2], spatial_1BND.shape[3])
            )

        return spatial_1BND

    def forward(
        self,
        spatial_1BND: ttnn.Tensor,
        N: int,
        prompt_1BLP: ttnn.Tensor | None = None,
        cross_attn_mask: ttnn.Tensor | None = None,
        rope_cos: ttnn.Tensor | None = None,
        rope_sin: ttnn.Tensor | None = None,
        trans_mat: ttnn.Tensor | None = None,
        addcmul_residual: ttnn.Tensor | None = None,
        addcmul_gate: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        spatial_1BND: fractured N on SP, fracturd D on TP
        prompt_1BLP: replicated on SP, replicated D on TP (optional)
        cross_attn_mask: (optional, cross-attn only) mask for SDPA, shape [B|1, nqh|1, Sq, Sk].
            Must be TILE layout, bf16/bfp8/bfp4 dtype, on device in DRAM.
        rope_cos: fractured on SP, TP
        rope_sin: fractured on SP, TP
        trans_mat: replicated
        addcmul_residual: (optional) residual tensor for fused matmul+addcmul (self-attn only)
        addcmul_gate: (optional) gate tensor for fused matmul+addcmul (self-attn only)

        If prompt_1BLP is not provided, run self-attention.
        Otherwise, run cross-attention on prompt.

        When addcmul_residual and addcmul_gate are both provided (self-attention only),
        the to_out projection and residual addcmul are fused into a single op:
            output = addcmul_residual + to_out(attn_output) * addcmul_gate

        Outputs:
        spatial_1BND: fractured N on SP, fractured D on TP
        """

        def create_heads(inp):
            out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                inp,
                num_heads=self.n_local_heads,
                num_kv_heads=0,
                transpose_k_heads=False,
            )
            return out

        cross_kv_key = None  # set in the cross-attention branch when the K/V cache is enabled
        cached_kv = None
        # CFG batching (WanTransformer3DModel.inner_step_cfg): the cond and uncond rows are stacked per device
        # ([1, 1, 2n, D]) and the cross-attention prompt is a (cond, uncond) pair. Row-wise ops run once on the
        # 2n rows; attention runs on [2, H, n, E].
        cfg_batch = getattr(self, "_cfg_batch", 1)
        prompt_pair = prompt_1BLP if isinstance(prompt_1BLP, tuple) else None
        # Self-attention folds the two passes into the head dim instead ([1, 2H, n, E], B=1): attention is per head,
        # so this is exact, keeps one ring SDPA call with its B=1 K/V chains, and the layout change is a free reshape
        # of the head-split norm output (no permute / slice / concat). On by default; WAN_CFG_HEADFOLD=0 attends the
        # two passes as [2, H, n, E], one ring SDPA call each.
        head_fold = cfg_batch == 2 and self.is_self and os.environ.get("WAN_CFG_HEADFOLD", "1") == "1"
        # EXPERIMENT (hkato): WAN_XATTN_FOLD=1 folds cross-attention the same way ([1, 2H, n, E]); the cached K/V are
        # interleaved per head (h0 cond, h0 uncond, h1 cond, ...) to match the free reshape of the query.
        x_fold = (
            cfg_batch == 2
            and not self.is_self
            and isinstance(prompt_1BLP, tuple)
            and cross_attn_mask is None
            and os.environ.get("WAN_XATTN_FOLD") == "1"
        )
        head_fold = head_fold or x_fold

        def rows_to_dim1(t):
            # [1, 1, 2n, F] -> [1, 2, n, F], the batched layout the fused distributed RMSNorm accepts (view)
            return ttnn.reshape(t, (1, 2, t.shape[2] // 2, t.shape[3])) if cfg_batch == 2 else t

        def rows_to_dim0(t):
            # [1, 1, 2n, F] -> [2, 1, n, F], the batched layout nlp_create_qkv_heads accepts (view)
            return ttnn.reshape(t, (2, 1, t.shape[2] // 2, t.shape[3])) if cfg_batch == 2 else t

        def norm_out_to_bhne(t):
            # The batched head-split norm output is [1, H, 2n, E] (batch-major within each head) -> [2, H, n, E]
            if cfg_batch != 2:
                return t
            h, n2, e = t.shape[1], t.shape[2], t.shape[3]
            if head_fold:
                return ttnn.reshape(t, (1, 2 * h, n2 // 2, e))
            return ttnn.permute(ttnn.reshape(t, (h, 2, n2 // 2, e)), (1, 0, 2, 3))

        if rope_cos is not None:
            # If ROPE is given, this is self-attention
            assert rope_sin is not None
            assert trans_mat is not None
            assert prompt_1BLP is None

        use_nonfused_agmm = (self.ccl_manager.topology == ttnn.Topology.Linear) and (
            self.parallel_config.tensor_parallel.factor > 1
        )
        if use_nonfused_agmm:
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        hk_v_BHNE = None  # EXPERIMENT (hkato): V already head-split by the QKV AGMM
        if self.is_self:
            # Fused QKV matmul with split output for self-attention
            q_1BNF, k_1BNF, v_1BNF = self.to_qkv(
                spatial_1BND,
                compute_kernel_config=self.mm_compute_kernel_config,
                parallel_config=None if use_nonfused_agmm else self.parallel_config,
                dtype=ttnn.bfloat8_b if os.environ.get("WAN_QKV_OUT_BF8") == "1" else None,  # EXPERIMENT
            )
            q_1BNF, k_1BNF = rows_to_dim1(q_1BNF), rows_to_dim1(k_1BNF)
            hk_v_heads = os.environ.get("TT_AGMM_V_HEADS")  # EXPERIMENT (hkato)
            if hk_v_heads is not None and int(hk_v_heads) > 0:
                # the QKV AGMM wrote V in head-split order [1, H, 2n, E]; as folded heads that is [1, 2H, n, E]
                assert head_fold and int(hk_v_heads) * 32 == v_1BNF.shape[3] // self.n_local_heads
                hk_v_BHNE = ttnn.experimental.view(
                    v_1BNF,
                    (1, 2 * self.n_local_heads, v_1BNF.shape[2] // 2, v_1BNF.shape[3] // self.n_local_heads),
                )
            v_1BNF = v_1BNF if head_fold else rows_to_dim0(v_1BNF)
        else:
            # Cross-attention: Q from spatial, fused KV from prompt
            assert prompt_1BLP is not None
            kv_input = prompt_1BLP
            q_1BNF = self.to_q(
                spatial_1BND,
                compute_kernel_config=self.mm_compute_kernel_config,
                parallel_config=None if use_nonfused_agmm else self.parallel_config,
            )
            q_1BNF = rows_to_dim1(q_1BNF)
            if prompt_pair is not None:
                # K/V per prompt, stacked on the batch dim; cached under the pair of prompt tensors
                if self._cross_kv_cache_enabled:
                    cross_kv_key = ("pairfold" if x_fold else "pair", id(prompt_pair[0]), id(prompt_pair[1]))
                cached_kv = self._cross_kv_cache.get(cross_kv_key) if cross_kv_key is not None else None
                if cached_kv is None:
                    ks, vs = [], []
                    for prompt in prompt_pair:
                        k_p, v_p = self.to_kv(prompt, compute_kernel_config=self.mm_compute_kernel_config)
                        ks.append(self.norm_k(k_p, num_heads_per_device=self.n_local_heads))
                        vs.append(create_heads(v_p))
                    if x_fold:
                        # [1, H, L, E] x2 -> [1, H, 2L, E] -> [1, 2H, L, E]: head 2h+b holds prompt b of head h
                        def _fold(ts):
                            t = ttnn.concat(ts, dim=2)
                            return ttnn.reshape(t, (1, 2 * t.shape[1], t.shape[2] // 2, t.shape[3]))

                        cached_kv = (_fold(ks), _fold(vs))
                    else:
                        cached_kv = (ttnn.concat(ks, dim=0), ttnn.concat(vs, dim=0))
                    if os.environ.get("WAN_XATTN_BF8") == "1":  # EXPERIMENT
                        cached_kv = tuple(ttnn.typecast(t, ttnn.bfloat8_b) for t in cached_kv)
                    if cross_kv_key is not None:
                        self._cross_kv_cache[cross_kv_key] = cached_kv
            else:
                cross_kv_key = id(prompt_1BLP) if self._cross_kv_cache_enabled else None
                cached_kv = self._cross_kv_cache.get(cross_kv_key) if cross_kv_key is not None else None
                if cached_kv is None:
                    k_1BNF, v_1BNF = self.to_kv(kv_input, compute_kernel_config=self.mm_compute_kernel_config)

        hk_out_concat = None  # EXPERIMENT (hkato): set when the ring SDPA writes concatenated heads
        # Set norm output dtype to the input dtype required for ring self-attn.
        sdpa_input_dtype = getattr(self, "_sdpa_input_dtype", None)
        use_ring_sdpa = self.parallel_config.sequence_parallel.factor > 1
        norm_output_dtype = sdpa_input_dtype if (use_ring_sdpa and prompt_1BLP is None) else None
        if prompt_1BLP is not None and os.environ.get("WAN_XATTN_BF8") == "1":  # EXPERIMENT: cross q in bf8
            norm_output_dtype = ttnn.bfloat8_b

        # Norm spatial before splitting heads
        q_BHNE = norm_out_to_bhne(
            self.norm_q(
                q_1BNF,
                num_heads_per_device=self.n_local_heads,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                trans_mat=trans_mat,
                dtype=norm_output_dtype,
            )
        )
        if cached_kv is None:
            k_BHNE = norm_out_to_bhne(
                self.norm_k(
                    k_1BNF,
                    num_heads_per_device=self.n_local_heads,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    trans_mat=trans_mat,
                    dtype=norm_output_dtype,
                )
            )
            if hk_v_BHNE is not None:
                v_BHNE = hk_v_BHNE
            else:
                v_BHNE = create_heads(v_1BNF)
            if head_fold and hk_v_BHNE is None:  # [1, H, 2n, E] -> [1, 2H, n, E], same head order as q / k
                v_BHNE = ttnn.reshape(v_BHNE, (1, 2 * v_BHNE.shape[1], v_BHNE.shape[2] // 2, v_BHNE.shape[3]))
            if prompt_1BLP is not None and os.environ.get("WAN_XATTN_BF8") == "1":  # EXPERIMENT
                k_BHNE, v_BHNE = ttnn.typecast(k_BHNE, ttnn.bfloat8_b), ttnn.typecast(v_BHNE, ttnn.bfloat8_b)
            if cross_kv_key is not None:
                self._cross_kv_cache[cross_kv_key] = (k_BHNE, v_BHNE)
        else:
            k_BHNE, v_BHNE = cached_kv

        if prompt_1BLP is None:
            # Self attention
            if self.parallel_config.sequence_parallel.factor > 1:
                # Q and K already cast by norm kernel; cast V and dummy joint inputs to match
                dummy_joint = self.dummy_joint_input
                if head_fold:
                    if getattr(self, "_dummy_joint_input_fold", None) is None:
                        self._dummy_joint_input_fold = bf16_tensor(
                            torch.zeros((1, 2 * self.n_local_heads, 0, self.head_dim)), device=self.mesh_device
                        )
                    dummy_joint = self._dummy_joint_input_fold
                if sdpa_input_dtype is not None:
                    if v_BHNE.dtype != sdpa_input_dtype:
                        v_BHNE = ttnn.typecast(v_BHNE, sdpa_input_dtype)
                    if dummy_joint.dtype != sdpa_input_dtype:
                        # cache the cast: a fresh tensor per call changes its address and misses the program cache
                        cache_key = (id(dummy_joint), sdpa_input_dtype)
                        if getattr(self, "_dummy_joint_cast_key", None) != cache_key:
                            self._dummy_joint_cast = ttnn.typecast(dummy_joint, sdpa_input_dtype)
                            self._dummy_joint_cast_key = cache_key
                        dummy_joint = self._dummy_joint_cast

                # HACK: pass null joint inputs to take advantage of ring attention, even though this is self-attention.
                fold_ccl_row_major = head_fold and self.fold_ccl_row_major

                def ring_sdpa(q, k, v):
                    if self.use_exp_ring_sdpa:
                        out, _, _ = ttnn.transformer.exp_ring_joint_scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            dummy_joint,
                            dummy_joint,
                            dummy_joint,
                            persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                                k.shape, 2, self.parallel_config.sequence_parallel.mesh_axis, dtype=k.dtype
                            ),
                            persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                                v.shape, 2, self.parallel_config.sequence_parallel.mesh_axis, dtype=v.dtype
                            ),
                            joint_strategy="rear",
                            logical_n=N,
                            program_config=self.exp_ring_sdpa_program_config,
                            compute_kernel_config=self.sdpa_compute_kernel_config,
                            dim=2,
                            multi_device_global_semaphore=self.ccl_manager.get_exp_ring_ping_pong_semaphore(
                                self.parallel_config.sequence_parallel.mesh_axis
                            ),
                            num_links=self.ccl_manager.num_links,
                            cluster_axis=self.parallel_config.sequence_parallel.mesh_axis,
                            mesh_device=self.mesh_device,
                            topology=self.ccl_manager.topology,
                            subdevice_id=self.ccl_manager.ccl_sub_device_id,
                            num_workers_per_link=5,
                            num_buffers_per_channel=32,
                        )
                    else:
                        out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            dummy_joint,
                            dummy_joint,
                            dummy_joint,
                            persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                                k.shape, 2, self.parallel_config.sequence_parallel.mesh_axis, dtype=k.dtype
                            ),
                            persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                                v.shape, 2, self.parallel_config.sequence_parallel.mesh_axis, dtype=v.dtype
                            ),
                            joint_strategy="rear",
                            logical_n=N,
                            program_config=(
                                self.ring_sdpa_program_config_fold if head_fold else self.ring_sdpa_program_config
                            ),
                            compute_kernel_config=self.sdpa_compute_kernel_config,
                            dim=2,
                            multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                                self.parallel_config.sequence_parallel.mesh_axis
                            ),
                            num_links=self.ccl_manager.num_links,
                            cluster_axis=self.parallel_config.sequence_parallel.mesh_axis,
                            mesh_device=self.mesh_device,
                            topology=self.ccl_manager.topology,
                            subdevice_id=self.ccl_manager.ccl_sub_device_id,
                            ccl_core_grid_offset=(
                                (0, self.fold_sdpa_worker_grid[1])
                                if fold_ccl_row_major
                                else ((self.fold_sdpa_worker_grid if head_fold else self.sdpa_worker_grid)[0], 0)
                            ),
                            use_column_major_ccl=not fold_ccl_row_major,
                        )
                    return out

                if cfg_batch == 2 and not head_fold:
                    # Ring SDPA with batch 2 takes ~4x a batch-1 call, not 2x: its K/V store-and-forward chains
                    # are only built for B == 1. So attend the cond and uncond passes one at a time.
                    halves = []
                    for b in range(2):
                        q_b, k_b, v_b = (
                            ttnn.slice(t, [b, 0, 0, 0], [b + 1, t.shape[1], t.shape[2], t.shape[3]])
                            for t in (q_BHNE, k_BHNE, v_BHNE)
                        )
                        halves.append(ring_sdpa(q_b, k_b, v_b))
                    spatial_BHNE = ttnn.concat(halves, dim=0)
                else:
                    spatial_BHNE = ring_sdpa(q_BHNE, k_BHNE, v_BHNE)
                    hk_out_concat = os.environ.get("TT_RING_SDPA_OUT_CONCAT")  # EXPERIMENT (hkato)
            else:
                spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
                    q_BHNE,
                    k_BHNE,
                    v_BHNE,
                    is_causal=False,
                    program_config=self.sdpa_program_config,
                    compute_kernel_config=self.sdpa_compute_kernel_config,
                )
        else:
            # Cross attention
            hk_pairfold = head_fold and os.environ.get("TT_SDPA_OUT_PAIRFOLD") == "1"  # EXPERIMENT (hkato)
            spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                is_causal=False,
                attn_mask=cross_attn_mask,
                program_config=self.sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
                **({"output_concat_heads": True} if hk_pairfold else {}),
            )
            if hk_pairfold:
                hk_out_concat = "x"

        if hk_out_concat == "x":
            # EXPERIMENT (hkato): the SDPA writer wrote [1, 1, n, 2H*E] storage as concat heads of [1, H, 2n, E]
            _, _, nn_, dd = spatial_BHNE.shape
            spatial_1BND = ttnn.experimental.view(spatial_BHNE, (1, 1, 2 * nn_, dd // 2))
        elif hk_out_concat in ("1", "2"):
            # EXPERIMENT (hkato): the ring SDPA writer already laid the output out in concatenated-heads order
            # (mode 2 also un-folds the CFG head pairs), so only the shape metadata changes
            assert hk_out_concat == ("2" if head_fold else "1"), "TT_RING_SDPA_OUT_CONCAT must match head folding"
            _, hh, nn_, ee = spatial_BHNE.shape
            if head_fold:
                hh, nn_ = hh // 2, nn_ * 2
            spatial_1BND = ttnn.experimental.view(spatial_BHNE, (1, 1, nn_, hh * ee))
            if (
                os.environ.get("WAN_SDPA_OUT_CAST_FIRST") == "1"
                and os.environ.get("WAN_AGMM_IN_BF8") == "1"
                and spatial_1BND.dtype == ttnn.bfloat16
            ):
                spatial_1BND = ttnn.typecast(spatial_1BND, ttnn.bfloat8_b)
        elif head_fold:  # [1, 2H, n, E] -> [1, H, 2n, E]: rows come out cond-then-uncond like the input
            spatial_BHNE = ttnn.reshape(
                spatial_BHNE, (1, spatial_BHNE.shape[1] // 2, spatial_BHNE.shape[2] * 2, spatial_BHNE.shape[3])
            )
        if hk_out_concat not in ("1", "2", "x"):
            spatial_1BND = self._hk_concat_out(spatial_BHNE, prompt_1BLP, cfg_batch)

        if use_nonfused_agmm:
            # Gather spatial on TP axis before projection
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        if addcmul_residual is not None and addcmul_gate is not None:
            # Fused to_out projection + addcmul (self-attention only)
            spatial_1BND = self._to_out_fused_addcmul(
                spatial_1BND,
                addcmul_residual,
                addcmul_gate,
                compute_kernel_config=self.mm_compute_kernel_config,
                parallel_config=None if use_nonfused_agmm else self.parallel_config,
            )
        else:
            spatial_1BND = self.to_out(
                spatial_1BND,
                compute_kernel_config=self.mm_compute_kernel_config,
                parallel_config=None if use_nonfused_agmm else self.parallel_config,
            )

        return spatial_1BND
