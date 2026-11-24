# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The DSA decode's per-head absorb matmuls (`TtDsa._absorb`: q~ = q W_kb, [1, 16, B, 256] @ [1, 16, 256, 512];
`TtDsa._values`: [1, 16, B, 512] @ W_vb^T [1, 16, 512, 256]) as one generic_op over more cores, the same values as
ttnn.matmul with `common.batched_matmul_config` (GLM53_DSA_ABSORB_OP, read at each call, default 2).

ttnn.matmul's batched config puts one head (and one tile row of tokens) on a core with the head's whole N: 16 cores at
8 / 32 users (1 tile row), 32 at 128 (2 tile rows, each core reading its head's whole weight again).  Each core reads
the head's 128 weight tiles (139 KB, BFP8) and then does 128 HiFi4 tile products: 17.6 / 17.2 us a call at 32 users and
26.8 / 26.7 at 128 users in the model, 2 calls a DSA layer x 11 layers.

Here each head's N is split over G cores (NC = N / G tiles: one of ttnn's output subblocks a core by default), each
core taking all the head's tile rows: a core reads its slice of w once (K x NC tiles) and a (R x K tiles), all in
flight with the compute starting on the first k rows (kernels/dsa/absorb_reader.cpp), computes its R x NC tiles
(absorb_compute.cpp) and writes them (absorb_writer.cpp).  The arithmetic of each output tile is ttnn's: the K tiles
in one block summed in fp32 Dest from zero by matmul_block in k order (in0_block_w = K, out_subblock_h 1, the
factory's start and the packer's reconfig), packed to bf16 once.  Only which core computes which tiles, and in
subblocks of how many, changes.

    absorb_matmul(mesh, a, w, memory_config=None, groups=None) -> [1, H, B, N * 32] bf16 (tiled, interleaved)
"""

from __future__ import annotations

import os

import ttnn

from .common import WEIGHT_MM, core_rects, fidelity

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/dsa/"
CB_A, CB_W, CB_OUT, CB_PART = 0, 1, 16, 24
MAX_CHUNKS = 8  # w's k rows pushed in at most this many chunks (transaction ids 2 ..)
TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.float32: 4096}


def absorb_op_on() -> bool:
    """GLM53_DSA_ABSORB_OP (read at each call; default 2, "0" ttnn.matmul): 1 or 2, TtDsa's decode absorb matmuls
    (up to ABSORB_ROWS tokens) by `absorb_matmul` in place of ttnn.matmul; the same values (the same arithmetic per
    output tile).  2: the heads-first decode's latent write in the same program as q W_kb (`latent_absorb`)."""
    return os.environ.get("GLM53_DSA_ABSORB_OP", "2") in ("1", "2")


def latent_absorb_on() -> bool:
    return os.environ.get("GLM53_DSA_ABSORB_OP", "2") == "2"


def absorb_ok(a, w) -> bool:
    """Whether `absorb_matmul` takes these (else the caller keeps ttnn.matmul): a bf16 tiled interleaved, w tiled
    interleaved in a dtype it knows."""
    return (
        a.dtype == ttnn.bfloat16
        and a.layout == ttnn.TILE_LAYOUT
        and not a.is_sharded()
        and w.layout == ttnn.TILE_LAYOUT
        and not w.is_sharded()
        and w.dtype in TILE_BYTES
        and int(w.shape[1]) == int(a.shape[1])
    )


def groups_for(heads: int, n_tiles: int, cores: int) -> int:
    """G, the cores a head: each core one of ttnn's output subblocks (out_subblock_w, 4 tiles) of the head's columns,
    G = N / 4 (W_kb 4, W_vb^T 2; W_vb^T at 2 subblocks of 2 tiles a core ran 28.6 us at 64 tokens against 20.2 at one
    of 4), at most GLM53_DSA_ABSORB_G (default 8) and heads x G <= the grid's cores."""
    top = int(os.environ.get("GLM53_DSA_ABSORB_G", "8"))
    sub = max(x for x in (4, 2, 1) if n_tiles % x == 0)
    best = 1
    for g in range(1, min(top, n_tiles // sub) + 1):
        if n_tiles % g == 0 and heads * g <= cores:
            best = g
    return best


def _dm(processor, noc):
    cfg = ttnn.DataMovementConfigDescriptor()
    cfg.processor, cfg.noc = processor, noc
    return cfg


def _cb(index, fmt, page, pages, cores):
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
    )


def absorb_parts(mesh, a, w, memory_config=None, groups: int | None = None, compute=WEIGHT_MM):
    """`absorb_matmul`'s kernels and CBs (on the grid's first H G cores, row-major) and its output -> (kernels, cbs,
    out, cores used)."""
    heads, rows, k32 = int(a.shape[1]), int(a.shape[-2]), int(a.shape[-1])
    n32 = int(w.shape[-1])
    assert a.dtype == ttnn.bfloat16 and a.layout == ttnn.TILE_LAYOUT, (a.dtype, a.layout)
    assert w.layout == ttnn.TILE_LAYOUT and int(w.shape[1]) == heads and int(w.shape[-2]) == k32, (w.shape, a.shape)
    assert k32 % 32 == 0 and n32 % 32 == 0
    kt, nt = k32 // 32, n32 // 32
    r_tiles = -(-int(a.padded_shape[-2]) // 32)
    grid = mesh.compute_with_storage_grid_size()
    cores_all = grid.x * grid.y
    g = groups or groups_for(heads, nt, cores_all)
    # each head's tile rows over rs cores too (GLM53_DSA_ABSORB_RSPLIT, default 1: a core all the head's rows, its
    # weight slice read once; rs > 1: each of the rs cores reads the slice again, its rows' products only;
    # slower, 64 tokens W_kb G 4 17.8 -> G 2 R 2 21.9 us, W_vb^T G 2 20.3 -> G 2 R 2 24.5 us)
    rs = int(os.environ.get("GLM53_DSA_ABSORB_RSPLIT", "1"))
    rs = rs if rs > 1 and r_tiles % rs == 0 and heads * g * rs <= cores_all else 1
    assert nt % g == 0 and heads * g * rs <= cores_all, (nt, g, rs, heads, cores_all)
    nc = nt // g
    rc = r_tiles // rs
    sub = max(x for x in (4, 2, 1) if nt % x == 0)  # ttnn's out_subblock_w (batched_matmul_config)
    cw = max(x for x in (4, 2, 1) if nc % x == 0 and x <= sub)
    kc = -(-kt // MAX_CHUNKS)
    buffer_type = ttnn.BufferType.DRAM
    if memory_config is not None and memory_config.buffer_type == ttnn.BufferType.L1:
        buffer_type = ttnn.BufferType.L1
    spec = ttnn.TensorSpec(ttnn.Shape([1, heads, rows, n32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=buffer_type)
    out = ttnn.allocate_tensor_on_device(spec, mesh)

    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(heads * g * rs)]
    cores = core_rects(coords)
    rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, c in enumerate(coords):
        h, rest = divmod(i, g * rs)
        j, q = divmod(rest, rs)
        rd[c.x][c.y] = [a.buffer_address(), w.buffer_address(), h, j * nc, q * rc]
        wr[c.x][c.y] = [out.buffer_address(), h, j * nc, q * rc]
    acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731
    math = getattr(compute, "math_fidelity", None)
    if math is None:
        math = fidelity("GLM53_MM_FIDELITY")
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=math, fp32_dest_acc_en=True)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "absorb_reader.cpp",
            core_ranges=cores,
            compile_time_args=[CB_A, CB_W, rc, kt, nt, nc, kc, r_tiles] + acc(a) + acc(w),
            runtime_args=rd,
            config=_dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "absorb_writer.cpp",
            core_ranges=cores,
            compile_time_args=[CB_OUT, rc, nt, nc, cw, r_tiles] + acc(out),
            runtime_args=wr,
            config=_dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "absorb_compute.cpp",
            core_ranges=cores,
            compile_time_args=[rc, kt, nc, cw, kc],
            runtime_args=ttnn.RuntimeArgs(),
            config=cfg,
        ),
    ]
    cbs = [
        _cb(CB_A, ttnn.bfloat16, TILE_BYTES[ttnn.bfloat16], rc * kt, cores),
        _cb(CB_W, w.dtype, TILE_BYTES[w.dtype], kt * nc, cores),
        _cb(CB_OUT, ttnn.bfloat16, TILE_BYTES[ttnn.bfloat16], rc * nc, cores),
        _cb(CB_PART, ttnn.float32, TILE_BYTES[ttnn.float32], 1, cores),
    ]
    return kernels, cbs, out, heads * g * rs


def _run(mesh, kernels, cbs, io) -> None:
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    mesh_program = ttnn.MeshProgramDescriptor()
    rows_m, cols_m = tuple(mesh.shape)  # (8 x 4, or a 1 x 1 mesh)
    whole = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(rows_m - 1, cols_m - 1))
    mesh_program[whole] = program
    ttnn.generic_op(io, mesh_program)


def absorb_matmul(mesh, a, w, memory_config=None, groups: int | None = None, compute=WEIGHT_MM):
    """a [1, H, B, K 32] bf16 tiled (interleaved, DRAM or L1) @ w [1, H, K 32, N 32] tiled in DRAM (bfp8 / bf16 / bfp4)
    -> [1, H, B, N 32] bf16 tiled, interleaved in `memory_config`'s buffer type (DRAM by default).  The same values as
    ttnn.matmul(a, w, program_config=batched_matmul_config(mesh, K, N), compute_kernel_config=compute) (`compute`: its
    math fidelity; fp32 Dest as WEIGHT_MM's)."""
    kernels, cbs, out, _ = absorb_parts(mesh, a, w, memory_config, groups, compute)
    _run(mesh, kernels, cbs, [a, w, out])
    return out


def latent_absorb(mesh, latent, cache, cur_pos, owners: int, pages: int, q, w_kb, memory_config=None):
    """`dsa_latent.latent_write(mesh, latent, cache, cur_pos, owners, pages)` and `absorb_matmul(mesh, q, w_kb,
    memory_config)` in one program on disjoint cores (GLM53_DSA_ABSORB_OP=2): the latents' copies on the cores after
    the matmul's (64 at most), running beside it; the same bytes and values as the two ops.  -> the matmul's output."""
    from .dsa_latent import latent_write_parts

    kernels, cbs, out, used = absorb_parts(mesh, q, w_kb, memory_config)
    grid = mesh.compute_with_storage_grid_size()
    lw, lcb, _ = latent_write_parts(mesh, latent, cache, cur_pos, owners, pages, start=used,
                                    max_cores=grid.x * grid.y - used)  # fmt: skip
    _run(mesh, kernels + [lw], cbs + [lcb], [latent, cur_pos, q, w_kb, cache, out])
    return out


__all__ = ["absorb_matmul", "absorb_ok", "absorb_op_on", "absorb_parts", "groups_for", "latent_absorb"]
