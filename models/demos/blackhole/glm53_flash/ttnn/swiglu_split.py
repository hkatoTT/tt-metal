# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The TP4 MLP's clamped SwiGLU (`TtMlp.partial`: the shared expert and the dense layers) from its gate | up matmul's
output in one op, through ttnn.generic_op, the same program on every chip:

  swiglu(mesh, gate_up, q, limit): gate_up [1, 1, S, 2 q] bf16 tiled interleaved (each row: its gate quarter, then its
      up quarter) -> h = silu(min(gate, limit)) * clamp(up, -limit, limit) [1, 1, S, q] bf16 tiled, in the same buffer
      type, in place of the two slices and `ttnn.multiply(gate, up, **swiglu_activations(limit))` (3 ops).

The compute is binary_ng's own kernel (`eltwise_binary_sfpu_no_bcast.cpp`, the path ttnn.multiply takes for bf16:
MUL is an SFPU op unless fast_and_approximate_mode) with the defines and compute config binary_ng gives that multiply
(read from its JIT build: the activations' init and call strings, the SFPU MUL, bf16 intermediates c_3 / c_4, 16-bit
dest, UnpackToDestFp32 on c_0, c_1, c_3 .. c_6, HiFi4, no approximation), one tile a cycle (its interleaved inputs'
case).  Only the reader differs: it reads the gate and up tiles straight from gate_up's columns c and q / 32 + c (the
slices are copies), so h is the same bit for bit.

(A compute kernel keeping the activated gate and up in dest, without binary_ng's round trips through c_3 / c_4, gave
the same bits and the same time: the op is not bound by its compute; not kept.)
"""

from __future__ import annotations

import struct

import ttnn

from .common import COLS, ROWS, core_rects

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/mlp/"
COMPUTE = "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp"
CB_A, CB_B, CB_O, CB_A_POST, CB_B_POST = 0, 1, 2, 3, 4
TILE = 2048  # a bf16 tile's bytes
PAGES = 4  # the reader's (and writer's) buffers, in tiles


def _bits(x: float) -> str:
    return f"0x{struct.unpack('<I', struct.pack('<f', float(x)))[0]:08x}u"


def _defines(limit: float) -> list:
    """binary_ng's compute defines for `ttnn.multiply(gate, up, **swiglu_activations(limit))` on bf16 (its JIT build's
    defines_generated.h, NOC_MODE aside: the framework adds it)."""
    return [
        ("BCAST_INPUT", ""),
        ("BINARY_SFPU_INIT", "mul_binary_tile_init();"),
        ("BINARY_SFPU_OP", "mul_binary_tile"),
        ("INP_FLOAT", "1"),
        (
            "PROCESS_LHS_ACTIVATIONS(i)",
            f"unary_min_tile_init();unary_min_tile(i, {_bits(limit)});silu_tile_init();silu_tile(i);",
        ),
        ("PROCESS_POST_ACTIVATIONS(i)", ""),
        ("PROCESS_RHS_ACTIVATIONS(i)", f"hardtanh_tile_init();hardtanh_tile(i, {_bits(-limit)}, {_bits(limit)});"),
        ("SCALAR_IS_LHS", "0"),
        ("SFPU_OP_COMPUTE_KERNEL_API_INCLUDE", "1"),
        ("SFPU_OP_HARDTANH_INCLUDE", "1"),
        ("WHERE_TST", "0"),
        ("WHERE_TTS", "0"),
    ]


def _cb(index: int, pages: int, cores) -> ttnn.CBDescriptor:
    return ttnn.CBDescriptor(
        total_size=TILE * pages,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=TILE)],
    )


def usable(gate_up) -> bool:
    """The op's case: bf16, tiled, interleaved, whole tile rows."""
    return (
        gate_up.dtype == ttnn.bfloat16
        and gate_up.layout == ttnn.TILE_LAYOUT
        and not gate_up.is_sharded()
        and gate_up.shape[-2] % 32 == 0
        and len(gate_up.shape) == 4
    )


def swiglu(mesh, gate_up, q: int, limit: float):
    """gate_up [1, 1, S, 2 q] bf16 tiled interleaved -> h [1, 1, S, q] (see the module's docstring)."""
    assert usable(gate_up) and gate_up.shape[-1] == 2 * q and q % 32 == 0, gate_up
    rows = gate_up.shape[-2]
    qt, rt = q // 32, rows // 32
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            ttnn.Shape([1, 1, rows, q]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            buffer_type=gate_up.memory_config().buffer_type,
        ),
        mesh,
    )
    grid = mesh.compute_with_storage_grid_size()
    total = rt * qt
    per = -(-total // min(grid.x * grid.y, total))
    ranges = [(i0, min(total, i0 + per)) for i0 in range(0, total, per)]
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(ranges))]
    cores = core_rects(coords)  # (rectangles: a program goes out range by range)
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (i0, i1) in zip(coords, ranges):
        rd[c.x][c.y] = [gate_up.buffer_address(), i0, i1]
        cp[c.x][c.y] = [i1 - i0]
        wr[c.x][c.y] = [out.buffer_address(), i0, i1]
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for i in (CB_A, CB_B, CB_A_POST, CB_B_POST, 5, 6):
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False
    )
    cfg.unpack_to_dest_mode = modes
    # (the shape and the per-core split in the compile-time args: the program cache does not see runtime values)
    key = [rt, qt, per, *struct.unpack("<I", struct.pack("<f", float(limit)))]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "swiglu_reader.cpp",
            core_ranges=cores,
            compile_time_args=[qt, CB_A, CB_B] + ttnn.TensorAccessorArgs(gate_up).get_compile_time_args() + key,
            runtime_args=rd,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=COMPUTE,
            core_ranges=cores,
            compile_time_args=[1, 0] + key,  # num_tiles_per_cycle, fill_with_value_int
            defines=_defines(limit),
            runtime_args=cp,
            config=cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=KDIR + "swiglu_writer.cpp",
            core_ranges=cores,
            compile_time_args=[CB_O] + ttnn.TensorAccessorArgs(out).get_compile_time_args() + key,
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    cbs = [
        _cb(CB_A, PAGES, cores),
        _cb(CB_B, PAGES, cores),
        _cb(CB_O, PAGES, cores),
        _cb(CB_A_POST, 1, cores),
        _cb(CB_B_POST, 1, cores),
    ]
    mesh_program = ttnn.MeshProgramDescriptor()
    whole = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(ROWS - 1, COLS - 1))
    mesh_program[whole] = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op([gate_up, out], mesh_program)
    return out
