# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DFlash2's next step's inputs made on the device (GLM53_DFLASH2_NEXT_INPUTS): the serving
step's host write (the token ids, DSA's positions, KDA's keep, the drafter's context positions and bases: one packed
write, `InputPack`) is skipped when the adapter's inputs are those that continue every seat from the step before.

`NextInputs.run` (the step's last op, inside its trace): from the step's gathered results (`dflash2_step.
gathered_results`: each row's n, bonus tokens and the drafter's next blocks), the step's own context positions and live
mask, the packed input row of the step that continues each live seat (its bonus at p + n + 1, its next drafts after it,
keep n; a seat without a request as the generator writes it: token 0 at position 0, keep K), written over the packed
tensor; the next step's `unpack` copies it to the fields' tensors.  `predict` makes the same values on the host from the
step's read-back results; the generator compares the adapter's next inputs with them (`Glm53Generator.dflash_verify`)
and writes only where they differ (a request's end, a new request, a cut proposal).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.blackhole.glm53_flash.ttnn.common import COLS, ROWS

KDIR = "models/demos/blackhole/glm53_flash/ttnn/kernels/"
SLOT = 64
TOKENS, KEEP, POSITIONS, CTX, BASE1 = 0, 1, 2, 3, 4


class NextInputs:
    """The device op for one DFlash2Step (its io, pack, context buffers and K), and the host's prediction."""

    def __init__(self, model, step) -> None:
        io, pack = step.io, step.io.pack
        assert pack is not None and step.packed, "GLM53_DFLASH2_NEXT_INPUTS needs the context positions packed"
        assert not step.relaxed, "GLM53_DFLASH2_NEXT_INPUTS: no looser acceptance"
        self.mesh, self.step, self.model = model.mesh, step, model
        self.u, self.k = step.users, step.k
        self.max_pos = model.dsa.max_seq - 1 if model.dsa is not None else (1 << 30)
        dsa_pos = io.positions
        fields = [(TOKENS, io.ids, 1), (KEEP, io.keep.flags, 1), (CTX, step.ctx_pos, 1), (BASE1, step.base1, 1)]
        if model.dsa is not None:
            from models.demos.blackhole.glm53_flash.ttnn import dsa as dsa_mod

            if model.dsa.long and dsa_mod.TRIM_WRITES:  # the long cache: params' word 0 (the position) alone
                fields.append((POSITIONS, dsa_pos.params, 16))
            else:
                assert len(dsa_pos.attend) == 1, "GLM53_DFLASH2_NEXT_INPUTS: one sdpa call (the dense form)"
                fields.append((POSITIONS, dsa_pos.cur_pos, 1))
        self.fields = []
        for kind, t, stride in fields:
            assert pack.has(t), (kind, "not in the input pack")
            _, offset, words, _, slot = pack.fields[id(t)]
            self.fields.append((kind, offset * 4, words, slot * 4, stride))
        self.packed = pack.device
        self.live = io.live.tensor
        self.cb_bytes = pack.width * 4 + 4 * SLOT + 4096  # the packed row, the results' row, ctx_pos, the mask

    def _program(self, r: int, results) -> ttnn.ProgramDescriptor:
        core = ttnn.CoreCoord(0, 0)
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        args = [self.packed.buffer_address(), self.packed.buffer_page_size()]
        args += [results.buffer_address(), results.buffer_page_size(), r]
        args += [self.step.ctx_pos.buffer_address(), self.step.ctx_pos.buffer_page_size()]
        args += [self.live.buffer_address(), self.live.buffer_page_size()]
        for kind, offset, words, slot, stride in self.fields:
            args += [kind, offset, words, slot, stride]
        rt = ttnn.RuntimeArgs()
        rt[0][0] = args
        kernel = ttnn.KernelDescriptor(
            kernel_source=KDIR + "next_pack.cpp",
            core_ranges=cores,
            compile_time_args=[self.u, self.k, self.max_pos, len(self.fields), 0],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        cb = ttnn.CBDescriptor(
            total_size=self.cb_bytes,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=self.cb_bytes)
            ],
        )
        return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])

    def run(self, results) -> None:
        """The packed tensor <- the next step's inputs (a device op: the step's last, inside its capture)."""
        assert results.buffer_page_size() == 10 * self.u * 4, (results.shape, "the one read's [1, 1, 8, 10 U]")
        mesh_program = ttnn.MeshProgramDescriptor()
        for r in range(ROWS):
            span = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, 0), ttnn.MeshCoordinate(r, COLS - 1))
            mesh_program[span] = self._program(r, results)
        ttnn.generic_op([self.packed, results], mesh_program)

    def predict(self, n, bonus, drafts, positions, live) -> tuple:
        """The step's read-back n, bonus [ROWS, U], drafts [ROWS, U, K] (int64) and its own inputs' positions [ROWS,
        (K + 1) U] and live [ROWS, (K + 1) U] bool -> the inputs `NextInputs.run` wrote for the next step, as
        `Glm53Generator._dflash_inputs` returns them: (tokens, positions, keep, live)."""
        u, k = self.u, self.k
        on = live[:, :u]
        base = positions[:, :u] + n + 1  # [ROWS, U]
        j = torch.arange(k + 1)
        pos = (base[:, None, :] + j[None, :, None]).clamp(max=self.max_pos)  # [ROWS, K + 1, U]
        tok = torch.cat([bonus[:, None, :], drafts.transpose(1, 2)], dim=1)  # [ROWS, K + 1, U]
        on3 = on[:, None, :].expand(-1, k + 1, -1)
        tok = torch.where(on3, tok, 0).reshape(ROWS, (k + 1) * u)
        pos = torch.where(on3, pos, 0).reshape(ROWS, (k + 1) * u)
        keep = torch.where(on, n, k)
        return tok, pos, keep, live.clone()


__all__ = ["NextInputs"]
