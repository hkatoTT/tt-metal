"""The GLM-5.3-Flash checkpoint: the pinned revision and zero-copy tensor access.

Tensors are views of the safetensors shards, mapped copy-on-write: reading one touches only its own pages, so the
643 GB checkpoint is usable on a host with less memory than that (the page cache keeps what was read last).  A view
never writes back to the file.
"""

from __future__ import annotations

import json
import mmap
import os
import re
import struct
import threading
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import torch

# Read-ahead of the layers about to be loaded (GLM53_PREFETCH, default 1; 0: off).  A model build reads every tensor once,
# layer by layer, through the mapped views, and a page cache smaller than the 600 GB checkpoint cannot keep it between
# processes.  A thread reads the next layers' byte ranges with large sequential reads into the page cache, at most
# PREFETCH_LAYERS ahead of the layer being read, so the views' page faults find their pages there.  The same bytes: no
# value changes.  The load is bound by the host's BFP8 conversion rather than the disk: this shortens it by ~6 %.
PREFETCH = os.environ.get("GLM53_PREFETCH", "1") == "1"
PREFETCH_LAYERS = int(os.environ.get("GLM53_PREFETCH_LAYERS", "4"))
PREFETCH_CHUNK = 64 << 20
_LAYER = re.compile(r"\.layers\.(\d+)\.")

HF_REPO = "zai-org/GLM-5.3-Flash-BF16"
# The pinned revision: 130 files, 642,676,400,791 bytes (120 safetensors shards, the tokenizer, the chat template).  The
# weights are unquantized bf16; the FP8 sibling `zai-org/GLM-5.3-Flash` (46a050bec130) is not used as the reference.
PINNED_REVISION = "f9fed3bbc82291073373d67452e9f25317ef7577"

_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "I64": torch.int64,
    "I32": torch.int32,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def checkpoint_root(root: Path | str | None = None) -> Path:
    """The checkpoint directory: the argument, else ``$GLM53_CHECKPOINT`` (`paths.checkpoint_dir`)."""
    from models.demos.blackhole.glm53_flash.paths import checkpoint_dir

    return checkpoint_dir(root)


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    begin: int  # absolute byte offset in the shard file
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


class Glm53Checkpoint:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = checkpoint_root(root)
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._maps: dict[str, mmap.mmap] = {}
        self._reading = -1  # the highest layer whose tensor was read (the read-ahead keeps PREFETCH_LAYERS ahead of it)
        self._ahead: threading.Thread | None = None
        self._wake = threading.Condition()

    @cached_property
    def config(self) -> dict:
        """The raw ``config.json`` (parsed and checked by ``config.Glm53Config``)."""
        return json.loads((self.root / "config.json").read_text())

    def _header(self, shard: str) -> tuple[int, dict]:
        with open(self.root / shard, "rb") as stream:
            (length,) = struct.unpack("<Q", stream.read(8))
            header = json.loads(stream.read(length))
        header.pop("__metadata__", None)
        return 8 + length, header

    @cached_property
    def infos(self) -> dict[str, TensorInfo]:
        """Every tensor's dtype, shape and file range, from the shard headers (no tensor data is read)."""
        out = {}
        for shard in sorted(set(self.weight_map.values())):
            base, header = self._header(shard)
            for name, entry in header.items():
                begin, end = entry["data_offsets"]
                out[name] = TensorInfo(name, shard, entry["dtype"], tuple(entry["shape"]), base + begin, base + end)
        if set(out) != set(self.weight_map):
            raise ValueError(
                f"shard headers and the index disagree: {len(set(self.weight_map) - set(out))} names only in the index,"
                f" {len(set(out) - set(self.weight_map))} only in the headers"
            )
        for info in out.values():
            if info.numel * _DTYPES[info.dtype].itemsize != info.nbytes:
                raise ValueError(f"{info.name}: {info.shape} {info.dtype} does not fill {info.nbytes} bytes")
        return out

    def names(self, prefix: str = "") -> list[str]:
        return sorted(name for name in self.weight_map if name.startswith(prefix))

    def _map(self, shard: str) -> mmap.mmap:
        if shard not in self._maps:
            with open(self.root / shard, "rb") as stream:
                self._maps[shard] = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
        return self._maps[shard]

    def tensor(self, name: str) -> torch.Tensor:
        """A zero-copy view of one tensor in its stored dtype (copy-on-write: writing to it never reaches the file)."""
        info = self.infos[name]
        dtype = _DTYPES[info.dtype]
        if PREFETCH:
            self._note_layer(name)
        if info.numel == 0:
            return torch.empty(info.shape, dtype=dtype)
        flat = torch.frombuffer(self._map(info.shard), dtype=dtype, count=info.numel, offset=info.begin)
        return flat.view(info.shape)

    def _note_layer(self, name: str) -> None:
        """The read-ahead's progress: started at the first layer tensor read, told each higher layer reached."""
        m = _LAYER.search(name)
        if m is None:
            return
        layer = int(m.group(1))
        with self._wake:  # (the uploads read tensors from several threads: one read-ahead thread, started once)
            if self._ahead is None:
                self._ahead = threading.Thread(
                    target=self._read_ahead, args=(layer,), daemon=True, name="glm53-read-ahead"
                )
                self._ahead.start()
            if layer > self._reading:
                self._reading = layer
                self._wake.notify()

    def _read_ahead(self, first: int) -> None:
        """Each layer from `first` on: its tensors' byte ranges, merged per shard, read in PREFETCH_CHUNK pieces (the page
        cache keeps them for the views), waiting while it is PREFETCH_LAYERS ahead of the layer being read."""
        ranges: dict[int, dict[str, list[tuple[int, int]]]] = {}
        for info in self.infos.values():
            m = _LAYER.search(info.name)
            if m is not None and int(m.group(1)) >= first:
                ranges.setdefault(int(m.group(1)), {}).setdefault(info.shard, []).append((info.begin, info.end))
        buf = bytearray(PREFETCH_CHUNK)
        for layer in sorted(ranges):
            with self._wake:
                while layer > self._reading + PREFETCH_LAYERS:
                    self._wake.wait(timeout=5.0)
            for shard, spans in ranges[layer].items():
                merged: list[list[int]] = []
                for begin, end in sorted(spans):
                    if merged and begin <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], end)
                    else:
                        merged.append([begin, end])
                fd = os.open(self.root / shard, os.O_RDONLY)
                try:
                    for begin, end in merged:
                        pos = begin
                        while pos < end:
                            n = os.preadv(fd, [memoryview(buf)[: min(PREFETCH_CHUNK, end - pos)]], pos)
                            if n <= 0:
                                break
                            pos += n
                finally:
                    os.close(fd)


class ShapeOnlyCheckpoint(Glm53Checkpoint):
    """The checkpoint's names, shapes and dtypes, without its data: every tensor is a torch "meta" tensor.

    For speed experiments: the device code builds its weights from these as usual, and `ttnn/common.py` turns a meta
    tensor into random values of the same shape, dtype and placement made on the device, so no weight is read from
    disk or sent over PCIe.  Timing is the same as with the real weights where the kernels do not depend on values;
    numerics are not meaningful.  The sparse MoE paths' timing depends on the routing: the random selection bias
    outweighs the random router's logits ~10x, so every token would pick about the same 8 experts and few experts'
    weights would be read; ttnn/moe.py zeroes it by default (GLM53_SYNTHETIC_ROUTING=spread) so each token picks its
    own, as real tokens do.
    """

    def tensor(self, name: str) -> torch.Tensor:
        info = self.infos[name]  # (no read-ahead: no data is read)
        return torch.empty(info.shape, dtype=_DTYPES[info.dtype], device="meta")
