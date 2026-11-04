"""Load .spiderw native weight bundles (no torch required)."""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

MAGIC = b"SPIDERW1"
HEADER_SIZE = 80

SECTION_NAMES = {
    1: "token_emb",
    2: "lm_head",
    30: "router",
    31: "loop_bias",
    32: "loop_embedding",
    33: "edge_generator",
    34: "edge_value",
    35: "expert_up",
    36: "expert_down",
    37: "moe_norm",
    38: "exit_head",
    40: "csr_row_ptr",
    41: "csr_col_idx",
    42: "csr_edge_w",
}

DTYPE_MAP = {
    0: np.float32,
    1: np.float16,
    2: np.float16,  # bf16 stored as raw u16; exposed as float16 view for Phase 1
    3: np.int32,
    4: np.int8,
}


@dataclass
class SpiderWMeta:
    version: int
    flags: int
    vocab_size: int
    hidden_size: int
    num_heads: int
    num_experts: int
    top_k: int
    edges_per_expert: int
    num_loops: int
    expert_ffn_size: int
    num_prelude_layers: int
    num_coda_layers: int

    @property
    def fp16_weights(self) -> bool:
        return bool(self.flags & 1)

    @property
    def has_csr(self) -> bool:
        return bool(self.flags & 2)

    @property
    def tied_embeddings(self) -> bool:
        return bool(self.flags & 4)


@dataclass
class SpiderWBundle:
    meta: SpiderWMeta
    tensors: Dict[str, np.ndarray] = field(default_factory=dict)
    csr: Optional[Dict[str, np.ndarray]] = None
    path: Optional[str] = None

    def require(self, name: str) -> np.ndarray:
        if name not in self.tensors:
            raise KeyError(f"missing section {name}")
        return self.tensors[name]


def _read_header(buf: bytes) -> Tuple[SpiderWMeta, int]:
    if len(buf) < HEADER_SIZE:
        raise ValueError("file too small for spiderw header")
    if buf[:8] != MAGIC:
        raise ValueError(f"bad magic: {buf[:8]!r}")
    (
        version,
        flags,
        vocab,
        hidden,
        heads,
        experts,
        top_k,
        edges,
        loops,
        h_up,
        n_pre,
        n_coda,
        _r0,
        _r1,
    ) = struct.unpack_from("<14I", buf, 8)
    # skip crc + data_nbytes at 64
    meta = SpiderWMeta(
        version=version,
        flags=flags,
        vocab_size=vocab,
        hidden_size=hidden,
        num_heads=heads,
        num_experts=experts,
        top_k=top_k,
        edges_per_expert=edges,
        num_loops=loops,
        expert_ffn_size=h_up,
        num_prelude_layers=n_pre,
        num_coda_layers=n_coda,
    )
    return meta, HEADER_SIZE


def _read_section(data: bytes, off: int) -> Tuple[str, np.ndarray, int]:
    if off + 24 > len(data):
        raise ValueError("truncated section header")
    section_id, dtype_id, nbytes, rank = struct.unpack_from("<IIQI", data, off)
    off += 20
    dims = struct.unpack_from("<4I", data, off)
    off += 16
    # header was 4+4+8+4+16 = 36
    # fix: we already consumed 20 then 16 = 36 from original off-20
    if off + nbytes > len(data):
        raise ValueError("truncated section payload")
    dt = DTYPE_MAP.get(dtype_id)
    if dt is None:
        raise ValueError(f"unknown dtype id {dtype_id}")
    raw = data[off : off + nbytes]
    shape = tuple(d for d in dims[: max(rank, 1)] if d > 0) or (nbytes // np.dtype(dt).itemsize,)
    arr = np.frombuffer(raw, dtype=dt).reshape(shape) if rank > 0 else np.frombuffer(raw, dtype=dt)
    name = SECTION_NAMES.get(section_id, f"section_{section_id}")
    return name, arr.copy(), off + int(nbytes)


def load_spiderw(path: str | Path) -> SpiderWBundle:
    path = Path(path)
    data = path.read_bytes()
    meta, off = _read_header(data)
    tensors: Dict[str, np.ndarray] = {}
    while off + 36 <= len(data):
        # detect end padding
        if off >= len(data):
            break
        try:
            name, arr, off = _read_section(data, off)
        except ValueError:
            break
        tensors[name] = arr

    csr = None
    if meta.has_csr or all(k in tensors for k in ("csr_row_ptr", "csr_col_idx", "csr_edge_w")):
        csr = {
            "row_ptr": tensors.get("csr_row_ptr"),
            "col_idx": tensors.get("csr_col_idx"),
            "edge_w": tensors.get("csr_edge_w"),
        }
    if meta.tied_embeddings and "lm_head" not in tensors and "token_emb" in tensors:
        tensors["lm_head"] = tensors["token_emb"]

    return SpiderWBundle(meta=meta, tensors=tensors, csr=csr, path=str(path))


def write_spiderw(
    path: str | Path,
    meta: SpiderWMeta,
    tensors: Dict[str, np.ndarray],
    section_ids: Optional[Dict[str, int]] = None,
) -> None:
    """Minimal writer used by export script."""
    inv = {v: k for k, v in SECTION_NAMES.items()}
    if section_ids:
        inv.update({v: k for k, v in section_ids.items()})

    parts = []
    for name, arr in tensors.items():
        sid = inv.get(name)
        if sid is None:
            continue
        arr = np.ascontiguousarray(arr)
        if arr.dtype == np.float32:
            dtype_id = 0
        elif arr.dtype == np.float16:
            dtype_id = 1
        elif arr.dtype == np.int32:
            dtype_id = 3
        else:
            arr = arr.astype(np.float32)
            dtype_id = 0
        rank = min(arr.ndim, 4)
        dims = list(arr.shape) + [1] * (4 - rank)
        payload = arr.tobytes(order="C")
        parts.append(struct.pack("<IIQI", sid, dtype_id, len(payload), rank))
        parts.append(struct.pack("<4I", *[int(d) for d in dims[:4]]))
        parts.append(payload)

    body = b"".join(parts)
    header = bytearray(HEADER_SIZE)
    header[0:8] = MAGIC
    struct.pack_into(
        "<14I",
        header,
        8,
        meta.version,
        meta.flags,
        meta.vocab_size,
        meta.hidden_size,
        meta.num_heads,
        meta.num_experts,
        meta.top_k,
        meta.edges_per_expert,
        meta.num_loops,
        meta.expert_ffn_size,
        meta.num_prelude_layers,
        meta.num_coda_layers,
        0,
        0,
    )
    struct.pack_into("<Q", header, 64, 0)  # crc unused
    struct.pack_into("<Q", header, 72, len(body))
    Path(path).write_bytes(bytes(header) + body)
