"""Rebuild spider_moe/model.py from base64 chunks.

Usage (from repo root, chunks in spider_moe/):
    python scripts/restore_model.py
"""
from __future__ import annotations
import base64
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent

def find_chunks_dir() -> Path:
    for p in (HERE, HERE.parent / "spider_moe", Path.cwd() / "spider_moe", Path.cwd()):
        if (p / "_model_chunk0.b64").exists():
            return p
    raise FileNotFoundError("could not find _model_chunk0.b64")

def main() -> None:
    d = find_chunks_dir()
    joined = "".join(
        (d / f"_model_chunk{i}.b64").read_text().replace("\n", "")
        for i in range(8)
    )
    data = base64.b64decode(joined)
    if d.name == "spider_moe":
        target = d / "model.py"
    else:
        target = Path.cwd() / "spider_moe" / "model.py"
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    text = data.decode("utf-8")
    if "sparse_expert_delta" not in text or "see_next" in text:
        raise SystemExit("integrity check failed")
    print(f"OK {target} ({len(data)} bytes)")
    print("sha256", hashlib.sha256(data).hexdigest())

if __name__ == "__main__":
    main()
