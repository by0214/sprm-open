from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HiddenStateRef:
    """Reference to one hidden-state vector stored in a binary file."""

    file: str
    offset_bytes: int
    dtype: str
    shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "offset_bytes": int(self.offset_bytes),
            "dtype": self.dtype,
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HiddenStateRef":
        return cls(
            file=str(data["file"]),
            offset_bytes=int(data["offset_bytes"]),
            dtype=str(data.get("dtype", "float16")),
            shape=tuple(int(x) for x in data.get("shape", [])),
        )


class HiddenStateBinWriter:
    """Append or overwrite a binary file of contiguous hidden-state vectors."""

    def __init__(self, bin_path: str | os.PathLike[str], dtype: str = "float16", mode: str = "wb"):
        if mode not in {"wb", "ab"}:
            raise ValueError("mode must be 'wb' or 'ab'")
        self.bin_path = Path(bin_path)
        self.dtype = np.dtype(dtype)
        self.bin_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.bin_path.open(mode)

    def tell(self) -> int:
        return int(self._fh.tell())

    def write_vector(self, vector: np.ndarray | list[float] | tuple[float, ...]) -> HiddenStateRef:
        arr = np.asarray(vector)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        arr = np.ascontiguousarray(arr.astype(self.dtype, copy=False))
        offset = self.tell()
        self._fh.write(arr.tobytes(order="C"))
        return HiddenStateRef(
            file=str(self.bin_path),
            offset_bytes=offset,
            dtype=str(self.dtype),
            shape=tuple(int(x) for x in arr.shape),
        )

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.flush()
        finally:
            self._fh.close()

    def __enter__(self) -> "HiddenStateBinWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class HiddenStateBinReader:
    """Memmap-backed reader for hidden-state references."""

    def __init__(self) -> None:
        self._mmap_cache: dict[tuple[str, str], np.memmap] = {}

    def _get_mmap(self, bin_path: str, dtype: str) -> np.memmap:
        key = (bin_path, dtype)
        mmap = self._mmap_cache.get(key)
        if mmap is None:
            mmap = np.memmap(bin_path, dtype=np.dtype(dtype), mode="r")
            self._mmap_cache[key] = mmap
        return mmap

    def read_ref(
        self,
        ref: HiddenStateRef | dict[str, Any],
        base_dir: str | os.PathLike[str] | None = None,
    ) -> np.ndarray:
        if isinstance(ref, dict):
            ref = HiddenStateRef.from_dict(ref)

        bin_path = ref.file
        if base_dir is not None and not os.path.isabs(bin_path):
            bin_path = os.path.join(str(base_dir), bin_path)

        dtype = ref.dtype or "float16"
        mmap = self._get_mmap(bin_path, dtype)
        itemsize = np.dtype(dtype).itemsize
        start = int(ref.offset_bytes // itemsize)
        length = int(np.prod(ref.shape))
        out = np.array(mmap[start : start + length], dtype=np.float32, copy=True)
        if len(ref.shape) > 1:
            out = out.reshape(ref.shape)
        return out
