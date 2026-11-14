"""Safe, framework-neutral tensor/blob trees for non-image training samples.

No pickle, imports named by a payload, or executable object reconstruction is
used. Framework tensors can cross this boundary explicitly via ``TensorValue``.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .sample_keys import validate_sample_key

TYPED_SAMPLE_FORMAT = "typed-npz-v1"
TYPED_EXTENSION = "ntensor"
DEFAULT_MAX_BYTES = 1024**3
_TREE = "__tree__"


def _dtype(value: str) -> tuple[str, int]:
    if value == "bfloat16":
        return value, 2
    dtype = np.dtype(value)
    if (dtype.hasobject or dtype.fields or dtype.subdtype or dtype.kind not in "biufc"
            or dtype.itemsize not in {1, 2, 4, 8, 16}):
        raise ValueError(f"Unsupported tensor dtype: {value!r}")
    return dtype.str, dtype.itemsize


def _shape(value) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) > 32:
        raise ValueError("Tensor shape must be a list/tuple of at most 32 dimensions")
    if any(type(n) is not int or n < 0 for n in value):
        raise ValueError("Tensor dimensions must be non-negative integers")
    return tuple(value)


@dataclass(frozen=True)
class TensorValue:
    """A dense C-order tensor represented by dtype, shape and immutable bytes.

    ``bfloat16`` stores the original two-byte words, without converting to
    float32. NumPy arrays otherwise remain NumPy arrays when decoded. Device,
    gradients, strides and alias relationships are deliberately not serialized.
    """

    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        dtype, itemsize = _dtype(self.dtype)
        shape = _shape(self.shape)
        if not isinstance(self.data, bytes) or len(self.data) != math.prod(shape) * itemsize:
            raise ValueError("Tensor bytes do not match dtype and shape")
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "TensorValue":
        return cls(array.dtype.str, tuple(array.shape), array.tobytes(order="C"))

    def to_numpy(self) -> np.ndarray:
        if self.dtype == "bfloat16":
            raise ValueError("NumPy has no standard bfloat16 dtype; use to_torch() or the original bytes")
        return np.frombuffer(self.data, dtype=np.dtype(self.dtype)).copy().reshape(self.shape)

    @classmethod
    def from_torch(cls, tensor) -> "TensorValue":
        """Copy an explicitly supplied dense torch tensor to CPU; torch is optional."""
        import torch

        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or tensor.is_quantized:
            raise TypeError("Expected a dense, non-quantized torch tensor")
        tensor = tensor.detach().cpu().resolve_conj().resolve_neg().contiguous()
        dtype = str(tensor.dtype).removeprefix("torch.")
        data = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return cls(dtype, tuple(tensor.shape), data)

    def to_torch(self):
        """Return a fresh CPU tensor without changing the global torch RNG."""
        import torch

        if self.dtype == "bfloat16":
            # frombuffer cannot consume a zero-byte buffer.
            if not self.data:
                return torch.empty(self.shape, dtype=torch.bfloat16)
            return torch.frombuffer(bytearray(self.data), dtype=torch.bfloat16).clone().reshape(self.shape)
        return torch.from_numpy(self.to_numpy())


def _compression_level(value: int) -> int:
    if type(value) is not int or not 0 <= value <= 9:
        raise ValueError("compression_level must be an integer from 0 (stored) to 9 (Deflate)")
    return value


def encode_typed_tree(value: Any, *, max_bytes: int = DEFAULT_MAX_BYTES, compression_level: int = 0) -> bytes:
    """Encode a nested tree as a lossless NPZ with an explicit JSON schema.

    Supported leaves are JSON scalars, bytes, numeric/bool NumPy arrays/scalars
    and ``TensorValue``. Dictionary keys may be JSON scalars, including integers.
    Ragged arrays are lists/tuples of dense arrays, never NumPy object arrays.
    Level 0 retains the original np.savez bytes; levels 1-9 Deflate each exact
    NPY member. The expanded-data and encoded-file limits apply in both modes.
    """
    level = _compression_level(compression_level)
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    arrays: dict[str, np.ndarray] = {}
    total = 0
    active: set[int] = set()

    def store(array):
        nonlocal total
        if len(arrays) >= 99_999:
            raise ValueError("Typed sample exceeds maximum array count")
        total += array.nbytes
        if total > max_bytes:
            raise ValueError("Typed sample exceeds max_bytes")
        name = f"v{len(arrays):08d}"
        arrays[name] = array
        return name

    def visit(item, depth=0):
        if depth > 64:
            raise ValueError("Typed sample exceeds maximum nesting depth")
        if item is None or type(item) in {str, bool, int, float}:
            if isinstance(item, float) and not math.isfinite(item):
                return ["float", item.hex()]
            return ["scalar", item]
        if isinstance(item, bytes):
            return ["bytes", store(np.frombuffer(item, dtype=np.uint8))]
        if isinstance(item, TensorValue):
            return ["tensor", item.dtype, list(item.shape), store(np.frombuffer(item.data, dtype=np.uint8))]
        if isinstance(item, (np.ndarray, np.generic)):
            array = np.asarray(item)
            _dtype(array.dtype.str)
            _shape(tuple(array.shape))
            return ["numpy_scalar" if isinstance(item, np.generic) else "array", store(array)]
        if type(item) not in {dict, list, tuple}:
            raise TypeError(f"Unsupported typed value: {type(item).__name__}; convert tensors with TensorValue")
        if id(item) in active:
            raise ValueError("Cyclic typed samples are not supported")
        active.add(id(item))
        try:
            if isinstance(item, dict):
                pairs = []
                for key, child in item.items():
                    if key is not None and type(key) not in {str, bool, int, float}:
                        raise TypeError("Dictionary keys must be JSON scalars")
                    if isinstance(key, float) and not math.isfinite(key):
                        raise ValueError("Dictionary keys must be finite")
                    pairs.append([key, visit(child, depth + 1)])
                return ["dict", pairs]
            return ["tuple" if isinstance(item, tuple) else "list", [visit(x, depth + 1) for x in item]]
        finally:
            active.remove(id(item))

    tree = {"format": TYPED_SAMPLE_FORMAT, "tree": visit(value)}
    encoded = json.dumps(tree, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    total += len(encoded)
    if total > max_bytes:
        raise ValueError("Typed sample exceeds max_bytes")
    arrays[_TREE] = np.frombuffer(encoded, dtype=np.uint8)
    output = io.BytesIO()
    if level == 0:
        np.savez(output, **arrays)  # keep the default wire bytes exactly unchanged
    else:
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=level, allowZip64=True) as archive:
            for name, array in arrays.items():
                # Stream one NPY member, never build/recompress a second whole
                # uncompressed NPZ. Object/pickle dtypes were already rejected.
                with archive.open(name + ".npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, array, allow_pickle=False)
    payload = output.getvalue()
    if len(payload) > max_bytes:
        raise ValueError("Encoded typed sample exceeds max_bytes (including NPZ headers)")
    return payload


def decode_typed_tree(payload: bytes, *, max_bytes: int = DEFAULT_MAX_BYTES) -> Any:
    """Validate and decode a typed NPZ. ``allow_pickle`` is always false."""
    if max_bytes <= 0 or len(payload) > max_bytes:
        raise ValueError("Typed payload exceeds max_bytes")
    stream = io.BytesIO(payload)
    with zipfile.ZipFile(stream) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or len(names) > 100_000:
            raise ValueError("Duplicate or excessive typed archive members")
        if any(name != f"{_TREE}.npy" and not (name.startswith("v") and name[1:-4].isdigit()
                                                and name.endswith(".npy")) for name in names):
            raise ValueError("Unexpected typed archive member")
        if sum(entry.file_size for entry in entries) > max_bytes + len(entries) * 256:
            raise ValueError("Expanded typed payload exceeds max_bytes")
        # np.load may allocate from an NPY header before discovering a short
        # malicious payload. Validate headers and byte counts before loading.
        for entry in entries:
            with archive.open(entry) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise ValueError("Unsupported NPY version in typed payload")
                _dtype(dtype.str)
                if dtype.hasobject:
                    raise ValueError("Object arrays are forbidden in typed payloads")
                expected = math.prod(_shape(shape)) * dtype.itemsize
                if expected > max_bytes or member.tell() + expected != entry.file_size:
                    raise ValueError("Typed array shape/byte count mismatch")
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as archive:
        header = archive[_TREE]
        if header.dtype != np.uint8 or header.ndim != 1 or header.nbytes > max_bytes:
            raise ValueError("Invalid typed JSON header")
        document = json.loads(header.tobytes().decode("utf-8"))
        if set(document) != {"format", "tree"} or document["format"] != TYPED_SAMPLE_FORMAT:
            raise ValueError("Unsupported typed sample format")
        used = {_TREE}
        total = header.nbytes

        def array(name):
            nonlocal total
            if not isinstance(name, str) or name in used or name not in archive.files:
                raise ValueError("Invalid or duplicate typed array reference")
            value = archive[name]
            _dtype(value.dtype.str)
            total += value.nbytes
            if total > max_bytes:
                raise ValueError("Decoded typed sample exceeds max_bytes")
            used.add(name)
            return value

        def restore(node, depth=0):
            if depth > 64 or not isinstance(node, list) or not node or not isinstance(node[0], str):
                raise ValueError("Invalid typed tree node")
            kind = node[0]
            if kind == "scalar" and len(node) == 2:
                value = node[1]
                if value is not None and type(value) not in {str, bool, int, float}:
                    raise ValueError("Invalid typed scalar")
                return value
            if kind == "float" and len(node) == 2 and node[1] in {"inf", "-inf", "nan"}:
                return float.fromhex(node[1])
            if kind in {"array", "numpy_scalar", "bytes"} and len(node) == 2:
                value = array(node[1])
                if kind == "array":
                    return value
                if kind == "numpy_scalar" and value.shape == ():
                    return value[()]
                if kind == "bytes" and value.dtype == np.uint8 and value.ndim == 1:
                    return value.tobytes()
                raise ValueError("Invalid typed leaf shape/dtype")
            if kind == "tensor" and len(node) == 4:
                value = array(node[3])
                if value.dtype != np.uint8 or value.ndim != 1:
                    raise ValueError("Tensor storage must be one-dimensional uint8")
                return TensorValue(node[1], _shape(node[2]), value.tobytes())
            if kind in {"list", "tuple"} and len(node) == 2 and isinstance(node[1], list):
                children = [restore(child, depth + 1) for child in node[1]]
                return tuple(children) if kind == "tuple" else children
            if kind == "dict" and len(node) == 2 and isinstance(node[1], list):
                result = {}
                for pair in node[1]:
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise ValueError("Invalid typed dictionary entry")
                    key, child = pair
                    if key is not None and type(key) not in {str, bool, int, float}:
                        raise ValueError("Invalid typed dictionary key")
                    if isinstance(key, float) and not math.isfinite(key):
                        raise ValueError("Invalid typed dictionary key")
                    if key in result:
                        raise ValueError("Duplicate typed dictionary key")
                    result[key] = restore(child, depth + 1)
                return result
            raise ValueError(f"Invalid typed tree node: {kind!r}")

        result = restore(document["tree"])
        if used != set(archive.files):
            raise ValueError("Unreferenced typed archive members")
        return result


@dataclass(frozen=True)
class TypedSample:
    """One tensor/blob sample; compatible with the existing build/publish path."""

    sample_key: str
    data: Any
    split: str = "train"
    source_id: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    compression_level: int = 0

    def __post_init__(self) -> None:
        validate_sample_key(self.sample_key)
        _compression_level(self.compression_level)
        # Unlike data, searchable sidecar metadata must be ordinary JSON.
        json.dumps(dict(self.metadata), allow_nan=False)

    def encode(self) -> bytes:
        return encode_typed_tree(self.data, compression_level=self.compression_level)

    def metadata_payload(self, payload: bytes) -> dict[str, Any]:
        return {**self.metadata, "schema_version": "1.0", "sample_format": TYPED_SAMPLE_FORMAT,
                "sample_key": self.sample_key, "source_id": self.source_id, "split": self.split,
                "payload_extension": TYPED_EXTENSION, "payload_sha256": hashlib.sha256(payload).hexdigest()}

    def to_training_dict(self, *, include_metadata: bool = True, **_kwargs) -> dict[str, Any]:
        result = {"id": self.sample_key, "data": self.data}
        if include_metadata:
            result["metadata"] = dict(self.metadata)
        return result
