import io
import random
import zipfile

import numpy as np
import pytest

from neko_data.contract import TensorValue, TypedSample, decode_typed_tree, encode_typed_tree


def sample_tree():
    return {"workers": [{"units": np.arange(n * 3, dtype=np.int64).reshape(n, 3),
                         "mask": np.ones((n, 17), dtype=np.bool_),
                         "ar": {"selected": (np.array([1, 3], np.int64), None),
                                "legal": np.zeros((16, n), np.bool_)},
                         "logp": np.array([-0.0, np.nan, np.inf], dtype=">f8")}
                        for n in (0, 1, 9)],
            "bf16": TensorValue("bfloat16", (3,), bytes.fromhex("803f0080c17f")),
            "empty_bf16": TensorValue("bfloat16", (0, 7), b""), "blob": b"\x00\xff" * 1024,
            "fortran": np.asfortranarray(np.arange(12, dtype=np.int32).reshape(3, 4)),
            "scalars": (True, 2**61, np.float16(-0.0), None)}


def npy_members(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_deflate_keeps_every_npy_member_and_decoded_value_bit_exact():
    tree = sample_tree()
    stored = encode_typed_tree(tree)
    compressed = encode_typed_tree(tree, compression_level=1)
    assert npy_members(compressed) == npy_members(stored)
    with zipfile.ZipFile(io.BytesIO(compressed)) as archive:
        assert all(item.compress_type == zipfile.ZIP_DEFLATED for item in archive.infolist())
    decoded = decode_typed_tree(compressed)
    # Deterministic canonical encoding proves the complete decoded tree,
    # including dtype/shape/order/tuple/bytes/NaN payload, is unchanged.
    assert encode_typed_tree(decoded) == stored
    assert decoded["bf16"] == tree["bf16"]
    assert decoded["empty_bf16"] == tree["empty_bf16"]
    assert type(decoded["scalars"]) is tuple


def test_default_bytes_are_exact_original_np_savez_output(monkeypatch):
    original = np.savez
    captured = {}

    def capture(output, **arrays):
        captured.update(arrays)
        return original(output, **arrays)

    monkeypatch.setattr(np, "savez", capture)
    actual = encode_typed_tree(sample_tree())
    expected = io.BytesIO()
    original(expected, **captured)
    assert actual == expected.getvalue()
    assert encode_typed_tree(sample_tree(), compression_level=0) == actual


@pytest.mark.parametrize("level", [-1, 10, True, False, 1.0, None, "1"])
def test_invalid_levels_fail_explicitly(level):
    with pytest.raises(ValueError, match="compression_level"):
        encode_typed_tree(None, compression_level=level)
    with pytest.raises(ValueError, match="compression_level"):
        TypedSample("safe", None, compression_level=level)


def test_compression_does_not_bypass_expanded_data_limit_or_bomb_guard(monkeypatch):
    values = np.zeros(1024**2, np.uint8)
    with pytest.raises(ValueError, match="max_bytes"):
        encode_typed_tree(values, max_bytes=64 * 1024, compression_level=1)
    encoded = encode_typed_tree(values, max_bytes=2 * 1024**2, compression_level=1)
    assert len(encoded) < 64 * 1024
    monkeypatch.setattr(np, "load", lambda *_a, **_kw: pytest.fail("expanded-size guard must precede NumPy allocation"))
    with pytest.raises(ValueError, match="Expanded"):
        decode_typed_tree(encoded, max_bytes=64 * 1024)


def test_deflated_object_payload_is_still_forbidden():
    payload = io.BytesIO()
    np.savez_compressed(payload, __tree__=np.array([], np.uint8), v00000000=np.array([None], object))
    with pytest.raises(ValueError, match="dtype|Object"):
        decode_typed_tree(payload.getvalue())


def test_deflate_roundtrip_preserves_global_python_numpy_and_torch_cpu_rng():
    torch = pytest.importorskip("torch")
    py, numpy_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    decode_typed_tree(encode_typed_tree(sample_tree(), compression_level=1))
    assert random.getstate() == py
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]
    assert torch.equal(cpu, torch.get_rng_state())
