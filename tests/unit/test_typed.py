import io
import json

import numpy as np
import pytest

from neko_data.contract.typed import TensorValue, TypedSample, decode_typed_tree, encode_typed_tree


def test_ragged_nested_tree_roundtrip_preserves_types_and_bits():
    value = {"trace": [{"units": np.arange(n * 3, dtype=np.int64).reshape(n, 3),
                         "mask": np.ones((n,), dtype=np.bool_), "targets": (None, b"\x00\xff", [n])}
                        for n in (0, 1, 7)], 3: np.float16(-0.0),
             "float": np.array([np.inf, np.nan, -0.0], dtype=">f8"),
             "bf16": TensorValue("bfloat16", (2,), bytes.fromhex("803f00c0"))}
    result = decode_typed_tree(encode_typed_tree(value))
    for actual, expected in zip(result["trace"], value["trace"]):
        assert actual["units"].dtype == expected["units"].dtype
        assert actual["units"].shape == expected["units"].shape
        assert actual["units"].tobytes() == expected["units"].tobytes()
        assert actual["mask"].dtype == np.bool_
        assert isinstance(actual["targets"], tuple)
        assert actual["targets"] == expected["targets"]
    assert type(result[3]) is np.float16
    assert result[3].tobytes() == value[3].tobytes()
    assert result["float"].dtype == value["float"].dtype
    assert result["float"].tobytes() == value["float"].tobytes()
    assert result["bf16"] == value["bf16"]


@pytest.mark.parametrize("value", [np.array([object()], dtype=object), np.array(["text"]), object()])
def test_rejects_unsafe_or_implicit_objects(value):
    with pytest.raises((ValueError, TypeError)):
        encode_typed_tree(value)


def test_size_cycle_metadata_and_malformed_shape_rejected():
    with pytest.raises(ValueError, match="max_bytes"):
        encode_typed_tree(np.ones(100), max_bytes=10)
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="Cyclic"):
        encode_typed_tree(cycle)
    with pytest.raises(TypeError):
        TypedSample("x", None, metadata={"array": np.zeros(2)})
    with pytest.raises(ValueError):
        TensorValue("float32", (True,), b"\0" * 4)
    with pytest.raises(ValueError):
        TensorValue("float32", (2,), b"\0" * 4)


def test_decoder_rejects_object_arrays_even_in_forged_archive():
    tree = json.dumps({"format": "typed-npz-v1", "tree": ["array", "v00000000"]}).encode()
    stream = io.BytesIO()
    np.savez(stream, __tree__=np.frombuffer(tree, dtype=np.uint8), v00000000=np.array([None], dtype=object))
    with pytest.raises(ValueError, match="dtype|Object arrays"):
        decode_typed_tree(stream.getvalue())


def test_forged_huge_shape_rejected_before_numpy_allocation(monkeypatch):
    import zipfile

    array = io.BytesIO()
    np.lib.format.write_array_header_1_0(array, {"descr": "|u1", "fortran_order": False, "shape": (2**50,)})
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("v00000000.npy", array.getvalue())
    monkeypatch.setattr(np, "load", lambda *_a, **_kw: pytest.fail("Header validation must run before np.load"))
    with pytest.raises(ValueError, match="shape/byte"):
        decode_typed_tree(stream.getvalue())


def test_optional_torch_roundtrip_has_no_rng_effect():
    torch = pytest.importorskip("torch")
    from neko_data.integrations.torch import from_torch_tree, to_torch_tree

    original = {"bf16": torch.tensor([1.5, -0.0], dtype=torch.bfloat16),
                "mask": torch.tensor([True, False]), "ids": torch.tensor([2**60], dtype=torch.int64),
                "scalar": torch.tensor(3.0), "empty": torch.empty((0, 2)), "tuple": (None, b"blob")}
    rng = torch.get_rng_state().clone()
    result = to_torch_tree(decode_typed_tree(encode_typed_tree(from_torch_tree(original))))
    assert torch.equal(rng, torch.get_rng_state())
    for name in ("bf16", "mask", "ids", "scalar", "empty"):
        assert result[name].dtype == original[name].dtype
        assert result[name].shape == original[name].shape
        assert torch.equal(result[name], original[name])
        assert result[name].device.type == "cpu"
    assert result["tuple"] == (None, b"blob")
