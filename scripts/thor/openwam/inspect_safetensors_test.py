import json
import struct

from inspect_safetensors import inspect
import pytest


def test_partial_and_complete(tmp_path):
    path = tmp_path / "model"
    header = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    assert not inspect(path)["size_complete"]
    path.write_bytes(path.read_bytes() + b"\0" * 4)
    result = inspect(path)
    assert result["size_complete"]
    assert result["dtype_tensor_counts"] == {"F32": 1}
    assert not result["finite_values_checked"]


def test_reject_short_header(tmp_path):
    path = tmp_path / "bad"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="Incomplete"):
        inspect(path)
