import json
import struct

from audit_download import scan


def test_bf16_finite_nan_inf(tmp_path):
    path = tmp_path / "weights.safetensors"
    header = json.dumps({"w": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<4H", 0, 0x3F80, 0x7F80, 0x7FC0))
    assert scan(path) == {"tensor_count": 1, "nonfinite_by_tensor": {"w": 2}}
