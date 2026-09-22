import json
import struct

from audit_openwam_foundation import audit
from audit_openwam_foundation import scan_safetensors


def test_scan_detects_bf16_nonfinite_values(tmp_path):
    path = tmp_path / "weights.safetensors"
    header = json.dumps({"w": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<4H", 0, 0x3F80, 0x7F80, 0x7FC0))
    assert scan_safetensors(path)["nonfinite_by_tensor"] == {"w": 2}


def test_scan_allows_mixed_float32_when_requested(tmp_path):
    path = tmp_path / "weights.safetensors"
    header = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<2I", 0x3F800000, 0x7F800000))
    assert scan_safetensors(path, require_bf16=False)["nonfinite_by_tensor"] == {"w": 1}


def test_audit_requires_verified_receipt(tmp_path):
    (tmp_path / "foundation-download-receipt.json").write_text(
        json.dumps({"repo": "OpenWAM/test", "revision": "r", "files": []})
    )
    try:
        audit(tmp_path, tmp_path / "audit.json")
    except ValueError as exc:
        assert "no safetensors" in str(exc)
    else:
        raise AssertionError("expected missing weight failure")


def test_audit_accepts_custom_receipt_name(tmp_path):
    (tmp_path / "video-backbone-download-receipt.json").write_text(
        json.dumps({"repo": "Wan-AI/test", "revision": "r", "files": []})
    )
    try:
        audit(
            tmp_path,
            tmp_path / "audit.json",
            "video-backbone-download-receipt.json",
        )
    except ValueError as exc:
        assert "no safetensors" in str(exc)
    else:
        raise AssertionError("expected missing weight failure")
