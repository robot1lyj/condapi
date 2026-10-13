import msgpack
import numpy as np
import pytest
import tree

from openpi_client import msgpack_numpy


@pytest.mark.parametrize("array", [
    np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
    np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2],
    np.asfortranarray(np.arange(24, dtype=np.float32).reshape(4, 6)),
    np.array([1, -2], dtype=">i4"),
    np.array(3.5, dtype=np.float64),
    np.empty((0, 3), dtype=np.uint8),
])
def test_buffer_fast_path_preserves_exact_wire_bytes(array):
    def original(obj):
        if isinstance(obj, np.ndarray):
            return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
        return msgpack_numpy.pack_array(obj)

    expected = msgpack.packb({"array": array}, default=original)
    actual = msgpack_numpy.packb({"array": array})
    assert actual == expected
    np.testing.assert_array_equal(msgpack_numpy.unpackb(actual)["array"], array)


def _check(expected, actual):
    if isinstance(expected, np.ndarray):
        assert expected.shape == actual.shape
        assert expected.dtype == actual.dtype
        assert np.array_equal(expected, actual, equal_nan=expected.dtype.kind == "f")
    else:
        assert expected == actual


@pytest.mark.parametrize(
    "data",
    [
        1,  # int
        1.0,  # float
        "hello",  # string
        np.bool_(True),  # boolean scalar
        np.array([1, 2, 3])[0],  # int scalar
        np.str_("asdf"),  # string scalar
        [1, 2, 3],  # list
        {"key": "value"},  # dict
        {"key": [1, 2, 3]},  # nested dict
        np.array(1.0),  # 0D array
        np.array([1, 2, 3], dtype=np.int32),  # 1D integer array
        np.array(["asdf", "qwer"]),  # string array
        np.array([True, False]),  # boolean array
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),  # 2D float array
        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16),  # 3D integer array
        np.array([np.nan, np.inf, -np.inf]),  # special float values
        {"arr": np.array([1, 2, 3]), "nested": {"arr": np.array([4, 5, 6])}},  # nested dict with arrays
        [np.array([1, 2]), np.array([3, 4])],  # list of arrays
        np.zeros((3, 4, 5), dtype=np.float32),  # 3D zeros
        np.ones((2, 3), dtype=np.float64),  # 2D ones with double precision
    ],
)
def test_pack_unpack(data):
    packed = msgpack_numpy.packb(data)
    unpacked = msgpack_numpy.unpackb(packed)
    tree.map_structure(_check, data, unpacked)
