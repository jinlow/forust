from __future__ import annotations

import pytest

from forust.serialize import CommonItem, CommonSerializer

common_values = [
    1,
    1.0,
    "a string",
    [1, 2, 3],
    [1.0, 4.0],
    ["a", "b", "c"],
    ("a", "b"),
    True,
    False,
]


@pytest.mark.parametrize("value", common_values)
def test_common(value: CommonItem):
    serializer = CommonSerializer()
    r = serializer.serialize(value)
    assert isinstance(r, str)
    assert value == serializer.deserialize(r)
