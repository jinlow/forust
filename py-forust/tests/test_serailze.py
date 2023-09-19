from __future__ import annotations

import pytest

from forust.serialize import ObjectItem, ObjectSerializer, Scaler, ScalerSerializer

scaler_values = [
    1,
    1.0,
    "a string",
    True,
    False,
]


@pytest.mark.parametrize("value", scaler_values)
def test_scaler(value: Scaler):
    serializer = ScalerSerializer()
    r = serializer.serialize(value)
    assert isinstance(r, str)
    assert value == serializer.deserialize(r)


object_values = [
    [1, 2, 3],
    [1.0, 4.0],
    ["a", "b", "c"],
    {"a": 1.0, "b": 2.0},
    {"a": "test", "b": "what"},
]


@pytest.mark.parametrize("value", object_values)
def test_object(value: ObjectItem):
    serializer = ObjectSerializer()
    r = serializer.serialize(value)
    assert isinstance(r, str)
    assert value == serializer.deserialize(r)
