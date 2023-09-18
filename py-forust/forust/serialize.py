from __future__ import annotations

import json
from abc import ABC, abstractmethod
from ast import literal_eval
from dataclasses import dataclass
from typing import Dict, Generic, List, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt

T = TypeVar("T")


class BaseSerializer(ABC, Generic[T]):
    def __call__(self, obj: Union[T, str]) -> Union[T, str]:
        """Serializer is callable, if it's a string we are deserializing, anything else we are serializing. For the string serializer, this works as well, because both serialize and deserialize just return itself.

        Args:
            obj (T | str): Object either to serialize, or deserialize.

        Returns:
            T | str: Object that is either serialized or deserialized.
        """
        if isinstance(obj, str):
            return self.deserialize(obj)
        else:
            return self.serialize(obj)

    @abstractmethod
    def serialize(self, obj: T) -> str:
        ...

    @abstractmethod
    def deserialize(self, obj_repr: str) -> T:
        ...


CommonScaler = Union[int, float, str]
CommonItem = Union[
    Tuple[CommonScaler, ...],
    List[CommonScaler],
    Dict[CommonScaler, CommonScaler],
    CommonScaler,
]


class CommonSerializer(BaseSerializer[CommonItem]):
    def serialize(self, obj: CommonItem) -> str:
        if isinstance(obj, str):
            obj_ = f"'{obj}'"
        else:
            obj_ = str(obj)
        return obj_

    def deserialize(self, obj_repr: str) -> CommonItem:
        return literal_eval(node_or_string=obj_repr)


@dataclass
class NumpyData:
    array: list[float] | list[int]
    dtype: str
    shape: tuple[int, ...]


class NumpySerializer(BaseSerializer[npt.NDArray]):
    def serialize(self, obj: npt.NDArray) -> str:
        return json.dumps(
            {"array": obj.tolist(), "dtype": str(obj.dtype), "shape": obj.shape}
        )

    def deserialize(self, obj_repr: str) -> npt.NDArray:
        data = NumpyData(**json.loads(obj_repr))
        return np.array(data.array, dtype=data.dtype, shape=data.shape)  # type: ignore
