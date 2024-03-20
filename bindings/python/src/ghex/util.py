#
# ghex-org
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
from __future__ import annotations
from enum import Enum
import inspect
from typing import TYPE_CHECKING

import ghex.pyghex as _pyghex

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from typing import Any, Union


class Architecture(Enum):
    CPU = "cpu"
    GPU = "gpu"


def unwrap(arg: Any) -> Any:
    return arg.__wrapped__ if isinstance(arg, CppWrapper) else arg


def dtype_to_cpp(dtype: DTypeLike) -> str:
    """Convert numpy dtype to c++ type"""
    import numpy as np

    return {np.float64: "double", np.float32: "float"}[dtype.type]


def cls_from_cpp_type_spec(cpp_type_spec: Union[str, tuple[str, ...]]) -> Any:
    if isinstance(cpp_type_spec, str):
        return getattr(_pyghex, cpp_type_spec)
    else:
        fq_cpp_type_name, *template_args = cpp_type_spec
        template_args = [
            targ if not isinstance(targ, int) else f"std__integral_constant_int_{targ}_"
            for targ in template_args
        ]
        fq_cpp_type_specialization_name = fq_cpp_type_name + "_" + "_".join(template_args) + "_"

        return getattr(_pyghex, fq_cpp_type_specialization_name)


class CppWrapper:
    __wrapped__ = None

    def __init__(
        self, cpp_type_spec: Union[str, tuple[str, ...]], *args: Any, **kwargs: Any
    ) -> None:
        wrapped_cls = cls_from_cpp_type_spec(cpp_type_spec)

        self.__wrapped__ = wrapped_cls(
            *(unwrap(arg) for arg in args),
            **{kw: unwrap(arg) for kw, arg in kwargs.items()},
        )

    def __wrapped_call__(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self.__wrapped__, method_name)
        return method(
            *(unwrap(arg) for arg in args),
            **{kw: unwrap(arg) for kw, arg in kwargs.items()},
        )

    def __getattr__(self, name: str) -> Any:
        if hasattr(self, "__wrapped__"):
            attr = getattr(self.__wrapped__, name)
            if inspect.ismethod(attr):
                return lambda *args, **kwargs: self.__wrapped_call__(name, *args, **kwargs)
            return attr

        raise AttributeError(name)
