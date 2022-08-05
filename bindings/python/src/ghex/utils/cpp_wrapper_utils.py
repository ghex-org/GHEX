# -*- coding: utf-8 -*-
#
# GridTools
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import inspect

from typing import Tuple, Union
from functools import reduce

import ghex_py_bindings as _ghex

def unwrap(arg):
    return arg.__wrapped__ if isinstance(arg, CppWrapper) else arg

def dtype_to_cpp(dtype):
    """Convert numpy dtype to c++ type"""
    import numpy as np

    return {np.float64: "double", np.float32: "float"}[dtype.type]

def cls_from_cpp_type_spec(cpp_type_spec: Union[str, Tuple[str, ...]]):
    if isinstance(cpp_type_spec, str):
        return getattr(_ghex, cpp_type_spec)
    else:
        fq_cpp_type_name, *template_args = cpp_type_spec
        template_args = [targ if not isinstance(targ,
                                                int) else f"std::integral_constant<int, {targ}> "
                         for targ in
                         template_args]
        fq_cpp_type_specialization_name = fq_cpp_type_name + "<" + ", ".join(
            template_args) + ">"

        return getattr(_ghex, fq_cpp_type_specialization_name)

class CppWrapper:
    __wrapped__ = None

    def __init__(self, cpp_type_spec: Union[str, Tuple[str, ...]], *args, **kwargs):
        wrapped_cls = cls_from_cpp_type_spec(cpp_type_spec)

        self.__wrapped__ = wrapped_cls(*(unwrap(arg) for arg in args), **{kw: unwrap(arg) for kw, arg in kwargs.items()})

    def __wrapped_call__(self, method_name, *args, **kwargs):
        method = getattr(self.__wrapped__, method_name)
        return method(*(unwrap(arg) for arg in args), **{kw: unwrap(arg) for kw, arg in kwargs.items()})

    def __getattr__(self, name):
        if hasattr(self, "__wrapped__"):
            attr = getattr(self.__wrapped__, name)
            if inspect.ismethod(attr):
                return lambda *args, **kwargs: self.__wrapped_call__(name, *args, **kwargs)
            return attr

        raise AttributeError(name)