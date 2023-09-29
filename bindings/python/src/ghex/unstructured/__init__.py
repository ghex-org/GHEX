#
# ghex-org
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
from typing import Optional

import numpy as np

import ghex as _ghex
from ghex.util.architecture import Architecture
from ghex.util.cpp_wrapper import (
    CppWrapper,
    dtype_to_cpp,
    unwrap,
    cls_from_cpp_type_spec,
)

# from ghex.structured.index_space import cartesian_set, product_set, union


def domain_descriptor(index, indices, halo_indices):
    type_spec = (
        "ghex::unstructured::domain_descriptor",
        "int",
        "int",
    )
    return cls_from_cpp_type_spec(type_spec)(index, indices, halo_indices)


def halo_generator():
    type_spec = (
        "ghex::unstructured::halo_generator",
        "int",
        "int",
    )
    return cls_from_cpp_type_spec(type_spec)()


def halo_generator_with_gids(gids):
    type_spec = (
        "ghex::unstructured::halo_generator",
        "int",
        "int",
    )
    return cls_from_cpp_type_spec(type_spec)(gids)


def field_descriptor(
    domain_desc: domain_descriptor,
    field: np.ndarray,
    *,
    arch: Optional[Architecture] = Architecture.CPU,
):
    if not arch:
        if hasattr(field, "__cuda_array_interface__"):
            arch = Architecture.GPU
        elif hasattr(field, "__array_interface__"):
            arch = Architecture.CPU
        else:
            raise ValueError()

    if arch == Architecture.CPU:
        assert hasattr(field, "__array_interface__")
    if arch == Architecture.GPU:
        assert hasattr(field, "__cuda_array_interface__")

    type_spec = (
        "ghex::unstructured::data_descriptor",
        arch.value,
        "int",
        "int",
        dtype_to_cpp(field.dtype),
    )
    return cls_from_cpp_type_spec(type_spec)(domain_desc, field)


def make_pattern(context, halo_gen, domain_range):
    return _ghex.make_pattern_unstructured(context, unwrap(halo_gen), domain_range)


def make_co(context):
    return _ghex.make_co_unstructured(context)
