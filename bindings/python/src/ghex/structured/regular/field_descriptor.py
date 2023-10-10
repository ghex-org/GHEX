# -*- coding: utf-8 -*-
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
from typing import TYPE_CHECKING

from ghex.util.architecture import Architecture
from ghex.util.cpp_wrapper import cls_from_cpp_type_spec, dtype_to_cpp, unwrap

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing import Any

    from ghex.structured.regular.domain_descriptor import DomainDescriptor


def _layout_order(field: NDArray, arch: Architecture) -> tuple[int, ...]:
    if arch == Architecture.CPU:
        strides = field.__array_interface__["strides"]
    elif arch == Architecture.GPU:
        strides = field.__cuda_array_interface__["strides"]
    else:
        raise ValueError()

    ordered_strides = list(reversed(sorted(strides)))
    layout_map = [ordered_strides.index(stride) for stride in strides]
    # ensure layout map has unique indices in case the size in dimension is one
    for i, val in enumerate(layout_map):
        if val in layout_map[:i]:
            layout_map[i] = max(layout_map) + 1
    return tuple(layout_map)


def make_field_descriptor(
    domain_desc: DomainDescriptor,
    field: NDArray,
    offsets: tuple[int, ...],
    extents: tuple[int, ...],
    *,
    arch: Architecture = Architecture.CPU,
) -> Any:
    if not arch:
        if hasattr(field, "__cuda_array_interface__"):
            arch = Architecture.GPU
        elif hasattr(field, "__array_interface__"):
            arch = Architecture.CPU
        else:
            raise ValueError()

    if arch == Architecture.CPU:
        assert hasattr(field, "__array_interface__")
    elif arch == Architecture.GPU:
        assert hasattr(field, "__cuda_array_interface__")

    type_spec = (
        "ghex::structured::regular::field_descriptor",
        dtype_to_cpp(field.dtype),
        arch.value,
        domain_desc.__cpp_type__,
        f"gridtools::layout_map_impl::layout_map<{', '.join(map(str, _layout_order(field, arch)))}> ",
    )
    return cls_from_cpp_type_spec(type_spec)(
        unwrap(domain_desc), unwrap(field), unwrap(offsets), unwrap(extents)
    )


def wrap_field(*args):
    return make_field_descriptor(*args)
