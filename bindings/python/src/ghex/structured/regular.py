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
<<<<<<< HEAD
from ghex.util.cpp_wrapper import CppWrapper, cls_from_cpp_type_spec, dtype_to_cpp, unwrap
from ghex.pyghex import make_co_regular as _make_co_regular
from ghex.util.architecture import Architecture

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing import Any

    from ghex.structured.index_space import CartesianSet
    from ghex.context import context
    #from ghex.structured.regular.domain_descriptor import DomainDescriptor

def make_communication_object(context: context, pattern):
    return _make_co_regular(context, pattern)

class DomainDescriptor(CppWrapper):
    def __init__(self, id_: int, sub_domain_indices: CartesianSet) -> None:
        super(DomainDescriptor, self).__init__(
            ("structured__regular__domain_descriptor", "int", sub_domain_indices.dim),
            id_,
            sub_domain_indices[tuple(0 for _ in range(sub_domain_indices.dim))],
            sub_domain_indices[tuple(-1 for _ in range(sub_domain_indices.dim))],
        )

def _layout_order(field: NDArray, arch: Architecture) -> tuple[int, ...]:
    if arch == Architecture.CPU:
        strides = getattr(field, "__array_interface__", {}).get("strides", None)
    elif arch == Architecture.GPU:
        if hasattr(field, "__hip_array_interface__"):
            strides = field.__hip_array_interface__.get("strides", None)
        else:
            strides = getattr(field, "__cuda_array_interface__", {}).get("strides", None)
    else:
        raise ValueError()

    # `strides` field of array interface protocol is empty for C-style contiguous arrays
    if strides is None:
        strides = getattr(field, "strides", None)
    assert strides is not None

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
        if hasattr(field, "__cuda_array_interface__") or hasattr(field, "__hip_array_interface__"):
            arch = Architecture.GPU
        elif hasattr(field, "__array_interface__"):
            arch = Architecture.CPU
        else:
            raise ValueError()

    if arch == Architecture.CPU:
        assert hasattr(field, "__array_interface__")
    elif arch == Architecture.GPU:
        assert hasattr(field, "__cuda_array_interface__") or hasattr(
            field, "__hip_array_interface__"
        )

    type_spec = (
        "structured__regular__field_descriptor",
        dtype_to_cpp(field.dtype),
        arch.value,
        domain_desc.__wrapped__.__class__.__name__,
        f"gridtools__layout_map_impl__layout_map_{'_'.join(map(str,_layout_order(field, arch)))}_",
    )
    return cls_from_cpp_type_spec(type_spec)(
        unwrap(domain_desc), unwrap(field), unwrap(offsets), unwrap(extents)
    )


def wrap_field(*args):
    return make_field_descriptor(*args)
