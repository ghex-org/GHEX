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
from typing import TYPE_CHECKING, overload, Optional

from ghex.util import Architecture
from ghex.util import CppWrapper, cls_from_cpp_type_spec, dtype_to_cpp, unwrap
from ghex.context import context
from ghex.pyghex import make_co_unstructured as _make_co_unstructured
from ghex.pyghex import make_pattern_unstructured as _make_pattern_unstructured

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from typing import Any, Optional, List


def make_communication_object(context: context):
    return _make_co_unstructured(context)


class DomainDescriptor(CppWrapper):
    def __init__(self, index: int, indices: ArrayLike, halo_indices: ArrayLike):
        super(DomainDescriptor, self).__init__(
            ("unstructured__domain_descriptor", "int", "int"), index, indices, halo_indices
        )


def make_field_descriptor(
    domain_desc: DomainDescriptor,
    field: NDArray,
    *,
    arch: Optional[Architecture] = Architecture.CPU,
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
        "unstructured__data_descriptor",
        arch.value,
        "int",
        "int",
        dtype_to_cpp(field.dtype),
    )
    return cls_from_cpp_type_spec(type_spec)(unwrap(domain_desc), field)


def wrap_field(*args):
    return make_field_descriptor(*args)


class HaloGenerator(CppWrapper):
    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, gids: ArrayLike) -> None:
        ...

    def __init__(self, gids: Optional[ArrayLike] = None) -> None:
        if gids is None:
            super(HaloGenerator, self).__init__(("unstructured__halo_generator", "int", "int"))
        else:
            super(HaloGenerator, self).__init__(
                ("unstructured__halo_generator", "int", "int"), gids
            )

    @classmethod
    def from_gids(cls, gids: ArrayLike) -> HaloGenerator:
        h = cls(gids)
        return h


def make_pattern(context: context, halo_gen: HaloGenerator, domain_range: List[DomainDescriptor]):
    return _make_pattern_unstructured(context, unwrap(halo_gen), [unwrap(d) for d in domain_range])
