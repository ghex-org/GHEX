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
    from typing import Any, Optional

    from ghex.unstructured.domain_descriptor import DomainDescriptor


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
