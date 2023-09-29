#
# ghex-org
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
from typing import Tuple, Optional

import numpy as np

import ghex as _ghex
from ghex.util.architecture import Architecture
from ghex.util.cpp_wrapper import (
    CppWrapper,
    dtype_to_cpp,
    unwrap,
    cls_from_cpp_type_spec,
)
from ghex.structured.index_space import CartesianSet, ProductSet, union


class HaloContainer:
    local: CartesianSet
    global_: CartesianSet

    def __init__(self, local: CartesianSet, global_: CartesianSet):
        self.local = local
        self.global_ = global_


class DomainDescriptor(CppWrapper):
    def __init__(self, id_: int, sub_domain_indices: CartesianSet):
        super(DomainDescriptor, self).__init__(
            (
                "ghex::structured::regular::domain_descriptor",
                "int",
                sub_domain_indices.dim,
            ),
            id_,
            sub_domain_indices[tuple(0 for _ in range(sub_domain_indices.dim))],
            sub_domain_indices[tuple(-1 for _ in range(sub_domain_indices.dim))],
        )


class HaloGenerator(CppWrapper):
    def __init__(self, glob_domain_indices: ProductSet, halos, periodicity):
        assert glob_domain_indices.dim == len(halos)
        assert glob_domain_indices.dim == len(periodicity)

        # canonicalize integer halos, e.g. turn (h0, (h1, h2), h3) into ((h0, h0), (h1, h2), ...)
        halos = ((halo, halo) if isinstance(halo, int) else halo for halo in halos)
        flattened_halos = tuple(h for halo in halos for h in halo)

        super(HaloGenerator, self).__init__(
            (
                "ghex::structured::regular::halo_generator",
                "int",
                glob_domain_indices.dim,
            ),
            glob_domain_indices[tuple(0 for _ in range(glob_domain_indices.dim))],
            glob_domain_indices[tuple(-1 for _ in range(glob_domain_indices.dim))],
            flattened_halos,
            periodicity,
        )

    def __call__(self, domain: DomainDescriptor):
        result = self.__wrapped_call__("__call__", domain)

        local = union(
            *(
                ProductSet.from_coords(tuple(box2.local.first), tuple(box2.local.last))
                for box2 in result
            )
        )
        global_ = union(
            *(
                ProductSet.from_coords(tuple(box2.global_.first), tuple(box2.global_.last))
                for box2 in result
            )
        )
        return HaloContainer(local, global_)


# todo: try importing gt4py to see if it's there, avoiding the dependency


def _layout_order(field: np.ndarray, arch: Architecture) -> tuple[int, ...]:
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


def field_descriptor(
    domain_desc: DomainDescriptor,
    field: np.ndarray,
    offsets: Tuple[int, ...],
    extents: Tuple[int, ...],
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
    return field_descriptor(*args)


def make_pattern(context, halo_gen, domain_range):
    return _ghex.make_pattern_regular(context, unwrap(halo_gen), [unwrap(d) for d in domain_range])


def make_co(context, pattern):
    return _ghex.make_co_regular(context, pattern)
