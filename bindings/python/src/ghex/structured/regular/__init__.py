# -*- coding: utf-8 -*-
#
# GridTools
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Tuple

import numpy as np

import ghex_py_bindings as _ghex
from ghex.utils.cpp_wrapper_utils import CppWrapper, dtype_to_cpp, unwrap, cls_from_cpp_type_spec
from ghex.utils.index_space import CartesianSet, ProductSet, union

class HaloContainer:
    local: CartesianSet
    global_: CartesianSet

    def __init__(self, local: CartesianSet, global_: CartesianSet):
        self.local = local
        self.global_ = global_

class DomainDescriptor(CppWrapper):
    def __init__(self, id_: int, sub_domain_indices: ProductSet):
        super(DomainDescriptor, self).__init__(
            ("gridtools::ghex::structured::regular::domain_descriptor", "int", 3),
            id_, sub_domain_indices[0, 0, 0], sub_domain_indices[-1, -1, -1])

class HaloGenerator(CppWrapper):
    def __init__(self, glob_domain_indices: ProductSet, halos, periodicity):
        assert glob_domain_indices.dim == len(halos)
        assert glob_domain_indices.dim == len(periodicity)

        # canonanicalize integer halos, e.g. turn (h0, (h1, h2), h3) into ((h0, h0), (h1, h2), ...)
        halos = ((halo, halo) if isinstance(halo, int) else halo for halo in halos)
        flattened_halos = tuple(h for halo in halos for h in halo)

        super(HaloGenerator, self).__init__(
            ("gridtools::ghex::structured::regular::halo_generator", "int", 3),
            glob_domain_indices[0, 0, 0], glob_domain_indices[-1, -1, -1], flattened_halos, periodicity)

    def __call__(self, domain: DomainDescriptor):
        result = self.__wrapped_call__("__call__", domain)

        local = union(*(ProductSet.from_coords(tuple(box2.local.first), tuple(box2.local.last)) for box2 in result))
        global_ = union(*(ProductSet.from_coords(tuple(box2.global_.first), tuple(box2.global_.last)) for box2 in result))
        return HaloContainer(local, global_)


# todo: try importing gt4py to see if it's there, avoiding the dependency

def _layout_order(field) -> tuple[int, ...]:
    ordered_strides = list(reversed(sorted(field.strides)))
    layout_map = [ordered_strides.index(stride) for stride in field.strides]
    # ensure layout map has unique indices in case the size in dimension is one
    for i, val in enumerate(layout_map):
        if val in layout_map[:i]:
            layout_map[i] = max(layout_map)+1
    return tuple(layout_map)


def FieldDescriptor(
    domain_desc: DomainDescriptor,
    field: np.ndarray,
    offsets: Tuple[int, ...],
    extents: Tuple[int, ...]
):
    type_spec = ("gridtools::ghex::structured::regular::field_descriptor",
                 dtype_to_cpp(field.dtype), "gridtools::ghex::cpu",
                 domain_desc.__cpp_type__,
                 f"gridtools::layout_map<{', '.join(map(str, _layout_order(field)))}> ")
    return cls_from_cpp_type_spec(type_spec)(
        unwrap(domain_desc),
        unwrap(field),
        unwrap(offsets),
        unwrap(extents)
    )

def wrap_field(*args):
    return FieldDescriptor(*args)