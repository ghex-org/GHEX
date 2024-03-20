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
from dataclasses import dataclass

from ghex.pyghex import make_co_regular as _make_co_regular
from ghex.pyghex import make_pattern_regular as _make_pattern_regular
from ghex.util import CppWrapper, cls_from_cpp_type_spec, dtype_to_cpp, unwrap
from ghex.util import Architecture
from ghex.structured.cartesian_sets import CartesianSet, ProductSet, union

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing import Any, Union
    from ghex.context import context


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


@dataclass
class HaloContainer:
    local: CartesianSet
    global_: CartesianSet


class HaloGenerator(CppWrapper):
    def __init__(
        self,
        glob_domain_indices: ProductSet,
        halos: tuple[Union[int, tuple[int, int]], ...],
        periodicity: tuple[bool, ...],
    ) -> None:
        assert glob_domain_indices.dim == len(halos)
        assert glob_domain_indices.dim == len(periodicity)

        # canonicalize integer halos, e.g. turn (h0, (h1, h2), h3) into ((h0, h0), (h1, h2), ...)
        halos2 = ((halo, halo) if isinstance(halo, int) else halo for halo in halos)
        flattened_halos = tuple(h for halo in halos2 for h in halo)

        super(HaloGenerator, self).__init__(
            (
                "structured__regular__halo_generator",
                "int",
                glob_domain_indices.dim,
            ),
            glob_domain_indices[tuple(0 for _ in range(glob_domain_indices.dim))],
            glob_domain_indices[tuple(-1 for _ in range(glob_domain_indices.dim))],
            flattened_halos,
            periodicity,
        )

    def __call__(self, domain: DomainDescriptor) -> HaloContainer:
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


def make_pattern(context: context, halo_gen: HaloGenerator, domain_range: List[DomainDescriptor]):
    return _make_pattern_regular(context, unwrap(halo_gen), [unwrap(d) for d in domain_range])
