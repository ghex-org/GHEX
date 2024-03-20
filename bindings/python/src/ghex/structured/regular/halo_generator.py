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
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ghex.structured.cartesian_sets import CartesianSet, ProductSet, union
from ghex.util.cpp_wrapper import CppWrapper

if TYPE_CHECKING:
    from typing import Union

    from ghex.structured.regular.domain_descriptor import DomainDescriptor


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
