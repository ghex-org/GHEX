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
from ghex.util.cpp_wrapper import CppWrapper
from ghex.pyghex import make_co_regular as _make_co_regular

if TYPE_CHECKING:
    from ghex.structured.index_space import CartesianSet
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
