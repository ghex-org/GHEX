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
from ghex.pyghex import make_co_unstructured as _make_co_unstructured

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ghex.context import context


def make_communication_object(context: context):
    return _make_co_unstructured(context)

class DomainDescriptor(CppWrapper):
    def __init__(self, index: int, indices: ArrayLike, halo_indices: ArrayLike):
        super(DomainDescriptor, self).__init__(
            ("unstructured__domain_descriptor", "int", "int"), index, indices, halo_indices
        )
