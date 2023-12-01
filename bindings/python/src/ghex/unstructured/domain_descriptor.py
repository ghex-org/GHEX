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

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class DomainDescriptor(CppWrapper):
    def __init__(self, index: int, indices: ArrayLike, halo_indices: ArrayLike):
        super(DomainDescriptor, self).__init__(
            ("ghex::unstructured::domain_descriptor", "int", "int"), index, indices, halo_indices
        )
