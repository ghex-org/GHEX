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

from ghex.util.cpp_wrapper import CppWrapper

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class HaloGenerator(CppWrapper):
    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, gids: ArrayLike) -> None:
        ...

    def __init__(self, gids: Optional[ArrayLike] = None) -> None:
        if gids is None:
            super(HaloGenerator, self).__init__(
                ("unstructured__halo_generator", "int", "int")
            )
        else:
            super(HaloGenerator, self).__init__(
                ("unstructured__halo_generator", "int", "int"), gids
            )

    @classmethod
    def from_gids(cls, gids: ArrayLike) -> HaloGenerator:
        h = cls(gids)
        return h
