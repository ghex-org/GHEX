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

import ghex as _pyghex
from ghex.util.cpp_wrapper import unwrap

if TYPE_CHECKING:
    from ghex.unstructured.domain_descriptor import DomainDescriptor
    from ghex.unstructured.halo_generator import HaloGenerator


def make_pattern(context, halo_gen: HaloGenerator, domain_range: List[DomainDescriptor]):
    return _pyghex.make_pattern_unstructured(
        context, unwrap(halo_gen), [unwrap(d) for d in domain_range]
    )
