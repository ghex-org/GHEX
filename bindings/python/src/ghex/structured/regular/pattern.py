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

from ghex.context import context
from ..._pyghex import make_pattern_regular as _make_pattern_regular
from ghex.util.cpp_wrapper import unwrap

if TYPE_CHECKING:
    from ghex.structured.regular.domain_descriptor import DomainDescriptor
    from ghex.structured.regular.halo_generator import HaloGenerator


def make_pattern(context, halo_gen, domain_range):
    return _make_pattern_regular(context, unwrap(halo_gen), [unwrap(d) for d in domain_range])
