# -*- coding: utf-8 -*-
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

import ghex as _pyghex
from ghex.util.cpp_wrapper import unwrap


def make_pattern(context, halo_gen, domain_range):
    return _pyghex.make_pattern_regular(
        context, unwrap(halo_gen), [unwrap(d) for d in domain_range]
    )
