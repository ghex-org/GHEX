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

import _pyghex


def make_communication_object(context, pattern):
    return _pyghex.make_co_regular(context, pattern)