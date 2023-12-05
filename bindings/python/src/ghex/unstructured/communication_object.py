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

from ghex.bind import make_co_unstructured as _make_co_unstructured

if TYPE_CHECKING:
    from ghex.context import context


def make_communication_object(context: context):
    return _make_co_unstructured(context)
