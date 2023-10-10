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
from typing import TYPE_CHECKING

import _pyghex

if TYPE_CHECKING:
    from mpi4py.MPI import Comm


def make_context(mpi_comm: Comm) -> _pyghex.context:
    return _pyghex.context(_pyghex.mpi_comm(mpi_comm))
