# -*- coding: utf-8 -*-
#
# GridTools
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import ghex_py_bindings as _ghex
from ghex.utils.cpp_wrapper_utils import CppWrapper, dtype_to_cpp

class Context(CppWrapper):
    def __init__(self, mpi_comm):
        super(Context, self).__init__("gridtools::ghex::tl::context<gridtools::ghex::tl::mpi::transport_context>", mpi_comm)