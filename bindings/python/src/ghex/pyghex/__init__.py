#
# ghex-org
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

# The Python wrapper generated using pybind11 is a compiled dynamic library,
# with a name like _pyghex.cpython-38-x86_64-linux-gnu.so
#
# The library will be installed in the same path as this file, which will
# import the compiled part of the wrapper from the _pyghex namespace.

from .._pyghex import *  # noqa:F403
