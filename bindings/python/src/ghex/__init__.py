# -*- coding: utf-8 -*-
#
# GridTools
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
__copyright__ = "Copyright (c) 2014-2021 ETH Zurich"
__license__ = "BSD-3-Clause"

import os
import sys
sys.path.append(os.environ.get('GHEX_PY_LIB_PATH', "/home/tille/Development/GHEX/build"))

from ghex.utils.cpp_wrapper_utils import unwrap
import ghex_py_bindings as _ghex

def make_pattern(context, halo_gen, domain_range):
    return _ghex.make_pattern(unwrap(context), unwrap(halo_gen), [unwrap(d) for d in domain_range])

CommunicationObject = _ghex.CommunicationObject

#wrap_field = _ghex.wrap_field

#def wrap_field(domain_desc: DomainDescriptor, field: np.ndarray, offsets: Sequence[int, ...], extents: Sequence[int, ...]):
#    FieldDescriptor
#    ghex.wrap_field(domain_desc.__wrapped__,
#                    field,
#                    offsets,
#                    extents)
from .structured.regular import *

from . import tl