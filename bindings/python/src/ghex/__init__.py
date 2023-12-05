#
# ghex-org
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

import ghex.bind as _pyghex
from ghex.bind import mpi_comm as mpi_comm
import os


# Parse version.txt file for the ghex version string
def get_version() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "version.txt")) as version_file:
        return version_file.read().strip()


__version__ = get_version()
__config__ = _pyghex.config()  # noqa:F405

# Remove get_version from module.
del get_version
