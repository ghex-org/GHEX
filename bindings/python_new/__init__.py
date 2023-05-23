# The Python wrapper generated using pybind11 is a compiled dynamic library,
# with a name like _pyghex.cpython-38-x86_64-linux-gnu.so
#
# The library will be installed in the same path as this file, which will
# import the compiled part of the wrapper from the _pyghex namespace.

from ._pyghex import *  # noqa: F403


# Parse version.txt file for the ghex version string
def get_version():
    import os

    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "version.txt")) as version_file:
        return version_file.read().strip()


__version__ = get_version()
__config__ = config()  # noqa:F405

# Remove get_version from module.
del get_version
