# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/mjs/src/GHEX/ext/googletest")
  file(MAKE_DIRECTORY "/home/mjs/src/GHEX/ext/googletest")
endif()
file(MAKE_DIRECTORY
  "/home/mjs/src/GHEX/_nix_build/googletest-ghex-build-prefix/src/googletest-ghex-build-build"
  "/home/mjs/src/GHEX/_nix_build/ext/googletest"
  "/home/mjs/src/GHEX/_nix_build/googletest-ghex-build-prefix/tmp"
  "/home/mjs/src/GHEX/_nix_build/googletest-ghex-build-prefix/src/googletest-ghex-build-stamp"
  "/home/mjs/src/GHEX/_nix_build/googletest-ghex-build-prefix/src"
  "/home/mjs/src/GHEX/_nix_build/googletest-ghex-build-prefix/src/googletest-ghex-build-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/mjs/src/GHEX/_nix_build/googletest-ghex-build-prefix/src/googletest-ghex-build-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/mjs/src/GHEX/_nix_build/googletest-ghex-build-prefix/src/googletest-ghex-build-stamp${cfgdir}") # cfgdir has leading slash
endif()
