# CMake generated Testfile for 
# Source directory: /home/mjs/src/GHEX/test
# Build directory: /home/mjs/src/GHEX/_nix_build/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[decomposition]=] "/home/mjs/src/GHEX/_nix_build/bin/decomposition")
set_tests_properties([=[decomposition]=] PROPERTIES  LABELS "serial" RUN_SERIAL "ON" _BACKTRACE_TRIPLES "/home/mjs/src/GHEX/cmake/ghex_reg_test.cmake;21;add_test;/home/mjs/src/GHEX/test/CMakeLists.txt;16;ghex_reg_test;/home/mjs/src/GHEX/test/CMakeLists.txt;0;")
add_test([=[context]=] "/nix/store/1gx7vrvkclgjc828hmxdscgcvhs5zi1k-python3-3.13.13-env/bin/mpiexec" "-n" "4" "/home/mjs/src/GHEX/_nix_build/bin/context")
set_tests_properties([=[context]=] PROPERTIES  LABELS "parallel-ranks-4" RUN_SERIAL "ON" _BACKTRACE_TRIPLES "/home/mjs/src/GHEX/cmake/ghex_reg_test.cmake;44;add_test;/home/mjs/src/GHEX/test/CMakeLists.txt;20;ghex_reg_parallel_test;/home/mjs/src/GHEX/test/CMakeLists.txt;0;")
add_test([=[mpi_communicator]=] "/nix/store/1gx7vrvkclgjc828hmxdscgcvhs5zi1k-python3-3.13.13-env/bin/mpiexec" "-n" "4" "/home/mjs/src/GHEX/_nix_build/bin/mpi_communicator")
set_tests_properties([=[mpi_communicator]=] PROPERTIES  LABELS "parallel-ranks-4" RUN_SERIAL "ON" _BACKTRACE_TRIPLES "/home/mjs/src/GHEX/cmake/ghex_reg_test.cmake;44;add_test;/home/mjs/src/GHEX/test/CMakeLists.txt;20;ghex_reg_parallel_test;/home/mjs/src/GHEX/test/CMakeLists.txt;0;")
add_test([=[context_mt]=] "/nix/store/1gx7vrvkclgjc828hmxdscgcvhs5zi1k-python3-3.13.13-env/bin/mpiexec" "-n" "4" "/home/mjs/src/GHEX/_nix_build/bin/context_mt")
set_tests_properties([=[context_mt]=] PROPERTIES  LABELS "parallel-ranks-4" RUN_SERIAL "ON" _BACKTRACE_TRIPLES "/home/mjs/src/GHEX/cmake/ghex_reg_test.cmake;44;add_test;/home/mjs/src/GHEX/test/CMakeLists.txt;24;ghex_reg_parallel_test;/home/mjs/src/GHEX/test/CMakeLists.txt;0;")
subdirs("mpi_runner")
subdirs("structured")
subdirs("unstructured")
subdirs("glue")
subdirs("bindings")
