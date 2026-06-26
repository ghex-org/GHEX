# CMake generated Testfile for 
# Source directory: /home/mjs/src/GHEX/test/glue/gridtools
# Build directory: /home/mjs/src/GHEX/_nix_build/test/glue/gridtools
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[gt_datastore]=] "/nix/store/1gx7vrvkclgjc828hmxdscgcvhs5zi1k-python3-3.13.13-env/bin/mpiexec" "-n" "4" "/home/mjs/src/GHEX/_nix_build/bin/gt_datastore")
set_tests_properties([=[gt_datastore]=] PROPERTIES  LABELS "parallel-ranks-4" RUN_SERIAL "ON" _BACKTRACE_TRIPLES "/home/mjs/src/GHEX/cmake/ghex_reg_test.cmake;44;add_test;/home/mjs/src/GHEX/test/glue/gridtools/CMakeLists.txt;3;ghex_reg_parallel_test;/home/mjs/src/GHEX/test/glue/gridtools/CMakeLists.txt;0;")
