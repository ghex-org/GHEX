# CMake generated Testfile for 
# Source directory: /home/mjs/src/GHEX/test/structured/cubed_sphere
# Build directory: /home/mjs/src/GHEX/_nix_build/test/structured/cubed_sphere
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[cubed_sphere_transform]=] "/home/mjs/src/GHEX/_nix_build/bin/cubed_sphere_transform")
set_tests_properties([=[cubed_sphere_transform]=] PROPERTIES  LABELS "serial" RUN_SERIAL "ON" _BACKTRACE_TRIPLES "/home/mjs/src/GHEX/cmake/ghex_reg_test.cmake;21;add_test;/home/mjs/src/GHEX/test/structured/cubed_sphere/CMakeLists.txt;3;ghex_reg_test;/home/mjs/src/GHEX/test/structured/cubed_sphere/CMakeLists.txt;0;")
add_test([=[cubed_sphere_exchange]=] "/nix/store/1gx7vrvkclgjc828hmxdscgcvhs5zi1k-python3-3.13.13-env/bin/mpiexec" "-n" "6" "/home/mjs/src/GHEX/_nix_build/bin/cubed_sphere_exchange")
set_tests_properties([=[cubed_sphere_exchange]=] PROPERTIES  LABELS "parallel-ranks-6" RUN_SERIAL "ON" _BACKTRACE_TRIPLES "/home/mjs/src/GHEX/cmake/ghex_reg_test.cmake;44;add_test;/home/mjs/src/GHEX/test/structured/cubed_sphere/CMakeLists.txt;6;ghex_reg_parallel_test;/home/mjs/src/GHEX/test/structured/cubed_sphere/CMakeLists.txt;0;")
