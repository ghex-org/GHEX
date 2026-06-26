# CMake generated Testfile for 
# Source directory: /home/mjs/src/GHEX/test/unstructured
# Build directory: /home/mjs/src/GHEX/_nix_build/test/unstructured
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[user_concepts]=] "/nix/store/1gx7vrvkclgjc828hmxdscgcvhs5zi1k-python3-3.13.13-env/bin/mpiexec" "-n" "4" "/home/mjs/src/GHEX/_nix_build/bin/user_concepts")
set_tests_properties([=[user_concepts]=] PROPERTIES  LABELS "parallel-ranks-4" RUN_SERIAL "ON" _BACKTRACE_TRIPLES "/home/mjs/src/GHEX/cmake/ghex_reg_test.cmake;44;add_test;/home/mjs/src/GHEX/test/unstructured/CMakeLists.txt;3;ghex_reg_parallel_test;/home/mjs/src/GHEX/test/unstructured/CMakeLists.txt;0;")
add_test([=[user_concepts_mt]=] "/nix/store/1gx7vrvkclgjc828hmxdscgcvhs5zi1k-python3-3.13.13-env/bin/mpiexec" "-n" "2" "/home/mjs/src/GHEX/_nix_build/bin/user_concepts_mt")
set_tests_properties([=[user_concepts_mt]=] PROPERTIES  LABELS "parallel-ranks-2" RUN_SERIAL "ON" _BACKTRACE_TRIPLES "/home/mjs/src/GHEX/cmake/ghex_reg_test.cmake;44;add_test;/home/mjs/src/GHEX/test/unstructured/CMakeLists.txt;4;ghex_reg_parallel_test;/home/mjs/src/GHEX/test/unstructured/CMakeLists.txt;0;")
