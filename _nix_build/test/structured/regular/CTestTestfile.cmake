# CMake generated Testfile for 
# Source directory: /home/mjs/src/GHEX/test/structured/regular
# Build directory: /home/mjs/src/GHEX/_nix_build/test/structured/regular
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[regular_domain]=] "/nix/store/1gx7vrvkclgjc828hmxdscgcvhs5zi1k-python3-3.13.13-env/bin/mpiexec" "-n" "4" "/home/mjs/src/GHEX/_nix_build/bin/regular_domain")
set_tests_properties([=[regular_domain]=] PROPERTIES  LABELS "parallel-ranks-4" RUN_SERIAL "ON" _BACKTRACE_TRIPLES "/home/mjs/src/GHEX/cmake/ghex_reg_test.cmake;44;add_test;/home/mjs/src/GHEX/test/structured/regular/CMakeLists.txt;3;ghex_reg_parallel_test;/home/mjs/src/GHEX/test/structured/regular/CMakeLists.txt;0;")
add_test([=[regular_domain_mt]=] "/nix/store/1gx7vrvkclgjc828hmxdscgcvhs5zi1k-python3-3.13.13-env/bin/mpiexec" "-n" "4" "/home/mjs/src/GHEX/_nix_build/bin/regular_domain_mt")
set_tests_properties([=[regular_domain_mt]=] PROPERTIES  LABELS "parallel-ranks-4" RUN_SERIAL "ON" _BACKTRACE_TRIPLES "/home/mjs/src/GHEX/cmake/ghex_reg_test.cmake;44;add_test;/home/mjs/src/GHEX/test/structured/regular/CMakeLists.txt;4;ghex_reg_parallel_test;/home/mjs/src/GHEX/test/structured/regular/CMakeLists.txt;0;")
add_test([=[simple_regular_domain]=] "/nix/store/1gx7vrvkclgjc828hmxdscgcvhs5zi1k-python3-3.13.13-env/bin/mpiexec" "-n" "4" "/home/mjs/src/GHEX/_nix_build/bin/simple_regular_domain")
set_tests_properties([=[simple_regular_domain]=] PROPERTIES  LABELS "parallel-ranks-4" RUN_SERIAL "ON" _BACKTRACE_TRIPLES "/home/mjs/src/GHEX/cmake/ghex_reg_test.cmake;44;add_test;/home/mjs/src/GHEX/test/structured/regular/CMakeLists.txt;7;ghex_reg_parallel_test;/home/mjs/src/GHEX/test/structured/regular/CMakeLists.txt;0;")
add_test([=[simple_regular_domain_mt]=] "/nix/store/1gx7vrvkclgjc828hmxdscgcvhs5zi1k-python3-3.13.13-env/bin/mpiexec" "-n" "4" "/home/mjs/src/GHEX/_nix_build/bin/simple_regular_domain_mt")
set_tests_properties([=[simple_regular_domain_mt]=] PROPERTIES  LABELS "parallel-ranks-4" RUN_SERIAL "ON" _BACKTRACE_TRIPLES "/home/mjs/src/GHEX/cmake/ghex_reg_test.cmake;44;add_test;/home/mjs/src/GHEX/test/structured/regular/CMakeLists.txt;8;ghex_reg_parallel_test;/home/mjs/src/GHEX/test/structured/regular/CMakeLists.txt;0;")
add_test([=[local_rma]=] "/nix/store/1gx7vrvkclgjc828hmxdscgcvhs5zi1k-python3-3.13.13-env/bin/mpiexec" "-n" "4" "/home/mjs/src/GHEX/_nix_build/bin/local_rma")
set_tests_properties([=[local_rma]=] PROPERTIES  LABELS "parallel-ranks-4" RUN_SERIAL "ON" _BACKTRACE_TRIPLES "/home/mjs/src/GHEX/cmake/ghex_reg_test.cmake;44;add_test;/home/mjs/src/GHEX/test/structured/regular/CMakeLists.txt;11;ghex_reg_parallel_test;/home/mjs/src/GHEX/test/structured/regular/CMakeLists.txt;0;")
add_test([=[local_rma_mt]=] "/nix/store/1gx7vrvkclgjc828hmxdscgcvhs5zi1k-python3-3.13.13-env/bin/mpiexec" "-n" "4" "/home/mjs/src/GHEX/_nix_build/bin/local_rma_mt")
set_tests_properties([=[local_rma_mt]=] PROPERTIES  LABELS "parallel-ranks-4" RUN_SERIAL "ON" _BACKTRACE_TRIPLES "/home/mjs/src/GHEX/cmake/ghex_reg_test.cmake;44;add_test;/home/mjs/src/GHEX/test/structured/regular/CMakeLists.txt;12;ghex_reg_parallel_test;/home/mjs/src/GHEX/test/structured/regular/CMakeLists.txt;0;")
