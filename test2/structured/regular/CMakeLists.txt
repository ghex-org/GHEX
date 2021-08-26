
include(ghex_gtest)
include(ghex_reg_test)


ghex_compile_test(regular_domain)
ghex_reg_parallel_test(regular_domain mpi 4 false)
ghex_reg_parallel_test(regular_domain mpi 4 true)
ghex_reg_parallel_test(regular_domain ucx 4 false)
ghex_reg_parallel_test(regular_domain ucx 4 true)
