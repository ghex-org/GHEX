
ghex_compile_test(regular_domain)
ghex_reg_parallel_test(regular_domain 4 false)
ghex_reg_parallel_test(regular_domain 4 true)

ghex_compile_test(simple_regular_domain)
ghex_reg_parallel_test(simple_regular_domain 4 false)
ghex_reg_parallel_test(simple_regular_domain 4 true)

ghex_compile_test(local_rma)
ghex_reg_parallel_test(local_rma 4 false)
ghex_reg_parallel_test(local_rma 4 true)
