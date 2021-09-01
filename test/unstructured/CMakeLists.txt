
include(ghex_gtest)
include(ghex_reg_test)

ghex_compile_test(user_concepts)
ghex_reg_parallel_test(user_concepts 4 false)
ghex_reg_parallel_test(user_concepts 2 true)

