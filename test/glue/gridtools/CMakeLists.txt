
include(ghex_gtest)
include(ghex_reg_test)

ghex_compile_test(gt_datastore)
ghex_reg_parallel_test(gt_datastore 4 false)

