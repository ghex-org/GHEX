
include(ghex_gtest)
include(ghex_reg_test)

ghex_compile_test(atlas_integration)
ghex_reg_parallel_test(atlas_integration 4 false)
