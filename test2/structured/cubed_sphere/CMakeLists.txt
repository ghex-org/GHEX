
include(ghex_gtest)
include(ghex_reg_test)


ghex_compile_test(cubed_sphere_transform)
ghex_reg_test(cubed_sphere_transform)

ghex_compile_test(cubed_sphere_exchange)
ghex_reg_parallel_test(cubed_sphere_exchange mpi 6 false)
ghex_reg_parallel_test(cubed_sphere_exchange ucx 6 false)

