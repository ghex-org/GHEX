
ghex_compile_test(cubed_sphere_transform)
ghex_reg_test(cubed_sphere_transform)

ghex_compile_test(cubed_sphere_exchange)
ghex_reg_parallel_test(cubed_sphere_exchange 6 false)

