# find_package(cpp_bindgen)
# bindgen_add_library(ghex2 SOURCES bindgen.cpp)
# target_include_directories(ghex2 PRIVATE ${CMAKE_SOURCE_DIR}/include)

# cannot use that, because GridTools already do export this target and we get conflicts
# include(FetchContent)
# FetchContent_Declare(
#   cpp_bindgen
#   GIT_REPOSITORY https://github.com/GridTools/cpp_bindgen.git
#   GIT_TAG        master # consider replacing master by a tagged version
# )
# FetchContent_MakeAvailable(cpp_bindgen)

find_package(MPI)

enable_language(Fortran)
add_library(ghex SHARED message_bind.cpp communicator_bind.cpp future_bind.cpp ghex_message_mod.f90 ghex_comm_mod.f90 ghex_future_mod.f90)
target_include_directories(ghex PRIVATE ${CMAKE_SOURCE_DIR}/include ${MPI_INCLUDE_PATH})

install (FILES "${PROJECT_BINARY_DIR}/bindings/ghex_message_mod.mod" DESTINATION include)
install (FILES "${PROJECT_BINARY_DIR}/bindings/ghex_comm_mod.mod" DESTINATION include)
install (FILES "${PROJECT_BINARY_DIR}/bindings/ghex_future_mod.mod" DESTINATION include)
