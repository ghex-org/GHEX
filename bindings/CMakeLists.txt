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

enable_language(Fortran)

add_library(fhex SHARED communicator_bind.cpp context_bind.cpp message_bind.cpp future_bind.cpp request_bind.cpp structured_staged_bind.cpp cubed_sphere_bind.cpp ghex_comm_mod.f90 ghex_message_mod.f90 ghex_future_mod.f90 ghex_request_mod.f90 ghex_structured_mod.f90 ghex_cubed_sphere_mod.f90 ghex_defs.f90 ghex_mod.f90)

if (GHEX_USE_PMIX)
   target_compile_definitions(fhex PRIVATE GHEX_USE_PMIX)
endif()

if (GHEX_USE_UCP)
   target_compile_definitions(fhex PRIVATE GHEX_USE_UCP)
endif()

if (GHEX_USE_XPMEM)
  target_compile_definitions(fhex PRIVATE GHEX_USE_XPMEM)
endif()

target_compile_definitions(fhex PRIVATE GHEX_USE_OPENMP)
target_include_directories(fhex PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(fhex PRIVATE Boost::boost GridTools::gridtools)
