
set(c_cxx_lang "$<COMPILE_LANGUAGE:C,CXX>")
set(c_cxx_lang_clang "$<COMPILE_LANG_AND_ID:CXX,Clang>")
set(cuda_lang "$<COMPILE_LANGUAGE:CUDA>")
set(fortran_lang "$<COMPILE_LANGUAGE:Fortran>")
set(fortran_lang_gnu "$<COMPILE_LANG_AND_ID:Fortran,GNU>")

function(ghex_target_compile_options target)
    target_compile_options(${target} PRIVATE
    # flags for CXX builds
    $<${c_cxx_lang}:$<BUILD_INTERFACE:-Wall -Wextra -Wpedantic -Wno-unknown-pragmas -Wno-unused-local-typedef>>
    $<${c_cxx_lang_clang}:$<BUILD_INTERFACE:-Wno-c++17-extensions -Wno-unused-lambda-capture>>
    # flags for CUDA builds
    $<${cuda_lang}:$<BUILD_INTERFACE:-Xcompiler=-Wall -Wextra -Wno-unknown-pragmas --default-stream per-thread>>
    # flags for Fortran builds
    $<${fortran_lang}:$<BUILD_INTERFACE:-cpp -fcoarray=single>>
    $<${fortran_lang_gnu}:$<BUILD_INTERFACE:-ffree-line-length-none>>)
endfunction()

function(compile_as_cuda)
    if (ghex_gpu_mode STREQUAL "cuda")
        set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)
    endif()
endfunction()

function(link_device_runtime target)
    if (ghex_gpu_mode STREQUAL "hip")
        target_link_libraries(${target} PRIVATE hip::device)
    endif()
endfunction()
