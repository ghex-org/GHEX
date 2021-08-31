
set(c_cxx_lang "$<COMPILE_LANGUAGE:C,CXX>")
set(cuda_lang "$<COMPILE_LANGUAGE:CUDA>")

function(ghex_target_compile_options target)
    target_compile_options(${target} PRIVATE
    # flags for CXX builds
    $<${c_cxx_lang}:$<BUILD_INTERFACE:-Wall -Wextra -Wpedantic -Wno-unknown-pragmas>>
    # flags for CUDA builds
    $<${cuda_lang}:$<BUILD_INTERFACE:-Xcompiler=-Wall -Wextra -Wno-unknown-pragmas --default-stream per-thread>>)
endfunction()

function(compile_as_cuda)
    if (ghex_gpu_mode STREQUAL "cuda")
        set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)
    endif()
endfunction()

