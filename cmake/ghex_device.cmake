
set(GHEX_USE_GPU "OFF" CACHE BOOL "use gpu")
set(GHEX_GPU_TYPE "AUTO" CACHE STRING "Choose the GPU type: AMD | NVIDIA | AUTO (environment-based) | EMULATE")
set_property(CACHE GHEX_GPU_TYPE PROPERTY STRINGS "AMD" "NVIDIA" "AUTO" "EMULATE")
set(GHEX_COMM_OBJ_USE_U "OFF" CACHE BOOL "uniform field optimization for gpu")

if (GHEX_USE_GPU)
    if (GHEX_GPU_TYPE STREQUAL "AUTO")
        find_package(hip)
        if (hip_FOUND)
            set(ghex_gpu_mode "hip")
        else() # assume cuda elsewhere; TO DO: might be refined
            set(ghex_gpu_mode "cuda")
        endif()
    elseif (GHEX_GPU_TYPE STREQUAL "AMD")
        set(ghex_gpu_mode "hip")
    elseif (GHEX_GPU_TYPE STREQUAL "NVIDIA")
        set(ghex_gpu_mode "cuda")
    else()
        set(ghex_gpu_mode "emulate")
    endif()

    if (ghex_gpu_mode STREQUAL "cuda")
        include(FindCUDAToolkit)
        # This fixes nvcc picking up a wrong host compiler for linking, causing
        # issues with outdated libraries, eg libstdc++ and std::filesystem. Must
        # happen before all calls to enable_language(CUDA)
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
        # enable cuda langauge in cmake
        enable_language(CUDA)
        # find cuda toolkit for version info
        find_package(CUDAToolkit)
        # set default cuda architecture
        if(${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 12)
            if(NOT CMAKE_CUDA_ARCHITECTURES)
                # Pascal, Volta, Turing, Ampere, Hopper
                set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 90)
            endif()
         elseif(${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 11)
            if(NOT CMAKE_CUDA_ARCHITECTURES)
                # Pascal, Volta, Turing, Ampere
                set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)
            endif()
         elseif(${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 10)
            if(NOT CMAKE_CUDA_ARCHITECTURES)
                # Pascal, Volta, Turing
                set(CMAKE_CUDA_ARCHITECTURES 60 70 75)
            endif()
        else()
            message(FATAL_ERROR "Need at least CUDA 10, got ${CUDAToolkit_VERSION_MAJOR}")
        endif()
        message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

        # TODO: still required?
        set(CMAKE_CUDA_FLAGS "" CACHE STRING "")
        string(APPEND CMAKE_CUDA_FLAGS " --cudart shared --expt-relaxed-constexpr")

        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_EXTENSIONS OFF)
        set(GHEX_GPU_MODE_EMULATE "OFF")
    elseif (ghex_gpu_mode STREQUAL "hip")
        enable_language(HIP)
        if(NOT CMAKE_HIP_ARCHITECTURES)
            set(CMAKE_HIP_ARCHITECTURES gfx900 gfx906)
        endif()
        message(STATUS "CMAKE_HIP_ARCHITECTURES = ${CMAKE_HIP_ARCHITECTURES}")
    else()
        set(GHEX_GPU_MODE_EMULATE "ON")
    endif()
else()
    set(ghex_gpu_mode "none")
    set(GHEX_GPU_MODE_EMULATE "OFF")
endif()

string(TOUPPER ${ghex_gpu_mode} ghex_gpu_mode_u)
set(GHEX_DEVICE "GHEX_DEVICE_${ghex_gpu_mode_u}")

