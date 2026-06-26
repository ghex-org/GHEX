
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was HWMALLOCConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################
set(HWMALLOC_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
list(APPEND CMAKE_MODULE_PATH ${HWMALLOC_MODULE_PATH})
include(CMakeFindDependencyMacro)
if(UNIX AND NOT APPLE)
    set(NUMA_LIBRARY     /nix/store/wjfhh11sfcdf97mvg7hbxickybxzk850-numactl-2.0.18/lib/libnuma.so)
    set(NUMA_INCLUDE_DIR /nix/store/c6yn4j8y6ngixhp6igfrm2bl77wf9pia-numactl-2.0.18-dev/include)
    find_dependency(NUMA)
endif()
include(${CMAKE_CURRENT_LIST_DIR}/HWMALLOC-targets.cmake)
