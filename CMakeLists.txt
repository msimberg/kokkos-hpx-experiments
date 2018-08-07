# Playground for testing HPX/Kokkos/CUDA/ROCm interoperability

cmake_minimum_required(VERSION 3.3.2)
project(hpx-kokkos-gpu CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(HPX REQUIRED)

if(KOKKOS_SOURCE_DIR)
    add_subdirectory(${KOKKOS_SOURCE_DIR} ${hpx-kokkos-gpu_BINARY_DIR}/kokkos)
    include_directories(${Kokkos_INCLUDE_DIRS_RET})
else()
    message(FATAL_ERROR "KOKKOS_SOURCE_DIR is not set")
endif()

if (NOT DEFINED LAPACK_LIBS)
  message("Trying to find MKL")
  # Find intel MKL if possible, checks ${MKL_ROOT} and $ENV{MKL_ROOT}
  # enable dump of info during find
  ### Need to be cleaned up
  set(MKL_VERBOSE=1)
  find_package(MKL)

  if (MKL_FOUND)
    set(LAPACK_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
    set(LAPACK_LIBRARIES
      mkl_intel_lp64 mkl_core mkl_sequential
      pthread m dl
      CACHE PATH "LAPACK_LIBS"
    )
    link_directories(${MKL_LIB_DIR})
    message("${PROJECT_NAME} MKL libraries are ${MKL_LIBRARIES_MT}")
  elseif (NOT DEFINED LAPACK_LIBS)
    message("${PROJECT_NAME} MKL NOT_FOUND and no LAPACK_LIBS set")
  endif()
else()
  set_property(CACHE "LAPACK_LIBS" PROPERTY TYPE PATH)
  set(LAPACK_LIBRARIES ${LAPACK_LIBS})
endif()

set(test_executables
  test_1
  test_2
  test_3
  test_4
  test_5
  test_6
  test_7
  test_8
  test_9
  test_10
  test_11
  test_12
  thread_mappings
  )

foreach(test ${test_executables})
  add_hpx_executable(${test}
    SOURCES ${test}.cpp
    COMPONENT_DEPENDENCIES iostreams)

  target_link_libraries(${test}_exe kokkos)
endforeach()

target_include_directories(test_12_exe PRIVATE ${LAPACK_INCLUDE_DIRS})
target_link_libraries(test_12_exe ${LAPACK_LIBRARIES})

# if(KOKKOS_ENABLE_CUDA)
#   target_link_libraries(test_12_exe cublas curand)
# endif()
