# Need gcc 6.2.0 for CUDA and C++11 for Kokkos
# Pascal60 = P100
# HSW = daint-gpu, BDW = daint-mc
# Need nvcc_wrapper for Kokkos to be happy

# pthreads backend only
cmake \
    -DHPX_DIR=$APPS/UES/simbergm/local/hpx-master-gcc-6.2.0-c++11-boost-1.67.0-gperftools-2.7-hwloc-2.0.1-Release/lib/cmake/HPX/ \
    -DKOKKOS_SOURCE_DIR=/apps/daint/UES/simbergm/src/kokkos \
    -DKOKKOS_ENABLE_PTHREAD=ON \
    -DKOKKOS_ARCH=HSW \
    ../..

# CUDA backend on device, serial on host
# ENABLE_CUDA_RELOCATABLE_DEVICE_CODE required for task graphs
cmake \
    -DHPX_DIR=$APPS/UES/simbergm/local/hpx-master-gcc-6.2.0-c++11-boost-1.67.0-gperftools-2.7-hwloc-2.0.1-Release/lib/cmake/HPX/ \
    -DKOKKOS_SOURCE_DIR=/apps/daint/UES/simbergm/src/kokkos \
    -DKOKKOS_ENABLE_CUDA=ON \
    -DKOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON \
    -DKOKKOS_ARCH=HSW,Pascal60 \
    -DCMAKE_CXX_COMPILER=/apps/daint/UES/simbergm/src/kokkos/bin/nvcc_wrapper \
    ../..

module load daint-gpu
module switch PrgEnv-cray PrgEnv-gnu
module switch gcc gcc/6.2.0
module load cudatoolkit/9.0.103_3.7-6.0.4.1_2.1__g72b395b
source /apps/daint/UES/sandbox/rasolca/intel/mkl/bin/mklvars.sh intel64 
module load CMake/3.11.4
module load jemalloc
