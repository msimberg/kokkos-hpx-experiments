# Pascal60 = P100
# HSW = daint-gpu, BDW = daint-mc
# Need nvcc_wrapper for Kokkos to be happy

# HPX backend
cmake \
    -DHPX_DIR=$HPX_DIR \
    -DKOKKOS_SOURCE_DIR=$KOKKOS_SOURCE_DIR \
    -DKOKKOS_ENABLE_HPX=ON \
    -DKOKKOS_ARCH=BDW \
    ../..

# OpenMP backend
cmake \
    -DHPX_DIR=$HPX_DIR \
    -DKOKKOS_SOURCE_DIR=$KOKKOS_SOURCE_DIR \
    -DKOKKOS_ENABLE_OPENMP=ON \
    -DKOKKOS_ARCH=BDW \
    ../..

# CUDA backend on device, serial on host
# ENABLE_CUDA_RELOCATABLE_DEVICE_CODE required for task graphs
cmake \
    -DHPX_DIR=$HPX_DIR \
    -DKOKKOS_SOURCE_DIR=/apps/daint/UES/simbergm/src/kokkos \
    -DKOKKOS_ENABLE_CUDA=ON \
    -DKOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON \
    -DKOKKOS_ARCH=HSW,Pascal60 \
    -DCMAKE_CXX_COMPILER=$KOKKOS_SOURCE_DIR/bin/nvcc_wrapper \
    ../..

# CUDA backend on device, HPX on host
# ENABLE_CUDA_RELOCATABLE_DEVICE_CODE required for task graphs
cmake \
    -DHPX_DIR=$HPX_DIR \
    -DKOKKOS_SOURCE_DIR=/apps/daint/UES/simbergm/src/kokkos \
    -DKOKKOS_ENABLE_CUDA=ON \
    -DKOKKOS_ENABLE_HPX=ON \
    -DKOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON \
    -DKOKKOS_ARCH=HSW,Pascal60 \
    -DCMAKE_CXX_COMPILER=$KOKKOS_SOURCE_DIR/bin/nvcc_wrapper \
    ../..

# Various modules on Piz Daint, useful for copy-pasting
module load daint-gpu
module switch PrgEnv-cray PrgEnv-gnu
module load CMake/3.12.0
module load jemalloc
source /apps/daint/UES/sandbox/rasolca/intel/mkl/bin/mklvars.sh intel64

module load cudatoolkit/9.0.103_3.15-6.0.7.0_14.1__ge802626
module switch gcc gcc/6.2.0

module load cudatoolkit/9.0.103_3.15-6.0.7.0_14.1__ge802626
module switch gcc gcc/5.3.0

module load cudatoolkit/8.0.61_2.4.3-6.0.4.0_3.1__gb475d12
module switch gcc gcc/4.9.3
