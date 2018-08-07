#include "blas_lapack.h"
// #include "hpx_kokkos.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Random.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
#include "cublas.h"
#include "cusolverRf.h"
#endif

// #include <hpx/hpx_init.hpp>
// #include <hpx/include/async.hpp>
// #include <hpx/include/iostreams.hpp>
#include <hpx/include/util.hpp>
// #include <hpx/parallel/executors/service_executors.hpp>

#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

// NOTE: All blocks are stored in column-major order.

template <typename E, typename ExecutionSpace>
using Matrix = Kokkos::View<E ***, ExecutionSpace>;

template <typename E, typename ExecutionSpace> class RandomSymmPosDefFiller {
public:
  using member_type = Kokkos::TeamPolicy<>::member_type;

  RandomSymmPosDefFiller(Kokkos::View<E ***, ExecutionSpace> matrix,
                         const std::size_t num_blocks,
                         const std::size_t block_size)
      : matrix_(matrix), num_blocks_(num_blocks), block_size_(block_size){};

  KOKKOS_INLINE_FUNCTION void operator()(const int &row, const int &col) const {
    std::size_t idx0 = row / block_size_;
    std::size_t idx1 = col / block_size_;
    std::size_t idx2 = (col % block_size_) * block_size_ + (row % block_size_);

    std::size_t idx0_tr = col / block_size_;
    std::size_t idx1_tr = row / block_size_;
    std::size_t idx2_tr =
        (row % block_size_) * block_size_ + (col % block_size_);

    if (row == col) {
      matrix_(idx0, idx1, idx2) += E(block_size_ * num_blocks_);
    } else if (row < col) {
      matrix_(idx0_tr, idx1_tr, idx2_tr) = matrix_(idx0, idx1, idx2);
    }
  }

private:
  const Kokkos::View<E ***, ExecutionSpace> matrix_;
  const std::size_t num_blocks_;
  const std::size_t block_size_;
};

template <typename E, typename ExecutionSpace>
KOKKOS_INLINE_FUNCTION std::size_t
get_block_size(Matrix<E, ExecutionSpace> matrix) {
  return std::sqrt(matrix.dimension_2());
}

template <typename E, typename L, typename ExecutionSpace>
KOKKOS_INLINE_FUNCTION std::size_t
get_block_size(Kokkos::View<E ***, L, ExecutionSpace> matrix) {
  return std::sqrt(matrix.dimension_2());
}

template <typename E, typename ExecutionSpace>
KOKKOS_INLINE_FUNCTION std::size_t
get_num_blocks(Matrix<E, ExecutionSpace> matrix) {
  return matrix.dimension_0();
}

template <typename E, typename L, typename ExecutionSpace>
KOKKOS_INLINE_FUNCTION std::size_t
get_num_blocks(Kokkos::View<E ***, L, ExecutionSpace> matrix) {
  return matrix.dimension_0();
}

template <typename E, typename ExecutionSpace>
KOKKOS_INLINE_FUNCTION void set_elem(Matrix<E, ExecutionSpace> matrix,
                                     std::size_t row, std::size_t col, E val) {
  const std::size_t block_size = get_block_size(matrix);
  const std::size_t idx0 = row / block_size;
  const std::size_t idx1 = col / block_size;
  const std::size_t idx2 = (col % block_size) * block_size + (row % block_size);

  matrix(idx0, idx1, idx2) = val;
}

template <typename E, typename L, typename ExecutionSpace>
KOKKOS_INLINE_FUNCTION void
set_elem(Kokkos::View<E ***, L, ExecutionSpace> matrix, std::size_t row,
         std::size_t col, E val) {
  const std::size_t block_size = get_block_size(matrix);
  const std::size_t idx0 = row / block_size;
  const std::size_t idx1 = col / block_size;
  const std::size_t idx2 = (col % block_size) * block_size + (row % block_size);

  matrix(idx0, idx1, idx2) = val;
}

template <typename E, typename ExecutionSpace>
KOKKOS_INLINE_FUNCTION E get_elem(Matrix<E, ExecutionSpace> matrix,
                                  std::size_t row, std::size_t col) {
  const std::size_t block_size = get_block_size(matrix);
  const std::size_t idx0 = row / block_size;
  const std::size_t idx1 = col / block_size;
  const std::size_t idx2 = (col % block_size) * block_size + (row % block_size);

  return matrix(idx0, idx1, idx2);
}

// TODO: Get the overloads right.
template <typename E, typename L, typename ExecutionSpace>
KOKKOS_INLINE_FUNCTION E get_elem(Kokkos::View<E ***, L, ExecutionSpace> matrix,
                                  std::size_t row, std::size_t col) {
  const std::size_t block_size = get_block_size(matrix);
  const std::size_t idx0 = row / block_size;
  const std::size_t idx1 = col / block_size;
  const std::size_t idx2 = (col % block_size) * block_size + (row % block_size);

  return matrix(idx0, idx1, idx2);
}

template <typename E, typename ExecutionSpace>
KOKKOS_INLINE_FUNCTION void print(Matrix<E, ExecutionSpace> matrix) {
  const std::size_t num_blocks = get_num_blocks(matrix);
  const std::size_t block_size = get_block_size(matrix);
  const std::size_t n = num_blocks * block_size;

  auto matrix_h = Kokkos::create_mirror_view(matrix);
  Kokkos::deep_copy(matrix_h, matrix);

  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      printf("%5.2lf ", get_elem(matrix_h, row, col));
    }
    printf("\n");
  }
}

template <typename E, typename ExecutionSpace>
void make_random_symmposdef(Matrix<E, ExecutionSpace> matrix) {
  const std::size_t num_blocks = get_num_blocks(matrix);
  const std::size_t block_size = get_block_size(matrix);
  const std::size_t n = num_blocks * block_size;

  Kokkos::Random_XorShift64_Pool<ExecutionSpace> g(1931);
  Kokkos::fill_random(matrix, g, 1.0);

  Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, n}),
                       RandomSymmPosDefFiller<E, ExecutionSpace>(
                           matrix, num_blocks, block_size));
}

template <typename ExecutionSpace>
void kpotrf(Kokkos::View<double *, ExecutionSpace> matrix,
            const int block_size);

#if defined(KOKKOS_ENABLE_CUDA)
template <>
void kpotrf<Kokkos::Cuda>(Kokkos::View<double *, Kokkos::Cuda> A,
                          const int block_size) {
  // cusolverDnDpotrf("L", /*m*/ block_size, x.data(), /*ld*/ block_size,
  // &info);
}
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
template <>
void kpotrf<Kokkos::OpenMP>(Kokkos::View<double *, Kokkos::OpenMP> A,
                            const int block_size) {
  int info = 0;
  potrf("L", block_size, A.data(), block_size, &info);
}
#endif

template <typename ExecutionSpace>
void kherk(const double a, Kokkos::View<double *, ExecutionSpace> A,
           const double b, Kokkos::View<double *, ExecutionSpace> B,
           const int block_size);

#if defined(KOKKOS_ENABLE_OPENMP)
template <>
void kherk<Kokkos::OpenMP>(const double a,
                           Kokkos::View<double *, Kokkos::OpenMP> A,
                           const double b,
                           Kokkos::View<double *, Kokkos::OpenMP> B,
                           const int block_size) {

  int info = 0;
  herk("L", "N", block_size, block_size, a, A.data(), block_size, b, B.data(),
       block_size);
}
#endif

template <typename ExecutionSpace>
void kgemm(const double a, Kokkos::View<double *, ExecutionSpace> A,
           Kokkos::View<double *, ExecutionSpace> B, const double b,
           Kokkos::View<double *, ExecutionSpace> C, const int block_size);

#if defined(KOKKOS_ENABLE_OPENMP)
template <>
void kgemm<Kokkos::OpenMP>(const double a,
                           Kokkos::View<double *, Kokkos::OpenMP> A,
                           Kokkos::View<double *, Kokkos::OpenMP> B,
                           const double b,
                           Kokkos::View<double *, Kokkos::OpenMP> C,
                           const int block_size) {

  int info = 0;
  gemm("N", "C", block_size, block_size, block_size, -1.0, A.data(), block_size,
       B.data(), block_size, 1.0, C.data(), block_size);
}
#endif

template <typename ExecutionSpace>
void ktrsm(const double a, Kokkos::View<double *, ExecutionSpace> A,
           Kokkos::View<double *, ExecutionSpace> B, const int block_size);

#if defined(KOKKOS_ENABLE_OPENMP)
template <>
void ktrsm<Kokkos::OpenMP>(const double a,
                           Kokkos::View<double *, Kokkos::OpenMP> A,
                           Kokkos::View<double *, Kokkos::OpenMP> B,
                           const int block_size) {

  int info = 0;
  trsm("R", "L", "C", "N", block_size, block_size, 1.0, A.data(), block_size,
       B.data(), block_size);
}
#endif

// This example constructs a DAG for a Cholesky factorization.

// Version 1: Spawn one task for each block from the root task. The task for
// each block will then choose its dependencies and respawn itself. This
// requires one respawn per task that would not be necessary if task_spawn could
// take a list of dependent futures. Update: task_spawn can take a list of
// futures, see next version.
template <typename ExecutionSpace> struct Cholesky {
  using sched_type = Kokkos::TaskScheduler<ExecutionSpace>;
  using value_type = void;
  using future_type = Kokkos::Future<value_type, ExecutionSpace>;
  using size_type = int;

  sched_type sched;
  Matrix<double, ExecutionSpace> matrix;
  const size_type block_size;
  const size_type num_blocks;
  Kokkos::View<future_type **, ExecutionSpace> futures;
  const size_type row;
  const size_type col;
  size_type diag;

  bool debug;

  KOKKOS_INLINE_FUNCTION
  Cholesky(const sched_type &arg_sched,
           Matrix<double, ExecutionSpace> arg_matrix, bool arg_debug = true)
      : sched(arg_sched), matrix(arg_matrix),
        block_size(get_block_size(arg_matrix)),
        num_blocks(get_num_blocks(arg_matrix)),
        futures("futures", num_blocks, num_blocks), row(-1), col(-1), diag(-1),
        debug(arg_debug) {}

  KOKKOS_INLINE_FUNCTION
  Cholesky(const sched_type &arg_sched,
           Matrix<double, ExecutionSpace> arg_matrix,
           Kokkos::View<future_type **, ExecutionSpace> arg_futures,
           const size_type arg_row, const size_type arg_col,
           const size_type arg_diag, bool arg_debug = true)
      : sched(arg_sched), matrix(arg_matrix),
        block_size(get_block_size(arg_matrix)),
        num_blocks(get_num_blocks(arg_matrix)), futures(arg_futures),
        row(arg_row), col(arg_col), diag(arg_diag), debug(arg_debug) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(typename sched_type::member_type &) {
    if (debug) {
      printf("in cholesky task for block (%d, %d)\n", row, col);
    }

    // Root task
    if (col < 0 || row < 0) {
      if (diag == 0) {
        if (debug) {
          printf("done with last task\n");
        }
        return;
      } else {
        if (debug) {
          printf("spawning all worker tasks\n");
        }
        for (size_type r = 0; r < num_blocks; ++r) {
          for (size_type c = 0; c <= r; ++c) {
            assert(futures(r, c).is_null());
            if (debug) {
              printf("spawning task for block (%d, %d)\n", r, c);
              printf("futures dimensions: %d\n", futures.size());
            }
            if (c == 0) {
              futures(r, c) = Kokkos::task_spawn(
                  Kokkos::TaskSingle(sched, Kokkos::TaskPriority::High),
                  Cholesky(sched, matrix, futures, r, c, -1, debug));
            } else {
              futures(r, c) = Kokkos::task_spawn(
                  Kokkos::TaskSingle(sched),
                  Cholesky(sched, matrix, futures, r, c, -1, debug));
            }

            check_future(futures(r, c));
          }
        }

        // Mark this as the final task
        if (debug) {
          printf("respawning task for block (%d, %d)\n", row, col);
        }
        diag = 0;
        Kokkos::respawn(this, futures(num_blocks - 1, num_blocks - 1),
                        Kokkos::TaskPriority::Low);
      }
    } else if (row == 0 && col == 0) { // Top left block
      if (debug) {
        printf("(%d, %d): potrf\n", row, col);
      }

      auto x = Kokkos::subview(matrix, 0, 0, Kokkos::ALL());
      kpotrf<ExecutionSpace>(x, block_size);

      return;                // Top left task is done
    } else if (row == col) { // Rest of diagonal
      if (diag >= 0) {
        if (debug) {
          printf("(%d, %d): herk\n", row, col);
        }

        auto diag_block = Kokkos::subview(matrix, row, col, Kokkos::ALL());
        auto panel_block = Kokkos::subview(matrix, row, diag, Kokkos::ALL());
        kherk<ExecutionSpace>(-1.0, panel_block, 1.0, diag_block, block_size);

        if (diag == col - 1) {
          if (debug) {
            printf("(%d, %d): potrf\n", row, col);
          }

          kpotrf<ExecutionSpace>(diag_block, block_size);
          return;
        }
      }

      diag = diag + 1;

      if (debug) {
        printf("setting up dependencies for block (%d, %d)\n", row, col);
      }
      Kokkos::respawn(this, futures(row, diag), Kokkos::TaskPriority::High);
    } else {             // Non-diagonal blocks
      if (diag == col) { // Do the final trsm
        if (debug) {
          printf("(%d, %d): trsm\n", row, col);
        }

        auto diag_block = Kokkos::subview(matrix, diag, diag, Kokkos::ALL());
        auto panel_block = Kokkos::subview(matrix, row, col, Kokkos::ALL());
        ktrsm<ExecutionSpace>(1.0, diag_block, panel_block, block_size);
      } else {
        if (diag >= 0) {
          if (debug) {
            printf("(%d, %d): gemm\n", row, col);
          }

          auto panel_block_row =
              Kokkos::subview(matrix, row, diag, Kokkos::ALL());
          auto panel_block_col =
              Kokkos::subview(matrix, col, diag, Kokkos::ALL());
          auto current_block = Kokkos::subview(matrix, row, col, Kokkos::ALL());
          kgemm<ExecutionSpace>(-1.0, panel_block_row, panel_block_col, 1.0,
                                current_block, block_size);
        }

        if (diag == col - 1) {
          Kokkos::respawn(this, futures(col, col), Kokkos::TaskPriority::High);
        } else {
          Kokkos::Future<ExecutionSpace> dependencies[] = {
              futures(row, diag + 1), futures(col, diag + 1)};
          Kokkos::Future<ExecutionSpace> fib_all =
              Kokkos::when_all(dependencies, 2);
          Kokkos::respawn(this, fib_all, Kokkos::TaskPriority::Regular);
        }

        diag = diag + 1;
      }
    }
  }

  void check_future(future_type f) {
    if (f.is_null()) {
      printf("Cholesky: insufficient memory alloc_capacity(%d) task_max(%d) "
             "task_accum(%ld)\n",
             sched.allocation_capacity(), sched.allocated_task_count_max(),
             sched.allocated_task_count_accum());

      Kokkos::abort("Cholesky insufficient memory\n");
    }
  }

  static void run(Matrix<double, ExecutionSpace> matrix, bool debug = true) {
    typedef typename sched_type::memory_space memory_space;

    if (debug) {
      std::cout << "sizeof(Cholesky) = " << sizeof(Cholesky) << std::endl;
    }

    const std::size_t num_blocks = get_num_blocks(matrix);
    const std::size_t MinBlockSize = 256; // sizeof(Cholesky);
    const std::size_t MaxBlockSize = 256; // MinBlockSize;
    const std::size_t SuperBlockSize =
        1 + MinBlockSize * num_blocks * (num_blocks + 1) / 2 * 2;
    const std::size_t MemoryCapacity = SuperBlockSize;

    if (debug) {
      printf("creating root_sched\n");
    }

    sched_type root_sched(memory_space(), MemoryCapacity, MinBlockSize,
                          MaxBlockSize, SuperBlockSize);

    if (debug) {
      printf("spawning Cholesky task\n");
    }
    future_type f = Kokkos::host_spawn(Kokkos::TaskSingle(root_sched),
                                       Cholesky(root_sched, matrix, debug));

    if (debug) {
      printf("waiting for root_sched\n");
    }
    Kokkos::wait(root_sched);
    if (debug) {
      printf("done waiting for root_sched\n");
    }
  }

  static bool check_result(Matrix<double, ExecutionSpace> original,
                           Matrix<double, ExecutionSpace> factored,
                           bool print_diff = true, bool print_result = false) {
    // TODO: Speed up.
    const std::size_t num_blocks = get_num_blocks(original);
    const std::size_t block_size = get_block_size(original);

    auto original_h = Kokkos::create_mirror_view(original);
    Kokkos::deep_copy(original_h, original);
    auto factored_h = Kokkos::create_mirror_view(factored);
    Kokkos::deep_copy(factored_h, factored);

    Matrix<double, Kokkos::HostSpace> output("output", num_blocks, num_blocks,
                                             block_size * block_size);

    for (std::size_t col = 0; col < block_size * num_blocks; ++col) {
      for (std::size_t row = col; row < block_size * num_blocks; ++row) {
        double tmp = 0.0;
        for (std::size_t i = 0; i <= (std::min)(row, col); ++i) {
          tmp += get_elem(factored_h, row, i) * get_elem(factored_h, col, i);
        }
        set_elem(output, row, col, tmp);
        if (row != col) {
          set_elem(output, col, row, tmp);
        }
      }
    }

    bool correct = true;
    for (std::size_t col = 0; col < block_size * num_blocks; ++col) {
      for (std::size_t row = col; row < block_size * num_blocks; ++row) {
        double diff =
            get_elem(original_h, row, col) - get_elem(output, row, col);
        if (std::abs(diff) > 0.0001) {
          correct = false;
          if (print_diff) {
            std::cout << "A(" << row << ", " << col
                      << ") = " << get_elem(original_h, row, col) << ", A'("
                      << row << ", " << col
                      << ") = " << get_elem(output, row, col) << std::endl;
          }
        }
      }
    }

    if (print_result) {
      print(output);
    }

    return correct;
  }
};

int main(int argc, char *argv[]) {
  bool debug = false;

  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout, debug);

  if (argc < 3) {
    std::cout << "Usage: test_12 <block_size> <num_blocks>" << std::endl;
    exit(1);
  }

  {
    std::cout << "Factorizing a random positive definite matrix" << std::endl;

    std::size_t block_size = std::stoi(argv[1]);
    std::size_t num_blocks = std::stoi(argv[2]);

    std::cout << "block_size = " << block_size << std::endl;
    std::cout << "num_blocks = " << num_blocks << std::endl;
    std::cout << "n          = " << num_blocks * block_size << std::endl;

    Matrix<double, Kokkos::DefaultExecutionSpace> A("A", num_blocks, num_blocks,
                                                    block_size * block_size);
    Matrix<double, Kokkos::DefaultExecutionSpace> B("B", num_blocks, num_blocks,
                                                    block_size * block_size);
    // std::cout << "generating matrix" << std::endl;
    make_random_symmposdef(A);
    Kokkos::deep_copy(B, A);

    if (debug) {
      std::cout << "input:" << std::endl;
      print(A);
      std::cout << std::endl;
    }

    std::cout << "running Cholesky decomposition" << std::endl;
    hpx::util::high_resolution_timer t;
    Cholesky<Kokkos::DefaultExecutionSpace>::run(A, debug);
    double time = t.elapsed();

    std::size_t n = num_blocks * block_size;
    double performance = (2 * std::pow(double(n), 3.0) / 6) / 1e9 / time;
    std::cout << "t = " << time << " s, performance (GFlop/s) = " << performance
              << std::endl;

    if (debug) {
      std::cout << std::endl;

      std::cout << "output:" << std::endl;
      print(A);
      std::cout << std::endl;

      std::cout << "reconstruction of original matrix:" << std::endl;
    }

    // std::cout << "checking result" << std::endl;
    // bool correct = Cholesky<Kokkos::DefaultExecutionSpace>::check_result(B, A);
    // std::cout << (correct ? "correct result" : "incorrect result") << std::endl;
  }

  std::cout << "----------" << std::endl;

  {
    std::cout << "Factorizing a 3x3 matrix with 1x1 block size" << std::endl;
    Matrix<double, Kokkos::DefaultExecutionSpace> A("A", 3, 3, 1);
    Matrix<double, Kokkos::DefaultExecutionSpace> B("B", 3, 3, 1);

    auto A_h = Kokkos::create_mirror_view(A);
    Kokkos::deep_copy(A_h, A);

    set_elem(A_h, 0, 0, 4.0);
    set_elem(A_h, 1, 0, 12.0);
    set_elem(A_h, 2, 0, -16.0);

    set_elem(A_h, 0, 1, 12.0);
    set_elem(A_h, 1, 1, 37.0);
    set_elem(A_h, 2, 1, -43.0);

    set_elem(A_h, 0, 2, -16.0);
    set_elem(A_h, 1, 2, -43.0);
    set_elem(A_h, 2, 2, 98.0);

    Kokkos::deep_copy(A, A_h);

    Kokkos::deep_copy(B, A);
    if (debug) {
      std::cout << "input:" << std::endl;
      print(A);
      std::cout << std::endl;
    }

    Cholesky<Kokkos::DefaultExecutionSpace>::run(A, false);
    if (debug) {
      std::cout << std::endl;

      std::cout << "output:" << std::endl;
      print(A);
      std::cout << std::endl;

      std::cout << "reconstruction of original matrix:" << std::endl;
    }
    bool correct = Cholesky<Kokkos::DefaultExecutionSpace>::check_result(B, A);
    std::cout << (correct ? "correct result" : "incorrect result") << std::endl;
  }

  std::cout << "----------" << std::endl;

  {
    std::cout << "Factorizing a 1x1 matrix with 3x3 block size" << std::endl;
    Matrix<double, Kokkos::DefaultExecutionSpace> A("A", 1, 1, 3 * 3);
    Matrix<double, Kokkos::DefaultExecutionSpace> B("B", 1, 1, 3 * 3);

    auto A_h = Kokkos::create_mirror_view(A);
    Kokkos::deep_copy(A_h, A);

    set_elem(A_h, 0, 0, 4.0);
    set_elem(A_h, 1, 0, 12.0);
    set_elem(A_h, 2, 0, -16.0);

    set_elem(A_h, 0, 1, 12.0);
    set_elem(A_h, 1, 1, 37.0);
    set_elem(A_h, 2, 1, -43.0);

    set_elem(A_h, 0, 2, -16.0);
    set_elem(A_h, 1, 2, -43.0);
    set_elem(A_h, 2, 2, 98.0);

    Kokkos::deep_copy(A, A_h);

    Kokkos::deep_copy(B, A);
    if (debug) {
      std::cout << "input:" << std::endl;
      print(A);
      std::cout << std::endl;
    }

    Cholesky<Kokkos::DefaultExecutionSpace>::run(A, false);
    if (debug) {
      std::cout << std::endl;

      std::cout << "output:" << std::endl;
      print(A);
      std::cout << std::endl;

      std::cout << "reconstruction of original matrix:" << std::endl;
    }
    bool correct = Cholesky<Kokkos::DefaultExecutionSpace>::check_result(B, A);
    std::cout << (correct ? "correct result" : "incorrect result") << std::endl;
  }
  Kokkos::finalize();

  return 0;
}
