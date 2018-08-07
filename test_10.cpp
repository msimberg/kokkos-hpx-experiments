#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/parallel/executors/service_executors.hpp>

#include <cassert>
#include <iostream>

#include "hpx_kokkos.hpp"

// This example constructs a dummy (made-up dependencies) DAG for a Cholesky
// factorization. It doesn't actually do anything, and it especially does not
// have the correct dependencies.

// Version 1: Spawn one task for each block from the root task. The task for
// each block will then choose its dependencies and respawn itself. This
// requires one respawn per task that would not be necessary if task_spawn could
// a list of dependent futures.
template <typename Space> struct Cholesky {
  using sched_type = Kokkos::TaskScheduler<Space>;
  using value_type = void;
  using future_type = Kokkos::Future<value_type, Space>;
  using size_type = int;

  sched_type sched;
  Kokkos::View<future_type **, Space> futures;
  const size_type n;
  const size_type row;
  const size_type col;
  bool do_work;

  KOKKOS_INLINE_FUNCTION
  Cholesky(const sched_type &arg_sched,
           Kokkos::View<future_type **, Space> arg_futures,
           const size_type arg_n, const size_type arg_row,
           const size_type arg_col)
      : sched(arg_sched), futures(arg_futures), n(arg_n), row(arg_row),
        col(arg_col), do_work(false) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(typename sched_type::member_type &) {
    printf("in cholesky task for block (%d, %d)\n", row, col, n);

    // Root task
    if (col < 0 || row < 0) {
      if (do_work) {
        printf("done with last task\n");
        return;
      } else {
        printf("spawning all worker tasks\n");
        for (size_type r = 0; r < n; ++r) {
          for (size_type c = 0; c <= r; ++c) {
            assert(futures(r, c).is_null());
            printf("spawning task for block (%d, %d)\n", r, c);
            printf("futures dimensions: %d\n", futures.size());
            if (c == 0) {
              futures(r, c) = Kokkos::task_spawn(
                  Kokkos::TaskSingle(sched, Kokkos::TaskPriority::High),
                  Cholesky(sched, futures, n, r, c));
            } else {
              futures(r, c) = Kokkos::task_spawn(
                  Kokkos::TaskSingle(sched), Cholesky(sched, futures, n, r, c));
            }
          }
        }

        // Mark this as the final task
        printf("respawning task for block (%d, %d)\n", row, col);
        do_work = true;
        Kokkos::respawn(this, futures(n - 1, n - 1));
      }
    } else if (row == col) {
      if (do_work || row == 0) {
        // Do the actual work here
        printf("doing work for block (%d, %d)\n", row, col);
      } else {
        // Set up dependencies for diagonal
        printf("setting up dependencies for block (%d, %d)\n", row, col);
        Kokkos::Future<Space> dependencies[] = {futures(row, 0),
                                                futures(row - 1, col - 1)};
        Kokkos::Future<Space> fib_all = Kokkos::when_all(dependencies, 2);
        Kokkos::respawn(this, fib_all, Kokkos::TaskPriority::High);
        do_work = true;
      }
    } else {
      if (do_work) {
        // Do the actual work here
        printf("doing work for block (%d, %d)\n", row, col);
      } else {
        // Set up dependencies for non-diagonal
        printf("setting up dependencies for block (%d, %d)\n", row, col);
        if (col == 0) {
          do_work = true;
          Kokkos::respawn(this, futures(col, col), Kokkos::TaskPriority::High);
        } else {
          Kokkos::Future<Space> dependencies[] = {futures(col, col),
                                                  futures(row, col - 1)};
          Kokkos::Future<Space> fib_all = Kokkos::when_all(dependencies, 2);

          do_work = true;
          Kokkos::respawn(this, fib_all, Kokkos::TaskPriority::High);
        }
      }
    }
  }

  static void run(int n, size_t MemoryCapacity = 16000) {
    typedef typename sched_type::memory_space memory_space;

    enum { MinBlockSize = 64 };
    enum { MaxBlockSize = 1024 };
    enum { SuperBlockSize = 4096 };

    sched_type root_sched(memory_space(), MemoryCapacity, MinBlockSize,
                          std::min(size_t(MaxBlockSize), MemoryCapacity),
                          std::min(size_t(SuperBlockSize), MemoryCapacity));

    Kokkos::View<future_type **> futures("futures", n, n);
    printf("spawning Cholesky task\n");
    future_type f =
        Kokkos::host_spawn(Kokkos::TaskSingle(root_sched),
                           Cholesky(root_sched, futures, n, -1, -1));

    printf("waiting for root_sched\n");
    Kokkos::wait(root_sched);
    printf("done waiting for root_sched\n");
  }
};

// int hpx_main(int argc, char *argv[]) {
//   hpx::Kokkos::launch(Cholesky<Kokkos::DefaultExecutionSpace>::run, 10);

//   return hpx::finalize();
// }

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  // Kokkos::print_configuration(std::cout, true);

  Cholesky<Kokkos::DefaultExecutionSpace>::run(3);
  // hpx::init(argc, argv);

  Kokkos::finalize();

  return 0;
}
