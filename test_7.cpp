#include <Kokkos_Core.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/parallel/executors/service_executors.hpp>

#include <iostream>
#include <stdio.h>

#include "hpx_kokkos.hpp"

// This example initializes Kokkos on the main thread, and provides wrappers to
// call Kokkos functionality on the main thread.

struct Work {
  using member_type = Kokkos::TeamPolicy<>::member_type;

  Kokkos::View<double *> a;

  Work(Kokkos::View<double *> a) : a(a){};

  KOKKOS_INLINE_FUNCTION void operator()(const int &i) const {
    // Can use normal Kokkos functionality here. Nested parallelism etc.
    a(i) = i;
  };

  KOKKOS_INLINE_FUNCTION void operator()(member_type thread) const {
    printf("parallel_for(TeamPolicy(2, 3)): %d/%d %d/%d", thread.league_rank(),
           thread.league_size(), thread.team_rank(), thread.team_size());
  }
};

struct WorkFillZeros {
  Kokkos::View<double *> a;

  WorkFillZeros(Kokkos::View<double *> a) : a(a){};

  KOKKOS_INLINE_FUNCTION void operator()(const int &i) const { a(i) = 0; };
};

struct WorkSquare {
  Kokkos::View<double *> a;

  WorkSquare(Kokkos::View<double *> a) : a(a){};

  KOKKOS_INLINE_FUNCTION void operator()(const int &i) const {
    a(i) = a(i) * a(i);
  };
};

struct WorkReduce {
  Kokkos::View<double *> a;

  WorkReduce(Kokkos::View<double *> a) : a(a){};

  KOKKOS_INLINE_FUNCTION void operator()(const int &i, double &update) const {
    update += a(i);
  };
};

int hpx_main(int argc, char *argv[]) {
  // Call Kokkos parallel for loops sequentially and synchronously.
  for (std::size_t i; i < 10; ++i) {
    auto a = hpx::Kokkos::make_view<double *>("A", 100);
    auto h_a = hpx::Kokkos::create_mirror_view(a);

    Kokkos::TeamPolicy<> policy(2, 3);
    hpx::Kokkos::parallel_for(policy, Work(a));
    hpx::Kokkos::parallel_for(a.size(), Work(a));

    hpx::Kokkos::deep_copy(h_a, a);
    hpx::cout << "h_a(9) = " << h_a(9) << " on thread "
              << hpx::get_worker_thread_num() << hpx::endl;

    double sum = 0.0;
    hpx::Kokkos::parallel_reduce(a.size(), WorkReduce(a), sum);

    hpx::Kokkos::deep_copy(h_a, a);
    hpx::cout << "sum = " << sum << " on thread "
              << hpx::get_worker_thread_num() << hpx::endl;
  }

  // Call async parallel for loop and chain multiple calls with then.
  {
    auto b = hpx::Kokkos::make_view<double *>("B", 10);
    auto h_b = hpx::Kokkos::create_mirror_view(b);

    auto f = hpx::Kokkos::parallel_for_async(b.size(), WorkFillZeros(b))
                 .then([b](hpx::future<void>) {
                   return hpx::Kokkos::parallel_for(b.size(), Work(b));
                 })
                 .then([b](hpx::future<void>) {
                   return hpx::Kokkos::parallel_for(b.size(), WorkSquare(b));
                 });

    f.wait();

    hpx::Kokkos::deep_copy(h_b, b);
    hpx::cout << "h_b(h_b.size() - 1) = " << h_b(h_b.size() - 1)
              << " on thread " << hpx::get_worker_thread_num() << hpx::endl;
  }

  // Call multiple wrapped parallel for loops in one async call.
  {
    auto b = hpx::Kokkos::make_view<double *>("B", 10);
    auto h_b = hpx::Kokkos::create_mirror_view(b);

    auto f = hpx::async([b]() {
      hpx::Kokkos::parallel_for(b.size(), WorkFillZeros(b));
      hpx::Kokkos::parallel_for(b.size(), Work(b));
      hpx::Kokkos::parallel_for(b.size(), WorkSquare(b));
    });

    f.wait();

    hpx::Kokkos::deep_copy(h_b, b);
    hpx::cout << "h_b(h_b.size() - 1) = " << h_b(h_b.size() - 1)
              << " on thread " << hpx::get_worker_thread_num() << hpx::endl;
  }

  // Call multiple native Kokkos parallel for loops in one async call with the
  // main thread executor.
  {
    auto b = hpx::Kokkos::make_view<double *>("B", 10);
    auto h_b = hpx::Kokkos::create_mirror_view(b);

    hpx::parallel::execution::service_executor main_thread_exec(
        hpx::threads::executors::service_executor_type::main_thread);

    auto f = hpx::async(main_thread_exec, [b]() {
      Kokkos::parallel_for(b.size(), WorkFillZeros(b));
      Kokkos::parallel_for(b.size(), Work(b));
      Kokkos::parallel_for(b.size(), WorkSquare(b));
    });

    f.wait();

    hpx::Kokkos::deep_copy(h_b, b);
    hpx::cout << "h_b(h_b.size() - 1) = " << h_b(h_b.size() - 1)
              << " on thread " << hpx::get_worker_thread_num() << hpx::endl;
  }

  // Kokkos enabled library functions would have to be dispatched explicitly to
  // the main thread.

  // HPX threads can still be used for asynchronously launching communication
  // threads.

  return hpx::finalize();
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout, true);

  hpx::init(argc, argv);

  Kokkos::finalize();

  return 0;
}
