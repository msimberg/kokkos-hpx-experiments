#include <Kokkos_Core.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/parallel_for_loop.hpp>

#include <iostream>

// This example initializes Kokkos on the main thread and calls Kokkos
// functionality from multiple HPX worker threads simultaneously. Oddly enough,
// no exceptions are thrown if run on CUDA backend.

struct Work {
  Kokkos::View<double *> a;

  Work(Kokkos::View<double *> a) : a(a){};

  KOKKOS_INLINE_FUNCTION void operator()(const int &i) const {
    a(i) = i + a(i);
  };
};

void work() {
  using hpx::parallel::execution::par;
  hpx::parallel::for_loop(par, 0, 10, [](std::size_t i) {
    Kokkos::View<double *> a("A", 100);
    auto h_a = Kokkos::create_mirror_view(a);

    // Does this block? Yes, it calls ThreadsExec::fence() which calls a
    // spinwait (sleeps etc.).
    Kokkos::parallel_for(a.size(), Work(a));

    Kokkos::deep_copy(h_a, a);
    hpx::cout << "h_a(9) = " << h_a(9) << "on thread "
              << hpx::get_worker_thread_num() << hpx::endl;
  });
}

int hpx_main(int argc, char *argv[]) {
  work();
  return hpx::finalize();
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout, true);
#if defined(KOKKOS_ENABLE_HPX)
  hpx::apply(work);
#else
  hpx::start(argc, argv);
  hpx::stop();
#endif
  Kokkos::finalize();

  return 0;
}
