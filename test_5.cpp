#include <Kokkos_Core.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/parallel_for_loop.hpp>

#include <iostream>

// This example tries to initialize Kokkos multiple times on each HPX worker
// thread. It fails.

struct Work {
  Kokkos::View<double *> a;

  Work(Kokkos::View<double *> a) : a(a){};

  KOKKOS_INLINE_FUNCTION void operator()(const int &i) const {
    a(i) = i + a(i);
  };
};

int hpx_main(int argc, char *argv[]) {
  using hpx::parallel::execution::par;

  hpx::parallel::for_loop(par, 0, 3, [argc, argv](std::size_t i) mutable {
    Kokkos::initialize(argc, argv);
    // Kokkos::print_configuration(std::cout, true);
    Kokkos::View<double *> a("A", 100);
    auto h_a = Kokkos::create_mirror_view(a);

    hpx::cout << "Calling Kokkos::parallel_for" << hpx::endl;
    Kokkos::parallel_for(a.size(), Work(a));
    hpx::cout << "Done calling Kokkos::parallel_for" << hpx::endl;

    hpx::cout << "Calling Kokkos::deep_copy" << hpx::endl;
    Kokkos::deep_copy(h_a, a);
    hpx::cout << "Done calling Kokkos::deep_copy" << hpx::endl;
    hpx::cout << "h_a(9) = " << h_a(9) << "on thread "
              << hpx::get_worker_thread_num() << hpx::endl;
    Kokkos::finalize();
  });

  return hpx::finalize();
}

int main(int argc, char *argv[]) {
  hpx::start(argc, argv);
  hpx::stop();

  return 0;
}
