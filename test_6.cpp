#include <Kokkos_Core.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/parallel/executors/service_executors.hpp>

#include <iostream>

// This example initializes Kokkos on the main thread, allocates and calls
// Kokkos::parallel_for on the main thread via the main thread executor. This
// example works with all backends.

struct Work {
  Kokkos::View<double *> a;

  Work(Kokkos::View<double *> a) : a(a){};

  KOKKOS_INLINE_FUNCTION void operator()(const int &i) const {
    a(i) = i + a(i);
  };
};

int hpx_main(int argc, char *argv[]) {
  using hpx::parallel::execution::par;
  using namespace hpx::parallel;
  using hpx::threads::executors::service_executor_type;
  execution::service_executor exec(service_executor_type::main_thread);

  hpx::parallel::for_loop(par, 0, 10, [](std::size_t i) {
    execution::service_executor exec(service_executor_type::main_thread);

    auto a = hpx::parallel::execution::sync_execute(
        exec, []() { return Kokkos::View<double *>("A", 100); });
    auto h_a = Kokkos::create_mirror_view(a);

    hpx::parallel::execution::sync_execute(exec, [a]() {
      hpx::cout << "Calling Kokkos::parallel_for" << hpx::endl;
      Kokkos::parallel_for(a.size(), Work(a));
      hpx::cout << "Done calling Kokkos::parallel_for" << hpx::endl;
    });

    hpx::cout << "Calling Kokkos::deep_copy" << hpx::endl;
    Kokkos::deep_copy(h_a, a);
    hpx::cout << "Done calling Kokkos::deep_copy" << hpx::endl;
    hpx::cout << "h_a(9) = " << h_a(9) << "on thread "
              << hpx::get_worker_thread_num() << hpx::endl;
  });

  return hpx::finalize();
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout, true);
  hpx::start(argc, argv);
  hpx::stop();
  Kokkos::finalize();

  return 0;
}
