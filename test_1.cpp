#include <Kokkos_Core.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/parallel/executors/service_executors.hpp>

// This example initializes Kokkos on the main thread and uses the main thread
// executor to dispatch work from HPX to Kokkos.

struct Work {
  Kokkos::View<double *> a;

  Work(Kokkos::View<double *> a) : a(a){};

  KOKKOS_INLINE_FUNCTION void operator()(const int &i) const {
    a(i) = i + a(i);
  };
};

void work() {
  using namespace hpx::parallel;
  using hpx::threads::executors::service_executor_type;
  execution::service_executor exec(service_executor_type::main_thread);

  std::vector<hpx::future<void>> fs;

  for (std::size_t i = 0; i < 10; ++i) {
    fs.push_back(hpx::parallel::execution::async_execute(exec, []() {
      Kokkos::View<double *> a("A", 100);
      auto h_a = Kokkos::create_mirror_view(a);

      // This is allowed to block, runs on main thread.
      Kokkos::parallel_for(a.size(), Work(a));

      Kokkos::deep_copy(h_a, a);
      hpx::cout << "h_a(9) = " << h_a(9) << hpx::endl;
    }));
  }

  hpx::wait_all(fs);
}

int hpx_main(int argc, char *argv[]) {
  work();
  return hpx::finalize();
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
#if defined(KOKKOS_ENABLE_HPX)
  hpx::apply(work);
#else
  hpx::start(argc, argv);
  hpx::stop();
#endif
  Kokkos::finalize();

  return 0;
}
