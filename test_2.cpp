#include <Kokkos_Core.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/include/iostreams.hpp>

// This example initializes Kokkos on the first HPX worker thread. Calling
// Kokkos::parallel_for does, however, block the underlying OS thread.

struct Work {
  Kokkos::View<double *> a;

  Work(Kokkos::View<double *> a) : a(a){};

  KOKKOS_INLINE_FUNCTION void operator()(const int &i) const {
    a(i) = i + a(i);
  };
};

int hpx_main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);

  {
    Kokkos::View<double *> a("A", 100);
    auto h_a = Kokkos::create_mirror_view(a);

    // Does this block? Yes, it calls ThreadsExec::fence() which calls a
    // spinwait (sleeps etc.).
    // NOTE: Ok with HPX backend.
    Kokkos::parallel_for(a.size(), Work(a));

    Kokkos::deep_copy(h_a, a);
    hpx::cout << "h_a(9) = " << h_a(9) << hpx::endl;
  }

  // Does this block?
  Kokkos::finalize();

  return hpx::finalize();
}

int main(int argc, char *argv[]) {
  hpx::start(argc, argv);
  hpx::stop();

  return 0;
}
