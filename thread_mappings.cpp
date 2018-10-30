#include <Kokkos_Core.hpp>
#include <hpx/hpx_init.hpp>

#include <iostream>
#include <stdio.h>

struct Work {
  using member_type = Kokkos::TeamPolicy<>::member_type;

  KOKKOS_INLINE_FUNCTION void operator()(member_type thread) const {
    // This gets called 2 * 3 times.
    printf("parallel_for(TeamPolicy(2, 3)):             %d/%d %d/%d\n",
           thread.league_rank(), thread.league_size(), thread.team_rank(),
           thread.team_size());

    Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, 4), [=](const int &j) {
      // This gets called 2 * 4 times, i.e. league_size * 4 times.
      printf("parallel_for(TeamThreadRange(thread, 4)):             %d/%d "
             "%d/%d, j = %d\n",
             thread.league_rank(), thread.league_size(), thread.team_rank(),
             thread.team_size(), j);

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, 5),
                           [=](const int &k) {
                             // This gets called 2 * 4 * 5 times, i.e.
                             // league_size * 4 * 5 times.
                             printf("parallel_for(ThreadVectorRange(thread, "
                                    "5)):             %d/%d %d/%d, k = %d\n",
                                    thread.league_rank(), thread.league_size(),
                                    thread.team_rank(), thread.team_size(), k);
                           });

      Kokkos::single(Kokkos::PerThread(thread), [=]() {
        // This gets called 2 * 4 times, i.e. league_size * 4 times.
        printf("parallel_for(single(PerThread) (1):             %d/%d %d/%d\n",
               thread.league_rank(), thread.league_size(), thread.team_rank(),
               thread.team_size());
      });

      // This is allowed, but probably not well defined.
      // Kokkos::parallel_for(
      //     Kokkos::TeamThreadRange(thread, 7), KOKKOS_LAMBDA(const int &i) {
      //       std::stringstream s;
      //       s << "parallel_for(TeamThreadRange(thread, 7)):   "
      //         << thread.league_rank() << "/" << thread.league_size() << " "
      //         << thread.team_rank() << "/" << thread.team_size()
      //         << ", i = " << j << std::endl;
      //       printf(s.str().c_str());
      //     });
    });

    Kokkos::single(Kokkos::PerThread(thread), [=]() {
      // This gets called 2 * 3 times, i.e. league_size * team_size times.
      printf("parallel_for(single(PerThread) (2):             %d/%d %d/%d\n",
             thread.league_rank(), thread.league_size(), thread.team_rank(),
             thread.team_size());
    });

    Kokkos::single(Kokkos::PerTeam(thread), [=]() {
      // This gets called 2 times, i.e. league_size times.
      printf("parallel_for(single(PerTeam):             %d/%d %d/%d\n",
             thread.league_rank(), thread.league_size(), thread.team_rank(),
             thread.team_size());
    });

    // This is not allowed, application hangs here.
    // Kokkos::parallel_for(Kokkos::TeamPolicy<>(3, 7),
    //                      KOKKOS_LAMBDA(member_type){});
  };
};

struct Work2 {
  using member_type = Kokkos::TeamPolicy<>::member_type;
  KOKKOS_INLINE_FUNCTION void operator()(member_type thread) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, 4), [=](const int &j) {
      Kokkos::single(Kokkos::PerThread(thread), [=]() { printf("|"); });
    });
  };
};

struct Work3 {
  KOKKOS_INLINE_FUNCTION void operator()(const int &i) const {
    printf("RangePolicy(2, 7): i = %d\n", i);
  };
};

struct Work4 {
  using member_type = Kokkos::TeamPolicy<>::member_type;

  KOKKOS_INLINE_FUNCTION void operator()(const int &i, const int &j,
                                         const int &k) const {
    printf("RangePolicy(2, 7): (i, j, k) = (%d, %d, %d)\n", i, j, k);

    // This is not allowed, application hangs here.
    // Kokkos::parallel_for(Kokkos::TeamPolicy<>(3, 7),
    //                      KOKKOS_LAMBDA(member_type){});
  };
};

struct WorkScratch {
  using member_type = Kokkos::TeamPolicy<>::member_type;

  KOKKOS_INLINE_FUNCTION void operator()(member_type thread) const {
    printf("parallel_for(TeamPolicy(2, Kokkos::AUTO_t)): %d/%d %d/%d\n",
           thread.league_rank(), thread.league_size(), thread.team_rank(),
           thread.team_size());
    printf("team_shmem() = %p\n", thread.team_shmem().get_shmem(4096));
    printf("team_scratch(0) = %p\n", thread.team_scratch(0).get_shmem(0));
    printf("thread_scratch(0) = %p\n",
           thread.thread_scratch(0).get_shmem(1024));
  };
};

struct WorkTeamBroadcast {
  using member_type = Kokkos::TeamPolicy<>::member_type;

  KOKKOS_INLINE_FUNCTION void operator()(member_type thread) const {
    printf("parallel_for(TeamPolicy(2, Kokkos::AUTO_t)): %d/%d %d/%d\n",
           thread.league_rank(), thread.league_size(), thread.team_rank(),
           thread.team_size());

    int value = thread.team_rank() * 5 * (thread.league_rank() + 1);
    thread.team_broadcast(value, 3);
    printf("team_broadcast: team_rank = %d, league_rank = %d, value = %d\n",
           thread.team_rank(), thread.league_rank(), value);
    value += thread.team_rank();
    thread.team_broadcast([&](int &var) { var *= 2; }, value, 2);
    printf("team_broadcast: team_rank = %d, league_rank = %d, value = %d\n",
           thread.team_rank(), thread.league_rank(), value);
  };
};

// This would be nice to have in HPX:
// - executor to restrict execution to a single NUMA node
// - executor to restrict execution to a single (arbitrary, scheduler can
// decide) NUMA node
// - executor to allow vectorization (but only in a single thread), i.e.
// hpx::parallel::execution::unseq
// - multidimensional arrays whose storage layout change depending on CPU or GPU
// execution
// - parallel algorithms on the GPU (thrust), but need not have futures *on* the
// GPU
// - GPU executors which efficiently chain multiple kernel calls

// int hpx_main(int argc, char* argv[]) {
//   hpx::parallel::for_loop(executor(distribute over numa nodes), league_start,
//     league_end, []() {
//     hpx::parallel::for_loop(this_numa_node, team_start, team_end, []() {
//     hpx::parallel::for_loop(unseq (i.e. simd on this thread))
//   })
// });
// }

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout, true);

  Kokkos::parallel_for(Kokkos::TeamPolicy<>(2, 3), Work());
  std::cout << std::endl;

  Kokkos::parallel_for(Kokkos::TeamPolicy<>(2, 3), Work2());
  std::cout << "\n\n" << std::flush;

  Kokkos::parallel_for(Kokkos::RangePolicy<>(2, 7), Work3());
  std::cout << std::endl;

  // Ranges are half-open (end not inclusive).
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 3, 2}, {5, 6, 3}), Work4());
  std::cout << std::endl;

  Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(2, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(4096), Kokkos::PerThread(1024)),
      WorkScratch());
  std::cout << std::endl;

  Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(2, Kokkos::AUTO),
      WorkTeamBroadcast());
  std::cout << std::endl;

  Kokkos::finalize();

  return 0;
}
