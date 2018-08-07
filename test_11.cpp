#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/parallel/executors/service_executors.hpp>

#include <cassert>
#include <iostream>

#include "hpx_kokkos.hpp"

// This example tests spawning a task with dependencies.

template <typename Space> struct TestTaskSpawn {
  using sched_type = Kokkos::TaskScheduler<Space>;
  using value_type = void;
  using future_type = Kokkos::Future<value_type, Space>;
  using size_type = int;

  sched_type sched;
  Kokkos::View<future_type **> futures;
  bool arbitrary_flag;

  KOKKOS_INLINE_FUNCTION
  TestTaskSpawn(const sched_type &arg_sched, const bool arg_arbitrary_flag)
      : sched(arg_sched), arbitrary_flag(arg_arbitrary_flag) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(typename sched_type::member_type &) {
    printf("in task, flag = %d\n", arbitrary_flag);
    if (arbitrary_flag) {
      future_type f1 =
          task_spawn(Kokkos::TaskSingle(sched, Kokkos::TaskPriority::High),
                     TestTaskSpawn(sched, false));
      future_type f2 =
          task_spawn(Kokkos::TaskSingle(f1, Kokkos::TaskPriority::High),
                     TestTaskSpawn(sched, false));
      future_type dependencies[] = {f1, f2};
      future_type f3 =
          task_spawn(Kokkos::TaskSingle(Kokkos::when_all(dependencies, 2),
                                        Kokkos::TaskPriority::High),
                     TestTaskSpawn(sched, false));

      // NOTE: Can also respawn oneself with dependencies.
      arbitrary_flag = false;
      Kokkos::respawn(this, Kokkos::when_all(dependencies, 2),
                      Kokkos::TaskPriority::High);

      // NOTE: Can't spawn a task using a lambda.
      // future_type f4 =
      //     task_spawn(Kokkos::TaskSingle(Kokkos::when_all(dependencies, 2),
      //                                   Kokkos::TaskPriority::High),
      //                KOKKOS_LAMBDA(typename sched_type::member_type &) {
      //                  std::cout << "in lambda" << std::endl;
      //                });
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

    future_type f = Kokkos::host_spawn(Kokkos::TaskSingle(root_sched),
                                       TestTaskSpawn(root_sched, true));

    Kokkos::wait(root_sched);
  }
};

// int hpx_main(int argc, char *argv[]) {
//   hpx::Kokkos::launch(TestTaskSpawn<Kokkos::DefaultExecutionSpace>::run, 10);

//   return hpx::finalize();
// }

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout, true);

  TestTaskSpawn<Kokkos::DefaultExecutionSpace>::run(10);
  // hpx::init(argc, argv);

  Kokkos::finalize();

  return 0;
}
