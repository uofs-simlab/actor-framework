#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

using namespace caf;
using namespace std::chrono_literals;

void run_mmul_test(caf::actor_system& sys, int matrix_size) {
  caf::cuda::manager::init(sys);
  caf::cuda::manager& mgr = caf::cuda::manager::get();

  auto t_total_start = std::chrono::steady_clock::now();

  int THREADS = 32;
  int BLOCKS = (matrix_size + THREADS - 1) / THREADS;
  caf::cuda::nd_range dim(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

  std::vector<int> h_a(matrix_size * matrix_size, 2);
  std::vector<int> h_b(matrix_size * matrix_size, 3);
  std::vector<int> h_c(matrix_size * matrix_size, 0);

  auto gpuActor = mgr.spawnFromCUBIN("mmul.cubin", "matrixMul", dim,
                                     in<int>{}, in<int>{}, out<int>{}, in<int>{});

  auto t_spawn_end = std::chrono::steady_clock::now();

  using clock = std::chrono::steady_clock;
  auto t_a_inarg_start = clock::now();
  auto arg1 = caf::cuda::create_in_arg(h_a);
  auto t_a_inarg_end = clock::now();

  auto t_b_inarg_start = clock::now();
  auto arg2 = caf::cuda::create_in_arg(h_b);
  auto t_b_inarg_end = clock::now();

  auto arg3 = caf::cuda::create_out_arg(h_c);
  auto arg4 = caf::cuda::create_in_arg(matrix_size);

  sys.spawn([=](event_based_actor* self_actor) {
    auto t_request_start = clock::now();
    self_actor->mail(gpuActor, arg1, arg2, arg3, arg4)
      .request(gpuActor, 100s).then(
        [=](const std::vector<output_buffer>& outputs) {
          auto t_response_received = clock::now();
          auto t_total_end = clock::now();

          using ms = std::chrono::duration<double, std::milli>;
          std::cout << "\n===== ACTOR FACADE BENCHMARK RESULTS (N=" << matrix_size << ") =====\n";
          std::cout << "spawn actor: " << ms(t_spawn_end - t_total_start).count() << " ms\n";
          std::cout << "create_in_arg A: " << ms(t_a_inarg_end - t_a_inarg_start).count() << " ms\n";
          std::cout << "create_in_arg B: " << ms(t_b_inarg_end - t_b_inarg_start).count() << " ms\n";
          std::cout << "request \xE2\x86\x92 response latency (includes transfers & exec): "
                    << ms(t_response_received - t_request_start).count() << " ms\n";
          std::cout << "TOTAL end-to-end: " << ms(t_total_end - t_total_start).count() << " ms\n";
          std::cout << "========================================================\n";

          self_actor->send_exit(gpuActor, exit_reason::user_shutdown);
          self_actor->quit();
        });
  });

  sys.await_all_actors_done();

  auto end = std::chrono::steady_clock::now();
  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - t_total_start).count();

  std::cout << "[MMUL FACADE TEST] matrix_size=" << matrix_size << ", time=" << duration_ms << " ms\n";

  caf::cuda::manager::shutdown();
}

void caf_main(caf::actor_system& sys) {
  run_mmul_test(sys, 1000);
  run_mmul_test(sys, 4000);
  run_mmul_test(sys, 8000);
  run_mmul_test(sys, 12000);
}

CAF_MAIN()
