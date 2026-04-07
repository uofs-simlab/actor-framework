#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>

using namespace caf;
using namespace std::chrono_literals;

static const unsigned int RANDOM_SEED = 42;

void run_mmul_test(caf::actor_system& sys, int matrix_size) {
  // F5: manager::init/shutdown moved to caf_main — called once for all sizes
  caf::cuda::manager& mgr = caf::cuda::manager::get();

  // F3: spawn GPU actor (loads cubin) BEFORE timing begins, so module-load
  //     overhead is excluded from the per-size measurement — matching
  //     cuda_native which loads the module once outside all per-size timing.
  int THREADS = 32;
  int BLOCKS = (matrix_size + THREADS - 1) / THREADS;
  caf::cuda::nd_range dim(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

  auto gpuActor = mgr.spawnFromCUBIN("mmul.cubin", "matrixMul", dim,
                                     in<int>{}, in<int>{}, out<int>{}, in<int>{});

  using clock = std::chrono::steady_clock;
  using ms = std::chrono::duration<double, std::milli>;

  // F4: use mt19937(42) to match cuda_native data initialisation
  std::mt19937 rng(RANDOM_SEED);
  std::uniform_int_distribution<int> dist(1, 10);
  std::vector<int> h_a(matrix_size * matrix_size);
  std::vector<int> h_b(matrix_size * matrix_size);
  std::vector<int> h_c(matrix_size * matrix_size, 0);
  for (auto& v : h_a) v = dist(rng);
  for (auto& v : h_b) v = dist(rng);

  // TIMING STARTS — after spawn/module-load, matching cuda_native's exclusion
  auto t_total_start = clock::now();

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

          std::cout << "\n===== ACTOR FACADE BENCHMARK RESULTS (N=" << matrix_size << ") =====\n";
          std::cout << "create_in_arg A: " << ms(t_a_inarg_end - t_a_inarg_start).count() << " ms\n";
          std::cout << "create_in_arg B: " << ms(t_b_inarg_end - t_b_inarg_start).count() << " ms\n";
          std::cout << "request \xE2\x86\x92 response latency (includes transfers & exec): "
                    << ms(t_response_received - t_request_start).count() << " ms\n";
          std::cout << "TOTAL end-to-end: " << ms(t_total_end - t_total_start).count() << " ms\n";
          std::cout << "=============================================\n";

          self_actor->send_exit(gpuActor, exit_reason::user_shutdown);
          self_actor->quit();
        });
  });

  sys.await_all_actors_done();
}

class config : public actor_system_config {
public:
  config() {
      set("caf.scheduler.max-threads", 1u);
  }
};

void caf_main(caf::actor_system& sys, const config& cfg) {
  caf::cuda::manager::init(sys);  // F5: init once before all sizes

  // F2: warmup run to prime CUDA context, JIT, and CAF infrastructure
  std::cout << "--- warmup starting ---\n";
  run_mmul_test(sys, 64);
  std::cout << "--- warmup complete ---\n";

  // F1: unified sizes matching cuda_native: {1000, 2000, 4000, 8000, 16000}
  run_mmul_test(sys, 1000);
  run_mmul_test(sys, 2000);
  run_mmul_test(sys, 4000);
  run_mmul_test(sys, 8000);
  run_mmul_test(sys, 16000);

  caf::cuda::manager::shutdown();  // F5: shutdown once after all sizes
}

CAF_MAIN()
