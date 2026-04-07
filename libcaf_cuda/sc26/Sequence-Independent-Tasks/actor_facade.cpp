#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

using namespace caf;
using namespace std::chrono_literals;

struct bench_state {
  int completed = 0;
  int total = 0;
  bool is_warmup = false;
  std::chrono::steady_clock::time_point start_time;
  caf::actor gpuActor;
  int matrix_size = 0;
  std::vector<int> h_a;
  std::vector<int> h_b;
  std::vector<int> h_c;
};

// Forward declaration
void send_next_request(caf::stateful_actor<bench_state>* self);

// S3 fix: back-pressure — sends exactly one request and re-chains on response,
// bounding in-flight memory to a single 8 MB pair of in-args at any time.
void send_next_request(caf::stateful_actor<bench_state>* self) {
  auto& st = self->state();
  auto arg1 = caf::cuda::create_in_arg(st.h_a);
  auto arg2 = caf::cuda::create_in_arg(st.h_b);
  auto arg3 = caf::cuda::create_out_arg(st.h_c);
  auto arg4 = caf::cuda::create_in_arg(st.matrix_size);

  self->mail(st.gpuActor, arg1, arg2, arg3, arg4)
    .request(st.gpuActor, infinite)
    .then(
      [=](const std::vector<output_buffer>& /*outputs*/) {
        auto& st2 = self->state();
        st2.completed++;

        // S6 fix: milestone reporting every 1000 completions
        if (!st2.is_warmup && st2.completed % 1000 == 0) {
          auto now = std::chrono::steady_clock::now();
          using ms = std::chrono::duration<double, std::milli>;
          double elapsed = ms(now - st2.start_time).count();
          std::cout << "[MILESTONE] " << st2.completed << " / " << st2.total
                    << " iterations, elapsed = " << elapsed << " ms\n";
        }

        if (st2.completed == st2.total) {
          if (!st2.is_warmup) {
            auto end_time = std::chrono::steady_clock::now();
            using ms = std::chrono::duration<double, std::milli>;
            double duration_ms = ms(end_time - st2.start_time).count();
            std::cout << "[SERIES RESULT] Matrix " << st2.matrix_size << "x" << st2.matrix_size
                      << ", iterations = " << st2.total
                      << ", total CPU/Actor time = " << duration_ms << " ms\n";
          }
          self->send_exit(st2.gpuActor, exit_reason::user_shutdown);
          self->quit();
        } else {
          send_next_request(self);
        }
      },
      [=](const error& err) {
        if (!self->state().is_warmup)
          std::cout << "Error in iteration: " << to_string(err) << std::endl;
        auto& st2 = self->state();
        st2.completed++;
        if (st2.completed == st2.total) {
          self->send_exit(st2.gpuActor, exit_reason::user_shutdown);
          self->quit();
        } else {
          send_next_request(self);
        }
      }
    );
}

caf::behavior bench_coordinator(caf::stateful_actor<bench_state>* self,
                                 caf::actor gpuActor,
                                 int matrix_size,
                                 int iterations,
                                 bool is_warmup) {
  auto& st = self->state();
  st.completed = 0;
  st.total = iterations;
  st.is_warmup = is_warmup;
  st.gpuActor = gpuActor;
  st.matrix_size = matrix_size;
  st.h_a.assign(matrix_size * matrix_size, 2);
  st.h_b.assign(matrix_size * matrix_size, 3);
  st.h_c.assign(matrix_size * matrix_size, 0);

  self->monitor(gpuActor, [self](const error& err) {
    if (!self->state().is_warmup)
      std::cout << "GPU Actor died unexpectedly! " << to_string(err) << std::endl;
    self->quit();
  });

  return {
    [=](int) {
      self->state().start_time = std::chrono::steady_clock::now();
      // S3 fix: fire one request; subsequent requests are chained via back-pressure
      send_next_request(self);
    }
  };
}

// S4 fix: manager::init/shutdown moved to caf_main — called once for all series
void run_series(caf::actor_system& sys, int matrix_size, int iterations,
                bool is_warmup = false) {
  caf::cuda::manager& mgr = caf::cuda::manager::get();

  int THREADS = 32;
  int BLOCKS = (matrix_size + THREADS - 1) / THREADS;
  caf::cuda::nd_range dim(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

  auto gpuActor = mgr.spawnFromCUBIN("mmul.cubin", "matrixMul", dim,
                                     in<int>{}, in<int>{}, out<int>{}, in<int>{});

  auto coordinator = sys.spawn(bench_coordinator, gpuActor, matrix_size, iterations, is_warmup);
  anon_mail(1).send(coordinator);
  sys.await_all_actors_done();
}

void caf_main(caf::actor_system& sys) {
  constexpr int matrix_size = 1000;
  constexpr int total_iterations = 10000;

  caf::cuda::manager::init(sys);  // S4 fix: init once before all series

  // S2 fix: warmup run to prime CUDA context and CAF infrastructure
  std::cout << "--- warmup starting ---\n";
  run_series(sys, matrix_size, 10, /*is_warmup=*/true);
  std::cout << "--- warmup complete ---\n";

  run_series(sys, matrix_size, total_iterations);

  caf::cuda::manager::shutdown();  // S4 fix: shutdown once after all series
}

CAF_MAIN()
