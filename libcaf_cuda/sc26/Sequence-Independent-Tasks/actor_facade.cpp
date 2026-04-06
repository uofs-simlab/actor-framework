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
  std::chrono::steady_clock::time_point start_time;
  caf::actor gpuActor;
  int matrix_size = 0;
  std::vector<int> h_a;
  std::vector<int> h_b;
  std::vector<int> h_c;
};

caf::behavior bench_coordinator(caf::stateful_actor<bench_state>* self, caf::actor gpuActor, int matrix_size, int iterations) {
  self->state().completed = 0;
  self->state().total = iterations;
  self->state().gpuActor = gpuActor;
  self->state().matrix_size = matrix_size;
  
  self->monitor(gpuActor, [self](const error& err) {
    std::cout << "GPU Actor died unexpectedly! " << to_string(err) << std::endl;
    self->quit();
  });

  return {
    [=](int) {
      self->state().start_time = std::chrono::steady_clock::now();
      
      // Persistent host vectors in state
      self->state().h_a.assign(matrix_size * matrix_size, 2);
      self->state().h_b.assign(matrix_size * matrix_size, 3);
      self->state().h_c.assign(matrix_size * matrix_size, 0);
      
      for (int i = 0; i < iterations; ++i) {
        auto arg1 = caf::cuda::create_in_arg(self->state().h_a);
        auto arg2 = caf::cuda::create_in_arg(self->state().h_b);
        auto arg3 = caf::cuda::create_out_arg(self->state().h_c);
        auto arg4 = caf::cuda::create_in_arg(matrix_size);
        
        self->mail(gpuActor, arg1, arg2, arg3, arg4)
          .request(gpuActor, infinite)
          .then(
            [=](const std::vector<output_buffer>& /*outputs*/) {
              self->state().completed++;
              if (self->state().completed == self->state().total) {
                auto end_time = std::chrono::steady_clock::now();
                using ms = std::chrono::duration<double, std::milli>;
                double duration_ms = ms(end_time - self->state().start_time).count();
                std::cout << "[SERIES RESULT] Matrix " << self->state().matrix_size << "x" << self->state().matrix_size
                          << ", iterations = " << self->state().total
                          << ", total CPU/Actor time = " << duration_ms << " ms\n";
                self->send_exit(self->state().gpuActor, exit_reason::user_shutdown);
                self->quit();
              }
            },
            [=](const error& err) {
              std::cout << "Error in iteration: " << to_string(err) << std::endl;
              self->state().completed++;
              if (self->state().completed == self->state().total) {
                self->send_exit(self->state().gpuActor, exit_reason::user_shutdown);
                self->quit();
              }
            }
          );
      }
    }
  };
}

void run_series(caf::actor_system& sys, int matrix_size, int iterations) {
  caf::cuda::manager::init(sys);
  caf::cuda::manager& mgr = caf::cuda::manager::get();

  int THREADS = 32;
  int BLOCKS = (matrix_size + THREADS - 1) / THREADS;
  caf::cuda::nd_range dim(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

  // Spawn the GPU actor
  auto gpuActor = mgr.spawnFromCUBIN("mmul.cubin", "matrixMul", dim,
                                     in<int>{}, in<int>{}, out<int>{}, in<int>{});

  // Spawn coordinator
  auto coordinator = sys.spawn(bench_coordinator, gpuActor, matrix_size, iterations);
  
  // Start benchmark
  anon_mail(1).send(coordinator);

  // Wait for it to finish and kill GPU actor internally
  sys.await_all_actors_done();

  caf::cuda::manager::shutdown();
}

void caf_main(caf::actor_system& sys) {
  for (int i = 1000; i <= 10000; i += 1000) {
    run_series(sys, 1000, i);
  }
}

CAF_MAIN()
