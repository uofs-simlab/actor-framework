#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <caf/component-actors/mmul_actor/mmul_actor.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <random>
#include "caf/actor_registry.hpp"
//#include <caf/atoms.hpp>



using namespace caf;
using namespace std::chrono_literals;


void serial_matrix_multiply(const std::vector<int>& a,
                            const std::vector<int>& b,
                            std::vector<int>& c,
                            int N) {
  

 for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      int sum = 0;
      for (int k = 0; k < N; ++k) {
        sum += a[i * N + k] * b[k * N + j];
      }
      c[i * N + j] = sum;
    }
  }
}

using command =
  caf::cuda::command_runner<>;

command mmul_command;

struct mmul_state {
};

caf::behavior mmul_actor_fun(caf::stateful_actor<mmul_state>* self) {
  return {

    [=](const std::vector<int>& matrixA,
        const std::vector<int>& matrixB,
        int N) {

      using clock = std::chrono::steady_clock;
      using ms = std::chrono::duration<double, std::milli>;

      auto t_total_start = clock::now();

      caf::cuda::manager& mgr = caf::cuda::manager::get();
      int device = 0;
      int stream = 1;

      auto program =
        mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

      // -------------------------
      // transfer A
      // -------------------------
      auto t_a_start = clock::now();

      auto arg1 = mmul_command.transfer_memory(
          device,
          stream,
          caf::cuda::create_in_arg(std::move(matrixA)));

      auto t_a_end = clock::now();

      // -------------------------
      // transfer B
      // -------------------------
      auto t_b_start = clock::now();

      auto arg2 = mmul_command.transfer_memory(
          device,
          stream,
          caf::cuda::create_in_arg(std::move(matrixB)));

      auto t_b_end = clock::now();

      // -------------------------
      // spawn
      // -------------------------
      auto t_spawn_start = clock::now();

      caf::actor mmul_actor =
        self->spawn(caf::cuda::mmul_actor_fun<int>, program);

      auto t_spawn_end = clock::now();

      // -------------------------
      // request
      // -------------------------
      auto t_request_start = clock::now();

      self->mail(arg1, arg2, N, device, stream)
        .request(mmul_actor, std::chrono::seconds(30))
        .then(
          [=](caf::cuda::mem_ptr<int> dC) {

            auto t_response_received = clock::now();

            // -------------------------
            // copy to host
            // -------------------------
            auto t_copy_start = clock::now();

            std::vector<int> matrixC = dC->copy_to_host();

            auto t_copy_end = clock::now();
            auto t_total_end = clock::now();

            // -------------------------
            // Print timings
            // -------------------------

            std::cout << "\n===== BENCHMARK RESULTS (N=" << N << ") =====\n";

            std::cout << "transfer A: "
                      << ms(t_a_end - t_a_start).count()
                      << " ms\n";

            std::cout << "transfer B: "
                      << ms(t_b_end - t_b_start).count()
                      << " ms\n";

            std::cout << "spawn actor: "
                      << ms(t_spawn_end - t_spawn_start).count()
                      << " ms\n";

            std::cout << "request → response latency: "
                      << ms(t_response_received - t_request_start).count()
                      << " ms\n";

            std::cout << "copy_to_host: "
                      << ms(t_copy_end - t_copy_start).count()
                      << " ms\n";

            std::cout << "TOTAL end-to-end: "
                      << ms(t_total_end - t_total_start).count()
                      << " ms\n";

            std::cout << "=============================================\n";

            self->quit();
          }
        );
    }

  };
}


void run_mmul_test(caf::actor_system& sys, int matrix_size) {

  // ------------------------------------
  // Start timing
  // ------------------------------------
  auto start = std::chrono::steady_clock::now();

  // Spawn num_actors actors running the mmul behavior
  std::vector<int> matrixA(matrix_size * matrix_size,2);
  std::vector<int> matrixB(matrix_size * matrix_size,3);
 
 using clock = std::chrono::steady_clock;

auto t_start = clock::now();

caf::actor a =sys.spawn(mmul_actor_fun);

anon_mail(matrixA,matrixB,matrix_size).send(a);

auto t_end = clock::now();



  // Wait for all actors to finish
  sys.await_all_actors_done();

  // ------------------------------------
  // Stop timing
  // ------------------------------------
  auto end = std::chrono::steady_clock::now();
  auto duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  std::cout << "[MMUL TEST] matrix_size=" << matrix_size
            << ", time=" << duration_ms << " ms\n";
}

void caf_main(caf::actor_system& sys) {
  caf::cuda::manager::init(sys);
  run_mmul_test(sys,1000);
  //run_mmul_test(sys,4000);
  //run_mmul_test(sys,8000);
  //run_mmul_test(sys,12000);

}




CAF_MAIN()
