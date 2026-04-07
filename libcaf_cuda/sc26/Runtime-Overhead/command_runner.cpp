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

using namespace caf;
using namespace std::chrono_literals;

using command = caf::cuda::command_runner<>;

command mmul_command;

struct mmul_state {
};

//global output buffer meant to disclude it from timing
//the other benchmark test do not include its memory allocations in it 
//so its only fair that we do not either 
std::vector<int> matrixC;

static const unsigned int RANDOM_SEED = 42;

caf::behavior mmul_actor(caf::stateful_actor<mmul_state>* self) {
  return {

    [=](const std::vector<int>& matrixA,
        const std::vector<int>& matrixB,
        int N) {
      
      using clock = std::chrono::steady_clock;
      using ms = std::chrono::duration<double, std::milli>;


      caf::cuda::manager& mgr = caf::cuda::manager::get();
      int device = 0;
      int stream = 1;

      auto program =
        mgr.create_program_from_cubin("mmul.cubin", "matrixMul");

      auto t_total_start = clock::now();
      // -------------------------
      // create_in_arg A
      // -------------------------
      auto t_a_inarg_start = clock::now();

      auto inA = caf::cuda::create_in_arg(std::move(matrixA));

      auto t_a_inarg_end = clock::now();

      // -------------------------
      // transfer A
      // -------------------------
      auto t_a_transfer_start = clock::now();

      auto arg1 = mmul_command.transfer_memory(
          device,
          stream,
          std::move(inA));

      auto t_a_transfer_end = clock::now();

      // -------------------------
      // create_in_arg B
      // -------------------------
      auto t_b_inarg_start = clock::now();

      auto inB = caf::cuda::create_in_arg(std::move(matrixB));

      auto t_b_inarg_end = clock::now();

      // -------------------------
      // transfer B
      // -------------------------
      auto t_b_transfer_start = clock::now();

      auto arg2 = mmul_command.transfer_memory(
          device,
          stream,
          std::move(inB));

      auto t_b_transfer_end = clock::now();
  
      const int THREADS = 32;
      const int BLOCKS = (N + THREADS - 1) / THREADS;

      caf::cuda::nd_range dims(
        BLOCKS, BLOCKS, 1,
        THREADS, THREADS, 1);

      // -------------------------
      // request
      // -------------------------
      auto t_request_start = clock::now();

      caf::cuda::mmul_async_command<int> command;
      auto output = command.run_async(
		      program,dims,
		      1,
		      arg1,arg2,out<int>{N*N},in<int>{N});

       auto t_response_received = clock::now();

            //std::vector<int> matrixC(N*N);
            // -------------------------
            // copy to host
            // -------------------------
            auto t_copy_start = clock::now();

	    caf::cuda::mem_ptr<int> dC = std::get<2>(output);

            //std::vector<int> matrixC = dC->copy_to_host();

	    dC->copy_to_host(matrixC.data(),N*N);

            auto t_copy_end = clock::now();
            auto t_total_end = clock::now();

            // -------------------------
            // Print timings
            // -------------------------

            std::cout << "\n===== BENCHMARK RESULTS (N=" << N << ") =====\n";

            std::cout << "create_in_arg A: "
                      << ms(t_a_inarg_end - t_a_inarg_start).count()
                      << " ms\n";

            std::cout << "transfer A: "
                      << ms(t_a_transfer_end - t_a_transfer_start).count()
                      << " ms\n";

            std::cout << "create_in_arg B: "
                      << ms(t_b_inarg_end - t_b_inarg_start).count()
                      << " ms\n";

            std::cout << "transfer B: "
                      << ms(t_b_transfer_end - t_b_transfer_start).count()
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

  };
}


void run_mmul_test(caf::actor_system& sys, int matrix_size) {
  // F5: manager::init/shutdown moved to caf_main — called once for all sizes

  // ------------------------------------
  // Start timing
  // ------------------------------------
  auto start = std::chrono::steady_clock::now();

  // F4: use mt19937(42) to match cuda_native data initialisation
  std::mt19937 rng(RANDOM_SEED);
  std::uniform_int_distribution<int> dist(1, 10);
  std::vector<int> matrixA(matrix_size * matrix_size);
  std::vector<int> matrixB(matrix_size * matrix_size);
  for (auto& v : matrixA) v = dist(rng);
  for (auto& v : matrixB) v = dist(rng);

  matrixC.resize(matrix_size*matrix_size);

  using clock = std::chrono::steady_clock;

  auto t_start = clock::now();

  caf::actor a =sys.spawn(mmul_actor);

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
  run_mmul_test(sys,1000);
  run_mmul_test(sys,2000);
  run_mmul_test(sys,4000);
  run_mmul_test(sys,8000);
  run_mmul_test(sys,16000);

  caf::cuda::manager::shutdown();  // F5: shutdown once after all sizes
}




CAF_MAIN()
