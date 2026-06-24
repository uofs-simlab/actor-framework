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

//global output buffer meant to disclude it from timing
//the other benchmark test do not include its memory allocations in it 
//so its only fair that we do not either 
std::vector<int> matrixC;




caf::behavior mmul_actor_fun_2(caf::stateful_actor<mmul_state>* self) {
  return {

    [=](const std::vector<int>& matrixA,
        const std::vector<int>& matrixB,
        int N) {

      using clock = std::chrono::steady_clock;
      using ms = std::chrono::duration<double, std::milli>;

      size_t bytes_a = matrixA.size() * sizeof(int);
      size_t bytes_b = matrixB.size() * sizeof(int);
      size_t bytes_c = matrixC.size() * sizeof(int);


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
		      1, 0, device,
		      arg1,arg2,out<int>{N*N},in<int>{N});

       auto t_response_received = clock::now();

            //std::vector<int> matrixC(N*N);
            // -------------------------
            // copy to host
            // -------------------------
            auto t_copy_start = clock::now();

	    caf::cuda::mem_ptr<int> dC = std::get<2>(output);

            mmul_command.copy_to_host_async(dC, matrixC.data(), N * N);
            dC->synchronize();

            auto t_copy_end = clock::now();
            auto t_total_end = clock::now();

            // -------------------------
            // Print timings
            // -------------------------

            std::cout << "\n===== BENCHMARK RESULTS (N=" << N << ") =====\n";
            std::cout << "  Transfer size A: " << bytes_a << " bytes\n";
            std::cout << "  Transfer size B: " << bytes_b << " bytes\n";
            std::cout << "  Transfer size C: " << bytes_c << " bytes\n";

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


  caf::cuda::manager::init(sys);
  // ------------------------------------
  // Start timing
  // ------------------------------------
  auto start = std::chrono::steady_clock::now();

  // Spawn num_actors actors running the mmul behavior
  std::vector<int> matrixA(matrix_size * matrix_size,2);
  std::vector<int> matrixB(matrix_size * matrix_size,3);

  matrixC.resize(matrix_size*matrix_size);

 using clock = std::chrono::steady_clock;

auto t_start = clock::now();

caf::actor a =sys.spawn(mmul_actor_fun_2);

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

  caf::cuda::manager::shutdown();

}


void caf_main(caf::actor_system& sys) {
    run_mmul_test(sys, 1000);
    run_mmul_test(sys, 4000);
    run_mmul_test(sys, 8000);
    run_mmul_test(sys, 12000);
}

int main(int argc, char** argv) {
  // Initialize user defined types and messages if needed.
  //init_global_meta_objects<custom_types_1>();
  
  // Initialize the global type information.
  core::init_global_meta_objects();

  // Create the config.
  actor_system_config cfg;

  // --- SINGLE THREAD CONFIGURATION ---
  cfg.set("caf.scheduler.max-threads", 1);
  cfg.set("caf.scheduler.policy", "sharing"); 
  // ------------------------------------

  // Read CLI options. (Note: CLI flags like --caf.scheduler.max-threads=4 
  // will override the hardcoded '1' above if provided by the user).
  auto err = cfg.parse(argc, argv);
  if (err)
    return EXIT_FAILURE;

  if (cfg.helptext_printed())
    return 0;

  // Create the actor system (the scheduler starts here).
  actor_system sys{cfg};

  // Run user-defined code.
  caf_main(sys);
  
  return 0;
}
