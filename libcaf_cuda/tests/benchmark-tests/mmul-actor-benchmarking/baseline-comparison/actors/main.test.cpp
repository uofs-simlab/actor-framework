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
	caf::cuda::program_ptr program;
	int total_expected = 0;
	int results_received = 0;
};

//global output buffer meant to disclude it from timing
//the other benchmark test do not include its memory allocations in it 
//so its only fair that we do not either 
std::vector<int> matrixC;




caf::behavior mmul_actor_fun(caf::stateful_actor<mmul_state>* self,
		caf::cuda::program_ptr mmul_kernel, int iterations) {

	self ->state().program = mmul_kernel;
	self ->state().total_expected = iterations;

return {
    [=](const std::vector<int>& matrixA,
        const std::vector<int>& matrixB,
        int N) {

      using clock = std::chrono::steady_clock;
      using ms = std::chrono::duration<double, std::milli>;


      caf::cuda::manager& mgr = caf::cuda::manager::get();
      int device = 0;
      int stream = 1;

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

      caf::cuda::mmul_async_command<int> command;
      auto output = command.run_async(
		      self->state().program,
		      dims,
		      1, 0, device,
		      arg1,arg2,out<int>{N*N},in<int>{N});

	    caf::cuda::mem_ptr<int> dC = std::get<2>(output);

        auto self_hdl = caf::actor_cast<caf::actor>(self);
      
        
        if (++self->state().results_received == self->state().total_expected) {
           mmul_command.copy_to_host_async(dC, matrixC.data(), N*N, [self_hdl](int*, size_t) {
                caf::anon_mail(kernel_done_atom_v).send(self_hdl);
            });
        }
        else {
           mmul_command.copy_to_host_async(dC, matrixC.data(), N*N, [self_hdl](int*, size_t) {
            //do nothing there is nothing to do
          });

        }
        
    },
    [=](kernel_done_atom) {
        if (++self->state().results_received == self->state().total_expected) {
            self->quit();
        }
    }
  };
}


void run_mmul_test(caf::actor_system& sys, int matrix_size,int iterations) {


  caf::cuda::manager::init(sys);
  // ------------------------------------
  // Start timing
  // ------------------------------------

  // Spawn num_actors actors running the mmul behavior
  std::vector<int> matrixA(matrix_size * matrix_size,2);
  std::vector<int> matrixB(matrix_size * matrix_size,3);

  matrixC.resize(matrix_size*matrix_size);
 
  auto& mgr = caf::cuda::manager::get();
  
  auto program =
        mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");


  using clock = std::chrono::steady_clock;

  auto start = std::chrono::steady_clock::now();

  caf::actor a = sys.spawn(mmul_actor_fun, program, iterations);

  for (int i = 0; i < iterations; i++) 
	  anon_mail(matrixA,matrixB,matrix_size).send(a);

  // Wait for all actors to finish
  sys.await_all_actors_done();

  // ------------------------------------
  // Stop timing
  // ------------------------------------
  auto end = std::chrono::steady_clock::now();
  auto duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  std::cout << "[MMUL TEST] matrix_size=" << matrix_size
            << " iterations = " << iterations << 
	    ", time=" << duration_ms << " ms\n";

  caf::cuda::manager::shutdown();

}


void caf_main(caf::actor_system& sys) {

	for (int i = 1000; i < 11000; i+=1000)
		run_mmul_test(sys,1000,i);

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

// CAF_MAIN()
