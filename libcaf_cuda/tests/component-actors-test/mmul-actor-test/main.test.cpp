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



struct mmul_state {
};

// Stateful actor behavior
caf::behavior mmul_actor_fun(caf::stateful_actor<mmul_state>* self) {
  return {

	
	  // 1st handler matrices +  N, launches a kenrel and sends its result to itself for verification
	  [=](const std::vector<int> matrixA,
			  const std::vector<int> matrixB, int N) {


		  caf::cuda::manager& mgr = caf::cuda::manager::get();

		  //create program and dims   
		  auto program = mgr.create_program_from_cubin("../mmul.cubin","matrixMul");
		  //create args
		  auto arg1 = caf::cuda::create_in_arg(matrixA);
		  auto arg2 = caf::cuda::create_in_arg(matrixB);

		  caf::actor mmul_actor = self -> spawn(caf::cuda::mmul_actor_fun<int>,program);

		  int device = 0;
		  int stream = 1;

		  self->request(mmul_actor,
				  std::chrono::seconds(10),
				  arg1,
				  arg2,
				  N,
				  device,
				  stream)
			  .then(
					  [&](caf::cuda::mem_ptr<int> dC) {

					  std::vector<int> matrixC =
					  dC->copy_to_host();
					  std::vector<int> result(N*N);

					  serial_matrix_multiply(matrixA,matrixB,result,N);

					  if (result == matrixC) {

					  std::cout << "actor matrix references match\n";

					  }

					  else {
					  std::cout << "actor matrix  references did not match\n";
					  }

					  self-> quit();

					  }
	  });


}


void run_mmul_test(caf::actor_system& sys, int matrix_size, int num_actors) {
  if (num_actors < 1) {
    std::cerr << "[ERROR] Number of actors must be >= 1\n";
    return;
  }

  // ------------------------------------
  // Start timing
  // ------------------------------------
  auto start = std::chrono::steady_clock::now();

  // Spawn num_actors actors running the mmul behavior
  std::vector<caf::actor> actors;
  actors.reserve(num_actors);
 
 using clock = std::chrono::steady_clock;

auto t_start = clock::now();

for (int i = 0; i < num_actors; ++i) {
    actors.push_back(sys.spawn(mmul_actor_fun));
}

auto t_end = clock::now();

auto elapsed_ms =
    std::chrono::duration_cast<std::chrono::milliseconds>(
        t_end - t_start).count();

std::cout << "[SPAWN] spawned "
          << num_actors
          << " actors in "
          << elapsed_ms
          << " ms\n";
 
 
 
  // Actor 0 generates matrices and broadcasts to others
  caf::anon_mail(matrix_size, actors).send(actors[0]);

  // Wait for all actors to finish
  sys.await_all_actors_done();

  // ------------------------------------
  // Stop timing
  // ------------------------------------
  auto end = std::chrono::steady_clock::now();
  auto duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  std::cout << "[MMUL TEST] matrix_size=" << matrix_size
            << ", actors=" << num_actors
            << ", time=" << duration_ms << " ms\n";
}

void caf_main(caf::actor_system& sys) {
  caf::cuda::manager::init(sys);

  run_mmul_test(sys,512,512);
  //run_async_mmul_test(sys,100,1);
  //run_async_mmul_perf_test(sys,1024,200);

  // run the async (no-shared) suite:
  //benchmark_async_perf_all(sys);

  // run the shared-memory suite:
  //benchmark_shared_perf_all(sys);

  //run_mmul_scaling_tests(sys);
}




CAF_MAIN()
