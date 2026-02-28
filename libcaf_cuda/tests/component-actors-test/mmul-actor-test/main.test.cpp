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

    // 1st handler matrices + N, launches a kernel and verifies result
    [=](const std::vector<int>& matrixA,
        const std::vector<int>& matrixB,
        int N) {

      caf::cuda::manager& mgr = caf::cuda::manager::get();

      int device = 0;
      int stream = 1;
      
      // Create program and dims
      auto program =
        mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

      auto arg1 = mmul_command.transfer_memory(device,
		      stream,
		      caf::cuda::create_in_arg(matrixA));
      auto arg2 = mmul_command.transfer_memory(device,
		      stream,
		      caf::cuda::create_in_arg(matrixB));

      caf::actor mmul_actor =
        self->spawn(caf::cuda::mmul_actor_fun<int>, program);


      self->mail(arg1, arg2, N, device, stream)
        .request(mmul_actor, std::chrono::seconds(10))
        .then(
          [=](caf::cuda::mem_ptr<int> dC) {

            std::vector<int> matrixC = dC->copy_to_host();
            std::vector<int> result(N * N);

            serial_matrix_multiply(matrixA, matrixB, result, N);

            if (result == matrixC)
              std::cout << "actor matrix references match\n";
            else
              std::cout << "actor matrix references did not match\n";

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
  run_mmul_test(sys,10);

}




CAF_MAIN()
