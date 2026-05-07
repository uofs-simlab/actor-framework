#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
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

// Define a custom type ID block for custom actors
CAF_ADD_ATOM(cuda,shared_mem)






#include <chrono>
#include <iostream>

// Extend your actor state to keep the start time
struct mmul_actor_state {
  static inline const char* name = "my_actor";
  int last_N = 0; // example state variable
  int id = rand(); 
  // per-actor timing start
  std::chrono::high_resolution_clock::time_point start_time;
  int times = 0;
};




//commands classes used to launch kernels 
using mmulCommand = caf::cuda::command_runner<in<int>,in<int>,out<int>,in<int>>;
using matrixGenCommand = caf::cuda::command_runner<out<int>,in<int>,in<int>,in<int>>;

using mmulAsyncCommand = caf::cuda::command_runner<caf::cuda::mem_ptr<int>,caf::cuda::mem_ptr<int>,out<int>,in<int>>;

mmulCommand mmul;
matrixGenCommand randomMatrix;
mmulAsyncCommand mmulAsync;


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


// Stateful actor behavior
caf::behavior mmul_async_actor_fun(caf::stateful_actor<mmul_actor_state>* self) {
  return {
    // 1. Initial trigger: Setup data, launch kernel, and register the async callback.
    [=](int N) {
      std::cout << "Starting Async Callback Test with N=" << N << std::endl;
      
      // Initialize test data on host
      std::vector<int> h_a(N * N);
      std::vector<int> h_b(N * N);
      std::iota(h_a.begin(), h_a.end(), 1);
      std::iota(h_b.begin(), h_b.end(), 1);

      caf::cuda::manager& mgr = caf::cuda::manager::get();
      auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

      const int THREADS = 32;
      const int BLOCKS = (N + THREADS - 1) / THREADS;
      caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

      // Launch the kernel asynchronously using host vectors.
      // run_async returns a tuple of mem_ptrs for each argument.
      auto results = mmul.run_async(program, dims, self->state().id, 
                                    caf::cuda::create_in_arg(h_a),
                                    caf::cuda::create_in_arg(h_b),
                                    caf::cuda::create_out_arg(N * N),
                                    caf::cuda::create_in_arg(N));

      // The output matrix 'c' is the 3rd argument (index 2).
      auto matrixC_ptr = std::get<2>(results);
      auto self_hdl = caf::actor_cast<caf::actor>(self);

      // Invoke the new async copy with a callback. 
      // When the GPU is done, this lambda runs on a driver thread.
      mmul.copy_to_host_async(matrixC_ptr, [self_hdl, h_a = std::move(h_a), h_b = std::move(h_b), N](std::vector<int> h_c) mutable {
        std::cout << "GPU Work Complete. Callback triggered. Notifying actor..." << std::endl;
        // Use anon_mail to safely send the data back to the actor system.
        caf::anon_mail(std::move(h_a), std::move(h_b), std::move(h_c), N).send(self_hdl);
      });
    },

    // 2. Verification handler: Called when the async callback sends the data.
    [=](const std::vector<int>& matrixA,
    const std::vector<int> &matrixB,
    const std::vector<int> &matrixC, int N) {

    std::cout << "Actor received results. Verifying correctness..." << std::endl;
    std::vector<int> result(N * N);
    serial_matrix_multiply(matrixA, matrixB, result, N);

    if (result == matrixC) {
        std::cout << "SUCCESS: Actor id=" << self->state().id << " results match!" << std::endl;
    }
    else {
        std::cout << "FAILURE: Actor id=" << self->state().id << " results do NOT match!" << std::endl;
    }
    self->quit();
    }
  };
}


void run_async_mmul_test(caf::actor_system& sys, int matrix_size, int num_actors) {
  auto worker = sys.spawn(mmul_async_actor_fun);
  caf::anon_mail(matrix_size).send(worker);
  sys.await_all_actors_done();
}

void caf_main(caf::actor_system& sys) {
  caf::cuda::manager::init(sys);
  run_async_mmul_test(sys, 1024, 1);
}

CAF_MAIN()
