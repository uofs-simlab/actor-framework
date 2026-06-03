/* An example file where we demonstrate how to use cuda actors using
 * the entry point command runner, enabling the user to create their own custom gpu
 * actor
 * Be sure to run compile_kernels.sh
 * We will show how to create an actor that runs matrix multiply and sends a message to itself
 * once the data has finished being asynchronously transfered 
 * Note that we will be using create_program_from_cubin and create_program_from_fatbin
 * methods these are the recommends and supported methods, as while you can
 * use create_program method, you are likely to run into unsupported toolchain 
 * errors
 */

#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
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



#include <chrono>
#include <iostream>

struct mmul_actor_state {
  static inline const char* name = "my_actor";
  int id = rand(); // an actor id, each actor uses an id to request gpu resources
		   // if you want actors to share the same gpu resources (such as custreams)
		   // then they must share the same id
  
  //mostly here for benchmarking if needed 
  // per-actor timing start 
  std::chrono::high_resolution_clock::time_point start_time;
  int times = 0;
};


// Command classes used to launch kernels.
// Templates describe the sequence of argument types for the GPU kernel.
using mmulCommand = caf::cuda::command_runner<in<int>, in<int>, out<int>, in<int>>;
mmulCommand mmul;



//a simple cpu matrix multiplication verification function
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


// Demonstration of a fully asynchronous GPU actor.
// It is recommended to use run_async and copy_to_host_async to avoid blocking worker threads.
caf::behavior mmul_async_actor_fun(caf::stateful_actor<mmul_actor_state>* self,int device_number, int stream_id) {
  return {
    // 1st handler: Receive host data, launch GPU kernel, and copy results back
    [=](std::vector<int> matrixA, std::vector<int> matrixB, int N) {
 
  caf::cuda::manager& mgr = caf::cuda::manager::get();

  //create program and dims   
  auto program = mgr.create_program_from_cubin("../mmul.cubin","matrixMul");
  const int THREADS = 32;
  const int BLOCKS = (N + THREADS - 1) / THREADS;
  caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

    //create args
    auto arg1 = caf::cuda::create_in_arg(matrixA);
    auto arg2 = caf::cuda::create_in_arg(matrixB);
    auto arg3 = caf::cuda::create_out_arg_with_size<int>(N*N);
    auto arg4 = caf::cuda::create_in_arg(N);


    // Launch kernel asynchronously
    auto results = mmul.run_async(program, dims, stream_id, 0, device_number, arg1, arg2, arg3, arg4);
    auto matrixC_ptr = std::get<2>(results); // Matrix C is the 3rd argument (index 2)

    // Copy result back to host asynchronously and send to verification handler
    mmul.copy_to_host_async(matrixC_ptr, [=, m1 = std::move(matrixA), m2 = std::move(matrixB)](std::vector<int> m3) {
      // Send host vectors to the 2nd handler for verification
      self->mail(std::move(m1), std::move(m2), std::move(m3), N).send(self);
    });
    },

    // 2nd handler: CPU verification
    [=](const std::vector<int>& matrixA,
    const std::vector<int> &matrixB,
    const std::vector<int> &matrixC, int N) {

    std::vector<int> result(N * N);

    serial_matrix_multiply(matrixA, matrixB, result, N);

    if (result == matrixC) {
        std::cout << "actor with id " << self->state().id << " references match\n";
    }
    else {
        std::cout << "actor with id " << self->state().id << " references did not match\n";

    }


    /*
    auto print_matrix = [N](const std::vector<int>& mat, const std::string& name) {
            std::cout << name << ":\n";
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    std::cout << mat[i * N + j] << " ";
                }
                std::cout << "\n";
            }
            std::cout << std::endl;
        };

        print_matrix(matrixA, "Matrix A");
        print_matrix(matrixB, "Matrix B");
        print_matrix(result, "Result Matrix");
        print_matrix(matrixC, "GPU Result Matrix");
	*/
    self->quit();
    }
  };
}


void run_async_mmul_test(caf::actor_system& sys, int matrix_size, int num_actors) {
  if (num_actors < 1) {
    std::cerr << "[ERROR] Number of actors must be >= 1\n";
    return;
  }

  std::vector<int> matrixA(matrix_size * matrix_size, 1);
  std::vector<int> matrixB(matrix_size * matrix_size, 2);

  // Spawn num_actors actors running the mmul behavior
  std::vector<caf::actor> actors;
  actors.reserve(num_actors);
  for (int i = 0; i < num_actors; ++i) {
    actors.push_back(sys.spawn(mmul_async_actor_fun,0,1));
  }

  //send a size to all actors 
  for (auto a : actors)
	  caf::anon_mail(matrixA, matrixB, matrix_size).send(a);

   sys.await_all_actors_done();
}


void caf_main(caf::actor_system& sys) {
  caf::cuda::manager::init(sys);
  run_async_mmul_test(sys, 100, 10);
}

CAF_MAIN()
