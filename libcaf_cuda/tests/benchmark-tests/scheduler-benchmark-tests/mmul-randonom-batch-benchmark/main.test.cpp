#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <caf/component-actors/all-component-actors.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <random>
#include "caf/actor_registry.hpp"
#include <random>
#include <unordered_set>
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
caf::cuda::command_runner<caf::cuda::mem_ptr<int>, caf::cuda::mem_ptr<int>,caf::cuda::mem_ptr<int>,caf::cuda::mem_ptr<int >> mmul;
using async_command = caf::cuda::mmul_async_command<int>;
async_command async_mmul;





struct MatrixPool {
    std::unordered_map<int, std::vector<int>> A;
    std::unordered_map<int, std::vector<int>> B;
};



MatrixPool create_matrix_pool_random(
    int num_sizes,
    int min_N,
    int max_N,
    unsigned int seed
) {
    MatrixPool pool;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(min_N, max_N);

    std::unordered_set<int> used;

    while (used.size() < static_cast<size_t>(num_sizes)) {
        int N = dist(rng);
        if (used.insert(N).second) {
            pool.A[N] = std::vector<int>(N * N, 1);
            pool.B[N] = std::vector<int>(N * N, 1);
        }
    }

    return pool;
}



struct supervisor_actor_state {
    int num_actors;
    int num_waves;
    int completed;
    int max_waves;

    MatrixPool pool;

    std::vector<int> sizes;   // cached keys
    std::mt19937 rng;         // RNG
};



//runs for FCFS behavior
caf::behavior supervisor_actor_fun(
    caf::stateful_actor<supervisor_actor_state>* self,
    int num_actors,
    int max_waves,
    MatrixPool pool,
    caf::cuda::program_ptr program
) {
    // --- Initialize state ---
    self->state().num_actors = num_actors;
    self->state().completed = 0;
    self->state().max_waves = max_waves;
    self->state().num_waves = 0;
    self->state().pool = std::move(pool);

    self->state().rng = std::mt19937(42); // reproducible

    // Extract sizes once
    for (const auto& [N, _] : self->state().pool.A) {
        self->state().sizes.push_back(N);
    }

     caf::cuda::manager& mgr = caf::cuda::manager::get();
     auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

    // Kick off first wave
    self->mail("spawn").send(self);

    return {

        // =========================
        // SPAWN WAVE
        // =========================
        [=](std::string cmd) {
            if (cmd != "spawn")
                return;

            self->state().completed = 0;

            std::uniform_int_distribution<size_t> dist(
                0, self->state().sizes.size() - 1);

            for (int i = 0; i < self->state().num_actors; ++i) {

                int N = self->state().sizes[dist(self->state().rng)];

                const auto& A = self->state().pool.A[N];
                const auto& B = self->state().pool.B[N];

                const int THREADS = 32;
                const int BLOCKS = (N + THREADS - 1) / THREADS;

                caf::cuda::nd_range dims(
                    BLOCKS, BLOCKS, 1,
                    THREADS, THREADS, 1
                );

                self->spawn(
                    mmul_actor_fun_scheduler,
                    self, // exit actor
                    N,
                    program,
                    dims,
                    caf::cuda::create_in_arg(A),
                    caf::cuda::create_in_arg(B)
                );
            }
        },

        // =========================
        // COMPLETION TRACKING
        // =========================
        [=](int done) {
            self->state().completed += done;

            if (self->state().completed >= self->state().num_actors) {

                self->state().num_waves++;

                std::cout << "Completed wave "
                          << self->state().num_waves
                          << "/" << self->state().max_waves
                          << std::endl;

                if (self->state().num_waves >= self->state().max_waves) {
                    caf::cuda::manager::shutdown();
                    self->quit();
                } else {
                    self->mail("spawn").send(self);
                }
            }
        }
    };
}





struct mmul_state {

	caf::cuda::program_ptr mmul_kernel;

};



caf::behavior mmul_actor_fun(caf::stateful_actor<mmul_state>* self,caf::cuda::program_ptr mmul_kernel,
		const in<int> matrixA,
		const in<int> matrixB,
		int N) {

	self->state().mmul_kernel = mmul_kernel;
	self->mail(N).send(self);
	return {
		
		[=](int N) {

			caf::cuda::manager& mgr = caf::cuda::manager::get();
			int device = 0;
			int stream = rand();

			//auto program =
			//mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

		    const int THREADS = 32;
		    const int BLOCKS = (N + THREADS - 1) / THREADS;
		    caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);


			auto program = self->state().mmul_kernel;

			auto inA = std::move(matrixA);
			auto arg1 = mmul_command.transfer_memory(
					device,
					stream,
					std::move(inA));

			auto inB = std::move(matrixB);
			auto arg2 = mmul_command.transfer_memory(
					device,
					stream,
					std::move(inB));

			out<int> arg3 = caf::cuda::create_out_arg<int>(N * N);
			in<int>  arg4 = caf::cuda::create_in_arg<int>(N);

			auto result = 
				async_mmul.run_async(
						program,
						dims,
						stream,
						0,
						device,
						arg1,
						arg2,
						arg3,
						arg4);

			std::get<2>(result) -> copy_to_host();
			self -> quit();

		}
	};
}




struct mmul_actor_with_scheduler_state {
  static inline const char* name = "my_actor";
};


// Stateful actor behavior
caf::behavior mmul_actor_fun_scheduler(
    caf::stateful_actor<mmul_actor_with_scheduler_state>* self,
    caf::actor exit_actor,
    int N,
    caf::cuda::program_ptr program,
    caf::cuda::nd_range dims,
    const in<int> matrixA,
    const in<int> matrixB)
{


        caf::cuda::manager& mgr = caf::cuda::manager::get();

        //caf::actor scheduler = mgr.get_scheduler_actor();

        //send a launch token
        caf::cuda::token_ptr launch_token = caf::cuda::make_launch_token(
                        program,
                        dims,
                        0,
                        "hello",
                        self,
			rand() //dependency number, can declare indepedent but want to see what happens when you do not 
                        );
        mgr.send_scheduler_actor_message(launch_token);

	return {

		// 1. Handle response token
		[=](caf::cuda::response_token_ptr res_token) {
			//        std::cout << "Got response\n";

			if (res_token->getType() == LAUNCH_RESPONSE) {
				self->mail(res_token, N).send(self);

			} else {
				//          std::cout << "Got a memory response token\n";
			}
		},

	    // 2. Handle memory buffers -> GPU
	    [=] (const caf::cuda::response_token_ptr& res_token,
			    int N) {

		    //    std::cout << "Working\n";
		    caf::cuda::manager& mgr = caf::cuda::manager::get();


		    const int THREADS = 32;
		    const int BLOCKS = (N + THREADS - 1) / THREADS;
		    caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

		    auto arg1 = mmul.transfer_memory(res_token, matrixA);
		    auto arg2 = mmul.transfer_memory(res_token, matrixB);
		    auto arg3 = mmul.transfer_memory(res_token, caf::cuda::create_out_arg(N*N));
		    auto arg4 = mmul.transfer_memory(res_token, caf::cuda::create_in_arg(N));


		    auto tempC = mmul.run_async(program, dims, res_token, arg1, arg2, arg3, arg4);
		    caf::cuda::mem_ptr<int> bufferC = std::get<2>(tempC);

		    bufferC -> synchronize();
		    res_token->release();
		    bufferC->copy_to_host();

		    self->mail(1).send(exit_actor);
		    self->quit();
         }


       };

}


template <class Fn>
double time_run(Fn&& fn) {
  auto start = std::chrono::steady_clock::now();
  fn();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  return elapsed.count();
}










void caf_main(caf::actor_system& sys) {
  

  caf::cuda::manager_config man_config(true);
  //caf::cuda::manager::init(sys,man_config);



}




CAF_MAIN()
