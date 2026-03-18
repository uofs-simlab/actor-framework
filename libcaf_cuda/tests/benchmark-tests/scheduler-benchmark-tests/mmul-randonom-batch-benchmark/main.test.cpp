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






struct mmul_state {

	caf::cuda::program_ptr mmul_kernel;

};



caf::behavior mmul_actor_fun(caf::stateful_actor<mmul_state>* self,
		caf::actor exit_actor,
		caf::cuda::program_ptr program,
		caf::cuda::nd_range dims,
		int N,
		const in<int> matrixA,
		const in<int> matrixB
		) {

	self->mail(N).send(self);
	return {
		
		[=](int N) {

			caf::cuda::manager& mgr = caf::cuda::manager::get();
			int device = 0;
			int stream = rand();

			//auto program =
			//mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

		    

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
			self->mail(1).send(exit_actor);
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




// ---------------------------- SUPERVISOR ACTOR ----------------------------
struct supervisor_actor_state {
    int num_actors;
    int num_waves;
    int completed;
    int max_waves;

    MatrixPool pool;

    // Precomputed sequence of N values
    std::vector<int> Ns;
    int next_task;

    // Timing
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point wave_start_time;
};

caf::behavior supervisor_actor_fun(
    caf::stateful_actor<supervisor_actor_state>* self,
    int num_actors,
    int max_waves,
    MatrixPool pool,
    const std::vector<int>& Ns,      // deterministic task sizes 
    bool use_scheduler
    ) {
    // Initialize state
    self->state().num_actors = num_actors;
    self->state().completed = 0;
    self->state().max_waves = max_waves;
    self->state().num_waves = 0;
    self->state().pool = std::move(pool);

    self->state().Ns = Ns;
    self->state().next_task = 0;

    // Start timing
    self->state().start_time = std::chrono::steady_clock::now();

    caf::cuda::manager& mgr = caf::cuda::manager::get();
    auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

    // Kick off first wave
    self->mail("spawn").send(self);

    return {
        // -------------------- SPAWN WAVE --------------------
        [=](std::string cmd) {
            if (cmd != "spawn") return;

            self->state().completed = 0;
            self->state().wave_start_time = std::chrono::steady_clock::now();

            for (int i = 0; i < self->state().num_actors; ++i) {
                if (self->state().next_task >= self->state().Ns.size())
                    break;

                int N = self->state().Ns[self->state().next_task++];


                const auto& A = self->state().pool.A[N];
                const auto& B = self->state().pool.B[N];

                const int THREADS = 32;
                const int BLOCKS = (N + THREADS - 1) / THREADS;

                caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

		if (use_scheduler) {
			self->spawn(mmul_actor_fun_scheduler, self, N, program, dims,
                            caf::cuda::create_in_arg(A),
                            caf::cuda::create_in_arg(B));
		}
		else {
			self->spawn(mmul_actor_fun, self, program, dims,N,
                            caf::cuda::create_in_arg(A),
                            caf::cuda::create_in_arg(B));
		}
            }
        },

        // -------------------- COMPLETION TRACKING --------------------
        [=](int done) {
            self->state().completed += done;

            if (self->state().completed >= self->state().num_actors) {
                auto wave_end = std::chrono::steady_clock::now();
                std::chrono::duration<double> wave_time =
                    wave_end - self->state().wave_start_time;

                self->state().num_waves++;
                //std::cout << "Wave "
                  //        << self->state().num_waves
                    //      << " completed in "
                      //    << wave_time.count() << " s\n";

                if (self->state().num_waves >= self->state().max_waves) {
                    auto end_time = std::chrono::steady_clock::now();
                    std::chrono::duration<double> total_time =
                        end_time - self->state().start_time;

                    std::cout << "\n===== SUPERVISOR TOTAL TIME =====\n";
                    std::cout << "Total runtime: "
                              << total_time.count() << " s\n";

                    caf::cuda::manager::shutdown();
                    self->quit();
                } else {
                    self->mail("spawn").send(self);
                }
            }
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



void run_mmul_random_scaling_tests(caf::actor_system& sys,
                                  caf::cuda::manager_config man_config) {

    const int min_N = 256;
    const int max_N = 1024;
    const int num_sizes = 200;

    const int max_waves = 1;

    const std::vector<int> actor_counts = {
	    10000
    };


    int num_actors = actor_counts[0];

        // Generate deterministic random pool
        MatrixPool pool = create_matrix_pool_random(
            num_sizes,
            min_N,
            max_N,
            42  // fixed seed
        );

    // Precompute all task Ns (total_tasks = num_actors * max_waves)
    std::vector<int> sizes;
    for (const auto& [N, _] : pool.A) sizes.push_back(N);

    int total_tasks = num_actors * max_waves;
    std::vector<int> Ns;
    Ns.reserve(total_tasks);

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, sizes.size() - 1);
    for (int i = 0; i < total_tasks; ++i)
	    Ns.push_back(sizes[dist(rng)]);


    //scheduler
    for (int num_actors : actor_counts) {

    
	    // Initialize CUDA manager
	    caf::cuda::manager::init(sys, man_config);
	    std::cout << "=====================================\n";
	    std::cout << "Running with Scheduler with " << num_actors << " actors\n";

	    auto& mgr = caf::cuda::manager::get();
	    for (int i = 0; i < mgr.get_num_devices(); i++) {
		    mgr.send_scheduler_actor_message("green",i); 
	    }



        double elapsed = time_run([&]() {

            auto sup = sys.spawn(
                supervisor_actor_fun,
                num_actors,
                max_waves,
                pool,
		Ns,
		true
            );

        
	      sys.await_all_actors_done();
	    
	    });

    
	caf::cuda::manager::shutdown();
    }
    
    //no scheduler
    for (int num_actors : actor_counts) {

    
	    // Initialize CUDA manager
	    caf::cuda::manager::init(sys);
	    std::cout << "=====================================\n";
	    std::cout << "Running no scheduler with " << num_actors << " actors\n";

      
	    double elapsed = time_run([&]() {

            auto sup = sys.spawn(
                supervisor_actor_fun,
                num_actors,
                max_waves,
                pool,
		Ns,
		false
            );

	      sys.await_all_actors_done();
	    
	    });

    
	caf::cuda::manager::shutdown();
    }
    caf::cuda::manager::shutdown();
}






void caf_main(caf::actor_system& sys) {
  

  caf::cuda::manager_config man_config(true);
  //caf::cuda::manager::init(sys,man_config);


  run_mmul_random_scaling_tests(sys,man_config);



}




CAF_MAIN()
