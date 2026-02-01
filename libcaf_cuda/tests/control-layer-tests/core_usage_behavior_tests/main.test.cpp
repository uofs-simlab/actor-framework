#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <caf/cuda/control-layer/all-control-layer.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <random>
#include <unistd.h>
#include "caf/actor_registry.hpp"
#include <chrono>
#include <iostream>
//#include <caf/atoms.hpp>




using namespace caf;
using namespace std::chrono_literals;


struct exit_actor_state {
	int completed = 0;
};


caf::behavior exit_actor_fun(caf::stateful_actor<exit_actor_state>* self,int limit) {


	return {
		[=](int num_completed) {
			self->state().completed += num_completed;
			
			std::cout << "Actors finished is " << self->state().completed << "\n";
			if (self->state().completed >= limit) {
			
				caf::cuda::manager::shutdown();
				self->quit();
			}
		}
	};


}






// Define a custom type ID block for custom actors
CAF_ADD_ATOM(cuda,shared_mem)





// Extend your actor state to keep the start time
struct mmul_actor_state {
  static inline const char* name = "mmul_actor";

  int N = 0;
  int id = rand();

  // timing / bookkeeping only
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
caf::behavior mmul_actor_fun(
    caf::stateful_actor<mmul_actor_state>* self,
    caf::actor exit_actor,
    int N,
    caf::cuda::program_ptr program,
    caf::cuda::nd_range dims)
{

	//set the value of N correctly to overide the base option.	
	self->state().N = N;

	caf::cuda::manager& mgr = caf::cuda::manager::get();

	caf::actor scheduler = mgr.get_scheduler_actor();

	//send a launch token
	caf::cuda::token_ptr launch_token = caf::cuda::make_launch_token(
			program,
			dims,
			0,
			"hello",
			self
			);
	self -> mail(launch_token).send(scheduler);

	return {

		[=] (caf::cuda::response_token_ptr res_token) {

			if (res_token -> getType() == LAUNCH_RESPONSE) {
				//std::cout << "GPU ACTOR RECEIVED PERMISSION TO LAUNCH\n"; 
				//assume N = 1024
				int N = self -> state().N;
				std::vector<int> matrix1(N*N);
				matrix1.reserve(N);
				std::vector<int> matrix2(N*N);
				matrix2.reserve(N);

				//std::cout << "GPU ACTOR sending data to compute\n";
				self -> mail(matrix1,matrix2,res_token,N).send(self);

			}
			else {
				std::cout << "Got a memory response token\n";
			}
			//token should drop out of scope now, triggering a response 
		},

			// 2nd handler: GPU atom + matrices + N, launches a kenrel and sends its result to itself for verification
			[=](const std::vector<int>& matrixA,
					const std::vector<int>& matrixB,
					const caf::cuda::response_token_ptr& res_token, int N) {


				//caf::cuda::kernel_launch_token kernelToken = caf::intrusive_ptr_cast<caf::cuda::token_ptr>(kToken);

				//caf::cuda::launch_response_token& kt =
				//	    static_cast<caf::cuda::launch_response_token&>(*kToken);

				//std::cout << "GPU ACTOR  computing\n";
				caf::cuda::manager& mgr = caf::cuda::manager::get();

				//create program and dims   
				auto program = mgr.create_program_from_cubin("../mmul.cubin","matrixMul");
				const int THREADS = 32;
				const int BLOCKS = (N + THREADS - 1) / THREADS;
				caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

				//create args
				auto arg1 = caf::cuda::create_in_arg(matrixA);
				auto arg2 = caf::cuda::create_in_arg(matrixB);
				auto arg3 = caf::cuda::create_out_arg(N*N);
				auto arg4 = caf::cuda::create_in_arg(N);

				auto tempC = mmul.run(program,dims,res_token,arg1,arg2,arg3,arg4);
				std::vector<int> matrixC = caf::cuda::extract_vector<int>(tempC);

				//std::cout << "GPU ACTOR done  computing\n";
				//verify its own result 
				self -> mail(matrixA,matrixB,matrixC,N).send(self);

			},

					// 3rd handler: CPU atom + matrices + N
					[=](const std::vector<int>& matrixA,
							const std::vector<int>& matrixB,
							const std::vector<int>& matrixC,
							int N) {

						using clock = std::chrono::high_resolution_clock;

						auto start = clock::now();

						//std::cout << "GPU ACTOR verifying\n";

						std::vector<int> result(N * N);

						serial_matrix_multiply(matrixA, matrixB, result, N);

						if (result == matrixC) {
							std::cout << "actor with id " << self->state().id
								<< " references match\n";
						} else {
							std::cout << "actor with id " << self->state().id
								<< " references did not match\n";
						}

						auto end = clock::now();

						auto ms =
							std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

						std::cout << "[TIMING] verification took "
							<< ms << " ms (actor id "
							<< self->state().id << ")\n";

						// signal exit actor and quit
						self->mail(1).send(exit_actor);
						self->quit();

					}
	};
}





// this actor will not verify its results
// great for performance analysis 
caf::behavior mmul_actor_fun_no_verify(
    caf::stateful_actor<mmul_actor_state>* self,
    caf::actor exit_actor,
    int N,
    caf::cuda::program_ptr program,
    caf::cuda::nd_range dims)
{

	//set the value of N correctly to overide the base option.	
	self->state().N = N;

	caf::cuda::manager& mgr = caf::cuda::manager::get();

	caf::actor scheduler = mgr.get_scheduler_actor();

	//send a launch token
	caf::cuda::token_ptr launch_token = caf::cuda::make_launch_token(
			program,
			dims,
			0,
			"hello",
			self
			);
	self -> mail(launch_token).send(scheduler);

	return {

		[=] (caf::cuda::response_token_ptr res_token) {

			if (res_token -> getType() == LAUNCH_RESPONSE) {
				//std::cout << "GPU ACTOR RECEIVED PERMISSION TO LAUNCH\n"; 
				//assume N = 1024
				int N = self -> state().N;
				std::vector<int> matrix1(N*N);
				matrix1.reserve(N);
				std::vector<int> matrix2(N*N);
				matrix2.reserve(N);

				//std::cout << "GPU ACTOR sending data to compute\n";
				self -> mail(matrix1,matrix2,res_token,N).send(self);

			}
			else {
				std::cout << "Got a memory response token\n";
			}
			//token should drop out of scope now, triggering a response 
		},

			// 2nd handler: GPU atom + matrices + N, launches a kenrel and sends its result to itself for verification
			[=](const std::vector<int>& matrixA,
					const std::vector<int>& matrixB,
					const caf::cuda::response_token_ptr& res_token, int N) {


				//caf::cuda::kernel_launch_token kernelToken = caf::intrusive_ptr_cast<caf::cuda::token_ptr>(kToken);

				//caf::cuda::launch_response_token& kt =
				//	    static_cast<caf::cuda::launch_response_token&>(*kToken);

				//std::cout << "GPU ACTOR  computing\n";
				caf::cuda::manager& mgr = caf::cuda::manager::get();

				//create program and dims   
				//create args
				auto arg1 = caf::cuda::create_in_arg(matrixA);
				auto arg2 = caf::cuda::create_in_arg(matrixB);
				auto arg3 = caf::cuda::create_out_arg(N*N);
				auto arg4 = caf::cuda::create_in_arg(N);

				auto tempC = mmul.run(program,dims,res_token,arg1,arg2,arg3,arg4);
				std::vector<int> matrixC = caf::cuda::extract_vector<int>(tempC);

				//std::cout << "GPU ACTOR done  computing\n";
				// signal exit actor and quit
				self->mail(1).send(exit_actor);
				self->quit();

			}

				
	};
}



// Stateful actor behavior
// this actor does not invoke the scheduler at all 
caf::behavior mmul_actor_fun_no_schedule(
    caf::stateful_actor<mmul_actor_state>* self,
    caf::actor exit_actor,
    int N,
    caf::cuda::program_ptr program,
    caf::cuda::nd_range dims) {

    self->state().N = N;

    std::vector<int> matrix1(N * N);
    std::vector<int> matrix2(N * N);

    // send initial mail to self
    self->mail(matrix1, matrix2, N).send(self);

    return {
        // GPU atom + matrices + N
        [=](const std::vector<int>& matrixA,
            const std::vector<int>& matrixB,
            int N_local) {   // avoid shadowing outer N
            
	    
    	    std::cout << "Hello\n";
	    caf::cuda::manager& mgr = caf::cuda::manager::get();

            auto arg1 = caf::cuda::create_in_arg(matrixA);
            auto arg2 = caf::cuda::create_in_arg(matrixB);
            auto arg3 = caf::cuda::create_out_arg(N_local * N_local);
            auto arg4 = caf::cuda::create_in_arg(N_local);

            auto tempC = mmul.run(program, dims, self->state().id, arg1, arg2, arg3, arg4);
            std::vector<int> matrixC = caf::cuda::extract_vector<int>(tempC);

	    self->mail(1).send(exit_actor);
            self->quit();
        },

        // CPU verification
        [=](const std::vector<int>& matrixA,
            const std::vector<int>& matrixB,
            const std::vector<int>& matrixC,
            int N_local) {

            std::vector<int> result(N_local * N_local);
            serial_matrix_multiply(matrixA, matrixB, result, N_local);

            if (result == matrixC) {
                std::cout << "actor with id " << self->state().id << " references match\n";
            } else {
                std::cout << "actor with id " << self->state().id << " references did not match\n";
            }

            self->quit();
        }
    };
}










void run_mmul_test(caf::actor_system& sys, int matrix_size, int num_actors) {
  if (num_actors < 1) {
    std::cerr << "[ERROR] Number of actors must be >= 1\n";
    return;
  }

  caf::cuda::manager& mgr = caf::cuda::manager::get();

  //change the scheduler to core_usage
  anon_mail(
	caf::cuda::make_behavior_token("core_usage")
	).send(mgr.get_scheduler_actor());

  // CREATE ONCE
  auto program =
      mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

  const int THREADS = 32;
  const int BLOCKS = (matrix_size + THREADS - 1) / THREADS;
  caf::cuda::nd_range dims(
      BLOCKS, BLOCKS, 1,
      THREADS, THREADS, 1);

  caf::actor exit_actor = sys.spawn(exit_actor_fun, num_actors);

  for (int i = 0; i < num_actors; ++i) {
    /*
    sys.spawn(
        mmul_actor_fun,
        exit_actor,
        matrix_size,
        program,
        dims);
    */
      sys.spawn(
        mmul_actor_fun_no_verify,
        exit_actor,
        matrix_size,
        program,
        dims);
    
  }

  sys.await_all_actors_done();
}


void run_mmul_test_no_scheduler(caf::actor_system& sys, int matrix_size, int num_actors) {
  if (num_actors < 1) {
    std::cerr << "[ERROR] Number of actors must be >= 1\n";
    return;
  }

  caf::cuda::manager& mgr = caf::cuda::manager::get();

  /*
  //change the scheduler to core_usage
  anon_mail(
	caf::cuda::make_behavior_token("core_usage")
	).send(mgr.get_scheduler_actor());

  */


  // CREATE ONCE
  auto program =
      mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

  const int THREADS = 32;
  const int BLOCKS = (matrix_size + THREADS - 1) / THREADS;
  caf::cuda::nd_range dims(
      BLOCKS, BLOCKS, 1,
      THREADS, THREADS, 1);

  caf::actor exit_actor = sys.spawn(exit_actor_fun, num_actors);

  for (int i = 0; i < num_actors; ++i) {
    
	sys.spawn(
        mmul_actor_fun_no_verify,
        exit_actor,
        matrix_size,
        program,
        dims);
    
  
	/*  
sys.spawn(
        mmul_actor_fun,
        exit_actor,
        matrix_size,
        program,
        dims);
  */
  }

  sys.await_all_actors_done();
}

void run_mmul_test_no_scheduler_actor(caf::actor_system& sys, int matrix_size, int num_actors) {
    if (num_actors < 1) {
        std::cerr << "[ERROR] Number of actors must be >= 1\n";
        return;
    }

    caf::cuda::manager& mgr = caf::cuda::manager::get();

    // CREATE ONCE
    auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

    const int THREADS = 32;
    const int BLOCKS = (matrix_size + THREADS - 1) / THREADS;
    caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

    caf::actor exit_actor = sys.spawn(exit_actor_fun, num_actors);

    for (int i = 0; i < num_actors; ++i) {
        sys.spawn(
            mmul_actor_fun_no_schedule,
            exit_actor,
            matrix_size,
            program,
            dims
        );
    }

    sys.await_all_actors_done();
}


template <class Fn>
double time_run(Fn&& fn) {
  auto start = std::chrono::steady_clock::now();
  fn();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  return elapsed.count();
}
void run_mmul_scaling_tests(caf::actor_system& sys,
                            caf::cuda::manager_config man_config) {
    const int max_size   = 1024;
    const int min_actors = 1;
    const int max_actors = 1024;

    std::vector<int> matrix_sizes = {10};
    for (int s = 32; s <= max_size; s *= 2)
        matrix_sizes.push_back(s);

    std::vector<int> actor_counts;
    for (int a = min_actors; a <= max_actors; a *= 2)
        actor_counts.push_back(a);

    std::cout << "=== MMUL Scaling Tests ===\n";
    std::cout << "Format:\n";
    std::cout << "scheduler matrix_size actors time_seconds\n";

    for (int size : matrix_sizes) {
        for (int actors : actor_counts) {

            /* ================= Scheduler-enabled (core_usage) ================= */
            caf::cuda::manager::init(sys, man_config); // green-light scheduler enabled
            std::cout << "\n[RUN] scheduler=core_usage "
                      << "matrix_size=" << size
                      << " actors=" << actors << "\n";

            double core_usage_time = time_run([&] {
                run_mmul_test(sys, size, actors); // uses mmul_actor_fun_no_verify
            });

            std::cout << std::fixed << std::setprecision(6)
                      << "RESULT core_usage "
                      << size << " "
                      << actors << " "
                      << core_usage_time << "\n";

            caf::cuda::manager::shutdown(); // make sure manager is cleaned up

            /* ================= Scheduler-disabled actor (still uses green-light) ================= */
            caf::cuda::manager::init(sys, man_config); // init with scheduler
            std::cout << "\n[RUN] scheduler=green_light_only "
                      << "matrix_size=" << size
                      << " actors=" << actors << "\n";

            double green_light_time = time_run([&] {
                run_mmul_test_no_scheduler(sys, size, actors); // your previous "no scheduler" actor
            });

            std::cout << std::fixed << std::setprecision(6)
                      << "RESULT green_light_only "
                      << size << " "
                      << actors << " "
                      << green_light_time << "\n";

            caf::cuda::manager::shutdown();

            /* ================= No scheduler at all actor ================= */
            caf::cuda::manager_config no_sched_config(false); // disable scheduler
            caf::cuda::manager::init(sys, no_sched_config);
            std::cout << "\n[RUN] scheduler=none "
                      << "matrix_size=" << size
                      << " actors=" << actors << "\n";

            double no_scheduler_time = time_run([&] {
                run_mmul_test_no_scheduler_actor(sys, size, actors); // mmul_actor_fun_no_schedule
            });

            std::cout << std::fixed << std::setprecision(6)
                      << "RESULT none "
                      << size << " "
                      << actors << " "
                      << no_scheduler_time << "\n";

            caf::cuda::manager::shutdown();
        }
    }

    std::cout << "\n=== MMUL Scaling Tests Complete ===\n";
}
void run_mmul_mixed_batch_one_mode(
    caf::actor_system& sys,
    const std::vector<int>& sizes,
    int num_actors,
    bool use_scheduler_actor,
    bool use_core_usage_behavior,
    bool randomize = false)
{
    caf::cuda::manager& mgr = caf::cuda::manager::get(); // SAFE NOW

    if (use_scheduler_actor && use_core_usage_behavior) {
        anon_mail(caf::cuda::make_behavior_token("core_usage"))
            .send(mgr.get_scheduler_actor());
    }

    auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

    caf::actor exit_actor = sys.spawn(exit_actor_fun, num_actors);

    std::mt19937 rng(123456);
    std::uniform_int_distribution<size_t> dist(0, sizes.size() - 1);

    const int THREADS = 32;

    for (int i = 0; i < num_actors; ++i) {
        int N = randomize ? sizes[dist(rng)] : sizes[i % sizes.size()];
        int BLOCKS = (N + THREADS - 1) / THREADS;

        caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

        if (use_scheduler_actor) {
            sys.spawn(
                mmul_actor_fun_no_verify,
                exit_actor,
                N,
                program,
                dims);
        } else {
            sys.spawn(
                mmul_actor_fun_no_schedule,
                exit_actor,
                N,
                program,
                dims);
        }
    }

    sys.await_all_actors_done();
}




void run_mmul_mixed_batch_comparison(
    caf::actor_system& sys,
    const std::vector<int>& sizes,
    int num_actors)
{
    std::cout << "\n=== MMUL Mixed-Size Batch Comparison ===\n";
    std::cout << "scheduler actors sizes time_seconds\n\n";

    /* ================= core_usage ================= */
    {
        caf::cuda::manager_config cfg(true);
        caf::cuda::manager::init(sys, cfg);

        double t = time_run([&] {
            run_mmul_mixed_batch_one_mode(
                sys, sizes, num_actors,
                /*use_scheduler_actor=*/true,
                /*use_core_usage_behavior=*/true);
        });

        std::cout << "RESULT core_usage "
                  << num_actors << " "
                  << t << "\n";

        caf::cuda::manager::shutdown();
    }

    /* ================= green-light only ================= */
    {
        caf::cuda::manager_config cfg(true);
        caf::cuda::manager::init(sys, cfg);

        double t = time_run([&] {
            run_mmul_mixed_batch_one_mode(
                sys, sizes, num_actors,
                /*use_scheduler_actor=*/true,
                /*use_core_usage_behavior=*/false);
        });

        std::cout << "RESULT green_light_only "
                  << num_actors << " "
                  << t << "\n";

        caf::cuda::manager::shutdown();
    }

    /* ================= no scheduler ================= */
    {
        caf::cuda::manager_config cfg(false);
        caf::cuda::manager::init(sys, cfg);

        double t = time_run([&] {
            run_mmul_mixed_batch_one_mode(
                sys, sizes, num_actors,
                /*use_scheduler_actor=*/false,
                /*use_core_usage_behavior=*/false);
        });

        std::cout << "RESULT none "
                  << num_actors << " "
                  << t << "\n";

        caf::cuda::manager::shutdown();
    }

    std::cout << "\n=== Comparison Complete ===\n";
}



void caf_main(caf::actor_system& sys) {
  
//	caf::cuda::manager_config man_config(true); //turns the scheduler on
//	caf::cuda::manager::init(sys,man_config);
       //	run_mmul_test(sys,256,1000);	
//	run_mmul_scaling_tests(sys,man_config);

   std::vector<int> sizes = {32, 64, 128, 256, 512, 1024};
    const int num_actors = 200;
 // Option A: round-robin distribution (deterministic)
    run_mmul_mixed_batch_comparison(sys, sizes, num_actors);    

//tests will delete the old manager so will have to reinit if you do this 
	//in conjunction with each other	
	//caf::cuda::manager::init(sys,man_config);
}




CAF_MAIN()
