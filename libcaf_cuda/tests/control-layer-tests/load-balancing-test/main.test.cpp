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
			
		//	std::cout << "Actors finished is " << self->state().completed << "\n";
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
    caf::cuda::nd_range dims,
    bool request
    )
{

	//set the value of N correctly to overide the base option.	
	self->state().N = N;

	caf::cuda::manager& mgr = caf::cuda::manager::get();

	caf::actor scheduler = mgr.get_scheduler_actor();

	if (request) {
		//send a launch token
		caf::cuda::token_ptr launch_token = caf::cuda::make_launch_token(
				program,
				dims,
				0,
				"hello",
				self
				);
		self -> mail(launch_token).send(scheduler);
	}
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


				caf::cuda::manager& mgr = caf::cuda::manager::get();

				//create program and dims   
				//create args
				auto arg1 = caf::cuda::create_in_arg(matrix1);
				auto arg2 = caf::cuda::create_in_arg(matrix2);
				auto arg3 = caf::cuda::create_out_arg(N*N);
				auto arg4 = caf::cuda::create_in_arg(N);

				auto tempC = mmul.run(program,dims,res_token,arg1,arg2,arg3,arg4);
			
			
				//mask the transfer back to the cpu for scheduler
				res_token -> release();
				std::vector<int> matrixC = caf::cuda::extract_vector<int>(tempC);

				//std::cout << "GPU ACTOR done  computing\n";
				// signal exit actor and quit
				self->mail(1).send(exit_actor);
				self->quit();

				//std::cout << "GPU ACTOR sending data to compute\n";
			//	self -> mail(matrix1,matrix2,res_token,N).send(self);

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
			
			
				//mask the transfer back to the cpu for scheduler
				res_token -> release();
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
            
	    
    	    //std::cout << "Hello\n";
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
        dims,
	true);
    
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
        dims,
	true);
    
  
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


//this test is meant to demonstrate the fact that scheduler actors can 
//migrate work to correct load imbalance
//the sizes should be large enough such that the tests exceed 4-5 seconds in total
//otherwise the schedulers wont care to do this fast enough
void run_load_balance_test(
    caf::actor_system& sys,
    const std::vector<int>& sizes,
    int num_actors,
    bool randomize = false)
{
    caf::cuda::manager& mgr = caf::cuda::manager::get(); 


    //set the behaviors of each scheduler actor
    for (int i = 0; i < mgr.get_num_devices();i++) {
	    mgr.send_scheduler_actor_message("multilevel",i);
    }

    auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

    caf::actor exit_actor = sys.spawn(exit_actor_fun, num_actors);

    std::vector<caf::cuda::token_ptr> tokens(num_actors);

    std::mt19937 rng(123456);
    std::uniform_int_distribution<size_t> dist(0, sizes.size() - 1);

    const int THREADS = 32;

    auto t_start = std::chrono::steady_clock::now();

for (int i = 0; i < num_actors; ++i) {
    int N = randomize ? sizes[dist(rng)] : sizes[i % sizes.size()];
    int BLOCKS = (N + THREADS - 1) / THREADS;

    caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1,
                             THREADS, THREADS, 1);

    caf::actor a = sys.spawn(
        mmul_actor_fun_no_verify,
        exit_actor,
        N,
        program,
        dims,
        false
    );

    tokens[i] = caf::cuda::make_launch_token(
        program,
        dims,
        0, // memory usage is zero for now, we still do not track it at all
        "hello",
        a
    );
}

auto t_end = std::chrono::steady_clock::now();

auto us = std::chrono::duration_cast<std::chrono::microseconds>(
              t_end - t_start
          ).count();

std::cout << "Actor spawn + token creation loop took "
          << us << " us\n";

   
	  //send the tokens to only 1 GPU and let them
	  //figure out that there is a load imbalance 
	  mgr.send_scheduler_actor_message(tokens);
          sys.await_all_actors_done();
}


//-------------------------------------try load balancing with actors with dependencies


using namespace caf;
using namespace std::chrono_literals;


// --- command runner types (put near top of file) -------------------------
using initCommand =
  caf::cuda::command_runner<caf::cuda::mem_ptr<float>, in<int>, in<unsigned long long>>;

using divCommand  =
  caf::cuda::command_runner<caf::cuda::mem_ptr<float>, caf::cuda::mem_ptr<float>, caf::cuda::mem_ptr<float>, in<int>>;

using sumCommand  =
  caf::cuda::command_runner<caf::cuda::mem_ptr<float>, caf::cuda::mem_ptr<float>, in<int>>;

// single instances (can be file-global)
static initCommand init_cmd;
static divCommand  div_cmd;
static sumCommand  sum_cmd;

// --- pipeline actor state (device buffers persist here) ------------------
struct pipeline_actor_state {
    int id = rand();

    // device-side buffers that must persist across stages:
    caf::cuda::mem_ptr<float> d_denoms;
    caf::cuda::mem_ptr<float> d_results;
    caf::cuda::mem_ptr<float> d_sum;
};

// --- corrected pipeline_actor -------------------------------------------
behavior pipeline_actor(caf::stateful_actor<pipeline_actor_state>* self,
                        actor supervisor,
                        caf::cuda::program_ptr p1,
                        caf::cuda::program_ptr p2,
                        caf::cuda::program_ptr p3,
                        int n)
{
    // host-side scratch (only used for post-stage2 NaN/Inf detection)
    std::vector<float> h_results;


    // nd_range used for all stages (adapt to your kernels as needed)
    caf::cuda::nd_range range{
        {(n + 255) / 256, 1, 1},
        {256, 1, 1}
    };

    // helper to create and send a launch token
    auto launch = [&](caf::cuda::program_ptr prog, const std::string& stage) {
        auto tok = make_launch_token(
            prog,
            range,
            /*memory_usage=*/static_cast<int>(sizeof(float) * n),
            stage,
            self,
            self->state().id  // dependency/demo id
        );

	//do not specifiy a device number to send it to, let it figure it out
	caf::cuda::manager::get().send_scheduler_actor_message(tok);
    };

    // fire all three tokens (scheduler will reply with response_token on grants)
    launch(p1, "stage1");
    launch(p2, "stage2");
    launch(p3, "stage3");

    return {

        // handle response tokens by name — opaque to reclaim payload
        [=](caf::cuda::response_token_ptr res_token) mutable {

            const auto& stage = res_token->name();

	    if (res_token->getType() == LAUNCH) {

		    // --------------------- Stage 1: init_denominators ---------------------
		    if (stage == "stage1") {
			    // allocate device buffer for denominators (persist in state)

			    std::cout << "Starting stage 1\n";
			    unsigned long long seed = static_cast<unsigned long long>(
					    std::chrono::high_resolution_clock::now().time_since_epoch().count()
					    );



			    out<float> buffer = caf::cuda::create_out_arg_with_size<float>(n);
			    self->state().d_denoms = init_cmd.transfer_memory(res_token,buffer);

			    // run kernel on the stream/device from res_token
			    // kernel signature: (float* denominators, int n, unsigned long long seed)
			    init_cmd.run_async(
					    p1,
					    range,
					    res_token,                            // uses token's stream/device
					    self->state().d_denoms,               // device buffer
					    caf::cuda::create_in_arg(n),          // n
					    caf::cuda::create_in_arg(seed)     // seed
					    );

			    std::cout << "Finished stage 1\n";
			    // stage1 intentionally no checks — data may contain zeros
			    return;
		    }

		    // --------------------- Stage 2: perform_division ---------------------
		    if (stage == "stage2") {
			    // allocate device buffer for results (persist in state)
			    std::vector<float> buffer1(n);

			    std::cout << "Starting stage 2\n";
			    self->state().d_results = div_cmd.transfer_memory(res_token,out<float>{buffer1});

			    // create a host numerators vector (all ones)
			    std::vector<float> h_nums(n, 1.0f);

			    // transfer numerators to device on the token's stream/device
			    // transfer_memory returns a caf::cuda::mem_ptr<float>
			    auto d_nums = div_cmd.transfer_memory(res_token, in_out<float>{h_nums});

			    // run division kernel on the token's stream/device:
			    // kernel signature: (float* numerators, float* denominators, float* results, int n)
			    div_cmd.run(
					    p2,
					    range,
					    res_token,
					    d_nums,
					    self->state().d_denoms,
					    self->state().d_results,
					    caf::cuda::create_in_arg(n)
				       );


			    //there could be a division by zero in here 
			    //but this is a load balancing test
			    //not a fault test,
			    //go see the fault tolerance test to see how thats handled	

			    /*
			    // extract the device results back to host for verification.
			    // extract_vector will synchronize as needed.
			    h_results = self->state().d_results -> copy_to_host();

			    // check for NaN/Inf AFTER the kernel finished
			    bool fault = false;
			    for (float v : h_results) {
				    if (!std::isfinite(v)) {
					    fault = true;
					    break;
				    }
			    }

			    if (fault) {
				    // inform supervisor and exit;
				    anon_mail(std::string("crash")).send(supervisor);
				    self->quit();
				    return;
			    }

			    // stage2 passed — keep d_results in state for stage3
			    */
			    return;
		    }

		    // --------------------- Stage 3: sum_results --------------------------
		    if (stage == "stage3") {
			    // allocate device scalar for sum result

			    //std::cout << "Starting stage 3\n";
			    std::vector<float> buffer1(1);

			    self->state().d_sum = div_cmd.transfer_memory(res_token,out<float>{buffer1});

			    // run reduction on the token's stream/device:
			    // kernel signature: (float* results, float* final_sum, int n)
			    sum_cmd.run(
					    p3,
					    range,
					    res_token,
					    self->state().d_results,
					    self->state().d_sum,
					    caf::cuda::create_in_arg(n)
				       );

			    // extract final scalar

			    std::vector<float> buf = self->state().d_sum -> copy_to_host();
			    float final_sum = buf[0];
			    std::cout << "[pipeline] completed, sum = " << final_sum << "\n";

			    anon_mail(1).send(supervisor);

			    // quit the pipeline actor
			    self->quit();
			    return;
		    }
	    }

	    else if (res_token->getType() == TRANSFER) {
	   
		    std::cout << "Got a transfer token\n";

		   if (stage == "stage1") {
			   res_token->release(); // no dependencies at this point clear to continue
		   } 

		   else if (stage == "stage2") {
		  
			   //at this point the d_results needs to be transfer over to the other device
			    
			    in_out<float> temp_buffer{self->state().d_results -> copy_to_host()};

			    self->state().d_results = div_cmd.transfer_memory(res_token,temp_buffer);
			    //all done
			    res_token->release();
		   
		   }

		   else if (stage == "stage3") {
		   
			   //at this point d_results needs to be copied over to the new GPU   
			    self->state().d_results = div_cmd.transfer_memory(res_token,in_out<float>{self->state().d_results->copy_to_host()});
			    //all done
			    res_token->release();
		   }

		   else { 
			   std::cout << "Error unrecognized transfer token\n";
		   }
		    
	    
	    }

            // unknown stage: ignore or log
            std::cerr << "[pipeline] received unknown response token: " << stage << "\n";
	}
    };
}


//this test is meant to demonstrate the fact that scheduler actors can 
//migrate work to correct load imbalance
//the sizes should be large enough such that the tests exceed 4-5 seconds in total
//otherwise the schedulers wont care to do this fast enough
void run_load_balance_test_with_dependencies(
    caf::actor_system& sys,
    const std::vector<int>& sizes,
    int num_actors,
    bool randomize = false)
{
    caf::cuda::manager& mgr = caf::cuda::manager::get(); 


    //set the behaviors of each scheduler actor
    for (int i = 0; i < mgr.get_num_devices();i++) {
	    mgr.send_scheduler_actor_message("multilevel",i);
    }

    auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

    caf::actor exit_actor = sys.spawn(exit_actor_fun, num_actors);

    std::vector<caf::cuda::token_ptr> tokens(num_actors);

    std::mt19937 rng(123456);
    std::uniform_int_distribution<size_t> dist(0, sizes.size() - 1);

    const int THREADS = 32;

    auto t_start = std::chrono::steady_clock::now();

for (int i = 0; i < num_actors; ++i) {
    int N = randomize ? sizes[dist(rng)] : sizes[i % sizes.size()];
    int BLOCKS = (N + THREADS - 1) / THREADS;

    caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1,
                             THREADS, THREADS, 1);

    caf::actor a = sys.spawn(
        mmul_actor_fun_no_verify,
        exit_actor,
        N,
        program,
        dims,
        false
    );

    tokens[i] = caf::cuda::make_launch_token(
        program,
        dims,
        0, // memory usage is zero for now, we still do not track it at all
        "hello",
        a
    );
}

auto t_end = std::chrono::steady_clock::now();

auto us = std::chrono::duration_cast<std::chrono::microseconds>(
              t_end - t_start
          ).count();

std::cout << "Actor spawn + token creation loop took "
          << us << " us\n";

   
	  //send the tokens to only 1 GPU and let them
	  //figure out that there is a load imbalance 
	  mgr.send_scheduler_actor_message(tokens);
          sys.await_all_actors_done();
}










void caf_main(caf::actor_system& sys) {
  
	caf::cuda::manager_config man_config(true); //turns the scheduler on
	caf::cuda::manager::init(sys,man_config);
	std::vector<int> sizes = {32, 64, 128, 256, 512, 1024,2048,4096};
	const int num_actors = 2000;
	run_load_balance_test(sys,sizes,num_actors);


}




CAF_MAIN()
