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





struct mmul_actor_state {
  static inline const char* name = "my_actor";
  int last_N = 0; // example state variable
  int id = rand(); // an actor id
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






// Stateful actor behavior
caf::behavior mmul_actor_fun(
    caf::stateful_actor<mmul_actor_state>* self,
    caf::actor exit_actor,
    int N,
    caf::cuda::program_ptr program,
    caf::cuda::nd_range dims)
{


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

    int finished_stage = 0;

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
	//caf::cuda::manager::get().send_scheduler_actor_message(tok);
    
	//forcefully send to the first scheduler actor
	caf::actor scheduler = caf::cuda::manager::get().get_scheduler_actor();
	anon_mail(tok).send(scheduler);

    };

    // fire all three tokens (scheduler will reply with response_token on grants)
    launch(p1, "stage1");
    launch(p2, "stage2");
    launch(p3, "stage3");

    return {

        // handle response tokens by name — opaque to reclaim payload
        [=](caf::cuda::response_token_ptr res_token) mutable {

            const auto& stage = res_token->name();

	    if (res_token->getType() == LAUNCH_RESPONSE) {

		    // --------------------- Stage 1: init_denominators ---------------------
		    if (stage == "stage1") {
			    // allocate device buffer for denominators (persist in state)

			    //std::cout << "Starting stage 1\n";
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

			    //std::cout << "Finished stage 1\n";
			    // stage1 intentionally no checks — data may contain zeros
			    
			    self->state().finished_stage++;
			    return;
		    }

		    // --------------------- Stage 2: perform_division ---------------------
		    if (stage == "stage2") {
			    // allocate device buffer for results (persist in state)
			    std::vector<float> buffer1(n);

			    //std::cout << "Starting stage 2\n";
			    self->state().d_results = div_cmd.transfer_memory(res_token,out<float>{buffer1});

			    // create a host numerators vector (all ones)
			    std::vector<float> h_nums(n, 1.0f);

			    // transfer numerators to device on the token's stream/device
			    // transfer_memory returns a caf::cuda::mem_ptr<float>
			    auto d_nums = div_cmd.transfer_memory(res_token, in_out<float>{h_nums});

			    // run division kernel on the token's stream/device:
			    // kernel signature: (float* numerators, float* denominators, float* results, int n)
			    
			    if (self->state().d_denoms == nullptr) {
			    
				    std::cout << "Error with pipeline actor d_denoms is nullptr\n";
			    
			    }


			    if (d_nums == nullptr) {
			    
				    std::cout << "Error with pipeline actor d_denoms is nullptr\n";
			    
			    }

			    
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

			    self->state().finished_stage++;
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
			    //std::cout << "[pipeline] completed, sum = " << final_sum << "\n";

			    anon_mail(1).send(supervisor);

			    // quit the pipeline actor
			    self->state().finished_stage++;
			    self->quit();
			    return;
		    }
	    }

	    else if (res_token->getType() == TRANSFER) {
	   
		    std::cout << "Got a transfer token\n";

		   if (stage == "stage1") {
			   res_token->release(); // no dependencies at this point clear to continue
		   	   return;
		   } 

		   else if (stage == "stage2") {
		  
			   //at this point the d_results needs to be transfer over to the other device
			   
		           std::cout << "Transfering at stage 2\n";
			      if (self->state().d_denoms == nullptr) {
			    
				    std::cout << "Error with pipeline actor d_denoms is nullptr during transfer\n";
				    std::cout << "Completed stage is " << self->state().finished_stage << "\n";
			    
			    }
			   
			   
			   
			   //TODO FIX SEGFAULT TRIGGERED BY THIS LINE 
			    in_out<float> temp_buffer{self->state().d_denoms -> copy_to_host()};

			    self->state().d_denoms = div_cmd.transfer_memory(res_token,temp_buffer);
			    //all done
			    res_token->release();
		   	    return;
		   }

		   else if (stage == "stage3") {
		   
			   //at this point d_results needs to be copied over to the new GPU   
			    self->state().d_results = div_cmd.transfer_memory(res_token,in_out<float>{self->state().d_results->copy_to_host()});
			    //all done
			    res_token->release();
			    return;
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
//As it turns out pipeline actor does not do enough work in order to convince the GPUs
//that it should even attempt to migrate it
void run_load_balance_test_with_dependencies(
    caf::actor_system& sys,
    const int n,
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
    caf::cuda::program_ptr p1 = caf::cuda::manager::get().create_program_from_cubin("../fault.cubin","init_denominators");
    caf::cuda::program_ptr p2 = caf::cuda::manager::get().create_program_from_cubin("../fault.cubin","perform_division");
    caf::cuda::program_ptr p3 = caf::cuda::manager::get().create_program_from_cubin("../fault.cubin","sum_results");






    auto t_start = std::chrono::steady_clock::now();

for (int i = 0; i < num_actors; ++i) {
	sys.spawn(pipeline_actor, exit_actor, p1, p2, p3, n);	
}
  
	  //this time the gpu actors can figure out how to send tokens to the correct GPU scheduler
          sys.await_all_actors_done();
}



struct mmul_async_actor_state {
  static inline const char* name = "mmul_actor";

  int N = 0;
  int id = rand();

  // timing / bookkeeping only
  std::chrono::high_resolution_clock::time_point start_time;
  int times = 0;

  // --- mmul_async state (added) -------------------------------------------
  caf::cuda::mem_ptr<int> d_genA;                // device buffer for generated A
  caf::cuda::mem_ptr<int> d_genB;                // device buffer for generated B
  bool have_genA = false;
  bool have_genB = false;
};



//we intentionally send to only 1 actor to force load balancing and also
//see what happens if an actor gets a request that it is not responsible for
caf::behavior mmul_async_actor_fun(caf::stateful_actor<mmul_async_actor_state>* self,
		caf::actor exit_actor) {
  return {

    // ------------------------------------------------------------------
    // 1) Initial request: generate two matrices
    // ------------------------------------------------------------------
    [=](int N) {
      caf::cuda::manager& mgr = caf::cuda::manager::get();

      self->state().N = N;
      self->state().have_genA = false;
      self->state().have_genB = false;
      self->state().d_genA = nullptr;
      self->state().d_genB = nullptr;

      // Explicit generator launch configuration
      const int THREADS = 256;
      const int BLOCKS  = (N * N + THREADS - 1) / THREADS;
      caf::cuda::nd_range gen_range(BLOCKS, 1, 1,
                                    THREADS, 1, 1);

      auto gen_program =
        mgr.create_program_from_fatbin(
          "../generate_random_matrix.fatbin",
          "generate_random_matrix");

      auto send_launch = [&](const std::string& name) {
        auto tok = caf::cuda::make_launch_token(
                     gen_program,
                     gen_range,
                     sizeof(int) * N * N,
                     name,
                     self
                   );
        anon_mail(tok).send(mgr.get_scheduler_actor());
      };

      send_launch("genA");
      send_launch("genB");
    },

    // ------------------------------------------------------------------
    // 2) Handle scheduler response tokens
    // ------------------------------------------------------------------
[=](caf::cuda::response_token_ptr res_token) mutable {
  try {
    const auto type = res_token->getType();
    const auto name = res_token->name();
    int N = self->state().N;

    // ----------------------------
    // TRANSFER handling
    // ----------------------------
    if (type == TRANSFER) {
      if (name == "genA" && self->state().d_genA) {
        auto host_copy = self->state().d_genA->copy_to_host();
        self->state().d_genA =
          randomMatrix.transfer_memory(res_token, in_out<int>{host_copy});
      }

      if (name == "genB" && self->state().d_genB) {
        // Transfer B
        auto host_copyB = self->state().d_genB->copy_to_host();
        self->state().d_genB =
          randomMatrix.transfer_memory(res_token, in_out<int>{host_copyB});

        // ALSO transfer A to the same device as B
        if (self->state().d_genA) {
          auto host_copyA = self->state().d_genA->copy_to_host();
          self->state().d_genA =
            randomMatrix.transfer_memory(res_token, in_out<int>{host_copyA});
        }
      }

      res_token->release();
      return;
    }

    // ----------------------------
    // LAUNCH_RESPONSE handling
    // ----------------------------
    if (type != LAUNCH_RESPONSE)
      return;

    // Generator completion (genA / genB)
    if (name == "genA" || name == "genB") {
      auto out_arg  = caf::cuda::create_out_arg(N * N);
      auto size_arg = caf::cuda::create_in_arg(N * N);
      auto seed_arg = caf::cuda::create_in_arg(rand());
      auto maxval_arg = caf::cuda::create_in_arg(9999);

      const int THREADS = 256;
      const int BLOCKS  = (N * N + THREADS - 1) / THREADS;

      caf::cuda::nd_range gen_range(BLOCKS,1,1, THREADS,1,1);

      auto gen_program =
        caf::cuda::manager::get().create_program_from_fatbin(
          "../generate_random_matrix.fatbin",
          "generate_random_matrix");

      auto result = randomMatrix.run_async(
        gen_program, gen_range, res_token,
        out_arg, size_arg, seed_arg, maxval_arg);

      auto device_buffer = std::get<0>(result);

      if (name == "genA") {
	      self->state().d_genA = device_buffer;
      	      self->state().have_genA = true;
      
      }
      else {
	      self->state().d_genB = device_buffer;
      	      self->state().have_genB = true;
      }

      // After handling genA / genB completion
if (self->state().have_genA && self->state().have_genB) {
    const int THREADS_M = 32;
    int BLOCKS_M = (N + THREADS_M - 1) / THREADS_M;

    caf::cuda::nd_range mmul_range(
      BLOCKS_M, BLOCKS_M, 1,
      THREADS_M, THREADS_M, 1
    );

    auto mmul_program =
      caf::cuda::manager::get().create_program_from_cubin(
        "../mmul.cubin",
        "matrixMul"
      );

    // Create a launch token for mmul
    auto mmul_token = caf::cuda::make_launch_token(
      mmul_program,
      mmul_range,
      sizeof(int) * N * N,
      "mmul",
      self
    );

    // Send the launch token to the scheduler actor
    anon_mail(mmul_token)
      .send(caf::cuda::manager::get().get_scheduler_actor());
  }


      res_token->release();
      return;
    }

    // ----------------------------
    // mmul completion / kernel launch
    // ----------------------------
    if (name == "mmul") {
      const int THREADS_M = 32;
      int BLOCKS_M = (N + THREADS_M - 1) / THREADS_M;

      caf::cuda::nd_range mmul_range(BLOCKS_M, BLOCKS_M, 1,
                                    THREADS_M, THREADS_M, 1);

      auto mmul_program =
        caf::cuda::manager::get().create_program_from_cubin(
          "../mmul.cubin",
          "matrixMul");

      auto outC = caf::cuda::create_out_arg(N * N);
      auto inN  = caf::cuda::create_in_arg(N);

      auto result = mmulAsync.run(
        mmul_program,
        mmul_range,
        res_token,
        self->state().d_genA,
        self->state().d_genB,
        outC,
        inN);

      std::vector<int> matrixC = caf::cuda::extract_vector<int>(result, 2);

      self->state().have_genA = false;
      self->state().have_genB = false;
      self->state().d_genA.reset();
      self->state().d_genB.reset();

      res_token->release();
      self->mail(1).send(exit_actor);
    }

  } catch (std::exception& e) {
    std::cerr << "*** Caught exception: " << e.what() << "\n";

    if (self->state().d_genA)
      std::cerr << "d_genA deviceID: " << self->state().d_genA->deviceID() << "\n";
    if (self->state().d_genB)
      std::cerr << "d_genB deviceID: " << self->state().d_genB->deviceID() << "\n";
    if (res_token)
      std::cerr << "res_token deviceID: " << res_token->getDeviceNumber() << "\n";
  }
}

  };
}




//this test is meant to demonstrate the fact that scheduler actors can 
//migrate work to correct load imbalance
//the sizes should be large enough such that the tests exceed 4-5 seconds in total
//otherwise the schedulers wont care to do this fast enough
void run_load_balance_test_with_large_dependencies(
    caf::actor_system& sys,
    const int n,
    int num_actors,
    bool randomize = false)
{
    caf::cuda::manager& mgr = caf::cuda::manager::get(); 


    //set the behaviors of each scheduler actor
    for (int i = 0; i < mgr.get_num_devices();i++) {
	    mgr.send_scheduler_actor_message("multilevel",i);
    }


    caf::actor exit_actor = sys.spawn(exit_actor_fun, num_actors);






    auto t_start = std::chrono::steady_clock::now();

for (int i = 0; i < num_actors; ++i) {
	caf::actor a = sys.spawn(mmul_async_actor_fun, exit_actor);
	anon_mail(n).send(a);	
}
  
	  //this time the gpu actors can figure out how to send tokens to the correct GPU scheduler
          sys.await_all_actors_done();
}



void caf_main(caf::actor_system& sys) {
  
	caf::cuda::manager_config man_config(true); //turns the scheduler on
	caf::cuda::manager::init(sys,man_config);
	
	
	//no dependencies
//	std::vector<int> sizes = {32, 64, 128, 256, 512, 1024,2048,4096};
//	const int num_actors = 2000;
//	run_load_balance_test(sys,sizes,num_actors);


	//dependencies
	run_load_balance_test_with_large_dependencies(sys,1024,2000);


}




CAF_MAIN()
