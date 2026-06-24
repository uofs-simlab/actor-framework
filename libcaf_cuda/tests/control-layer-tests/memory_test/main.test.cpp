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
  static inline const char* name = "my_actor";
  int N = 1024; // example state variable
  int id = rand(); // an actor id
  // per-actor timing start
  std::chrono::high_resolution_clock::time_point start_time;
  int times = 0;
  caf::cuda::program_ptr program = caf::cuda::manager::get().create_program_from_cubin("../mmul.cubin","matrixMul");
  int THREADS = 32;
  int BLOCKS = (N + THREADS - 1) / THREADS;
  caf::cuda::nd_range dims = caf::cuda::nd_range(BLOCKS,BLOCKS,1,THREADS,THREADS,THREADS);
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
caf::behavior mmul_actor_fun(caf::stateful_actor<mmul_actor_state>* self,caf::actor exit_actor,int N) {
 

	//set the value of N correctly to overide the base option.	
	self->state().N = N;

  caf::cuda::manager& mgr = caf::cuda::manager::get();
	
  	caf::actor scheduler = mgr.get_scheduler_actor();
	

	//send a memory transfer token
	int bytes = N*N * sizeof(int);
	caf::cuda::token_ptr memory_token = caf::cuda::make_memory_token(bytes,H2D,self);
	self -> mail(memory_token).send(scheduler);	

	//send a launch token
	caf::cuda::token_ptr launch_token = caf::cuda::make_launch_token(self ->state().program,
			self -> state().dims,
			0,
			"hello",
			self
			);
	self -> mail(launch_token).send(scheduler);

	return {

	  [=] (caf::cuda::response_token_ptr launch_response_token) {
	
		 if (launch_response_token -> getType() == LAUNCH_RESPONSE) {
		  //std::cout << "GPU ACTOR RECEIVED PERMISSION TO LAUNCH\n"; 
		  //assume N = 1024
		  int N = self -> state().N;
		  std::vector<int> matrix1(N*N);
		  matrix1.reserve(N);
		  std::vector<int> matrix2(N*N);
		  matrix2.reserve(N);

		  //std::cout << "GPU ACTOR sending data to compute\n";
		  self -> mail(matrix1,matrix2,N).send(self);
	 
		 }
		 else {
			 std::cout << "Got a memory response token\n";
		 }
		 //token should drop out of scope now, triggering a response 
	  },

    // 2nd handler: GPU atom + matrices + N, launches a kenrel and sends its result to itself for verification
    [=](const std::vector<int>& matrixA,
        const std::vector<int>& matrixB, int N) {
 

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

    auto tempC = mmul.run(program,dims,self -> state().id,arg1,arg2,arg3,arg4);
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



void run_mmul_test(caf::actor_system& sys, int matrix_size, int num_actors) {
  if (num_actors < 1) {
    std::cerr << "[ERROR] Number of actors must be >= 1\n";
    return;
  }

  int limit = 1;

  caf::actor exit_actor = sys.spawn(exit_actor_fun,num_actors);

  for (int i = 0; i < limit; i++) {
  // Spawn num_actors actors running the mmul behavior
  std::vector<caf::actor> actors;
  actors.reserve(num_actors);
  for (int i = 0; i < num_actors; ++i) {
    actors.push_back(sys.spawn(mmul_actor_fun,exit_actor,matrix_size));
  }


 // std::cout << actors.size() << "\n";

  }

   sys.await_all_actors_done();
}



//this test will spawn more actors over time to demonstrate 
//changing scheduling algorithims at runtime 
void run_red_light_green_light_test(caf::actor_system& sys, int matrix_size, int num_actors) {
  if (num_actors < 1) {
    std::cerr << "[ERROR] Number of actors must be >= 1\n";
    return;
  }

  std::cout << "Starting RED LIGHT GREEN LIGHT TEST\n";
  int limit = 10;

  caf::actor exit_actor = sys.spawn(exit_actor_fun,num_actors * limit);

  for (int i = 0; i < limit; i++) {
  // Spawn num_actors actors running the mmul behavior
  std::vector<caf::actor> actors;
  actors.reserve(num_actors);
  for (int i = 0; i < num_actors; ++i) {
    actors.push_back(sys.spawn(mmul_actor_fun,exit_actor,matrix_size));
  }


  sleep(1);
 // std::cout << actors.size() << "\n";

  }

   sys.await_all_actors_done();
}



void caf_main(caf::actor_system& sys) {
  
	

	caf::cuda::manager_config man_config(true); //turns the scheduler on
	caf::cuda::manager::init(sys,man_config);
       	run_mmul_test(sys,10,10);
	
	//tests will delete the old manager so will have to reinit if you do this 
	//in conjunction with each other	
	//caf::cuda::manager::init(sys,man_config);
	//run_red_light_green_light_test(sys,10,1000);
}




CAF_MAIN()
