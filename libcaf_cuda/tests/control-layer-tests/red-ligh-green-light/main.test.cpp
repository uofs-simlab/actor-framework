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
caf::behavior mmul_actor_fun(caf::stateful_actor<mmul_actor_state>* self) {
  

  caf::cuda::manager& mgr = caf::cuda::manager::get();
	
  	caf::actor scheduler = mgr.get_scheduler_actor();
	caf::cuda::token_ptr launch_token = caf::cuda::make_launch_token(self ->state().program,
			self -> state().dims,
			0,
			"hello",
			self
			);
	self -> mail(launch_token).send(scheduler);

	return {

	  [=] (caf::cuda::token_ptr launch_response_token) {
	 
		  std::cout << "GPU ACTOR RECEIVED PERMISSION TO LAUNCH\n"; 
		  //assume N = 1024
		  int N = self -> state().N;
		  std::vector<int> matrix1(N*N);
		  matrix1.reserve(N);
		  std::vector<int> matrix2(N*N);
		  matrix2.reserve(N);

		  std::cout << "GPU ACTOR sending data to compute\n";
		  self -> mail(matrix1,matrix2,N).send(self);
	 
		 //token should drop out of scope now, triggering a response 
	  
	  },

    // 2nd handler: GPU atom + matrices + N, launches a kenrel and sends its result to itself for verification
    [=](const std::vector<int> matrixA,
        const std::vector<int> matrixB, int N) {
 

		  std::cout << "GPU ACTOR  computing\n";
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

		  std::cout << "GPU ACTOR done  computing\n";
    //verify its own result 
    self -> mail(matrixA,matrixB,matrixC,N).send(self);

    },

    // 3rd handler: CPU atom + matrices + N
    [=](const std::vector<int>& matrixA,
        const std::vector<int>& matrixB, const std::vector<int> matrixC, int N) {
       
		  std::cout << "GPU ACTOR  verfiying\n";
	 std::vector<int> result(N*N);

	 serial_matrix_multiply(matrixA,matrixB,result,N);

	 if (result == matrixC) {
	 
		 std::cout << "actor with id " <<  self->state().id << " references match\n";
	 
	 }

	 else {
	    std::cout << "actor with id " <<  self->state().id << " references did not match\n";
	 }

	 self-> quit();

    }
  };
}



void run_mmul_test(caf::actor_system& sys, int matrix_size, int num_actors) {
  if (num_actors < 1) {
    std::cerr << "[ERROR] Number of actors must be >= 1\n";
    return;
  }

  while (true) {
  // Spawn num_actors actors running the mmul behavior
  std::vector<caf::actor> actors;
  actors.reserve(num_actors);
  for (int i = 0; i < num_actors; ++i) {
    actors.push_back(sys.spawn(mmul_actor_fun));
  }
  
  sleep(1);
  }

  //caf::anon_mail(matrix_size, actors).send(actors[0]);

   sys.await_all_actors_done();
}
// Stateful actor behavior
// tests if mem_ptr can be sent correctly
caf::behavior mmul_actor_fun2(caf::stateful_actor<mmul_actor_state>* self) {
  

  caf::cuda::manager& mgr = caf::cuda::manager::get();
/*		
  caf::cuda::token_ptr bad_launch_token = caf::cuda::make_launch_token(self ->state().program,
			self -> state().dims,
			0,
			"hello",
			self
			);

	self -> mail(bad_launch_token).send(scheduler);
			*/

  	caf::actor scheduler = mgr.get_scheduler_actor();
	caf::cuda::mem_ptr<int> launch_token = caf::cuda::make_mem_ptr(16);
			
	self -> mail(launch_token).send(scheduler);

	self -> mail("Stay alive\n").send(self);



	return {

		[=] (std::string hello){
		while(1) {
	
		std::cout << "Staying alive\n";
		sleep(1);
	}	
		},
	  [=] (caf::cuda::token_ptr launch_response_token) {
	 
		  std::cout << "GPU ACTOR RECEIVED PERMISSION TO LAUNCH\n"; 
		  //assume N = 1024
		  int N = self -> state().N;
		  std::vector<int> matrix1(N*N);
		  matrix1.reserve(N);
		  std::vector<int> matrix2(N*N);
		  matrix2.reserve(N);

		  std::cout << "GPU ACTOR sending data to compute\n";
		  self -> mail(matrix1,matrix2,N).send(self);
	 
		 //token should drop out of scope now, triggering a response 
	  
	  },

    // 2nd handler: GPU atom + matrices + N, launches a kenrel and sends its result to itself for verification
    [=](const std::vector<int> matrixA,
        const std::vector<int> matrixB, int N) {
 

		  std::cout << "GPU ACTOR  computing\n";
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

		  std::cout << "GPU ACTOR done  computing\n";
    //verify its own result 
    self -> mail(matrixA,matrixB,matrixC,N).send(self);

    },

    // 3rd handler: CPU atom + matrices + N
    [=](const std::vector<int>& matrixA,
        const std::vector<int>& matrixB, const std::vector<int> matrixC, int N) {
       
		  std::cout << "GPU ACTOR  verfiying\n";
	 std::vector<int> result(N*N);

	 serial_matrix_multiply(matrixA,matrixB,result,N);

	 if (result == matrixC) {
	 
		 std::cout << "actor with id " <<  self->state().id << " references match\n";
	 
	 }

	 else {
	    std::cout << "actor with id " <<  self->state().id << " references did not match\n";
	 }

	 self-> quit();

    }
  };
}





void caf_main(caf::actor_system& sys) {
  
	

	caf::cuda::manager_config man_config(true); //turns the scheduler on
	caf::cuda::manager::init(sys,man_config);

	 //caf::init_global_meta_objects<caf::id_block::cuda_control>();

	sys.spawn(mmul_actor_fun2);
  //run_mmul_test(sys,100,1);
}




CAF_MAIN()
