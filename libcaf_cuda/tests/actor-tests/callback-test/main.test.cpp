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
    // 1st handler: Just int N, and who to send the matrices to
    [=](int N, std::vector<caf::actor> receivers) {

        caf::cuda::manager& mgr = caf::cuda::manager::get();
        //create the program and configure the dimesnions of the kernel
        auto program = mgr.create_program_from_fatbin("../generate_random_matrix.fatbin","generate_random_matrix");
	int THREADS = 256;
	int BLOCKS = (N*N + THREADS - 1) / THREADS;
  	caf::cuda::nd_range dim(BLOCKS,1, 1, THREADS,1, 1);

	//tag the arguments so that caf::cuda knows what to do with them	
         auto arg1 = caf::cuda::create_out_arg(N*N); //output buffer indicate its size, caf::cuda will handle the rest
          auto arg2 = caf::cuda::create_in_arg(N*N); //matrix size
          auto arg3 = caf::cuda::create_in_arg(rand()); //seed
	  auto arg4 = caf::cuda::create_in_arg(9999); //max valux
	  
          auto arg3B = caf::cuda::create_in_arg(rand()); //seed
	  int device_number= 74; //arbitary number to show that 
				 //can give illusion of selecting gpus that are
				 //not there


	  //launch kernels and collect their outputs
	  auto tempA = randomMatrix.run_async(program,dim, self -> state().id,0,device_number,arg1,arg2,arg3,arg4);
	  auto tempB = randomMatrix.run_async(program,dim, self -> state().id,0,device_number,arg1,arg2,arg3B,arg4);
	  caf::cuda::mem_ptr<int> matrixA =  std::get<0>(tempA);
	  caf::cuda::mem_ptr<int> matrixB = std::get<0>(tempB);

	  //ensure the data is actually done being worked on
	  matrixA -> synchronize();
	  matrixB -> synchronize();

	  std::cout << "Broadcasting\n";
	  //broadcast the result out to receviers.
	  for (auto actor: receivers) {
	  
		  self->mail(3,matrixA,matrixB,N,device_number).send(actor);
	  }

    },

    // 2nd handler: GPU atom + matrices + N, launches a kenrel and sends its result to itself for verification
    [=](const caf::cuda::mem_ptr<int> matrixA,
        const caf::cuda::mem_ptr<int> matrixB, int N,int device_number) {
 

  caf::cuda::manager& mgr = caf::cuda::manager::get();

  //create program and dims   
  auto program = mgr.create_program_from_cubin("../mmul.cubin","matrixMul");
  const int THREADS = 32;
  const int BLOCKS = (N + THREADS - 1) / THREADS;
  caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

    //create args
    auto arg1 = matrixA;
    auto arg2 = matrixB;
    auto arg3 = caf::cuda::create_out_arg(N*N);
    auto arg4 = caf::cuda::create_in_arg(N);


    auto tempC = mmulAsync.run(program,dims,self -> state().id,0,device_number,arg1,arg2,arg3,arg4);

    std::vector<int> matrix1 = matrixA -> copy_to_host();
    std::vector<int> matrix2 = matrixB -> copy_to_host();    
    std::vector<int> matrixC = caf::cuda::extract_vector<int>(tempC,2);

    //verify its own result 
    self -> mail(matrix1,matrix2,matrixC,N).send(self);

    },

 // 3nd handler: GPU atom + matrices + N, launches a kenrel using shared memory and sends its result to itself for verification
    [=](int x,const caf::cuda::mem_ptr<int> matrixA,
        const caf::cuda::mem_ptr<int> matrixB, int N,int device_number) {
 

  caf::cuda::manager& mgr = caf::cuda::manager::get();

  //create program and dims   
  auto program = mgr.create_program_from_cubin("../shared_mmul.cubin","matrixMul");
  const int THREADS = 32;
  const int BLOCKS = (N + THREADS - 1) / THREADS;
  caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

  int shared_mem = 8192; //we need 8KB of shared memory here
    //create args
    auto arg1 = matrixA;
    auto arg2 = matrixB;
    auto arg3 = caf::cuda::create_out_arg(N*N);
    auto arg4 = caf::cuda::create_in_arg(N);


    auto tempC = mmulAsync.run(program,dims,self -> state().id,shared_mem,device_number,arg1,arg2,arg3,arg4);

    std::vector<int> matrix1 = matrixA -> copy_to_host();
    std::vector<int> matrix2 = matrixB -> copy_to_host();    
    std::vector<int> matrixC = caf::cuda::extract_vector<int>(tempC,2);

    //verify its own result 
    self -> mail(matrix1,matrix2,matrixC,N).send(self);

    },



    // 3rd handler: CPU atom + matrices + N
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

  // Spawn num_actors actors running the mmul behavior
  std::vector<caf::actor> actors;
  actors.reserve(num_actors);
  for (int i = 0; i < num_actors; ++i) {
    actors.push_back(sys.spawn(mmul_async_actor_fun));
  }

  // Actor 0 generates matrices and broadcasts to others 
  caf::anon_mail(matrix_size, actors).send(actors[0]);

   sys.await_all_actors_done();
}


void caf_main(caf::actor_system& sys) {
  caf::cuda::manager::init(sys);

  //run_async_mmul_test(sys,100,1);

 
}



CAF_MAIN()
