#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <caf/cuda/control-layer/all-control-layer.hpp>
#include <caf/component-actors/all-component-actors.hpp>
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
  caf::actor sync_actor;
  caf::actor mem_transfer_actor;
  caf::cuda::response_token_ptr r; //holding onto a response token is forbidden, as this can cause scheduler actor to never reply to itself
			       //however we are doing it to see if the sync_actor works DO NOT TRY IN production environment 
};




//commands classes used to launch kernels
using mmulCommand = caf::cuda::command_runner<in<int>,in<int>,out<int>,in<int>>;
using matrixGenCommand = caf::cuda::command_runner<out<int>,in<int>,in<int>,in<int>>;

using mmulAsyncCommand = caf::cuda::command_runner<caf::cuda::mem_ptr<int>,caf::cuda::mem_ptr<int>,caf::cuda::mem_ptr<int>,caf::cuda::mem_ptr<int>>;

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

	self->state().sync_actor = self -> spawn(caf::cuda::sync_actor_fun<int>);
	self->state().mem_transfer_actor = self -> spawn(caf::cuda::mem_transfer_actor_fun<int>);

       return {

    // 1. Handle response token
    [=](caf::cuda::response_token_ptr res_token) {
        std::cout << "Got response\n";

        if (res_token->getType() == LAUNCH_RESPONSE) {
            std::vector<int> matrix1(N*N);
            std::vector<int> matrix2(N*N);

            self->mail(matrix1, matrix2, res_token, N).send(self);

        } else {
            std::cout << "Got a memory response token\n";
        }
    },

    // 2. Handle memory buffers -> GPU
    [=](const std::vector<int>& matrixA,
        const std::vector<int>& matrixB,
        const caf::cuda::response_token_ptr& res_token,
        int N) {

        std::cout << "Working\n";
        caf::cuda::manager& mgr = caf::cuda::manager::get();

        auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");
        const int THREADS = 32;
        const int BLOCKS = (N + THREADS - 1) / THREADS;
        caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

        auto arg1 = mmul.transfer_memory(res_token, caf::cuda::create_in_arg(matrixA));
        auto arg2 = mmul.transfer_memory(res_token, caf::cuda::create_in_arg(matrixB));
        auto arg3 = mmul.transfer_memory(res_token, caf::cuda::create_out_arg(N*N));
        auto arg4 = mmul.transfer_memory(res_token, caf::cuda::create_in_arg(N));

        auto tempC = mmulAsync.run_async(program, dims, res_token, arg1, arg2, arg3, arg4);
        caf::cuda::mem_ptr<int> bufferA = std::get<0>(tempC);

        self->state().r = res_token; // hold token

        using namespace std::chrono_literals;
        self->mail(bufferA, res_token)
            .request(self->state().sync_actor, 10s)
            .then(
                [=](caf::cuda::mem_ptr<int> /*syncedA*/) {
                    auto bufferC = std::get<2>(tempC);
                    self->mail(bufferC)
                        .request(self->state().mem_transfer_actor, 10s)
                        .then(
                            [=](std::vector<int> matrixC) {
                                self->mail(matrixA,matrixB,matrixC, N).send(self);
                            },
                            [=](caf::error& err) {
                                std::cout << "Transfer C failed: " << to_string(err) << "\n";
                                self->quit(err);
                            });
                },
                [=](caf::error& err) {
                    std::cout << "Sync failed: " << to_string(err) << "\n";
                    self->quit(err);
                });
    },

    // 3. Final CPU verification
    [=](const std::vector<int>& matrixA,
        const std::vector<int>& matrixB,
        const std::vector<int>& matrixC,
        int N) {

        std::vector<int> result(N * N);
        serial_matrix_multiply(matrixA, matrixB, result, N);

        if (result == matrixC) {
            std::cout << "actor with id " << self->state().id << " references match\n";
        } else {
            std::cout << "actor with id " << self->state().id << " references did not match\n";
        }

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


//this test is meant to demonstrate the fact that sync_actors work by selecting the
//single_usage_behavior. If nothing bad goes wrong we take that as a win 
//maybe a performance analysis in the future 
void run_sync_actor_test(
    caf::actor_system& sys,
    int N,
    int num_actors,
    bool randomize = false)
{
    caf::cuda::manager& mgr = caf::cuda::manager::get(); 


    //set the behaviors of each scheduler actor
    for (int i = 0; i < mgr.get_num_devices();i++) {
	    mgr.send_scheduler_actor_message("single_usage",i);
    }

    auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

     const int THREADS = 32;
      const int BLOCKS = (N + THREADS - 1) / THREADS;

      caf::cuda::nd_range dims(
        BLOCKS, BLOCKS, 1,
        THREADS, THREADS, 1);

    caf::actor exit_actor = mgr.spawn_exit_actor(num_actors);

    auto t_start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_actors; ++i) {
	    caf::actor a = sys.spawn(
			    mmul_actor_fun,
			    exit_actor,
			    N,
			    program,
			    dims
			    );

    }

    auto t_end = std::chrono::steady_clock::now();

    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
		    t_end - t_start
		    ).count();

    sys.await_all_actors_done();
}


void caf_main(caf::actor_system& sys) {
  
	caf::cuda::manager_config man_config(true); //turns the scheduler on
	caf::cuda::manager::init(sys,man_config);
	

	run_sync_actor_test(sys,1024,10);

	//no dependencies
//	std::vector<int> sizes = {32, 64, 128, 256, 512, 1024,2048,4096};
//	const int num_actors = 2000;
//	run_load_balance_test(sys,sizes,num_actors);




}




CAF_MAIN()
