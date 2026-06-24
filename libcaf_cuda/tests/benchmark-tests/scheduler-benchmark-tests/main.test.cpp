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
command mmul;

struct mmul_state {

	caf::cuda::program_ptr mmul_kernel;

};



caf::behavior mmul_actor_fun(caf::stateful_actor<mmul_state>* self,caf::cuda::program_ptr mmul_kernel,int N) {

	self->state().mmul_kernel = mmul_kernel;
	std::vector<int> matrix1(N*N);
	std::vector<int> matrix2(N*N);
	self->mail(matrix1, matrix2, N).send(self);
	return {
		[=](const std::vector<int>& matrixA,
				const std::vector<int>& matrixB,
				int N) {

			caf::cuda::manager& mgr = caf::cuda::manager::get();
			int device = 0;
			int stream = rand();

			//auto program =
			//mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

			auto program = self->state().mmul_kernel;

				// -------------------------
				// create_in_arg A
				// -------------------------

			auto inA = caf::cuda::create_in_arg(std::move(matrixA));


			auto arg1 = mmul_command.transfer_memory(
					device,
					stream,
					std::move(inA));





			auto inB = caf::cuda::create_in_arg(std::move(matrixB));
			auto arg2 = mmul_command.transfer_memory(
					device,
					stream,
					std::move(inB));



			caf::actor mmul_actor =
				self->spawn(caf::cuda::mmul_actor_fun<int>, program);

			caf::actor mem_transfer_actor = self->spawn(caf::cuda::mem_transfer_actor_fun<int>);


			self->mail(arg1, arg2, N, device, stream)
				.request(mmul_actor, std::chrono::seconds(30))
				.then([=](caf::cuda::mem_ptr<int> dC) {

						self->mail(dC).request(mem_transfer_actor,std::chrono::seconds(4000))
						.then([=] (std::vector<int>& matrixC) {
								//std::vector<int> matrixC = dC->copy_to_host();
								self->quit();

								});
						});
		}

	};
}




struct mmul_actor_with_scheduler_state {
  static inline const char* name = "my_actor";
  caf::actor sync_actor;
  caf::actor mem_transfer_actor;
  caf::actor mmul_actor;
};


// Stateful actor behavior
caf::behavior mmul_actor_fun_scheduler(
    caf::stateful_actor<mmul_actor_with_scheduler_state>* self,
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
	self->state().mmul_actor = self -> spawn(caf::cuda::mmul_actor_fun<int>,program);

       return {

    // 1. Handle response token
    [=](caf::cuda::response_token_ptr res_token) {
//        std::cout << "Got response\n";

        if (res_token->getType() == LAUNCH_RESPONSE) {
            std::vector<int> matrix1(N*N);
            std::vector<int> matrix2(N*N);

            self->mail(matrix1, matrix2, res_token, N).send(self);

        } else {
  //          std::cout << "Got a memory response token\n";
        }
    },

    // 2. Handle memory buffers -> GPU
    [=](const std::vector<int>& matrixA,
        const std::vector<int>& matrixB,
        const caf::cuda::response_token_ptr& res_token,
        int N) {

    //    std::cout << "Working\n";
        caf::cuda::manager& mgr = caf::cuda::manager::get();

       
	/*
	const int THREADS = 32;
        const int BLOCKS = (N + THREADS - 1) / THREADS;
        caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

        auto arg1 = mmul.transfer_memory(res_token, caf::cuda::create_in_arg(matrixA));
        auto arg2 = mmul.transfer_memory(res_token, caf::cuda::create_in_arg(matrixB));
        auto arg3 = mmul.transfer_memory(res_token, caf::cuda::create_out_arg(N*N));
        auto arg4 = mmul.transfer_memory(res_token, caf::cuda::create_in_arg(N));


        auto tempC = mmulAsync.run_async(program, dims, res_token, arg1, arg2, arg3, arg4);
        caf::cuda::mem_ptr<int> bufferA = std::get<0>(tempC);
	*/

	caf::cuda::mem_ptr<int> arg1 = mmul.transfer_memory(res_token, caf::cuda::create_in_arg(matrixA));
	caf::cuda::mem_ptr<int> arg2 = mmul.transfer_memory(res_token, caf::cuda::create_in_arg(matrixB));

	self->mail(arg1,arg2,N,res_token -> getDeviceNumber(), res_token -> getStreamId())
		.request(self->state().mmul_actor,1000s)
		.then(
		 [=](caf::cuda::mem_ptr<int> dC) {


        self->mail(dC, res_token)
            .request(self->state().sync_actor, 1000s)
            .then(
                [=](caf::cuda::mem_ptr<int> /*syncedC*/) {
                    self->mail(dC)
                        .request(self->state().mem_transfer_actor, 1000s)
                        .then(
                            [=](std::vector<int> matrixC) {
                                //self->mail(matrixA,matrixB,matrixC, N).send(self);
                            
			     self->mail(1).send(exit_actor);
        		     self->quit();
			    
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
		 });
    },
       };

}












void run_mmul_test(caf::actor_system& sys, int matrix_size, int num_actors) {
  if (num_actors < 1) {
    std::cerr << "[ERROR] Number of actors must be >= 1\n";
    return;
  }

  caf::cuda::manager& mgr = caf::cuda::manager::get();

  //change the scheduler to mulitlevle_usage
  anon_mail(
        caf::cuda::make_behavior_token("multilevel")
        ).send(mgr.get_scheduler_actor());

  // CREATE ONCE
  auto program =
      mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

  const int THREADS = 32;
  const int BLOCKS = (matrix_size + THREADS - 1) / THREADS;
  caf::cuda::nd_range dims(
      BLOCKS, BLOCKS, 1,
      THREADS, THREADS, 1);

  caf::actor exit_actor = mgr.spawn_exit_actor(num_actors);

  for (int i = 0; i < num_actors; ++i) {
    
      sys.spawn(
        mmul_actor_fun_scheduler,
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

    mgr.send_scheduler_actor_message("green",0);

  // CREATE ONCE
  auto program =
      mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

  const int THREADS = 32;
  const int BLOCKS = (matrix_size + THREADS - 1) / THREADS;
  caf::cuda::nd_range dims(
      BLOCKS, BLOCKS, 1,
      THREADS, THREADS, 1);

  caf::actor exit_actor = mgr.spawn_exit_actor(num_actors);

  for (int i = 0; i < num_actors; ++i) {
    
      sys.spawn(
        mmul_actor_fun_scheduler,
        exit_actor,
        matrix_size,
        program,
        dims);
    
  }

 

  sys.await_all_actors_done();
}

void run_mmul_test_no_scheduler_actor(caf::actor_system& sys, int matrix_size, int num_actors) {
    if (num_actors < 1) {
        std::cerr << "[ERROR] Number of actors must be >= 1\n";
        return;
    }

    
    caf::cuda::manager& mgr = caf::cuda::manager::get();

    auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

    const int THREADS = 32;
    const int BLOCKS = (matrix_size + THREADS - 1) / THREADS;
    caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);


    for (int i = 0; i < num_actors; ++i) {
        sys.spawn(
            mmul_actor_fun,
            program,
            matrix_size
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
    const int max_size   = 2048;
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
            caf::cuda::manager::init(sys, man_config); // scheduler enabled
            std::cout << "\n[RUN] scheduler=multilevel_usage "
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

            /* ================= Scheduler-disabled actor ( uses green-light) ================= */

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




void caf_main(caf::actor_system& sys) {
  

  caf::cuda::manager_config man_config(true);
  //caf::cuda::manager::init(sys,man_config);
  run_mmul_scaling_tests(sys,man_config);

}




CAF_MAIN()
