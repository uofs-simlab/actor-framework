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
caf::cuda::command_runner<caf::cuda::mem_ptr<int>, caf::cuda::mem_ptr<int>,caf::cuda::mem_ptr<int>,caf::cuda::mem_ptr<int >> mmul;
using async_command = caf::cuda::mmul_async_command<int>;
async_command async_mmul;





struct MatrixPool {
    std::unordered_map<int, std::vector<int>> A;
    std::unordered_map<int, std::vector<int>> B;
};

MatrixPool create_matrix_pool(const std::vector<int>& sizes) {
    MatrixPool pool;

    for (int N : sizes) {
        pool.A[N] = std::vector<int>(N*N, 1);
        pool.B[N] = std::vector<int>(N*N, 1);
    }

    return pool;
}









struct mmul_state {

	caf::cuda::program_ptr mmul_kernel;

};



caf::behavior mmul_actor_fun(caf::stateful_actor<mmul_state>* self,caf::cuda::program_ptr mmul_kernel,
		const std::vector<int>& matrix1,
		const std::vector<int>& matrix2,
		int N) {

	self->state().mmul_kernel = mmul_kernel;
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

		    const int THREADS = 32;
		    const int BLOCKS = (N + THREADS - 1) / THREADS;
		    caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);


			auto program = self->state().mmul_kernel;

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
    const std::vector<int>& matrix1,
    const std::vector<int> & matrix2)
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


		    const int THREADS = 32;
		    const int BLOCKS = (N + THREADS - 1) / THREADS;
		    caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

		    auto arg1 = mmul.transfer_memory(res_token, caf::cuda::create_in_arg(matrixA));
		    auto arg2 = mmul.transfer_memory(res_token, caf::cuda::create_in_arg(matrixB));
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



void run_mmul_mixed_batch_cuda_scheduler(
    caf::actor_system& sys,
    const std::vector<int>& sizes,
    int num_actors,
    MatrixPool pool,
    bool randomize = false)
{
    caf::cuda::manager& mgr = caf::cuda::manager::get();

    auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

    std::mt19937 rng(123456);
    std::uniform_int_distribution<size_t> dist(0, sizes.size() - 1);

    for (int i = 0; i < num_actors; ++i) {

        int N = randomize ? sizes[dist(rng)] : sizes[i % sizes.size()];

	const auto& A = pool.A[N];
	const auto& B = pool.B[N];

        sys.spawn(
            mmul_actor_fun,
            program,
            A,
	    B,
	    N);
    }

    sys.await_all_actors_done();
}


void run_mmul_mixed_batch_caf_cuda_scheduler(
    caf::actor_system& sys,
    const std::vector<int>& sizes,
    int num_actors,
    MatrixPool pool,
    bool FCFS,
    bool randomize = false)
{
    caf::cuda::manager& mgr = caf::cuda::manager::get();

    //set the scheduler actor behavior
    if (FCFS) {
	    for (int i = 0; i < mgr.get_num_devices(); i++) {

		    mgr.send_scheduler_actor_message("green",i);

	    }

    }
    else { 
	    for (int i = 0; i < mgr.get_num_devices(); i++) {

		    mgr.send_scheduler_actor_message("multilevel",i);

	    }
    }

    auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

    caf::actor exit_actor = mgr.spawn_exit_actor(num_actors);

    std::mt19937 rng(123456);
    std::uniform_int_distribution<size_t> dist(0, sizes.size() - 1);

    const int THREADS = 32;
// timestamp before actor creation
auto start_time = std::chrono::high_resolution_clock::now();

for (int i = 0; i < num_actors; ++i) {

    int N = randomize ? sizes[dist(rng)] : sizes[i % sizes.size()];
    int BLOCKS = (N + THREADS - 1) / THREADS;

    caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

    const auto& A = pool.A[N];
    const auto& B = pool.B[N];
  // time the single actor creation
    //auto start = std::chrono::high_resolution_clock::now();

    caf::actor a = sys.spawn(
        mmul_actor_fun_scheduler,
        exit_actor,
        N,
        program,
        dims,
        A,
        B
    );

    //auto end = std::chrono::high_resolution_clock::now();
    //auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    //std::cout << "[INFO] Actor " << i << " creation time: " << ms << " ms\n";
}

// timestamp after actor creation
auto end_time = std::chrono::high_resolution_clock::now();
auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

std::cout << "[INFO] Actor creation time for " << num_actors
          << " actors: " << ms << " ms\n";
    sys.await_all_actors_done();
}








void run_mmul_mixed_batch_comparison(
    caf::actor_system& sys)
{
    std::vector<int> sizes = {32,64,128,256,512,1024,2048};
    std::vector<int> actor_counts = {5000};
    MatrixPool pool = create_matrix_pool(sizes);



    std::cout << "\n=== MMUL Mixed Batch Comparison ===\n";
    std::cout << "scheduler actors time_seconds\n";

    for (auto actors : actor_counts) {

        /* ========= core_usage BULK ========= */

        {
            caf::cuda::manager_config cfg(true);
            caf::cuda::manager::init(sys, cfg);

            double t = time_run([&] {
                run_mmul_mixed_batch_caf_cuda_scheduler(sys, sizes, actors,pool,false);
            });

            std::cout << "RESULT CAF CUDA DEFAULT SCHEDULER "
                      << actors << " "
                      << t << "seconds\n";

            caf::cuda::manager::shutdown();
        }
        /* ========= green light BULK  ========= */

        {
            caf::cuda::manager_config cfg(true);
            caf::cuda::manager::init(sys, cfg);

            double t = time_run([&] {
                run_mmul_mixed_batch_caf_cuda_scheduler(sys, sizes, actors,pool,true);
            });

            std::cout << "RESULT CAF CUDA FCFS SCHEDULER "
                      << actors << " "
                      << t << "seconds\n";

            caf::cuda::manager::shutdown();
        }


        /* ========= no scheduler ========= */

        {
            caf::cuda::manager_config cfg(false);
            caf::cuda::manager::init(sys, cfg);

            double t = time_run([&] {
                run_mmul_mixed_batch_cuda_scheduler(sys, sizes, actors,pool);
            });

            std::cout << "RESULT CUDA SCHEDULER "
                      << actors << " "
                      << t << "seconds\n";

            caf::cuda::manager::shutdown();
        }
    }

    std::cout << "\n=== Mixed Batch Comparison Complete ===\n";
}


// Spawn actors memory-efficiently using counters
void run_mmul_spawn_counter(
    actor_system& sys,
    const std::vector<int>& sizes,
    int num_actors,
    const MatrixPool& pool,
    bool largest_first = false,
    bool smallest_first = false)
{
    caf::cuda::manager& mgr = caf::cuda::manager::get();
    auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

    const int THREADS = 32;

    // Determine spawn order
    std::vector<int> spawn_order = sizes;
    if (largest_first) std::sort(spawn_order.rbegin(), spawn_order.rend());
    if (smallest_first) std::sort(spawn_order.begin(), spawn_order.end());

    // Initialize counters
    std::unordered_map<int,int> spawned_count;
    for (auto N : spawn_order) spawned_count[N] = 0;

    int total_spawned = 0;
    int num_sizes = spawn_order.size();
    int base_quota = num_actors / num_sizes;
    int remainder = num_actors % num_sizes;

    for (size_t idx = 0; idx < spawn_order.size(); ++idx) {
        int N = spawn_order[idx];
        int limit = base_quota + (idx == spawn_order.size() - 1 ? remainder : 0);

        while (spawned_count[N] < limit && total_spawned < num_actors) {
            const auto& A = pool.A.at(N);
            const auto& B = pool.B.at(N);
            caf::cuda::nd_range dims((N+THREADS-1)/THREADS, (N+THREADS-1)/THREADS, 1, THREADS, THREADS, 1);

	    sys.spawn(mmul_actor_fun, program, A, B, N);
	    spawned_count[N]++;
	    total_spawned++;
	}
    }

    sys.await_all_actors_done();
}


void run_actor_spawn_order_comparison(actor_system& sys) {
    std::vector<int> sizes = {10,32,64,128,256,512,1024,2048};
    int num_actors = 5000;
    MatrixPool pool = create_matrix_pool(sizes);

    std::cout << "\n=== Actor Spawn Order Comparison ===\n";
    std::cout << "order num_actors time_seconds\n";

    // Round-robin
    {
        caf::cuda::manager_config cfg(false);
        caf::cuda::manager::init(sys, cfg);
         double t = time_run([&] {
                run_mmul_mixed_batch_cuda_scheduler(sys, sizes, num_actors , pool);
            });
        std::cout << "round_robin " << num_actors << " " << t << "\n";
        caf::cuda::manager::shutdown();
    }

    // Largest-first
    {
        caf::cuda::manager_config cfg(false);
        caf::cuda::manager::init(sys, cfg);
        double t = time_run([&] {
            run_mmul_spawn_counter(sys, sizes, num_actors, pool, true, false);
        });
        std::cout << "largest_first " << num_actors << " " << t << "\n";
        caf::cuda::manager::shutdown();
    }

    // Smallest-first
    {
        caf::cuda::manager_config cfg(false);
        caf::cuda::manager::init(sys, cfg);
        double t = time_run([&] {
            run_mmul_spawn_counter(sys, sizes, num_actors, pool, false, true);
        });
        std::cout << "smallest_first " << num_actors << " " << t << "\n";
        caf::cuda::manager::shutdown();
    }

    std::cout << "=== Spawn Order Comparison Complete ===\n";
}









void caf_main(caf::actor_system& sys) {
  

  caf::cuda::manager_config man_config(true);
  //caf::cuda::manager::init(sys,man_config);

 // run_mmul_mixed_batch_comparison(sys);

  run_actor_spawn_order_comparison(sys);

}




CAF_MAIN()
