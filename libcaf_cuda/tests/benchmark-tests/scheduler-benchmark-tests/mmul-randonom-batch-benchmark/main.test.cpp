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
#include <unordered_set>
//#include <caf/atoms.hpp>



using namespace caf;
using namespace std::chrono_literals;

// Command runners for GPU operations
caf::cuda::command_runner<> mmul_command;
using mmul_async_t = caf::cuda::command_runner<caf::cuda::mem_ptr<int>, caf::cuda::mem_ptr<int>, out<int>, in<int>>;
mmul_async_t async_mmul;

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
    int N_val = 0;

};



caf::behavior mmul_actor_fun(caf::stateful_actor<mmul_state>* self,
		caf::actor exit_actor,
		caf::cuda::program_ptr program,
		caf::cuda::nd_range dims,
		int stream,
		int N,
		const in<int> matrixA,
		const in<int> matrixB
		) {


	int device = stream % caf::cuda::manager::get().get_num_devices();
	self->mail(N).send(self);
    self->state().N_val = N;

	return {
		[=](int N) {
			auto arg1 = mmul_command.transfer_memory(device, stream, std::move(matrixA));
			auto arg2 = mmul_command.transfer_memory(device, stream, std::move(matrixB));

			out<int> arg3 = caf::cuda::create_out_arg<int>(N * N);
			in<int>  arg4 = caf::cuda::create_in_arg<int>(N);

			auto result = async_mmul.run_async(
					program, dims, stream, 0, device,
					arg1, arg2, arg3, arg4);

			auto bufferC = std::get<2>(result);
            auto self_hdl = caf::actor_cast<caf::actor>(self);

            // Asynchronously copy data back to host. This triggers a message to 'self' 
            // when the transfer is finished, instead of blocking the actor thread.
            mmul_command.copy_to_host_async(bufferC, [self_hdl](std::vector<int>&& /*data*/) {
                caf::anon_mail(uint64_t{1}).send(self_hdl);
            });
		},
        [=](uint64_t /* completion_token */) {
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
    const std::vector<int>& Ns      // deterministic task sizes 
    ) {
    // Initialize state
    self->state().num_actors = num_actors;
    self->state().completed = 0;
    self->state().max_waves = max_waves;
    self->state().num_waves = 0;
    self->state().pool = std::move(pool);

    self->state().Ns = Ns;
    self->state().next_task = 0;


    caf::cuda::manager& mgr = caf::cuda::manager::get();
    auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

    // Kick off first wave
    self->mail("spawn").send(self);

    // Start timing
    self->state().start_time = std::chrono::steady_clock::now();
    return {
        // -------------------- SPAWN WAVE --------------------
        [=](std::string cmd) {
            if (cmd != "spawn") return;

	     std::chrono::steady_clock::time_point cmd_start_time = std::chrono::steady_clock::now();
	    

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
                self->spawn(mmul_actor_fun, self, program, dims,i,N,
                            caf::cuda::create_in_arg(A),
                            caf::cuda::create_in_arg(B));
            }
        
	

    std::chrono::steady_clock::time_point cmd_end_time = std::chrono::steady_clock::now();

      std::chrono::duration<double> total_time =
                        cmd_end_time - cmd_start_time;

                    std::cout << "\n===== SUPERVISOR TOTAL TIME spawn =====\n";
                    std::cout << "Total runtime: "
                              << total_time.count() << " s\n";
        },

        // -------------------- COMPLETION TRACKING --------------------
        [=](int done) {
            self->state().completed += done;

            if (self->state().completed >= self->state().num_actors) {
                auto wave_end = std::chrono::steady_clock::now();
                std::chrono::duration<double> wave_time =
                    wave_end - self->state().wave_start_time;

                self->state().num_waves++;
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

    const int min_N = 32;
    const int max_N = 2048;
    const int num_sizes = 10;

    const int max_waves = 1;

    const std::vector<int> actor_counts = {
	  1,30000,40000,50000
    };


    int num_actors = actor_counts[actor_counts.size()-1];

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
    caf::cuda::manager_config scheduler_off(false);
    for (int num_actors : actor_counts) {
	    // Initialize CUDA manager
	    caf::cuda::manager::init(sys, scheduler_off);
        std::cout << "=====================================\n";
        std::cout << "Random Scaling | actors=" << num_actors << "\n";

        // Execute the supervisor which manages the asynchronous workload
        double elapsed = time_run([&]() {

            auto sup = sys.spawn(
                supervisor_actor_fun,
                num_actors,
                max_waves,
                pool,
		Ns
            );

	      sys.await_all_actors_done();
	    });

        caf::cuda::manager::shutdown();
    }
}
void caf_main(caf::actor_system& sys) {
  caf::cuda::manager_config man_config(false);
  run_mmul_random_scaling_tests(sys, man_config);
}
CAF_MAIN()
