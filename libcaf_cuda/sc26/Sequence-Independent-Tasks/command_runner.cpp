#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <caf/component-actors/mmul_actor/mmul_actor.hpp>
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


using command =
  caf::cuda::command_runner<>;

command mmul_command;

struct mmul_state {
	caf::cuda::program_ptr program;
	// S3 fix: matrices stored in actor state; messages carry only an int trigger,
	// eliminating the ~8 MB per-message vector copy (80 GB total for 10 k iters).
	std::vector<int> matrixA;
	std::vector<int> matrixB;
	int N = 0;
	int completed = 0;
	int total = 0;
	bool is_warmup = false;
	std::chrono::steady_clock::time_point start_time;
};

// S3 fix: global output buffer excludes its allocation from timing,
// consistent with the other benchmarks.
std::vector<int> matrixC;




caf::behavior mmul_actor_fun(caf::stateful_actor<mmul_state>* self,
		caf::cuda::program_ptr mmul_kernel,
		std::vector<int> matrixA_,
		std::vector<int> matrixB_,
		int N_,
		int total_,
		bool is_warmup_) {

	auto& st = self->state();
	st.program = mmul_kernel;
	st.matrixA = std::move(matrixA_);
	st.matrixB = std::move(matrixB_);
	st.N = N_;
	st.total = total_;
	st.completed = 0;
	st.is_warmup = is_warmup_;

return {

    // S1 fix: actor receives a trigger (int) instead of full matrix vectors.
    // It self-quits when all iterations are processed — no external kill needed.
    [=](int /*trigger*/) {
      auto& st = self->state();

      // Record start time on the first iteration
      if (st.completed == 0) {
        st.start_time = std::chrono::steady_clock::now();
      }

      using clock = std::chrono::steady_clock;
      using ms = std::chrono::duration<double, std::milli>;

      caf::cuda::manager& mgr = caf::cuda::manager::get();
      int device = 0;
      int stream = 1;
      const int N = st.N;

      auto inA = caf::cuda::create_in_arg(st.matrixA);
      auto arg1 = mmul_command.transfer_memory(device, stream, std::move(inA));

      auto inB = caf::cuda::create_in_arg(st.matrixB);
      auto arg2 = mmul_command.transfer_memory(device, stream, std::move(inB));

      const int THREADS = 32;
      const int BLOCKS = (N + THREADS - 1) / THREADS;

      caf::cuda::nd_range dims(
        BLOCKS, BLOCKS, 1,
        THREADS, THREADS, 1);

      caf::cuda::mmul_async_command<int> cmd;
      auto output = cmd.run_async(
		      st.program,
		      dims,
		      1, 0, device,
		      arg1, arg2, out<int>{N * N}, in<int>{N});

	    caf::cuda::mem_ptr<int> dC = std::get<2>(output);
	    dC->copy_to_host(matrixC.data(), N * N);

      st.completed++;

      // S6 fix: milestone reporting every 1000 completions
      if (!st.is_warmup && st.completed % 1000 == 0) {
        auto now = clock::now();
        double elapsed = ms(now - st.start_time).count();
        std::cout << "[MILESTONE] " << st.completed << " / " << st.total
                  << " iterations, elapsed = " << elapsed << " ms\n";
      }

      if (st.completed == st.total) {
        if (!st.is_warmup) {
          auto end = clock::now();
          double total_ms = ms(end - st.start_time).count();
          // S5 fix: standardised tag matches cuda_native and actor_facade output
          std::cout << "[SERIES RESULT] Matrix " << N << "x" << N
                    << ", iterations = " << st.total
                    << ", total time = " << total_ms << " ms\n";
        }
        // S1 fix: graceful self-quit — all prior mailbox messages already processed
        self->quit();
      }
    }

  };
}


// S4 fix: program loaded once in caf_main and passed in; manager::init/shutdown
// called once rather than once-per-series.
void run_mmul_test(caf::actor_system& sys,
                   caf::cuda::program_ptr program,
                   int matrix_size, int iterations,
                   bool is_warmup = false) {

  std::vector<int> matA(matrix_size * matrix_size, 2);
  std::vector<int> matB(matrix_size * matrix_size, 3);
  matrixC.resize(matrix_size * matrix_size);

  // S3 fix: spawn actor with matrices in state; send trigger-only int messages
  caf::actor a = sys.spawn(mmul_actor_fun, program,
                            std::move(matA), std::move(matB),
                            matrix_size, iterations, is_warmup);

  // S3 fix: int triggers carry no matrix data — no OOM risk regardless of iteration count
  // S1 fix: actor self-quits on completion; no anon_send_exit(kill) needed
  for (int i = 0; i < iterations; i++)
	  anon_mail(i).send(a);

  sys.await_all_actors_done();
}


void caf_main(caf::actor_system& sys) {
  constexpr int matrix_size = 1000;
  constexpr int total_iterations = 10000;

  caf::cuda::manager::init(sys);  // S4 fix: init once before all series

  auto program = caf::cuda::manager::get()
                     .create_program_from_cubin("mmul.cubin", "matrixMul");

  // S2 fix: warmup run to prime CUDA context before timed series
  std::cout << "--- warmup starting ---\n";
  run_mmul_test(sys, program, matrix_size, 10, /*is_warmup=*/true);
  std::cout << "--- warmup complete ---\n";

  run_mmul_test(sys, program, matrix_size, total_iterations);

  caf::cuda::manager::shutdown();  // S4 fix: shutdown once after all series

}




CAF_MAIN()
