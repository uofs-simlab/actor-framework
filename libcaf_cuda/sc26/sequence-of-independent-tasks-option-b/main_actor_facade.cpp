// main_actor_facade.cpp
// Option B: Throughput mode — all iterations sent non-blocking before the actor
// processes any of them. The worker counts completed GPU results and quits after
// 'iterations' are done. The outer scoped_actor calls wait_for() once — a
// single barrier matching native's single cuStreamSynchronize per series. This
// demonstrates the actor model's throughput ceiling and recovers near-native
// performance when per-iteration results are not required.
#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>

static const unsigned int RANDOM_SEED = 42;
using namespace std::chrono_literals;
using clock_t_ = std::chrono::steady_clock;


class MatMult {
  caf::event_based_actor* self_;
  std::vector<int> A_;
  std::vector<int> B_;
  caf::actor gpuActor_;
  int total_;       // total messages expected
  int completed_;   // GPU results received so far
  double acc_create_inA_ms_  = 0;
  double acc_create_inB_ms_  = 0;
  double acc_create_outC_ms_ = 0;
  double acc_create_inN_ms_  = 0;
  double acc_dispatch_ms_    = 0;
  double acc_extract_ms_     = 0;

public:
  MatMult(caf::event_based_actor* self, int N, int total)
      : self_(self), total_(total), completed_(0) {
    std::mt19937 rng(RANDOM_SEED);
    std::uniform_int_distribution<int> dist(1, 10);

    A_.resize(N * N);
    B_.resize(N * N);

    for (auto& v : A_) v = dist(rng);
    for (auto& v : B_) v = dist(rng);

    int THREADS = 32;
    int BLOCKS = (N + THREADS - 1) / THREADS;
    caf::cuda::nd_range dim(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

    gpuActor_ = self_->system().cuda_manager().spawnFromCUBIN(
        "mmul.cubin", "matrixMul", dim,
        in<int>{}, in<int>{}, out<int>{}, in<int>{});
  }

  caf::behavior make_behavior() {
    return {
      // Fire-and-forget handler: dispatch to GPU actor, count completions,
      // quit when all expected results have arrived.
      [this](int N) {
        using ms = std::chrono::duration<double, std::milli>;

        // -------------------------
        // create_in_arg A
        // -------------------------
        auto t_a_start = clock_t_::now();
        auto inA = caf::cuda::create_in_arg(A_);
        auto t_a_end = clock_t_::now();

        // -------------------------
        // create_in_arg B
        // -------------------------
        auto t_b_start = clock_t_::now();
        auto inB = caf::cuda::create_in_arg(B_);
        auto t_b_end = clock_t_::now();

        // -------------------------
        // create_out_arg C
        // -------------------------
        auto t_outC_start = clock_t_::now();
        auto outC = caf::cuda::create_out_arg_with_size<int>(N * N);
        auto t_outC_end = clock_t_::now();

        // -------------------------
        // create_in_arg N
        // -------------------------
        auto t_inN_start = clock_t_::now();
        auto inN = caf::cuda::create_in_arg(N);
        auto t_inN_end = clock_t_::now();

        // Accumulate synchronous pre-dispatch timings immediately
        acc_create_inA_ms_  += ms(t_a_end    - t_a_start).count();
        acc_create_inB_ms_  += ms(t_b_end    - t_b_start).count();
        acc_create_outC_ms_ += ms(t_outC_end - t_outC_start).count();
        acc_create_inN_ms_  += ms(t_inN_end  - t_inN_start).count();

        // -------------------------
        // request (dispatch + GPU execution + D2H)
        // -------------------------
        auto t_request_start = clock_t_::now();

        self_->mail(inA, inB, outC, inN)
            .request(gpuActor_, caf::infinite)
            .then(
              [this, t_request_start, N](const std::vector<output_buffer>& result) {
                using ms2 = std::chrono::duration<double, std::milli>;
                auto t_response_end = clock_t_::now();

                // -------------------------
                // extract_vector
                // -------------------------
                auto t_extract_start = clock_t_::now();
                (void)caf::cuda::extract_vector<int>(result);
                auto t_extract_end = clock_t_::now();

                acc_dispatch_ms_ += ms2(t_response_end - t_request_start).count();
                acc_extract_ms_  += ms2(t_extract_end  - t_extract_start).count();

                ++completed_;
                if (completed_ >= total_) {
                  double n = static_cast<double>(total_);
                  std::cout << "\n[PHASE BREAKDOWN] Series iters=" << total_
                            << " (N=" << N << "):\n";
                  std::cout << "  Avg create_in_arg A:                     " << acc_create_inA_ms_  / n << " ms\n";
                  std::cout << "  Avg create_in_arg B:                     " << acc_create_inB_ms_  / n << " ms\n";
                  std::cout << "  Avg create_out_arg C:                    " << acc_create_outC_ms_ / n << " ms\n";
                  std::cout << "  Avg create_in_arg N:                     " << acc_create_inN_ms_  / n << " ms\n";
                  std::cout << "  Avg request\u2192response (dispatch+GPU+D2H): " << acc_dispatch_ms_    / n << " ms\n";
                  std::cout << "  Avg extract_vector:                      " << acc_extract_ms_     / n << " ms\n";
                  self_->quit();
                }
              },
              [](const caf::error& err) {
                std::cerr << "GPU error: " << to_string(err) << "\n";
              }
            );
      }
    };
  }
};


void caf_main(caf::actor_system& sys) {
  caf::scoped_actor self{sys};
  const int N = 1000;
  std::vector<int> iteration_series = {1000, 2000, 3000, 4000, 5000,
                                       6000, 7000, 8000, 9000, 10000};

  // Warmup: send one fire-and-forget and wait for worker to finish
  {
    auto w = self->spawn(caf::actor_from_state<MatMult>, N, 1);
    caf::anon_mail(N).send(w);
    self->wait_for(w);
  }
  std::cout << "--- warmup complete ---\n";

  for (int iterations : iteration_series) {
    // Worker knows how many results to expect before quitting
    auto worker = self->spawn(caf::actor_from_state<MatMult>, N, iterations);

    auto start = clock_t_::now();

    // [OPTION B] Fire all iterations non-blocking — actor queues them and
    // processes back-to-back, matching native stream pipelining
    for (int i = 0; i < iterations; ++i) {
      caf::anon_mail(N).send(worker);
    }

    // Single barrier — blocks until worker has processed all iterations
    self->wait_for(worker);

    auto end = clock_t_::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "[SERIES RESULT] Matrix " << N << "x" << N
              << ", iterations = " << iterations
              << ", total GPU time = " << total_ms << " ms\n";
  }
}

CAF_MAIN(caf::cuda::manager)
