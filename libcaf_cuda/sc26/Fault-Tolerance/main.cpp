// example_14: Fault-tolerant Monte Carlo π estimation with GPU workers.
//
// Demonstrates:
//  1. Multiple GPU worker actors each owning a separate CUDA stream, running
//     Monte Carlo batches concurrently on the same physical device.
//  2. A supervisor actor that distributes batches, monitors workers, and
//     recovers automatically when a worker is killed mid-computation.
//  3. Fault injection: halfway through the run the supervisor force-kills
//     Worker 0, then detects the down_msg via the monitor callback, reschedules
//     the lost batch to a surviving worker, and respawns a replacement.
//  4. Final π estimate converges to the correct answer despite the disruption.
//
// Architecture:
//   main → Supervisor : start_atom
//   Supervisor spawns N WorkerActors, monitors each with callback.
//   Supervisor → Worker[k] : (seed, num_samples)   [request().then()]
//   Worker[k] → GPU via run_async_notify() on its private stream.
//   GPU → Worker[k] : gpu_done_atom  (from cuLaunchHostFunc on CUDA thread)
//   Worker[k] → Supervisor : int M   (number of in-circle hits)
//   [halfway] Supervisor kills Worker[0] → monitor callback fires immediately
//   Supervisor reschedules in-flight batch, respawns Worker[0], continues.
//
// Note on single-GPU machines: all workers share one physical device but each
// gets a distinct CUDA stream (keyed by actor_id inside get_stream_for_actor).
// Stream-level concurrency is real; for multi-GPU, pin worker N to device N
// by using the find_device(N) overload of create_program_from_cubin.

#include <caf/all.hpp>
#include <caf/actor_cast.hpp>
#include <caf/cuda/all.hpp>
#include "../common/kernel_paths.hpp"

#include <deque>
#include <map>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>

using namespace caf;
using namespace caf::cuda;

// ─────────────────────────────────────────────────────────────────────────────
// Atoms
// ─────────────────────────────────────────────────────────────────────────────
CAF_BEGIN_TYPE_ID_BLOCK(monte_carlo_app, caf::id_block::cuda::end)
    CAF_ADD_ATOM(monte_carlo_app, start_atom)
    CAF_ADD_ATOM(monte_carlo_app, kill_atom)
    CAF_ADD_ATOM(monte_carlo_app, done_atom)
CAF_END_TYPE_ID_BLOCK(monte_carlo_app)

// ─────────────────────────────────────────────────────────────────────────────
// Work batch description
// ─────────────────────────────────────────────────────────────────────────────
struct WorkBatch {
    int batch_id;
    int seed;         // deterministic: batch_id * 1000003
    int num_samples;  // constant per run
};

// ─────────────────────────────────────────────────────────────────────────────
// command_runner type alias for monteCarloKernel
//   arg 0: in<int>  seed
//   arg 1: in<int>  num_samples
//   arg 2: out<int> hit_count
// ─────────────────────────────────────────────────────────────────────────────
// Use in_out<int> (not out<int>) for the hit counter so the framework
// copies value 0 to device memory before each kernel launch.  out<int> uses
// cuMemAlloc without zero-init, which would corrupt the atomicAdd result.
// Use in_out<int> for the hit counter so 0 is uploaded to device before
// each launch — out<int> (scratch_argument) uses bare cuMemAlloc with no
// zero-init, which would corrupt the atomicAdd result.
using mc_runner = command_runner<in<int>, in<int>, in_out<int>>;

// ─────────────────────────────────────────────────────────────────────────────
// WorkerActor
//
// Owns one mc_runner (and therefore one private CUDA stream).
// Handles a single kernel request at a time:
//   receive (seed, num_samples) → launch GPU → reply int M to sender
// ─────────────────────────────────────────────────────────────────────────────
class WorkerActor {
    event_based_actor* self_;
    int worker_index_;  // for logging

    mc_runner runner_;

    // Keep all mem_ptrs alive until the stream is idle (gpu_done_atom).
    // Using a tuple matching the runner return: <mem_ptr<int>, mem_ptr<int>, mem_ptr<int>>
    std::tuple<mem_ptr<int>, mem_ptr<int>, mem_ptr<int>> pending_refs_;

    // The out<int> mem_ptr (alias into pending_refs_ for convenience).
    mem_ptr<int> result_ptr_ = nullptr;

    // Response promise to deliver the hit count back to the supervisor.
    typed_response_promise<int> pending_rp_;

public:
    WorkerActor(event_based_actor* self, int index)
        : self_(self), worker_index_(index) {}

    behavior make_behavior() {
        return {
            // ── Kernel launch request from the supervisor ─────────────────
            [this](int seed, int num_samples) -> result<int> {
                pending_rp_ = self_->make_response_promise<int>();

                auto& mgr = self_->system().cuda_manager();
                // All workers use device 0 on a single-GPU machine.
                // On multi-GPU hardware, replace 0 with worker_index_.
                auto program = mgr.create_program_from_cubin(
                    actor_tests::paths::monte_carlo_cubin, "monteCarloKernel");

                nd_range dims(/*grid*/64, 1, 1, /*block*/256, 1, 1);

                auto arg_seed    = create_in_arg(seed);
                auto arg_samples = create_in_arg(num_samples);
                // Zero-init the hit counter (in_out so value 0 is copied to
                // device before the kernel runs; out<int> is NOT zero-inited).
                const std::vector<int> zero_buf{0};
                auto arg_out = create_in_out_arg(zero_buf);

                // run_async_notify: kernel is launched non-blocking; when the
                // CUDA stream goes idle the runtime calls gpu_done_atom on this
                // actor from its internal thread.
                pending_refs_ = runner_.run_async_notify(
                    program, dims,
                    actor_cast<actor>(self_),
                    arg_seed, arg_samples, arg_out);

                // The in_out<int> result is the third (last) element.
                result_ptr_ = std::get<2>(pending_refs_);

                self_->println("[Worker {:2d}] Launched GPU kernel: "
                               "seed={}, samples={}", worker_index_, seed, num_samples);
                return pending_rp_;
            },

            // ── GPU stream idle callback ───────────────────────────────────
            [this](gpu_done_atom) {
                std::vector<int> host = result_ptr_->copy_to_host();
                int M = host.empty() ? 0 : host[0];

                self_->println("[Worker {:2d}] gpu_done_atom: M={}", worker_index_, M);

                // Safe to release device memory now (stream confirmed idle).
                result_ptr_  = nullptr;
                pending_refs_ = {};

                pending_rp_.deliver(M);
            }
        };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Supervisor
//
// Manages the work queue, dispatches batches to workers, handles fault
// injection at mid-run, recovers lost batches, and accumulates the final
// π estimate.
// ─────────────────────────────────────────────────────────────────────────────
class Supervisor {
    event_based_actor* self_;

    // Configuration
    int num_workers_;
    int total_batches_;
    int samples_per_batch_;

    // Worker handles (index-stable; replaced on respawn)
    std::vector<actor> workers_;

    // Pending work
    std::deque<WorkBatch> work_queue_;

    // In-flight tracking: actor address → batch currently running in that worker
    std::map<actor_addr, WorkBatch> in_flight_;

    // Accumulation
    long long total_hits_     = 0;
    long long total_expected_ = 0; // total samples we PLAN to complete
    int       batches_done_   = 0;

    // Fault control
    bool fault_injected_ = false;

    // The scoped_actor that main() uses to wait for us.
    actor parent_;

public:
    Supervisor(event_based_actor* self,
               int num_workers, int total_batches, int samples_per_batch,
               actor parent)
        : self_(self)
        , num_workers_(num_workers)
        , total_batches_(total_batches)
        , samples_per_batch_(samples_per_batch)
        , parent_(std::move(parent))
    {}

    behavior make_behavior() {
        return {
            // ── Boot ──────────────────────────────────────────────────────
            [this](start_atom) {
                self_->println("[Supervisor] Starting. Workers={}, Batches={}, "
                               "Samples/batch={}", num_workers_, total_batches_, samples_per_batch_);

                // Pre-load work queue.
                for (int b = 0; b < total_batches_; ++b) {
                    work_queue_.push_back({b, b * 1000003, samples_per_batch_});
                }
                total_expected_ = (long long)total_batches_ * samples_per_batch_;

                // Spawn workers and immediately dispatch the first batch each.
                workers_.resize(num_workers_);
                for (int k = 0; k < num_workers_; ++k) {
                    workers_[k] = spawn_worker(k);
                    if (!work_queue_.empty())
                        dispatch_to(k);
                }
            },

            // ── Externally-triggered fault injection (e.g., from main) ───
            // Also fired internally when half the batches complete.
            [this](kill_atom) {
                if (!fault_injected_ && !workers_.empty()) {
                    fault_injected_ = true;
                    self_->println("[Supervisor] *** Injecting fault: killing Worker 0 ***");
                    anon_send_exit(workers_[0], exit_reason::kill);
                }
            }
        };
    }

private:
    // ── Helpers ──────────────────────────────────────────────────────────────

    actor spawn_worker(int idx) {
        actor w = self_->spawn(actor_from_state<WorkerActor>, idx);

        // Monitor with callback — non-deprecated, preferred API.
        self_->monitor(w, [this, idx](const error& reason) {
            self_->println("[Supervisor] Worker {} died: {}", idx, to_string(reason));

            actor_addr dead_addr = workers_[idx].address();

            auto it = in_flight_.find(dead_addr);
            if (it != in_flight_.end()) {
                WorkBatch lost = it->second;
                self_->println("[Supervisor] Re-queuing lost batch {} (seed={})",
                               lost.batch_id, lost.seed);
                work_queue_.push_front(lost);  // high-priority: run next
                in_flight_.erase(it);
            }

            // Respawn and immediately give it work if available.
            workers_[idx] = spawn_worker(idx);
            self_->println("[Supervisor] Worker {} respawned.", idx);
            if (!work_queue_.empty())
                dispatch_to(idx);
        });

        return w;
    }

    void dispatch_to(int idx) {
        WorkBatch batch = work_queue_.front();
        work_queue_.pop_front();

        self_->println("[Supervisor] Dispatch batch {} → Worker {} (seed={})",
                       batch.batch_id, idx, batch.seed);

        // Snapshot the address NOW so the in-flight record and the then()
        // cleanup both refer to the same worker instance, even if workers_[idx]
        // is replaced by the time the response arrives.
        actor_addr src = workers_[idx].address();
        in_flight_.emplace(src, batch);

        self_->mail(batch.seed, batch.num_samples)
            .request(workers_[idx], infinite)
            .then(
                [this, idx, src](int M) {
                    total_hits_ += M;
                    batches_done_++;
                    in_flight_.erase(src);

                    double pi_est = 4.0 * total_hits_
                                    / ((double)batches_done_ * samples_per_batch_);
                    self_->println("[Supervisor] Batch complete ({}/{}). π ≈ {:.6f}",
                                   batches_done_, total_batches_, pi_est);

                    // Inject fault exactly at the halfway point.
                    if (batches_done_ == total_batches_ / 2) {
                        self_->mail(kill_atom_v).send(self_);
                    }

                    if (batches_done_ == total_batches_) {
                        finish();
                        return;
                    }
                    if (!work_queue_.empty())
                        dispatch_to(idx);
                },
                [this, idx](const error& err) {
                    // Worker died while we were waiting for its result.
                    // The monitor callback already rescheduled the in-flight batch.
                    self_->println("[Supervisor] Request to Worker {} failed: {} "
                                   "(batch rescheduled via monitor callback)", idx, to_string(err));
                });
    }

    void finish() {
        long long total_samples_completed =
            (long long)batches_done_ * samples_per_batch_;
        double pi_final = 4.0 * total_hits_ / (double)total_samples_completed;
        double error_pct = std::abs(pi_final - M_PI) / M_PI * 100.0;

        self_->println("\n=== RESULT ===");
        self_->println("Batches completed : {}", batches_done_);
        self_->println("Total samples     : {}", total_samples_completed);
        self_->println("Total hits        : {}", total_hits_);
        self_->println("π estimate        : {:.8f}", pi_final);
        self_->println("Error vs π        : {:.4f}%\n", error_pct);

        // Shut down all workers.
        for (auto& w : workers_)
            anon_send_exit(w, exit_reason::user_shutdown);

        // Signal completion to main.
        self_->mail(done_atom_v).send(parent_);
        self_->quit();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Config & entry point
// ─────────────────────────────────────────────────────────────────────────────
class config : public actor_system_config {
public:
    config() {
        // Deliberately use more than 1 thread so concurrent workers are real.
        // Uncomment to pin to 1 thread and confirm everything still works:
        // set("caf.scheduler.max-threads", 1u);
    }
};

void caf_main(actor_system& sys, const config&) {
    constexpr int NUM_WORKERS      = 2;
    constexpr int TOTAL_BATCHES    = 20;
    constexpr int SAMPLES_PER_BATCH = 10'000'000;

    sys.println("=== Monte Carlo π Estimation (Fault-Tolerant) ===");
    sys.println("Workers: {}   Batches: {}   Samples/batch: {}",
                NUM_WORKERS, TOTAL_BATCHES, SAMPLES_PER_BATCH);
    sys.println("Total samples planned: {}\n",
                (long long)TOTAL_BATCHES * SAMPLES_PER_BATCH);

    scoped_actor self{sys};

    actor supervisor = self->spawn(
        actor_from_state<Supervisor>,
        NUM_WORKERS, TOTAL_BATCHES, SAMPLES_PER_BATCH,
        actor_cast<actor>(self));

    self->mail(start_atom_v).send(supervisor);

    // Block until the supervisor signals done (or terminates cleanly).
    self->receive([](done_atom) {});
}

CAF_MAIN(caf::cuda::manager, id_block::monte_carlo_app)
