// Fault-Tolerant Monte Carlo π estimation with GPU workers.
// SC26-hardened — implements all Opus review recommendations:
//
//  1. Parameterized fault count (--num-faults) with evenly-spaced injection
//     across different worker slots, proving recovery works at arbitrary points
//     and repeatedly.
//  2. Scalable defaults: 4 workers, 200 batches, 50M samples/batch.
//  3. Per-recovery latency timing:
//       t_fault_detected  (monitor callback fires)
//       t_recovery_dispatched (recovery batch sent to new worker)
//       t_recovery_completed  (recovery batch reply received)
//  4. No-fault baseline mode (--no-fault) for direct throughput comparison.
//  5. Structured output lines for benchmark script parsing:
//       [FAULT_DETECTED]  [FAULT_DISPATCH]  [FAULT_RECOVERED]
//       [PROGRESS]        [RESULT]
//
// Architecture: main → Supervisor → WorkerActor → command_runner → GPU.
// Workers share device 0 via distinct CUDA streams (one per worker index).

#include <caf/all.hpp>
#include <caf/actor_cast.hpp>
#include <caf/cuda/all.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <map>
#include <vector>
#include <iostream>

using namespace caf;
using namespace caf::cuda;

using sc = std::chrono::steady_clock;
using tp = std::chrono::time_point<sc>;

static double ms_between(tp a, tp b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ─────────────────────────────────────────────────────────────────────────────
// Atoms
// ─────────────────────────────────────────────────────────────────────────────
CAF_BEGIN_TYPE_ID_BLOCK(monte_carlo_app, caf::id_block::cuda::end)
    CAF_ADD_ATOM(monte_carlo_app, start_atom)
    CAF_ADD_ATOM(monte_carlo_app, done_atom)
CAF_END_TYPE_ID_BLOCK(monte_carlo_app)

// ─────────────────────────────────────────────────────────────────────────────
// WorkBatch: one unit of Monte Carlo work.
//   fault_index >= 0 marks a re-dispatched recovery batch and links it to the
//   corresponding FaultRecord so latency timestamps can be filled in.
// ─────────────────────────────────────────────────────────────────────────────
struct WorkBatch {
    int batch_id;
    int seed;           // deterministic: batch_id * 1000003
    int num_samples;
    int fault_index = -1; // -1 = normal; >= 0 = recovery for fault_records_[i]
};

// ─────────────────────────────────────────────────────────────────────────────
// FaultRecord: per-fault timing and recovery metadata
// ─────────────────────────────────────────────────────────────────────────────
struct FaultRecord {
    int  fault_index;
    int  worker_slot;
    int  batch_id    = -1;   // batch that was in-flight when the fault fired
    tp   t_detected;          // when monitor callback fired
    tp   t_dispatched;        // when the recovery batch was re-sent
    tp   t_completed;         // when the recovery batch reply arrived
    bool dispatched  = false;
    bool completed   = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// mc_runner type alias for monteCarloKernel
//   arg 0: in<int>      seed
//   arg 1: in<int>      num_samples
//   arg 2: in_out<int>  hit_count — zero-inited before each launch so the
//                                    atomicAdd accumulation starts clean.
// ─────────────────────────────────────────────────────────────────────────────
using mc_runner = command_runner<in<int>, in<int>, in_out<int>>;

// ─────────────────────────────────────────────────────────────────────────────
// WorkerActor
//
// Owns one mc_runner (and therefore one private CUDA stream).
// Handles one kernel request at a time:
//   receive (seed, num_samples) → launch GPU → reply int M (in-circle hits)
// ─────────────────────────────────────────────────────────────────────────────
class WorkerActor {
    event_based_actor* self_;
    int                worker_index_;
    mc_runner          runner_;

public:
    WorkerActor(event_based_actor* self, int index)
        : self_(self), worker_index_(index) {}

    behavior make_behavior() {
        return {
            [this](int seed, int num_samples) -> int {
                auto& mgr = caf::cuda::manager::get();
                auto program = mgr.create_program_from_cubin(
                    "monte_carlo.cubin", "monteCarloKernel");

                nd_range dims(/*grid*/64, 1, 1, /*block*/256, 1, 1);

                auto arg_seed    = create_in_arg(seed);
                auto arg_samples = create_in_arg(num_samples);
                // Zero-init the hit counter (in_out copies 0 to device first).
                const std::vector<int> zero_buf{0};
                auto arg_out = create_in_out_arg(zero_buf);

                // stream key = worker_index_ + 1 to keep streams distinct per slot.
                auto output = runner_.run_async(
                    program, dims,
                    worker_index_ + 1, /*shared_memory=*/0, /*device_number=*/0,
                    arg_seed, arg_samples, arg_out);

                auto result_ptr = std::get<2>(output);
                std::vector<int> host = result_ptr->copy_to_host();
                return host.empty() ? 0 : host[0];
            }
        };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Supervisor
//
// Distributes batches across workers, tracks in-flight work, injects faults
// at evenly-spaced completion milestones (cycling through worker slots), and
// measures per-fault recovery latency from detection through completion.
// ─────────────────────────────────────────────────────────────────────────────
class Supervisor {
    event_based_actor* self_;

    // ── configuration ────────────────────────────────────────────────────────
    int  num_workers_;
    int  total_batches_;
    int  samples_per_batch_;
    int  num_faults_;
    bool no_fault_;

    // ── worker handles (index-stable; replaced on respawn) ───────────────────
    std::vector<actor> workers_;

    // ── work tracking ─────────────────────────────────────────────────────────
    std::deque<WorkBatch>            work_queue_;
    std::map<actor_addr, WorkBatch>  in_flight_;  // addr → batch in that worker

    // ── accumulation ──────────────────────────────────────────────────────────
    long long total_hits_   = 0;
    int       batches_done_ = 0;

    // ── fault injection ───────────────────────────────────────────────────────
    int              faults_injected_ = 0;
    std::vector<int> fault_triggers_; // batches_done_ values at which to fire each fault
    std::vector<FaultRecord> fault_records_;

    // ── timing ────────────────────────────────────────────────────────────────
    tp t_start_;

    // ── parent (main's scoped_actor) ──────────────────────────────────────────
    actor parent_;

public:
    Supervisor(event_based_actor* self,
               int num_workers, int total_batches, int samples_per_batch,
               int num_faults, bool no_fault,
               actor parent)
        : self_(self)
        , num_workers_(num_workers)
        , total_batches_(total_batches)
        , samples_per_batch_(samples_per_batch)
        , num_faults_(no_fault ? 0 : num_faults)
        , no_fault_(no_fault)
        , parent_(std::move(parent))
    {}

    behavior make_behavior() {
        return {
            [this](start_atom) {
                // Evenly-spaced triggers: fault i fires after batch completion count
                // (i+1) * total_batches / (num_faults + 1), clamped to [1, total-1].
                fault_triggers_.resize(num_faults_);
                for (int i = 0; i < num_faults_; ++i) {
                    int t = (i + 1) * total_batches_ / (num_faults_ + 1);
                    fault_triggers_[i] = std::max(1, std::min(total_batches_ - 1, t));
                }

                self_->println("[Supervisor] Starting: workers={} batches={} "
                               "samples/batch={} faults={} mode={}",
                               num_workers_, total_batches_, samples_per_batch_,
                               num_faults_, no_fault_ ? "baseline" : "fault-injection");
                for (int i = 0; i < num_faults_; ++i)
                    self_->println("[Supervisor] Fault {} target: batches_done={}",
                                   i, fault_triggers_[i]);

                // Populate work queue.
                for (int b = 0; b < total_batches_; ++b)
                    work_queue_.push_back({b, b * 1000003, samples_per_batch_, -1});

                t_start_ = sc::now();

                // Spawn workers and immediately dispatch the first batch to each.
                workers_.resize(num_workers_);
                for (int k = 0; k < num_workers_; ++k) {
                    workers_[k] = spawn_worker(k);
                    if (!work_queue_.empty())
                        dispatch_to(k);
                }
            }
        };
    }

private:
    // ── spawn_worker ── creates a WorkerActor with a monitor recovery callback

    actor spawn_worker(int idx) {
        actor w = self_->spawn(actor_from_state<WorkerActor>, idx);

        // Monitor with callback — preferred non-deprecated API.
        self_->monitor(w, [this, idx](const error& reason) {
            tp t_detected = sc::now();
            self_->println("[Supervisor] Worker {} died: {}", idx, to_string(reason));

            actor_addr dead_addr = workers_[idx].address();

            auto it = in_flight_.find(dead_addr);
            if (it != in_flight_.end()) {
                WorkBatch lost = it->second;
                in_flight_.erase(it);

                // Locate the most recently created fault record for this slot
                // that has not yet been dispatched (it was just injected).
                int fi = -1;
                for (int i = (int)fault_records_.size() - 1; i >= 0; --i) {
                    if (fault_records_[i].worker_slot == idx
                            && !fault_records_[i].dispatched) {
                        fi = i;
                        fault_records_[fi].batch_id   = lost.batch_id;
                        fault_records_[fi].t_detected = t_detected;
                        break;
                    }
                }

                double elapsed_ms = ms_between(t_start_, t_detected);
                self_->println("[FAULT_DETECTED] fault={} worker={} batch={} time_ms={:.3f}",
                               fi, idx, lost.batch_id, elapsed_ms);

                // Tag the re-queued batch for recovery tracking.
                lost.fault_index = fi;
                work_queue_.push_front(lost);  // high-priority: run next
            } else {
                self_->println("[Supervisor] Worker {} was idle when it died.", idx);
            }

            // Respawn and immediately give it work (recovery batch or next).
            workers_[idx] = spawn_worker(idx);
            self_->println("[Supervisor] Worker {} respawned.", idx);
            self_->println("[WORKER_RESPAWN] worker={} time_s={:.6f}",
                           idx, ms_between(t_start_, sc::now()) / 1000.0);
            if (!work_queue_.empty())
                dispatch_to(idx);
        });

        return w;
    }

    // ── dispatch_to ── pulls the next batch from work_queue_ and sends it to
    //   workers_[idx].  Recovery batches (fault_index >= 0) get their dispatch
    //   timestamp recorded here.

    void dispatch_to(int idx) {
        WorkBatch batch = work_queue_.front();
        work_queue_.pop_front();

        // Recovery dispatch: stamp the dispatch timestamp.
        if (batch.fault_index >= 0) {
            int fi = batch.fault_index;
            if (fi < (int)fault_records_.size()) {
                fault_records_[fi].t_dispatched = sc::now();
                fault_records_[fi].dispatched   = true;
                double d2d = ms_between(fault_records_[fi].t_detected,
                                        fault_records_[fi].t_dispatched);
                self_->println("[FAULT_DISPATCH] fault={} worker={} batch={} "
                               "detect_to_dispatch_ms={:.3f}",
                               fi, idx, batch.batch_id, d2d);
            }
        }

        self_->println("[Supervisor] Dispatch batch {:3d}/{} → Worker {}{}",
                       batch.batch_id, total_batches_ - 1, idx,
                       batch.fault_index >= 0 ? " [RECOVERY]" : "");

        // Snapshot address now so in-flight record survives a potential worker replace.
        actor_addr src = workers_[idx].address();
        in_flight_.emplace(src, batch);
        self_->println("[BATCH_START] batch={} worker={} time_s={:.6f}",
                       batch.batch_id, idx, ms_between(t_start_, sc::now()) / 1000.0);

        self_->mail(batch.seed, batch.num_samples)
            .request(workers_[idx], infinite)
            .then(
                [this, idx, src, batch](int M) {
                    total_hits_ += M;
                    batches_done_++;
                    in_flight_.erase(src);
                    self_->println("[BATCH_END] batch={} worker={} time_s={:.6f}",
                                   batch.batch_id, idx, ms_between(t_start_, sc::now()) / 1000.0);

                    // Recovery completion: stamp t_completed and print latency.
                    if (batch.fault_index >= 0) {
                        int fi = batch.fault_index;
                        if (fi < (int)fault_records_.size()) {
                            fault_records_[fi].t_completed = sc::now();
                            fault_records_[fi].completed   = true;
                            double d2d   = ms_between(fault_records_[fi].t_detected,
                                                      fault_records_[fi].t_dispatched);
                            double d2c   = ms_between(fault_records_[fi].t_dispatched,
                                                      fault_records_[fi].t_completed);
                            double total = ms_between(fault_records_[fi].t_detected,
                                                      fault_records_[fi].t_completed);
                            self_->println("[FAULT_RECOVERED] fault={} worker={} batch={} "
                                           "detect_to_dispatch_ms={:.3f} "
                                           "dispatch_to_complete_ms={:.3f} "
                                           "total_recovery_ms={:.3f}",
                                           fi, fault_records_[fi].worker_slot,
                                           batch.batch_id, d2d, d2c, total);
                        }
                    }

                    // Periodic progress reporting (~every 5% of total batches).
                    int interval = std::max(1, total_batches_ / 20);
                    if (batches_done_ % interval == 0 || batches_done_ == total_batches_) {
                        double elapsed_s = ms_between(t_start_, sc::now()) / 1000.0;
                        double rate      = batches_done_ / elapsed_s;
                        double pi_est    = 4.0 * total_hits_
                                           / ((double)batches_done_ * samples_per_batch_);
                        self_->println("[PROGRESS] batch={}/{} pi={:.6f} "
                                       "time_s={:.3f} rate={:.2f} batches/s",
                                       batches_done_, total_batches_, pi_est, elapsed_s, rate);
                    }

                    // Check whether any fault trigger point has been reached.
                    check_fault_injection();

                    if (batches_done_ == total_batches_) {
                        finish();
                        return;
                    }
                    if (!work_queue_.empty())
                        dispatch_to(idx);
                },
                [this, idx](const error& err) {
                    // Worker died mid-request; monitor callback already re-queued the batch.
                    self_->println("[Supervisor] Request to Worker {} failed: {} "
                                   "(batch rescheduled via monitor callback)",
                                   idx, to_string(err));
                });
    }

    // ── check_fault_injection ── called on every batch completion.
    //   Injects the next scheduled fault when batches_done_ hits the trigger.
    //   Cycles through worker slots so different workers are targeted.

    void check_fault_injection() {
        while (faults_injected_ < num_faults_
               && batches_done_ >= fault_triggers_[faults_injected_]) {
            int slot = faults_injected_ % num_workers_;
            self_->println("[Supervisor] *** Injecting fault {}/{}: killing Worker {} "
                           "at batches_done={}/{} ***",
                           faults_injected_ + 1, num_faults_,
                           slot, batches_done_, total_batches_);

            // Pre-register record; t_detected is set inside the monitor callback.
            FaultRecord fr;
            fr.fault_index = faults_injected_;
            fr.worker_slot = slot;
            fault_records_.push_back(fr);

            anon_send_exit(workers_[slot], exit_reason::kill);
            faults_injected_++;
        }
    }

    // ── finish ── prints final statistics and signals the main thread.

    void finish() {
        tp     t_end      = sc::now();
        double elapsed_s  = ms_between(t_start_, t_end) / 1000.0;
        long long total_s = (long long)batches_done_ * samples_per_batch_;
        double pi_final   = 4.0 * total_hits_ / (double)total_s;
        double error_pct  = std::abs(pi_final - M_PI) / M_PI * 100.0;
        double throughput = batches_done_ / elapsed_s;

        self_->println("\n=== RESULT ===");
        self_->println("Mode              : {}",
                       no_fault_ ? "no-fault baseline" : "fault-injection");
        self_->println("Workers           : {}", num_workers_);
        self_->println("Batches completed : {}", batches_done_);
        self_->println("Total samples     : {}", total_s);
        self_->println("Total hits        : {}", total_hits_);
        self_->println("π estimate        : {:.8f}", pi_final);
        self_->println("Error vs π        : {:.4f}%", error_pct);
        self_->println("Elapsed time      : {:.3f} s", elapsed_s);
        self_->println("Throughput        : {:.2f} batches/s", throughput);
        self_->println("Faults injected   : {}", faults_injected_);

        if (!fault_records_.empty()) {
            self_->println("\n=== RECOVERY LATENCIES ===");
            for (auto& fr : fault_records_) {
                if (fr.completed) {
                    double d2d   = ms_between(fr.t_detected, fr.t_dispatched);
                    double d2c   = ms_between(fr.t_dispatched, fr.t_completed);
                    double total = ms_between(fr.t_detected, fr.t_completed);
                    self_->println("  Fault {:2d}: worker={} batch={:3d}  "
                                   "detect→dispatch={:.3f}ms  "
                                   "dispatch→complete={:.3f}ms  "
                                   "total_recovery={:.3f}ms",
                                   fr.fault_index, fr.worker_slot, fr.batch_id,
                                   d2d, d2c, total);
                } else {
                    self_->println("  Fault {:2d}: worker={} "
                                   "(incomplete — batch never recovered)",
                                   fr.fault_index, fr.worker_slot);
                }
            }
        }

        // Machine-readable summary line for benchmark script parsing.
        self_->println("\n[RESULT] mode={} workers={} batches={} total_samples={} "
                       "pi={:.8f} error_pct={:.4f} elapsed_s={:.3f} "
                       "throughput={:.2f} faults={}",
                       no_fault_ ? "baseline" : "fault",
                       num_workers_, batches_done_, total_s,
                       pi_final, error_pct, elapsed_s, throughput, faults_injected_);

        // Shut down all workers gracefully.
        for (auto& w : workers_)
            anon_send_exit(w, exit_reason::user_shutdown);

        self_->mail(done_atom_v).send(parent_);
        self_->quit();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Config & entry point
// ─────────────────────────────────────────────────────────────────────────────
class config : public actor_system_config {
public:
    int  num_workers       = 4;
    int  total_batches     = 200;
    int  samples_per_batch = 50'000'000;
    int  num_faults        = 1;
    bool no_fault          = false;

    config() {
        opt_group{custom_options_, "global"}
            .add(num_workers,       "num-workers,W",
                 "number of GPU worker actors (default: 4)")
            .add(total_batches,     "num-batches,B",
                 "total Monte Carlo batches (default: 200)")
            .add(samples_per_batch, "samples-per-batch,S",
                 "RNG samples per batch (default: 50 000 000)")
            .add(num_faults,        "num-faults,F",
                 "number of faults to inject, evenly spaced (default: 1)")
            .add(no_fault,          "no-fault,n",
                 "disable fault injection — baseline throughput run");
    }
};

void caf_main(actor_system& sys, const config& cfg) {
    caf::cuda::manager::init(sys);

    sys.println("=== Monte Carlo π Estimation (Fault-Tolerant, SC26) ===");
    sys.println("Workers: {}  Batches: {}  Samples/batch: {}  Faults: {}  Mode: {}",
                cfg.num_workers, cfg.total_batches, cfg.samples_per_batch,
                cfg.num_faults, cfg.no_fault ? "baseline" : "fault-injection");
    sys.println("Total samples planned: {}\n",
                (long long)cfg.total_batches * cfg.samples_per_batch);

    scoped_actor self{sys};

    actor supervisor = self->spawn(
        actor_from_state<Supervisor>,
        cfg.num_workers, cfg.total_batches, cfg.samples_per_batch,
        cfg.num_faults, cfg.no_fault,
        actor_cast<actor>(self));

    self->mail(start_atom_v).send(supervisor);

    // Block until the supervisor signals done.
    self->receive([](done_atom) {});

    caf::cuda::manager::shutdown();
}

CAF_MAIN(id_block::monte_carlo_app)
