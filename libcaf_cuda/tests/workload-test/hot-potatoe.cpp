#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <deque>
#include <vector>
#include <random>
#include <chrono>
#include <filesystem>
#include <memory>

#include "caf/actorSOLVE/actorSOLVE.hpp"
#include "sparse_utils.hpp"

using namespace caf;
using namespace caf::cuda;
namespace fs = std::filesystem;

template <class Inspector>
bool inspect(Inspector& f, SolverType& x) {
    auto val = static_cast<int>(x);
    if (f.apply(val)) {
        if constexpr (Inspector::is_loading)
            x = static_cast<SolverType>(val);
        return true;
    }
    return false;
}

// ============================================================
// TYPE BLOCK (unchanged + extended)
// ============================================================

CAF_BEGIN_TYPE_ID_BLOCK(workload_test, caf::id_block::cuda::end)

    CAF_ADD_ATOM(workload_test, get_work_atom)
    CAF_ADD_ATOM(workload_test, started_atom)
    CAF_ADD_ATOM(workload_test, neighbor_atom)
    CAF_ADD_ATOM(workload_test, request_work_atom)
    CAF_ADD_ATOM(workload_test, worker_done_atom)
    CAF_ADD_ATOM(workload_test, work_tick_atom)
    CAF_ADD_ATOM(workload_test, add_work_atom)

    CAF_ADD_TYPE_ID(workload_test, (SolverType))
    CAF_ADD_TYPE_ID(workload_test, (MatrixTask))
    CAF_ADD_TYPE_ID(workload_test, (std::vector<MatrixTask>))
    CAF_ADD_TYPE_ID(workload_test, (std::shared_ptr<MatrixData>))
    CAF_ADD_TYPE_ID(workload_test, (std::deque<MatrixTask>))

CAF_END_TYPE_ID_BLOCK(workload_test)

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(MatrixTask)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::vector<MatrixTask>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::shared_ptr<MatrixData>)

// ============================================================
// TASK ACTOR STATE
// ============================================================

struct worker_state {
    MatrixTask task;
    actor supervisor;
    actor neighbor;

    int device_id;
    int stream_id;

    caf::actor cg_facade;
    
    bool solve_done = false;
    bool neighbor_received = false;
};

// ============================================================
// TASK ACTOR
// ============================================================

behavior worker_actor(stateful_actor<worker_state>* self, MatrixTask t, actor supervisor) {
    auto& st = self->state();
    st.task = std::move(t);
    st.supervisor = supervisor;
    st.cg_facade = self->spawn<sparse_cg_facade<float>, linked>(0);
    
    return {
        [=](start_atom, int dev, int stream) {
            auto& s = self->state();
            s.device_id = dev;
            s.stream_id = stream;

            // Alert master that we have begun work
            self->mail(started_atom_v).send(s.supervisor);

            // Execute solver
            auto& d = *s.task.data;
            self->mail(
                create_in_arg(d.row_ptr),
                create_in_arg(d.col_indices),
                create_in_arg(d.values),
                create_in_arg(d.b),
                create_in_out_arg(d.x_guess),
                matrix_format::csr,
                (int)d.row_ptr.size() - 1,
                (int)d.values.size(),
                1e-5f,
                2000,
                dev,
                stream
            ).send(s.cg_facade);
        },

        [=](neighbor_atom, actor n) {
            auto& s = self->state();
            s.neighbor = n;
            s.neighbor_received = true;

            if (s.solve_done && s.neighbor) {
                self->mail(start_atom_v, s.device_id, s.stream_id).send(s.neighbor);
                self->quit();
            }
        },

        [=](uint32_t, int, std::vector<float>&, solver_result_meta) {
            auto& s = self->state();
            s.solve_done = true;
            if (s.neighbor_received && s.neighbor) {
                self->mail(start_atom_v, s.device_id, s.stream_id).send(s.neighbor);
                self->quit();
            }
        }
    };
}

// ============================================================
// PRODUCER ACTOR
// ============================================================

struct producer_state {
    std::vector<MatrixTask> matrix_pool;
    int num_batches;
    int batch_size;
    double mean_arrival_ms;
    actor supervisor;
    std::mt19937 rng;
    int batches_sent = 0;
};

behavior producer_actor(stateful_actor<producer_state>* self,
                       std::vector<MatrixTask> pool,
                       int num_batches, int batch_size, 
                       double mean_ms, actor supervisor) {
    auto& st = self->state();
    st.matrix_pool = std::move(pool);
    st.num_batches = num_batches;
    st.batch_size = batch_size;
    st.mean_arrival_ms = mean_ms;
    st.supervisor = std::move(supervisor);
    st.rng.seed(42);

    return {
        [=](work_tick_atom) {
            auto& s = self->state();
            if (s.batches_sent < s.num_batches) {
                auto batch = generate_batch(s.matrix_pool, s.rng, s.batch_size);
                self->mail(add_work_atom_v, std::move(batch)).send(s.supervisor);
                s.batches_sent++;

                if (s.batches_sent < s.num_batches) {
                    auto delay = generate_random_interval(s.rng, s.mean_arrival_ms);
                    self->mail(work_tick_atom_v).delay(delay).send(self);
                }
            }
        }
    };
}

// ============================================================
// SUPERVISOR STATE
// ============================================================

struct resource_slot {
    int device_id;
    int stream_id;
};

struct supervisor_state {
    std::deque<MatrixTask> pending_queue;
    std::deque<resource_slot> available_slots;
    
    size_t total_expected;
    size_t completed = 0;
    bool initialized = false;
};

// ============================================================
// SUPERVISOR
// ============================================================

behavior supervisor_actor(stateful_actor<supervisor_state>* self,
                          size_t total_tasks,
                          int num_gpus,
                          int streams_per_gpu) {
    self->state().total_expected = total_tasks;

    return {
        [=](add_work_atom, std::vector<MatrixTask>& batch) {
            auto& s = self->state();
            for (auto& t : batch)
                s.pending_queue.push_back(std::move(t));

            // Bootstrapping: fill GPU slots with the first available tasks
            if (!s.initialized) {
                s.initialized = true;
                size_t total_slots = static_cast<size_t>(num_gpus * streams_per_gpu);
                for (size_t i = 0; i < total_slots; ++i) {
                    int dev = static_cast<int>(i / streams_per_gpu);
                    int stream = static_cast<int>(i % streams_per_gpu);
                    
                    if (!s.pending_queue.empty()) {
                        auto w = self->spawn(worker_actor, std::move(s.pending_queue.front()), actor_cast<actor>(self));
                        s.pending_queue.pop_front();
                        self->mail(start_atom_v, dev, stream).send(w);
                    } else {
                        s.available_slots.push_back({dev, stream});
                    }
                }
            } else {
                // New work arrived, check if we have idle GPU slots to fill
                while (!s.available_slots.empty() && !s.pending_queue.empty()) {
                    auto slot = s.available_slots.front();
                    s.available_slots.pop_front();
                    auto w = self->spawn(worker_actor, std::move(s.pending_queue.front()), actor_cast<actor>(self));
                    s.pending_queue.pop_front();
                    self->mail(start_atom_v, slot.device_id, slot.stream_id).send(w);
                }
            }
        },

        [=](started_atom) {
            auto& s = self->state();
            if (!s.pending_queue.empty()) {
                auto next_worker = self->spawn(worker_actor, std::move(s.pending_queue.front()), actor_cast<actor>(self));
                s.pending_queue.pop_front();
                self->mail(neighbor_atom_v, next_worker).send(actor_cast<actor>(self->current_sender()));
            } else {
                self->mail(neighbor_atom_v, actor_cast<actor>(self)).send(actor_cast<actor>(self->current_sender()));
            }
        },

        [=](start_atom, int dev, int stream) {
            auto& s = self->state();
            s.completed++;

            if (s.completed % 25 == 0 || s.completed == s.total_expected) {
                self->println("[PROGRESS] Completed {}/{} tasks", s.completed, s.total_expected);
            }

            if (!s.pending_queue.empty()) {
                auto w = self->spawn(worker_actor, std::move(s.pending_queue.front()), actor_cast<actor>(self));
                s.pending_queue.pop_front();
                self->mail(start_atom_v, dev, stream).send(w);
            } else {
                s.available_slots.push_back({dev, stream});
            }

            if (s.completed >= s.total_expected) {
                self->println("[INFO] All {} tasks completed. Shutting down.", s.total_expected);
                self->quit();
            }
        }
    };
}

// ============================================================
// MAIN
// ============================================================

void caf_main(actor_system& sys) {

    manager::init(sys, manager_config(true, true));

    int streams = 4;
    int batches = 25;
    int batch_size = 100;
    double mean_arrival = 1000.0;

    auto tasks = scan_for_matrices(
        "/scratch/nqr159/matrix-collection/matrix_corpus_v2/matrices/spd",
        CGS_SOLVER
    );

    manager& mgr = manager::get();
    int gpus = mgr.get_num_devices();

    auto supervisor = sys.spawn(supervisor_actor,
              static_cast<size_t>(batches * batch_size),
              gpus,
              streams);

    auto producer = sys.spawn(producer_actor,
                             std::move(tasks), batches, batch_size, mean_arrival,
                             supervisor);

    anon_mail(work_tick_atom_v).send(producer);

    sys.await_all_actors_done();
    manager::shutdown();
}

CAF_MAIN(id_block::cuda, id_block::workload_test)