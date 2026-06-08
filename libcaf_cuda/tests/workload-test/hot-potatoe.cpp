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

// ============================================================
// TYPE BLOCK (unchanged + extended)
// ============================================================

CAF_BEGIN_TYPE_ID_BLOCK(workload_test, caf::id_block::cuda::end)

    CAF_ADD_ATOM(workload_test, get_work_atom)
    CAF_ADD_ATOM(workload_test, start_atom)
    CAF_ADD_ATOM(workload_test, started_atom)
    CAF_ADD_ATOM(workload_test, neighbor_atom)
    CAF_ADD_ATOM(workload_test, request_work_atom)
    CAF_ADD_ATOM(workload_test, worker_done_atom)
    CAF_ADD_ATOM(workload_test, work_tick_atom)

    CAF_ADD_TYPE_ID(workload_test, (SolverType))
    CAF_ADD_TYPE_ID(workload_test, (MatrixTask))
    CAF_ADD_TYPE_ID(workload_test, (std::vector<MatrixTask>))
    CAF_ADD_TYPE_ID(workload_test, (std::shared_ptr<MatrixData>))

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
            self->send(s.supervisor, started_atom_v);

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
                self->send(s.neighbor, start_atom_v, s.device_id, s.stream_id);
                self->quit();
            }
        },

        [=](uint32_t, int, std::vector<float>&, solver_result_meta) {
            auto& s = self->state();
            s.solve_done = true;
            if (s.neighbor_received && s.neighbor) {
                self->send(s.neighbor, start_atom_v, s.device_id, s.stream_id);
                self->quit();
            }
        }
    };
}

// ============================================================
// SUPERVISOR STATE
// ============================================================

struct supervisor_state {
    std::vector<MatrixTask> batch;
    size_t next_task_idx = 0;
    size_t returned_potatoes = 0;
    size_t active_slots = 0;
};

// ============================================================
// SUPERVISOR
// ============================================================

behavior supervisor_actor(stateful_actor<supervisor_state>* self,
                          std::vector<MatrixTask> batch,
                          int num_gpus,
                          int streams_per_gpu) {
    auto& st = self->state();
    st.batch = std::move(batch);

    size_t total_slots = static_cast<size_t>(num_gpus * streams_per_gpu);

    // Dynamically spawn only the first set of workers to fill the GPU slots
    for (size_t i = 0; i < total_slots; ++i) {
        if (st.next_task_idx < st.batch.size()) {
            int dev = static_cast<int>(i / streams_per_gpu);
            int stream = static_cast<int>(i % streams_per_gpu);
            auto w = self->spawn(worker_actor, st.batch[st.next_task_idx++], actor_cast<actor>(self));
            self->send(w, start_atom_v, dev, stream);
            st.active_slots++;
        }
    }

    return {
        [=](started_atom) {
            auto& s = self->state();
            // Master Decision: Assign the next available task as a neighbor to the worker that just started
            if (s.next_task_idx < s.batch.size()) {
                auto next_worker = self->spawn(worker_actor, s.batch[s.next_task_idx++], actor_cast<actor>(self));
                self->send(self->current_sender(), neighbor_atom_v, next_worker);
            } else {
                // No more tasks to assign: the Supervisor becomes the neighbor (to reclaim resources)
                self->send(self->current_sender(), neighbor_atom_v, actor_cast<actor>(self));
            }
        },

        [=](start_atom, int, int) {
            auto& s = self->state();
            // Potato returned! One processing chain has finished.
            if (++s.returned_potatoes == s.active_slots) {
                self->println("[INFO] All {} GPU streams returned. Shutting down.", s.active_slots);
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

    int streams = 8;
    int batches = 25;
    int batch_size = 100;

    auto tasks = scan_for_matrices(
        "/scratch/nqr159/matrix-collection",
        CGS_SOLVER
    );

    auto batch = generate_batch(tasks, std::mt19937{42}, batch_size);

    manager& mgr = manager::get();
    int gpus = mgr.get_num_devices();

    sys.spawn(supervisor_actor,
              batch,
              gpus,
              streams);

    sys.await_all_actors_done();

    manager::shutdown();
}

CAF_MAIN(id_block::cuda, workload_test)