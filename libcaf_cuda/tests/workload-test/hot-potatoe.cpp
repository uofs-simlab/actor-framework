#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <deque>
#include <vector>
#include <random>
#include <chrono>
#include <filesystem>
#include <memory>
#include <cstdint>
#include <atomic>

#include "caf/actorSOLVE/actorSOLVE.hpp"
#include "sparse_utils.hpp"

using namespace caf;
using namespace caf::cuda;
namespace fs = std::filesystem;

constexpr uint32_t WORKLOAD_SEED = 42;

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

struct PrefetchedTask {
    MatrixTask task;
    int device_id;
    size_t memory_usage;
    mem_ptr<int32_t> row_ptr;
    mem_ptr<int32_t> col_indices;
    mem_ptr<float> values;
    mem_ptr<float> b;
    mem_ptr<float> x_guess;
    bool is_valid = false;
};

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
    CAF_ADD_ATOM(workload_test, memory_report_atom)
    CAF_ADD_ATOM(workload_test, prefetch_done_atom)
    CAF_ADD_ATOM(workload_test, shutdown_atom)

    CAF_ADD_TYPE_ID(workload_test, (SolverType))
    CAF_ADD_TYPE_ID(workload_test, (MatrixTask))
    CAF_ADD_TYPE_ID(workload_test, (std::vector<MatrixTask>))
    CAF_ADD_TYPE_ID(workload_test, (std::shared_ptr<MatrixData>))
    CAF_ADD_TYPE_ID(workload_test, (std::deque<MatrixTask>))
    CAF_ADD_TYPE_ID(workload_test, (PrefetchedTask))

CAF_END_TYPE_ID_BLOCK(workload_test)

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(MatrixTask)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::vector<MatrixTask>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::shared_ptr<MatrixData>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(PrefetchedTask)

// ============================================================
// TASK ACTOR STATE
// ============================================================

struct worker_state {
    MatrixTask task;
    actor supervisor;
    actor neighbor;

    int device_id;
    int stream_id;

    actor cg_facade;
    PrefetchedTask pref_task;
    bool solve_done = false;
    bool neighbor_received = false;
    size_t memory_usage = 0;
};

// ============================================================
// TASK ACTOR
// ============================================================

size_t get_task_memory(const MatrixTask& t) {
    if (!t.data) return 0;
    auto& d = *t.data;
    return (d.row_ptr.size() + d.col_indices.size()) * sizeof(int32_t) 
           + (d.values.size() + d.b.size() + d.x_guess.size()) * sizeof(float);
}

behavior worker_actor(stateful_actor<worker_state>* self, MatrixTask t, actor supervisor) {
    auto& st = self->state();
    st.task = std::move(t);
    st.supervisor = supervisor;
    st.cg_facade = self->spawn<sparse_cg_facade<float>, linked>(0);
    st.memory_usage = get_task_memory(st.task);
    
    return {
        [=](start_atom, int dev, int stream) {
            auto& s = self->state();
            s.device_id = dev;
            s.stream_id = stream;

            // Alert master that we have begun work so it can find us a neighbor
            self->mail(started_atom_v, dev).send(s.supervisor);

            // Execute solver
            if (s.pref_task.is_valid) {
                auto& pt = s.pref_task;
                self->mail(
                    pt.row_ptr,
                    pt.col_indices,
                    pt.values,
                    pt.b,
                    pt.x_guess,
                    matrix_format::csr,
                    (int)pt.task.data->row_ptr.size() - 1,
                    (int)pt.task.data->values.size(),
                    1e-5f, 2000, dev, stream
                ).send(s.cg_facade);
            } else {
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
                    1e-5f, 2000, dev, stream
                ).send(s.cg_facade);
            }
        },

        [=](PrefetchedTask& pt) {
            auto& s = self->state();
            s.pref_task = std::move(pt);
            s.task = s.pref_task.task;
            s.memory_usage = s.pref_task.memory_usage;
        },

        [=](neighbor_atom, actor n) {
            auto& s = self->state();
            s.neighbor = n;
            s.neighbor_received = true;

            if (s.solve_done && s.neighbor) {
                self->mail(start_atom_v, s.device_id, s.stream_id).send(s.neighbor);
                self->mail(memory_report_atom_v, s.device_id, static_cast<int64_t>(s.memory_usage)).send(s.supervisor);
                self->quit();
            }
        },

        [=](uint32_t, int, std::vector<float>&, solver_result_meta) {
            auto& s = self->state();
            s.solve_done = true;
            if (s.neighbor_received && s.neighbor) {
                self->mail(start_atom_v, s.device_id, s.stream_id).send(s.neighbor);
                self->mail(memory_report_atom_v, s.device_id, static_cast<int64_t>(s.memory_usage)).send(s.supervisor);
                self->quit();
            }
        }
    };
}

// ============================================================
// PREFETCHER ACTOR
// ============================================================
behavior prefetcher_actor(stateful_actor<bool>* self, MatrixTask task, int dev, actor supervisor, int prefetch_stream) {
    auto runner = std::make_shared<command_runner<>>();
    auto& d = *task.data;

    auto rp = runner->transfer_memory(dev, prefetch_stream, create_in_arg(d.row_ptr));
    auto ci = runner->transfer_memory(dev, prefetch_stream, create_in_arg(d.col_indices));
    auto val = runner->transfer_memory(dev, prefetch_stream, create_in_arg(d.values));
    auto b = runner->transfer_memory(dev, prefetch_stream, create_in_arg(d.b));
    auto x = runner->transfer_memory(dev, prefetch_stream, create_in_out_arg(d.x_guess));

    size_t mem = get_task_memory(task);
    PrefetchedTask pt{std::move(task), dev, mem, rp, ci, val, b, x, true};

    runner->add_callback(prefetch_stream, dev, [pt_moved = std::move(pt), supervisor, self_handle = actor_cast<actor>(self)]() mutable {
        anon_mail(prefetch_done_atom_v, std::move(pt_moved)).send(supervisor);
        anon_mail(shutdown_atom_v).send(self_handle);
    });

    return {
        [=](shutdown_atom) {
            self->quit();
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
    st.rng.seed(WORKLOAD_SEED);

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
                } else {
                    self->quit();
                }
            } else {
                self->quit();
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
    std::vector<std::deque<actor>> warm_workers;
    std::vector<size_t> gpu_free_mem;
    
    size_t total_expected;
    size_t completed = 0;
    int num_gpus;
    int streams_per_gpu;
    bool initialized = false;
    int prefetch_stream_counter = 0;
    actor parent;
};

// ============================================================
// SUPERVISOR
// ============================================================

behavior supervisor_actor(stateful_actor<supervisor_state>* self,
                          size_t total_tasks,
                          int num_gpus,
                          int streams_per_gpu,
                          actor parent) {
    self->state().total_expected = total_tasks;
    self->state().parent = std::move(parent);
    self->state().num_gpus = num_gpus;
    self->state().streams_per_gpu = streams_per_gpu;
    self->state().warm_workers.resize(num_gpus);

    for (int i = 0; i < num_gpus; ++i) {
        auto dev = manager::get().find_device(i);
        self->state().gpu_free_mem.push_back(dev ? dev->total_memory_bytes() : 0);
    }

    if (total_tasks == 0) {
        self->mail(worker_done_atom_v).send(self->state().parent);
        self->quit();
        return {};
    }

    auto check_prefetch = [=]() {
        auto& s = self->state();
        // Only prefetch if we don't have idle slots (saturated)
        if (!s.available_slots.empty())
            return;

        for (int i = 0; i < s.num_gpus; ++i) {
            while (s.gpu_free_mem[i] > 3ULL * 1024 * 1024 * 1024 && !s.pending_queue.empty()) {
                auto task = std::move(s.pending_queue.front());
                s.pending_queue.pop_front();
                
                size_t cost = get_task_memory(task);
                s.gpu_free_mem[i] -= cost; 
                s.completed++;
                
                int p_stream = -1 - (s.prefetch_stream_counter++ % 5);
                self->spawn(prefetcher_actor, std::move(task), i, actor_cast<actor>(self), p_stream);
            }
        }
    };

    return {
        [=](add_work_atom, std::vector<MatrixTask>& batch) {
            auto& s = self->state();
            for (auto& t : batch)
                s.pending_queue.push_back(std::move(t));

            // Bootstrapping: fill available GPU slots first (cold start)
            if (!s.initialized) {
                s.initialized = true;
                size_t total_slots = static_cast<size_t>(num_gpus * streams_per_gpu);
                for (size_t i = 0; i < total_slots; ++i) {
                    int dev = static_cast<int>(i / streams_per_gpu);
                    int stream = static_cast<int>(i % streams_per_gpu);
                    
                    if (!s.pending_queue.empty()) {
                        auto t = std::move(s.pending_queue.front());
                        s.pending_queue.pop_front();
                        s.gpu_free_mem[dev] -= get_task_memory(t);
                        s.completed++;

                        auto w = self->spawn(worker_actor, std::move(t), actor_cast<actor>(self));
                        self->mail(start_atom_v, dev, stream).send(w);
                    } else {
                        s.available_slots.push_back({dev, stream});
                    }
                }
                check_prefetch();
            } else {
                // New work arrived, check if we have idle GPU slots to fill
                while (!s.available_slots.empty() && !s.pending_queue.empty()) {
                    auto slot = s.available_slots.front();
                    s.available_slots.pop_front();
                    
                    auto t = std::move(s.pending_queue.front());
                    s.pending_queue.pop_front();
                    s.gpu_free_mem[slot.device_id] -= get_task_memory(t);
                    s.completed++;

                    auto w = self->spawn(worker_actor, std::move(t), actor_cast<actor>(self));
                    self->mail(start_atom_v, slot.device_id, slot.stream_id).send(w);
                }
                check_prefetch();
            }
        },

        [=](started_atom, int dev) {
            auto& s = self->state();
            if (!s.warm_workers[dev].empty()) {
                auto next_worker = s.warm_workers[dev].front();
                s.warm_workers[dev].pop_front();
                self->mail(neighbor_atom_v, next_worker).send(actor_cast<actor>(self->current_sender()));
            } else if (!s.pending_queue.empty()) {
                auto t = std::move(s.pending_queue.front());
                s.pending_queue.pop_front();
                s.gpu_free_mem[dev] -= get_task_memory(t);
                s.completed++;

                auto next_worker = self->spawn(worker_actor, std::move(t), actor_cast<actor>(self));
                self->mail(neighbor_atom_v, next_worker).send(actor_cast<actor>(self->current_sender()));
            } else {
                self->mail(neighbor_atom_v, actor_cast<actor>(self)).send(actor_cast<actor>(self->current_sender()));
            }
            check_prefetch();
        },

        [=](start_atom, int dev, int stream) {
            auto& s = self->state();
            if (!s.warm_workers[dev].empty()) {
                auto w = s.warm_workers[dev].front();
                s.warm_workers[dev].pop_front();
                self->mail(start_atom_v, dev, stream).send(w);
            } else if (!s.pending_queue.empty()) {
                auto t = std::move(s.pending_queue.front());
                s.pending_queue.pop_front();
                s.gpu_free_mem[dev] -= get_task_memory(t);
                s.completed++;

                auto w = self->spawn(worker_actor, std::move(t), actor_cast<actor>(self));
                self->mail(start_atom_v, dev, stream).send(w);
            } else {
                s.available_slots.push_back({dev, stream});
            }

            size_t total_slots = static_cast<size_t>(s.num_gpus * s.streams_per_gpu);
            if (s.completed == s.total_expected && s.available_slots.size() == total_slots) {
                self->println("[INFO] All {} tasks finished. Shutting down.", s.total_expected);
                self->mail(worker_done_atom_v).send(s.parent);
                self->quit();
            }
            check_prefetch();
        },

        [=](memory_report_atom, int dev, int64_t delta) {
            auto& s = self->state();
            if (dev >= 0 && dev < static_cast<int>(s.gpu_free_mem.size())) {
                s.gpu_free_mem[dev] += static_cast<size_t>(delta);
            }
            check_prefetch();
        },

        [=](prefetch_done_atom, PrefetchedTask& pt) {
            auto& s = self->state();
            int dev = pt.device_id;
            // Create the actor but don't solve yet
            auto w = self->spawn(worker_actor, MatrixTask{}, actor_cast<actor>(self));
            self->mail(std::move(pt)).send(w);
            s.warm_workers[dev].push_back(w);
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

    //  auto tasks = scan_for_matrices(
    //     "/scratch/nqr159/matrix-collection/matrices/spd",
    //     CGS_SOLVER
    // );

    manager& mgr = manager::get();
    int gpus = mgr.get_num_devices();

    std::cout << "[INFO] Streams/GPU:      " << streams << "\n";
    std::cout << "[INFO] Batches:          " << batches << "\n";
    std::cout << "[INFO] Batch size:       " << batch_size << "\n";
    std::cout << "[INFO] Mean arrival:     " << mean_arrival << " ms\n";

    auto start = std::chrono::steady_clock::now();
    scoped_actor self{sys};

    auto supervisor = self->spawn(supervisor_actor,
              static_cast<size_t>(batches * batch_size),
              gpus,
              streams,
              actor_cast<actor>(self));

    auto producer = self->spawn(producer_actor,
                             std::move(tasks), batches, batch_size, mean_arrival,
                             supervisor);

    anon_mail(work_tick_atom_v).send(producer);

    self->receive(
        [&](worker_done_atom) {
            std::cout << "[INFO] Supervisor signaled completion. Shutting down...\n";
        }
    );

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    int total_tasks = batches * batch_size;

    std::cout << "\n===== BENCHMARK COMPLETE =====\n";
    std::cout << "Seed:               " << WORKLOAD_SEED << "\n";
    std::cout << "Streams per GPU:    " << streams << "\n";
    std::cout << "Batches:            " << batches << "\n";
    std::cout << "Batch Size:         " << batch_size << "\n";
    std::cout << "Mean Arrival (ms):  " << mean_arrival << "\n";
    std::cout << "Tasks Processed:    " << total_tasks << "\n";
    std::cout << "Total Runtime:      " << elapsed.count() << " s\n";
    std::cout << "Throughput:         " << total_tasks / elapsed.count() << " tasks/s\n";
    std::cout << "==============================\n";

    manager::shutdown();
}

CAF_MAIN(id_block::cuda, id_block::workload_test)
