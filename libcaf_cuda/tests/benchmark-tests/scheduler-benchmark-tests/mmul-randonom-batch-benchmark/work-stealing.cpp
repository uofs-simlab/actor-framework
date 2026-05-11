#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <unordered_map>
#include <deque>

using namespace caf;
using namespace std::chrono_literals;

// ─────────────────────────────────────────────────────────────────────────────
// Atoms
// ─────────────────────────────────────────────────────────────────────────────
CAF_BEGIN_TYPE_ID_BLOCK(dynamic_work_stealing, caf::id_block::cuda::end)
    CAF_ADD_ATOM(dynamic_work_stealing, get_work_atom)
    CAF_ADD_ATOM(dynamic_work_stealing, task_done_atom)
    CAF_ADD_ATOM(dynamic_work_stealing, release_memory_atom)
    CAF_ADD_ATOM(dynamic_work_stealing, request_work_atom)
    CAF_ADD_ATOM(dynamic_work_stealing, worker_done_atom)
    CAF_ADD_ATOM(dynamic_work_stealing, refill_buffer_atom)
CAF_END_TYPE_ID_BLOCK(dynamic_work_stealing)

// Command runners for GPU operations
using mmul_kernel_t = caf::cuda::command_runner<caf::cuda::mem_ptr<int>, caf::cuda::mem_ptr<int>, out<int>, in<int>>;
mmul_kernel_t mmul_kernel;
caf::cuda::command_runner<> mmul_command;

struct MatrixPool {
    std::unordered_map<int, std::vector<int>> A;
    std::unordered_map<int, std::vector<int>> B;
};

// ---------------------------- GLOBAL TASK POOL ----------------------------
// The central source of truth for work. Implements a pull-based model.
struct task_pool_state {
    std::vector<int> tasks;
    size_t next_task_idx = 0;
};

caf::behavior global_task_pool(caf::stateful_actor<task_pool_state>* self, std::vector<int> tasks) {
    self->state().tasks = std::move(tasks);
    return {
        [=](get_work_atom, size_t batch_size) -> result<std::vector<int>> {
            auto& st = self->state();
            if (st.next_task_idx >= st.tasks.size())
                return sec::end_of_stream;
            size_t count = std::min(batch_size, st.tasks.size() - st.next_task_idx);
            std::vector<int> batch(st.tasks.begin() + st.next_task_idx, 
                                   st.tasks.begin() + st.next_task_idx + count);
            st.next_task_idx += count;
            return batch;
        }
    };
}

// ---------------------------- DEVICE BROKER ----------------------------
// Manages memory for a specific GPU and steals (pulls) work from the Global Pool.
struct device_actor_state {
    MatrixPool pool;
    caf::actor global_pool;
    std::deque<int> local_tasks; // Local buffer to keep GPU busy
    size_t total_mem = 0;
    size_t current_mem = 0;
    int device_id = -1;
    int active_workers = 0;
    size_t batch_size = 0;
    size_t low_water_mark = 0;
    bool fetching = false;
};

caf::behavior gpu_device_actor(caf::stateful_actor<device_actor_state>* self,
                               MatrixPool pool, caf::actor global_pool, int num_workers, int dev_id, int max_in_flight) {
    self->state().pool = std::move(pool);
    self->state().global_pool = global_pool;
    self->state().device_id = dev_id;
    self->state().active_workers = num_workers;

    // Dynamically calculate prefetch markers based on the total pipeline capacity
    self->state().low_water_mark = static_cast<size_t>(num_workers * max_in_flight);
    self->state().batch_size = self->state().low_water_mark * 2;

    auto dev_obj = caf::cuda::manager::get().find_device(dev_id);
    if (dev_obj)
        self->state().total_mem = dev_obj->total_memory_bytes();

    // Helper to refill the local task buffer from the global pool
    auto refill = [=]() {
        auto& st = self->state();
        if (st.fetching || st.local_tasks.size() >= st.low_water_mark + st.batch_size)
            return;

        st.fetching = true;
        self->mail(get_work_atom_v, (size_t)st.batch_size).request(st.global_pool, infinite).then(
            [=](std::vector<int>& batch) {
                auto& st_inner = self->state();
                for (int N : batch)
                    st_inner.local_tasks.push_back(N);
                st_inner.fetching = false;
                if (st_inner.local_tasks.size() < st_inner.low_water_mark)
                    self->mail(refill_buffer_atom_v).send(self);
            },
            [=](error& err) {
                self->state().fetching = false;
            }
        );
    };

    return {
        [=](refill_buffer_atom) {
            refill();
        },
        [=](get_work_atom) -> caf::result<int, in<int>, in<int>> {
            auto& st = self->state();

            // If we have tasks locally, satisfy the request immediately
            if (!st.local_tasks.empty()) {
                int N = st.local_tasks.front();
                size_t needed = (size_t)N * N * sizeof(int) * 3;
                if (st.current_mem + needed > st.total_mem)
                    return make_error(sec::runtime_error, "Out of GPU Memory");
                
                st.local_tasks.pop_front();
                st.current_mem += needed;
                
                // Proactively steal more work if the buffer is getting low
                if (st.local_tasks.size() < st.low_water_mark)
                    refill();

                return {N, caf::cuda::create_in_arg(st.pool.A[N]), 
                           caf::cuda::create_in_arg(st.pool.B[N])};
            }

            // Buffer empty: must fetch from global pool reactively
            auto promise = self->make_response_promise<int, in<int>, in<int>>();
            self->mail(get_work_atom_v, (size_t)st.batch_size).request(st.global_pool, infinite).then(
                [=](std::vector<int>& batch) mutable {
                    auto& st_inner = self->state();
                    int N = batch.front();
                    for(size_t i = 1; i < batch.size(); ++i) st_inner.local_tasks.push_back(batch[i]);
                    
                    st_inner.current_mem += (size_t)N * N * sizeof(int) * 3;
                    promise.deliver(N, caf::cuda::create_in_arg(st_inner.pool.A[N]), 
                                       caf::cuda::create_in_arg(st_inner.pool.B[N]));
                },
                [=](error& err) mutable { promise.deliver(err); }
            );
            return promise;
        },
        [=](release_memory_atom, int N) {
            self->state().current_mem -= (size_t)N * N * sizeof(int) * 3;
            refill(); // Try to get more work now that memory is free
        },
        [=](worker_done_atom) {
            if (--self->state().active_workers <= 0)
                self->quit();
        }
    };
}

// ---------------------------- WORKER ----------------------------
struct worker_state {
    caf::actor device_actor;
    caf::actor supervisor;
    caf::cuda::program_ptr program;
    int dev_id;
    int stream_id;
    int in_flight = 0;
    int max_in_flight = 2; // Keep the pipeline full
    bool draining = false;
};

caf::behavior mmul_worker(caf::stateful_actor<worker_state>* self, 
                          caf::actor supervisor, caf::actor device_actor, 
                          caf::cuda::program_ptr prog, int dev, int stream, int max_in_flight) {
    self->state().supervisor = supervisor;
    self->state().device_actor = device_actor;
    self->state().program = prog;
    self->state().dev_id = dev;
    self->state().stream_id = stream;
    self->state().max_in_flight = max_in_flight;

    for (int i = 0; i < self->state().max_in_flight; ++i)
        self->mail(request_work_atom_v).send(self);

    return {
        [=](request_work_atom) {
            auto& st = self->state();
            if (st.in_flight >= st.max_in_flight || st.draining) return;
            st.in_flight++;
            self->mail(get_work_atom_v).request(st.device_actor, infinite).then(
                [=](int N, in<int> A, in<int> B) mutable {
                    auto& st_inner = self->state();
                    auto a1 = mmul_command.transfer_memory(st_inner.dev_id, st_inner.stream_id, std::move(A));
                    auto a2 = mmul_command.transfer_memory(st_inner.dev_id, st_inner.stream_id, std::move(B));
                    caf::cuda::nd_range dims((N + 31) / 32, (N + 31) / 32, 1, 32, 32, 1);
                    auto res = mmul_kernel.run_async(st_inner.program, dims, st_inner.stream_id, 0, st_inner.dev_id,
                                                     a1, a2, caf::cuda::create_out_arg<int>(N * N),
                                                     caf::cuda::create_in_arg<int>(N));
                    auto self_ptr = caf::actor_cast<caf::actor>(self);
                    mmul_command.copy_to_host_async(std::get<2>(res), [self_ptr, N](std::vector<int>&&) {
                        caf::anon_mail(task_done_atom_v, N).send(self_ptr);
                    });
                },
                [=](error& err) mutable {
                    auto& st_inner = self->state();
                    st_inner.in_flight--;
                    if (err == sec::end_of_stream) {
                        st_inner.draining = true;
                        if (st_inner.in_flight == 0) {
                            self->mail(worker_done_atom_v).send(st_inner.device_actor);
                            self->quit();
                        }
                    } else {
                        self->delayed_anon_send(self, 100ms, request_work_atom_v);
                    }
                }
            );
        },
        [=](task_done_atom, int N) {
            self->state().in_flight--;
            self->mail(1).send(self->state().supervisor);
            self->mail(release_memory_atom_v, N).send(self->state().device_actor);
            self->mail(request_work_atom_v).send(self);
        }
    };
}

// ---------------------------- SUPERVISOR ----------------------------
struct supervisor_state {
    int total;
    int done = 0;
    std::chrono::steady_clock::time_point start;
};

caf::behavior supervisor_actor(caf::stateful_actor<supervisor_state>* self, 
                               int total, 
                               std::vector<int> Ns,
                               int workers_per_gpu,
                               int max_in_flight) {
    self->state().total = total;
    self->state().start = std::chrono::steady_clock::now();
    auto pool = self->spawn(global_task_pool, std::move(Ns));
    
    MatrixPool m_pool;
    for (int n : {256, 2048}) {
        m_pool.A[n] = std::vector<int>(n * n, 1);
        m_pool.B[n] = std::vector<int>(n * n, 1);
    }

    auto& mgr = caf::cuda::manager::get();
    auto prog = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

    for (int i = 0; i < mgr.get_num_devices(); ++i) {
        auto broker = self->spawn(gpu_device_actor, m_pool, pool, workers_per_gpu, i, max_in_flight);
        for (int j = 0; j < workers_per_gpu; ++j)
            self->spawn(mmul_worker, self, broker, prog, i, (i * 100) + j, max_in_flight);
    }

    return {
        [=](int count) {
            self->state().done += count;
            if (self->state().done >= self->state().total) {
                auto elapsed = std::chrono::steady_clock::now() - self->state().start;
                std::cout << "Dynamic Work-Stealing Complete.\n"
                          << "Makespan: " << std::chrono::duration<double>(elapsed).count() << "s\n";
                caf::cuda::manager::shutdown();
                self->quit();
            }
        }
    };
}

void caf_main(caf::actor_system& sys) {
    caf::cuda::manager_config cfg(false);
    caf::cuda::manager::init(sys, cfg);

    int total_tasks = 2000;
    int workers_per_gpu = 8;
    int max_in_flight = 3;

    std::vector<int> Ns;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0); // Power Law setup

    // Power Law simulation: 10% Heavy (2048), 90% Light (256)
    for (int i = 0; i < total_tasks; ++i) {
        if (dist(rng) < 0.1) Ns.push_back(2048);
        else Ns.push_back(256);
    }

    sys.spawn(supervisor_actor, total_tasks, std::move(Ns), workers_per_gpu, max_in_flight);
    sys.await_all_actors_done();
}

CAF_MAIN(id_block::dynamic_work_stealing)
