#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <string>
#include <chrono>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include "caf/actorSOLVE/actorSOLVE.hpp"
#include "sparse_utils.hpp"

using namespace caf;
using namespace caf::cuda;
namespace fs = std::filesystem;

enum SolverType { CGS_SOLVER, BICSTAB_SOLVER };

struct MatrixTask {
    std::string path;
    SolverType type;
};

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

template <class Inspector>
bool inspect(Inspector& f, MatrixTask& x) {
    return f.object(x).fields(f.field("path", x.path), f.field("type", x.type));
}

CAF_BEGIN_TYPE_ID_BLOCK(workload_test, caf::id_block::bicgstab_actor::end)
    CAF_ADD_ATOM(workload_test, get_work_atom)
    CAF_ADD_ATOM(workload_test, release_memory_atom)
    CAF_ADD_ATOM(workload_test, request_work_atom)
    CAF_ADD_ATOM(workload_test, worker_done_atom)
    CAF_ADD_TYPE_ID(workload_test, (SolverType))
    CAF_ADD_TYPE_ID(workload_test, (MatrixTask))
    CAF_ADD_TYPE_ID(workload_test, (std::vector<MatrixTask>))
CAF_END_TYPE_ID_BLOCK(workload_test)

// ---------------------------- GLOBAL TASK POOL ----------------------------
struct pool_state {
    std::vector<MatrixTask> tasks;
    size_t next_task_idx = 0;
};

behavior global_task_pool(stateful_actor<pool_state>* self, std::vector<MatrixTask> tasks) {
    self->state().tasks = std::move(tasks);
    return {
        [=](get_work_atom, size_t batch_size) -> result<std::vector<MatrixTask>> {
            auto& st = self->state();
            if (st.next_task_idx >= st.tasks.size())
                return sec::end_of_stream;
            size_t count = std::min(batch_size, st.tasks.size() - st.next_task_idx);
            std::vector<MatrixTask> batch(st.tasks.begin() + st.next_task_idx, 
                                          st.tasks.begin() + st.next_task_idx + count);
            st.next_task_idx += count;
            return batch;
        }
    };
}

// ---------------------------- DEVICE/GPU ACTOR ----------------------------
struct MatrixData {
    std::vector<int> row_ptr;
    std::vector<int> col_indices;
    std::vector<float> values;
    std::vector<float> b;
    std::vector<float> x_guess;
};

struct device_actor_state {
    caf::actor global_pool;
    std::deque<MatrixTask> local_tasks;
    std::unordered_map<std::string, MatrixData> cache;
    int active_workers = 0;
    int device_id = -1;
    bool fetching = false;
    size_t low_water_mark;
    size_t batch_size;
};

behavior gpu_device_actor(stateful_actor<device_actor_state>* self,
                          caf::actor global_pool, int num_workers, int dev_id, int max_in_flight) {
    self->state().global_pool = global_pool;
    self->state().device_id = dev_id;
    self->state().active_workers = num_workers;

    // Dynamically calculate prefetch markers based on the total pipeline capacity
    self->state().low_water_mark = static_cast<size_t>(num_workers * max_in_flight);
    self->state().batch_size = self->state().low_water_mark * 2;

    auto refill = [=]() {
        auto& st = self->state();
        if (st.fetching || st.local_tasks.size() >= st.low_water_mark)
            return;

        st.fetching = true;
        self->mail(get_work_atom_v, st.batch_size).request(st.global_pool, infinite).then(
            [=](std::vector<MatrixTask>& batch) {
                for (auto& task : batch)
                    self->state().local_tasks.push_back(std::move(task));
                self->state().fetching = false;
            },
            [=](error& err) {
                self->state().fetching = false;
            }
        );
    };

    refill();

    return {
        [=](get_work_atom) -> result<SolverType, std::string, in<int>, in<int>, in<float>, in<float>, in_out<float>, int, int> {
            auto& st = self->state();

            // If local tasks are available, process immediately
            if (!st.local_tasks.empty()) {
                MatrixTask t = std::move(st.local_tasks.front());
                st.local_tasks.pop_front();
                refill(); // Try to refill if below low water mark

                auto& data = st.cache[t.path];
                if (data.row_ptr.empty()) {
                    auto coo = load_binary_coo(t.path);
                    auto A = convert_coo_to_csr(coo);
                    data.b = compute_rhs_spmv(A, std::vector<float>(A.cols, 1.0f));
                    data.row_ptr = std::move(A.row_ptr);
                    data.col_indices = std::move(A.col_indices);
                    data.values = std::move(A.values);
                    data.x_guess.assign(A.cols, 0.0f);
                }

                return {t.type, t.path, create_in_arg(data.row_ptr), create_in_arg(data.col_indices), 
                        create_in_arg(data.values), create_in_arg(data.b), create_in_out_arg(data.x_guess), 
                        (int)data.row_ptr.size() - 1, (int)data.values.size()};
            }

            auto promise = self->make_response_promise<SolverType, std::string, in<int>, in<int>, in<float>, in<float>, in_out<float>, int, int>();
            self->mail(get_work_atom_v, st.batch_size).request(st.global_pool, infinite).then(
                [=](std::vector<MatrixTask>& batch) mutable {
                    auto& st_inner = self->state();
                    if (batch.empty()) { // Global pool returned empty batch, meaning no more work
                        promise.deliver(make_error(sec::end_of_stream));
                        return;
                    }

                    // Add remaining tasks to local queue
                    for (size_t i = 1; i < batch.size(); ++i) {
                        st_inner.local_tasks.push_back(std::move(batch[i]));
                    }

                    MatrixTask t = std::move(batch.front()); // Process the first task from the batch
                    refill(); // Try to refill if below low water mark (after adding tasks)

                    auto& data = st_inner.cache[t.path];
                    if (data.row_ptr.empty()) {
                        auto coo = load_binary_coo(t.path);
                        auto A = convert_coo_to_csr(coo);
                        data.b = compute_rhs_spmv(A, std::vector<float>(A.cols, 1.0f));
                        data.row_ptr = std::move(A.row_ptr);
                        data.col_indices = std::move(A.col_indices);
                        data.values = std::move(A.values);
                        data.x_guess.assign(A.cols, 0.0f);
                    }

                    promise.deliver(t.type, t.path, create_in_arg(data.row_ptr), create_in_arg(data.col_indices), 
                                    create_in_arg(data.values), create_in_arg(data.b), create_in_out_arg(data.x_guess), 
                                    (int)data.row_ptr.size() - 1, (int)data.values.size());
                },
                [=](error& err) mutable {
                    promise.deliver(err); // Propagate error from global pool
                }
            );
            return promise;
        },
        [=](release_memory_atom, std::string path) {
            self->state().cache.erase(path);
        },
        [=](worker_done_atom) {
            if (--self->state().active_workers <= 0)
                self->quit();
        }
    };
}

// ---------------------------- WORKER ACTOR ----------------------------
struct worker_state {
    caf::actor device_actor;
    caf::actor supervisor;
    int device_id;
    int stream_id;
    std::string current_matrix_path;
};

behavior sparse_worker_fun(stateful_actor<worker_state>* self,
                           caf::actor supervisor, caf::actor device_actor, int dev_id, int stream_id) {
    self->state().supervisor = supervisor;
    self->state().device_actor = device_actor;
    self->state().device_id = dev_id;
    self->state().stream_id = stream_id;

    self->mail(request_work_atom_v).send(self);

    return {
        [=](request_work_atom) {
            self->mail(get_work_atom_v).request(self->state().device_actor, infinite).then(
                [=](SolverType type, std::string path, in<int> rp, in<int> ci, in<float> val,
                    in<float> b, in_out<float> x, int rows, int nnz) {
                    self->state().current_matrix_path = path;
                    actor solver;
                    if (type == CGS_SOLVER) {
                        solver = self->spawn<sparse_cg_actor>(
                            std::move(rp), std::move(ci), std::move(val),
                            std::move(b), std::move(x),
                            matrix_format::csr, rows, nnz, 1e-5f, 2000, dev_id, stream_id, actor_cast<actor>(self));
                    } else {
                        solver = self->spawn<sparse_bicgstab_actor>(
                            std::move(rp), std::move(ci), std::move(val),
                            std::move(b), std::move(x),
                            matrix_format::csr, rows, nnz, 1e-5f, 2000, dev_id, stream_id, actor_cast<actor>(self));
                    }
                    self->mail(start_atom_v).send(solver);
                },
                [=](error& err) {
                    if (err == sec::end_of_stream) {
                        self->mail(worker_done_atom_v).send(self->state().device_actor);
                        self->quit();
                    }
                }
            );
        },
        [=](std::vector<float>& solution) {
            self->mail(1).send(self->state().supervisor);
            self->mail(release_memory_atom_v, self->state().current_matrix_path).send(self->state().device_actor);
            self->mail(request_work_atom_v).send(self);
        }
    };
}

// ---------------------------- SUPERVISOR ACTOR ----------------------------
struct supervisor_state {
    int total_tasks;
    int completed = 0;
};

behavior supervisor_actor_fun(stateful_actor<supervisor_state>* self, int total, std::vector<MatrixTask> tasks) {
    self->state().total_tasks = total;
    auto pool = self->spawn(global_task_pool, std::move(tasks));
    
    manager& mgr = manager::get();
    int num_gpus = mgr.get_num_devices();
    int workers_per_gpu = 8; // Adjustable worker count per physical GPU
    int max_in_flight_tasks_per_worker = 2; // How many tasks each worker can have in flight

    for (int i = 0; i < num_gpus; ++i) {
        auto broker = self->spawn(gpu_device_actor, pool, workers_per_gpu, i, max_in_flight_tasks_per_worker);
        for (int j = 0; j < workers_per_gpu; ++j) {
            self->spawn(sparse_worker_fun, self, broker, i, (i * 100) + j);
        }
    }

    return {
        [=](int done) {
            self->state().completed += done;
            if (self->state().completed >= self->state().total_tasks) {
                std::cout << "\n[DONE] All " << self->state().total_tasks << " matrices processed.\n";
                self->quit();
            }
        }
    };
}

void caf_main(actor_system& sys) {
    manager::init(sys, manager_config(true, true));
    std::vector<MatrixTask> tasks;
    
    auto scan = [&](const std::string& dir, SolverType type) {
        if (!fs::exists(dir)) return;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ".bin")
                tasks.push_back({entry.path().string(), type});
        }
    };

    scan("/scratch/nqr159/matrix-collection/matrices/spd", CGS_SOLVER);
    scan("/scratch/nqr159/matrix-collection/matrices/unsymmetric", BICSTAB_SOLVER);

    if (tasks.empty()) {
        std::cerr << "No matrix files found in search paths.\n";
        manager::shutdown();
        return;
    }

    auto task_count = tasks.size();
    std::cout << "[INFO] Found " << task_count << " matrices. Spawning workload...\n";

    auto start = std::chrono::steady_clock::now();

    sys.spawn(supervisor_actor_fun, static_cast<int>(task_count), std::move(tasks));
    sys.await_all_actors_done();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "\n===== BENCHMARK COMPLETE =====\n";
    std::cout << "Tasks Processed: " << task_count << "\n";
    std::cout << "Total Runtime:   " << elapsed.count() << " s\n";
    std::cout << "==============================\n";

    manager::shutdown();
}
CAF_MAIN(id_block::cuda, id_block::cg_actor, id_block::bicgstab_actor, id_block::workload_test)
