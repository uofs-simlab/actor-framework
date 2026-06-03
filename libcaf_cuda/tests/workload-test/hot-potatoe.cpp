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
#include <memory>
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

CAF_BEGIN_TYPE_ID_BLOCK(workload_test, caf::id_block::cuda::end)
    CAF_ADD_ATOM(workload_test, get_work_atom)
    CAF_ADD_ATOM(workload_test, release_memory_atom)
    CAF_ADD_ATOM(workload_test, request_work_atom)
    CAF_ADD_ATOM(workload_test, worker_done_atom)
    CAF_ADD_ATOM(workload_test, work_tick_atom)
    CAF_ADD_TYPE_ID(workload_test, (SolverType))
    CAF_ADD_TYPE_ID(workload_test, (MatrixTask))
    CAF_ADD_TYPE_ID(workload_test, (std::vector<MatrixTask>))
    CAF_ADD_TYPE_ID(workload_test, (std::shared_ptr<MatrixData>))
CAF_END_TYPE_ID_BLOCK(workload_test)

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(MatrixData)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::shared_ptr<MatrixData>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(MatrixTask)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::vector<MatrixTask>)



void caf_main(actor_system& sys) {
    manager::init(sys, manager_config(true, true));
    
    int num_streams = 8;
    int num_batches = 25;
    int batch_size = 100;
    double mean_arrival_ms = 1000.0;

    // // Basic command line argument parsing similar to native
    // auto& args = sys.config().remainder;
    // if (args.size() > 0) num_streams = std::max(1, std::stoi(args[0]));
    // if (args.size() > 1) num_batches = std::max(1, std::stoi(args[1]));
    // if (args.size() > 2) batch_size = std::max(1, std::stoi(args[2]));
    // if (args.size() > 3) mean_arrival_ms = std::stod(args[3]);

    std::cout << "[INFO] Loading matrices into memory...\n";
    auto tasks_vec = scan_for_matrices("/scratch/nqr159/matrix-collection/matrix_corpus_v2/matrices/spd", CGS_SOLVER);
    //scan("/scratch/nqr159/matrix-collection/matrices/unsymmetric", BICSTAB_SOLVER);
    
    if (tasks_vec.empty()) {
        std::cerr << "No matrix files found in search paths.\n";
        manager::shutdown();
        return;
    }

    std::cout << "[INFO] Matrix pool size: " << tasks_vec.size() << "\n";
    std::cout << "[INFO] Streams/GPU:      " << num_streams << "\n";
    std::cout << "[INFO] Batches:          " << num_batches << "\n";
    std::cout << "[INFO] Batch size:       " << batch_size << "\n";
    std::cout << "[INFO] Mean arrival:     " << mean_arrival_ms << " ms\n";

    auto start = std::chrono::steady_clock::now();

    // sys.spawn(supervisor_actor_fun, std::move(tasks_vec), num_streams, num_batches, batch_size, mean_arrival_ms);
    // sys.await_all_actors_done();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    int total_tasks = num_batches * batch_size;

    std::cout << "\n===== BENCHMARK COMPLETE =====\n";
    std::cout << "Seed:               " << WORKLOAD_SEED << "\n";
    std::cout << "Streams per GPU:    " << num_streams << "\n";
    std::cout << "Batches:            " << num_batches << "\n";
    std::cout << "Batch Size:         " << batch_size << "\n";
    std::cout << "Mean Arrival (ms):  " << mean_arrival_ms << "\n";
    std::cout << "Tasks Processed:    " << total_tasks << "\n";
    std::cout << "Total Runtime:      " << elapsed.count() << " s\n";
    std::cout << "Throughput:         " << total_tasks / elapsed.count() << " tasks/s\n";
    std::cout << "==============================\n";

    manager::shutdown();
}
CAF_MAIN(id_block::cuda,id_block::workload_test)