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
#include "sparse_utils.hpp"
#include "atoms.hpp"
#include "ft_cg_actor.hpp"
#include "supervisor_actor.hpp"

using namespace caf;
using namespace caf::cuda;


constexpr uint32_t WORKLOAD_SEED = 42;
void caf_main(actor_system& sys) {
    manager::init(sys, manager_config(true, true));
    std::cout << "[INFO] Loading matrices...\n";
    {
         //auto tasks_vec = scan_for_matrices("/scratch/nqr159/matrix-collection/matrices/spd", CGS_SOLVER);
        //auto tasks_vec = scan_for_matrices("/scratch/nqr159/matrix-collection/matrices/unsymmetric", CGS_SOLVER);
         //auto tasks_vec = scan_for_matrices("/scratch/nqr159/matrix-collection/matrix_corpus_v2/matrices/unsymmetric", CGS_SOLVER);
         //auto tasks_vec = scan_for_matrices("/scratch/nqr159/matrix-collection/matrices/mixed", CGS_SOLVER);

         auto tasks_vec = scan_for_matrices("/scratch/nqr159/matrices/workloadA", CGS_SOLVER);


        int num_gpus = manager::get().get_num_devices();
        std::cout << "[INFO] Found " << num_gpus << " GPUs\n";
        std::cout << "[INFO] Matrix pool size: " << tasks_vec.size() << "\n";

        if (tasks_vec.empty()) {
            std::cerr << "No matrices found. Running dummy test task." << std::endl;
            auto data = std::make_shared<MatrixData>();
            data->row_ptr = {0, 1, 2};
            data->col_indices = {0, 1};
            data->values = {10.0f, 10.0f};
            data->b = {100.0f, 100.0f};
            data->x_guess = {0.0f, 0.0f};
            tasks_vec.push_back({"dummy_task", CGS_SOLVER, data});
        }

        init_benchmark_timer();
        for (auto& task : tasks_vec) {
            task.enqueue_time = std::chrono::steady_clock::now();
        }

        // Workload partitioning logic (Timed)
        auto part_start = std::chrono::steady_clock::now();
        auto partitions = make_contiguous_partitions(tasks_vec.size(), num_gpus, 1);
        
        std::vector<std::vector<MatrixTask>> partitioned_workloads(num_gpus);
        for (int i = 0; i < num_gpus; ++i) {
            auto& p = partitions[i];
            partitioned_workloads[i].reserve(p.end - p.begin);
            for (size_t j = p.begin; j < p.end; ++j) {
                partitioned_workloads[i].push_back(std::move(tasks_vec[j]));
            }
        }
        auto part_end = std::chrono::steady_clock::now();
        auto part_ms = std::chrono::duration_cast<std::chrono::milliseconds>(part_end - part_start).count();
        std::cout << "[INFO] Workload partitioning completed in " << part_ms << " ms\n";

        auto benchmark_start = std::chrono::steady_clock::now();
        for (int i = 0; i < num_gpus; ++i) {
            sys.spawn(supervisor_actor, std::move(partitioned_workloads[i]), 4, benchmark_start, i);
        }

        sys.await_all_actors_done();
    }
    manager::shutdown();
}
CAF_MAIN(id_block::cuda, id_block::workload_test)