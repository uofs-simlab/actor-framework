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
         auto tasks_vec = scan_for_matrices("/scratch/nqr159/matrix-collection/matrices/mixed", CGS_SOLVER);

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

        auto benchmark_start = std::chrono::steady_clock::now();
        int admission_control_limit = 4 * num_gpus; // 4 concurrent tasks per GPU

        sys.spawn(supervisor_actor, std::move(tasks_vec), admission_control_limit, benchmark_start);
        sys.await_all_actors_done();
    }
    manager::shutdown();
}
CAF_MAIN(id_block::cuda, id_block::workload_test)