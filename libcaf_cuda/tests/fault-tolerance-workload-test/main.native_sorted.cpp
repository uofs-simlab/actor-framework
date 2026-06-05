#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <filesystem>
#include <algorithm>
#include "native_utils.hpp"

void producer(ThreadSafeQueue<MatrixTask>& queue, std::vector<MatrixTask> matrix_pool) {
    // Order tasks from lowest NNZ to highest NNZ
    std::sort(matrix_pool.begin(), matrix_pool.end(), [](const MatrixTask& a, const MatrixTask& b) {
        return a.data->nnz < b.data->nnz;
    });

    for (const auto& task : matrix_pool) {
        queue.push(task);
    }
    queue.signal_shutdown();
}

int main(int argc, char** argv) {
    constexpr uint32_t WORKLOAD_SEED = 42;
    int num_streams = 4;
    if (argc > 1) num_streams = std::max(1, std::atoi(argv[1]));

    std::cout << "[INFO] Loading matrices...\n";
    std::vector<MatrixTask> matrix_pool = scan_for_matrices("/scratch/nqr159/matrix-collection/matrices/spd", CGS_SOLVER);

    if (matrix_pool.empty()) {
        std::cerr << "No matrices found.\n";
        return 1;
    }

    int num_gpus = 0;
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));

    std::cout << "[INFO] Found " << num_gpus << " GPUs\n";
    std::cout << "[INFO] Matrix pool size: " << matrix_pool.size() << "\n";
    std::cout << "[INFO] Streams/GPU: " << num_streams << "\n";

    ThreadSafeQueue<MatrixTask> work_queue;
    auto benchmark_start = std::chrono::steady_clock::now();
    std::thread producer_thread(producer, std::ref(work_queue), matrix_pool);
    std::vector<std::thread> workers;
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        for (int stream = 0; stream < num_streams; ++stream) {
            workers.emplace_back(gpu_stream_worker, gpu, gpu * num_streams + stream, std::ref(work_queue));
        }
    }
    producer_thread.join();
    for (auto& worker : workers) worker.join();
    auto benchmark_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_time = benchmark_end - benchmark_start;

    std::cout << "All tasks in the pool have been processed." << std::endl;
    std::cout << "\n";
    std::cout << "=====================================\n";
    std::cout << "IRREGULAR WORKLOAD BENCHMARK (NATIVE - SORTED)\n";
    std::cout << "=====================================\n";
    std::cout << "Seed:               " << WORKLOAD_SEED << "\n";
    std::cout << "GPUs:               " << num_gpus << "\n";
    std::cout << "Streams per GPU:    " << num_streams << "\n";
    std::cout << "Worker Threads:     " << num_gpus * num_streams << "\n";
    std::cout << "Total Runtime:      " << total_time.count() << " s\n";
    std::cout << "=====================================\n";

    return 0;
}