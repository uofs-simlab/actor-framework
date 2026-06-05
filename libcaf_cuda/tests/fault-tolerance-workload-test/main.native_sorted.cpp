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
    int num_streams = 1;
    if (argc > 1) num_streams = std::max(1, std::atoi(argv[1]));
    std::vector<MatrixTask> matrix_pool = scan_for_matrices("/scratch/nqr159/matrix-collection/matrices/spd", CGS_SOLVER);
    if (matrix_pool.empty()) return 1;

    int num_gpus = 0;
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));
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
    std::cout << "All tasks processed in " << total_time.count() << " s\n";
    return 0;
}