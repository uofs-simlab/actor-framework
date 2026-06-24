#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <filesystem>
#include <algorithm>
#include <atomic>
#include "native_utils.hpp"

void producer(ThreadSafeQueue<MatrixTask>& queue, std::vector<MatrixTask> matrix_pool) {
    for (auto& task : matrix_pool) {
        task.enqueue_time = std::chrono::steady_clock::now();
        queue.push(task);
    }
    queue.signal_shutdown();
}

int main(int argc, char** argv)
{
    constexpr uint32_t WORKLOAD_SEED = 42;
    int num_streams = 4;
    // if (argc > 1) num_streams = std::max(num_streams, std::atoi(argv[1]));

    std::cout << "[INFO] Loading matrices...\n";
    std::vector<MatrixTask> matrix_pool = scan_for_matrices("/scratch/nqr159/matrix-collection/matrices/mixed", CGS_SOLVER);

    if (matrix_pool.empty()) {
        std::cerr << "No matrices found.\n";
        return 1;
    }

    int num_gpus = 0;
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));
    
    std::cout << "[INFO] Found " << num_gpus << " GPUs\n";
    std::cout << "[INFO] Matrix pool size: " << matrix_pool.size() << "\n";
    std::cout << "[INFO] Streams/GPU: " << num_streams << "\n";

    std::atomic<int> tasks_succeeded{0};
    std::atomic<int> tasks_failed{0};
    ThreadSafeQueue<MatrixTask> work_queue;
    
    init_benchmark_timer();
    std::thread producer_thread(producer, std::ref(work_queue), matrix_pool);
    
    std::vector<std::thread> workers;
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        for (int stream = 0; stream < num_streams; ++stream) {
            workers.emplace_back(gpu_stream_worker, gpu, gpu * num_streams + stream, std::ref(work_queue), std::ref(tasks_succeeded), std::ref(tasks_failed));
        }
    }

    producer_thread.join();
    for (auto& worker : workers) worker.join();

    report_workload_stats();

    return 0;
}
