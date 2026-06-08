#include "sparse_utils.hpp"
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <random>
#include <mutex>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace fs = std::filesystem;

SparseMatrixCOO load_binary_coo(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open matrix file: " + filepath);
    }

    SparseMatrixCOO coo;
    
    file.read(reinterpret_cast<char*>(&coo.rows), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&coo.cols), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&coo.nnz), sizeof(int32_t));

    coo.row_indices.resize(coo.nnz);
    coo.col_indices.resize(coo.nnz);
    coo.values.resize(coo.nnz);

    file.read(reinterpret_cast<char*>(coo.row_indices.data()), coo.nnz * sizeof(int32_t));
    file.read(reinterpret_cast<char*>(coo.col_indices.data()), coo.nnz * sizeof(int32_t));
    file.read(reinterpret_cast<char*>(coo.values.data()), coo.nnz * sizeof(float));

    return coo;
}

SparseMatrixCSR convert_coo_to_csr(const SparseMatrixCOO& coo) {
    SparseMatrixCSR csr;
    csr.rows = coo.rows;
    csr.cols = coo.cols;
    csr.nnz = coo.nnz;
    
    csr.row_ptr.assign(csr.rows + 1, 0);
    csr.col_indices.resize(csr.nnz);
    csr.values.resize(csr.nnz);

    for (int32_t i = 0; i < coo.nnz; ++i) {
        csr.row_ptr[coo.row_indices[i] + 1]++;
    }

    for (int32_t i = 0; i < csr.rows; ++i) {
        csr.row_ptr[i + 1] += csr.row_ptr[i];
    }

    std::vector<int32_t> current_row_pos = csr.row_ptr;

    for (int32_t i = 0; i < coo.nnz; ++i) {
        int32_t row = coo.row_indices[i];
        int32_t dest_pos = current_row_pos[row]++;
        csr.col_indices[dest_pos] = coo.col_indices[i];
        csr.values[dest_pos] = coo.values[i];
    }

    return csr;
}

std::vector<float> compute_rhs_spmv(const SparseMatrixCSR& A, const std::vector<float>& x) {
    std::vector<float> b(A.rows, 0.0f);
    
    for (int32_t i = 0; i < A.rows; ++i) {
        float sum = 0.0f;
        int32_t row_start = A.row_ptr[i];
        int32_t row_end = A.row_ptr[i + 1];
        
        for (int32_t j = row_start; j < row_end; ++j) {
            sum += A.values[j] * x[A.col_indices[j]];
        }
        b[i] = sum;
    }
    return b;
}

std::vector<MatrixTask> scan_for_matrices(const std::string& dir, SolverType type) {
    std::vector<MatrixTask> tasks;
    if (!fs::exists(dir)) return tasks;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".bin") {
            auto coo = load_binary_coo(entry.path().string());
            auto csr = convert_coo_to_csr(coo);
            auto data = std::make_shared<MatrixData>();
            data->rows = csr.rows;
            data->cols = csr.cols;
            data->nnz = csr.nnz;
            data->b = compute_rhs_spmv(csr, std::vector<float>(csr.cols, 1.0f));
            data->row_ptr = std::move(csr.row_ptr);
            data->col_indices = std::move(csr.col_indices);
            data->values = std::move(csr.values);
            data->x_guess.assign(data->cols, 0.0f);
            
            tasks.push_back({entry.path().string(), type, data});
        }
    }
    return tasks;
}


std::vector<Partition>
make_contiguous_partitions(size_t num_tasks, size_t rows, size_t cols)
{
    size_t num_parts = rows * cols;

    std::vector<Partition> parts;
    parts.reserve(num_parts);

    size_t base = num_tasks / num_parts;
    size_t rem  = num_tasks % num_parts;

    size_t current = 0;

    for (size_t p = 0; p < num_parts; ++p) {
        size_t size = base + (p < rem ? 1 : 0);

        parts.push_back({
            current,
            current + size
        });

        current += size;
    }

    return parts;
}


int generate_random_sleep_ms(int min_ms, int max_ms) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min_ms, max_ms);
    return dis(gen);
}

std::chrono::milliseconds generate_random_interval(
    std::mt19937& rng,
    double mean_ms)
{
    std::exponential_distribution<double> dist(
        1.0 / mean_ms);

    return std::chrono::milliseconds(
        static_cast<int>(dist(rng)));
}

std::vector<MatrixTask> generate_batch(
    const std::vector<MatrixTask>& matrix_pool,
    std::mt19937& rng,
    size_t batch_size)
{
    std::vector<MatrixTask> batch;
    batch.reserve(batch_size);

    std::uniform_int_distribution<size_t> dist(
        0,
        matrix_pool.size() - 1);

    for (size_t i = 0; i < batch_size; ++i) {
        batch.push_back(matrix_pool[dist(rng)]);
    }

    return batch;
}

static std::vector<JobStats> global_stats;
static std::mutex stats_mutex;
static std::chrono::steady_clock::time_point benchmark_start_tp;

void init_benchmark_timer() {
    benchmark_start_tp = std::chrono::steady_clock::now();
}

void record_job(const std::string& name,
                std::chrono::steady_clock::time_point enqueue_time,
                std::chrono::steady_clock::time_point pick_time,
                std::chrono::steady_clock::time_point finish_time,
                int iterations, bool success) {
    std::lock_guard<std::mutex> lock(stats_mutex);
    
    auto wait = std::chrono::duration<double, std::milli>(pick_time - enqueue_time).count();
    auto total = std::chrono::duration<double, std::milli>(finish_time - enqueue_time).count();
    auto finish_rel = std::chrono::duration<double, std::milli>(finish_time - benchmark_start_tp).count();

    global_stats.push_back({name, wait, total, iterations, success, finish_rel});
}

void report_workload_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex);
    if (global_stats.empty()) {
        std::cout << "No job statistics recorded.\n";
        return;
    }

    int total_iters = 0;
    int success_count = 0;
    double wasted_gpu_time_ms = 0;
    int failed_count = 0;
    std::vector<double> completions;
    completions.reserve(global_stats.size());

    // Ensure total_jobs is not zero to avoid division by zero
    size_t total_jobs = global_stats.size();

    for (const auto& s : global_stats) {
        total_iters += s.iterations;
        if (s.success) success_count++;
        else wasted_gpu_time_ms += (s.completion_time_ms - s.wait_time_ms);
        
        completions.push_back(s.completion_time_ms);
    }

    std::sort(completions.begin(), completions.end());
    failed_count = total_jobs - success_count;
    double success_percentage = (total_jobs > 0) ? (100.0 * success_count / total_jobs) : 0.0;
    double failed_percentage = (total_jobs > 0) ? (100.0 * failed_count / total_jobs) : 0.0;
    double mean = (total_jobs > 0) ? std::accumulate(completions.begin(), completions.end(), 0.0) / total_jobs : 0.0;
    double median = (total_jobs > 0) ? completions[total_jobs / 2] : 0.0;
    double p95 = completions[static_cast<size_t>(completions.size() * 0.95)];

    auto max_finish = std::max_element(global_stats.begin(), global_stats.end(), [](const JobStats& a, const JobStats& b) {
        return a.finish_relative_ms < b.finish_relative_ms;
    });

    std::cout << "\n" << std::string(45, '=') << "\n";
    std::cout << "          WORKLOAD PERFORMANCE REPORT\n";
    std::cout << std::string(45, '=') << "\n";
    std::cout << std::left << std::setw(25) << "Total Jobs:" << total_jobs << "\n";
    std::cout << std::left << std::setw(25) << "Succeeded Jobs:" << success_count << " (" << std::fixed << std::setprecision(2) << success_percentage << "%)\n";
    std::cout << std::left << std::setw(25) << "Failed Jobs:" << failed_count << " (" << std::fixed << std::setprecision(2) << failed_percentage << "%)\n";
    std::cout << std::left << std::setw(25) << "Total Iterations:" << total_iters << "\n";
    std::cout << std::left << std::setw(25) << "Cumul. Wasted GPU Time:" << wasted_gpu_time_ms / 1000.0 << " s (all streams)\n";
    std::cout << std::left << std::setw(25) << "Makespan:" << max_finish->finish_relative_ms / 1000.0 << " s\n";
    std::cout << std::left << std::setw(25) << "Mean Completion:" << mean << " ms\n";
    std::cout << std::left << std::setw(25) << "Median Completion:" << median << " ms\n";
    std::cout << std::left << std::setw(25) << "95th Percentile:" << p95 << " ms\n";
    
    std::cout << "\nThroughput Timeline (Job Completion & Success vs Wall-clock):\n";
    std::sort(global_stats.begin(), global_stats.end(), [](const JobStats& a, const JobStats& b) {
        return a.finish_relative_ms < b.finish_relative_ms;
    });
    
    double total_time = max_finish->finish_relative_ms;
    for (int i = 1; i <= 10; ++i) {
        double threshold = (total_time / 10.0) * i;
        auto it = std::upper_bound(global_stats.begin(), global_stats.end(), threshold, 
            [](double val, const JobStats& s) {
                return val < s.finish_relative_ms;
            });
        size_t total_at_t = std::distance(global_stats.begin(), it);
        size_t success_at_t = 0;
        for (auto s_it = global_stats.begin(); s_it != it; ++s_it) {
            if (s_it->success) success_at_t++;
        }

        std::cout << "  T + " << std::setw(5) << std::fixed << std::setprecision(0) << threshold 
                  << " ms | Total Progress: " << std::setw(3) << (100 * total_at_t / total_jobs) 
                  << "% | Successful solves: " << std::setw(3) << (100 * success_at_t / total_jobs) << "%\n";
    }
    std::cout << std::string(45, '=') << "\n\n";
}