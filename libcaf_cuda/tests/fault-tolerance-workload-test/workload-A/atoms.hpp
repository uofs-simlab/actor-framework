#pragma once

#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include "sparse_utils.hpp"

constexpr int MAX_ITERATIONS = 16000;

CAF_BEGIN_TYPE_ID_BLOCK(workload_test, caf::id_block::cuda::end)
    CAF_ADD_ATOM(workload_test, get_work_atom)
    CAF_ADD_ATOM(workload_test, release_memory_atom)
    CAF_ADD_ATOM(workload_test, request_work_atom)
    CAF_ADD_ATOM(workload_test, worker_done_atom)
    CAF_ADD_ATOM(workload_test, work_tick_atom)
    CAF_ADD_ATOM(workload_test, add_work_atom)
    CAF_ADD_ATOM(workload_test, steal_work_atom)
    CAF_ADD_ATOM(workload_test, update_stream_atom)
    CAF_ADD_ATOM(workload_test, shutdown_atom)
    CAF_ADD_TYPE_ID(workload_test, (SolverType))
    CAF_ADD_TYPE_ID(workload_test, (MatrixTask))
    CAF_ADD_TYPE_ID(workload_test, (MatrixData))
    CAF_ADD_TYPE_ID(workload_test, (std::vector<MatrixTask>))
    CAF_ADD_TYPE_ID(workload_test, (std::shared_ptr<MatrixData>))
CAF_END_TYPE_ID_BLOCK(workload_test)

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(MatrixData)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::shared_ptr<MatrixData>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(MatrixTask)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::vector<MatrixTask>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(SolverType)

enum cg_error_type : int {
  CG_SUCCESS = 0,
  CG_MAX_ITER = 1,
  CG_NAN_INF = 2,
  CG_STAGNATION = 3,
  CG_BREAKDOWN = 4,
  CG_RESIDUAL_FACTOR_FAIL = 5,
  CG_JACOBI_RETRY = 6
};

inline std::string to_string(cg_error_type err) {
  switch (err) {
    case CG_SUCCESS: return "Success";
    case CG_MAX_ITER: return "Maximum Iterations Reached";
    case CG_NAN_INF: return "Stability Check Failed (NaN/Inf Detected)";
    case CG_STAGNATION: return "Stagnation Detected (Residual stopped changing)";
    case CG_BREAKDOWN: return "Solver Breakdown (Division by zero/near-zero)";
    case CG_RESIDUAL_FACTOR_FAIL: return "Residual Factor Check Failed";
    case CG_JACOBI_RETRY: return "Falling back to Jacobi Preconditioner";
    default: return "Unknown Error (" + std::to_string(static_cast<int>(err)) + ")";
  }
}