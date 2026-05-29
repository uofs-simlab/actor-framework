#include <caf/all.hpp>
#include <caf/cuda/all.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "caf/actorSOLVE/sparse-matrix-solvers/sparse-CGS-actor/sparse-CGS-actor.hpp"
#include "caf/actorSOLVE/sparse-matrix-solvers/sparse-BiCGSTAB-actor/sparse-BiCGSTAB-actor.hpp"
#include "caf/actorSOLVE/sparse-matrix-solvers/sparse-GMRES-actor/sparse-GMRES-actor.hpp"

using namespace caf;
using namespace caf::cuda;
namespace fs = std::filesystem;

// ------------------------------------------------------------
// Matrix utilities
// ------------------------------------------------------------

template <class T = float>
struct LocalCSR {
  int rows = 0;
  int cols = 0;
  int nnz = 0;
  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<T> values;
};

template <class T = float>
LocalCSR<T> load_binary_matrix(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Could not open " + path);

  int32_t r = 0, c = 0, n = 0;
  file.read(reinterpret_cast<char*>(&r), sizeof(int32_t));
  file.read(reinterpret_cast<char*>(&c), sizeof(int32_t));
  file.read(reinterpret_cast<char*>(&n), sizeof(int32_t));

  if (!file || r <= 0 || c <= 0 || n < 0)
    throw std::runtime_error("Bad header in " + path);

  std::vector<int32_t> rows_coo(n), cols_coo(n);
  std::vector<float> vals_coo(n);
  file.read(reinterpret_cast<char*>(rows_coo.data()), n * sizeof(int32_t));
  file.read(reinterpret_cast<char*>(cols_coo.data()), n * sizeof(int32_t));
  file.read(reinterpret_cast<char*>(vals_coo.data()), n * sizeof(float));

  if (!file)
    throw std::runtime_error("Truncated matrix file: " + path);

  // Detect 1-based COO and normalize to 0-based.
  int max_idx = 0;
  for (auto v : rows_coo)
    if (v > max_idx)
      max_idx = v;
  for (auto v : cols_coo)
    if (v > max_idx)
      max_idx = v;

  if (max_idx == r || max_idx == c) {
    for (auto& v : rows_coo)
      --v;
    for (auto& v : cols_coo)
      --v;
  }

  LocalCSR<T> csr;
  csr.rows = static_cast<int>(r);
  csr.cols = static_cast<int>(c);
  csr.nnz = static_cast<int>(n);
  csr.row_ptr.assign(csr.rows + 1, 0);
  csr.col_ind.resize(csr.nnz);
  csr.values.resize(csr.nnz);

  for (int i = 0; i < csr.nnz; ++i) {
    if (rows_coo[i] < 0 || rows_coo[i] >= csr.rows)
      throw std::runtime_error("bad row index in " + path);
    if (cols_coo[i] < 0 || cols_coo[i] >= csr.cols)
      throw std::runtime_error("bad col index in " + path);
    csr.row_ptr[rows_coo[i] + 1]++;
  }

  for (int i = 0; i < csr.rows; ++i)
    csr.row_ptr[i + 1] += csr.row_ptr[i];

  std::vector<int> current_pos = csr.row_ptr;
  for (int i = 0; i < csr.nnz; ++i) {
    int row = rows_coo[i];
    int dest = current_pos[row]++;
    csr.col_ind[dest] = cols_coo[i];
    csr.values[dest] = static_cast<T>(vals_coo[i]);
  }

  return csr;
}

template <class T>
std::vector<T> compute_rhs(const LocalCSR<T>& A, const std::vector<T>& x) {
  std::vector<T> b(A.rows, T{0});
  for (int i = 0; i < A.rows; ++i) {
    T sum = T{0};
    for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j)
      sum += A.values[j] * x[A.col_ind[j]];
    b[i] = sum;
  }
  return b;
}

template <class T>
bool is_finite_vector(const std::vector<T>& x) {
  for (auto v : x)
    if (!std::isfinite(static_cast<double>(v)))
      return false;
  return true;
}

std::vector<fs::path> collect_matrix_files(const fs::path& root) {
  std::vector<fs::path> files;
  if (!fs::exists(root))
    return files;

  for (const auto& entry : fs::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file())
      continue;
    const auto ext = entry.path().extension().string();
    if (ext == ".bin")
      files.push_back(entry.path());
  }

  std::sort(files.begin(), files.end());
  return files;
}

// ------------------------------------------------------------
// Solver sweep helpers
// ------------------------------------------------------------

static constexpr uint32_t kReplyId = 0xA11CE001u;

// NOTE:
// This driver expects a BiCGSTAB Jacobi facade named
// caf::cuda::sparse_bicgstab_jacobi_facade<T> to be available from
// sparse-BiCGSTAB-actor.hpp.

template <class Facade, class T>
bool run_one_variant(actor_system& sys,
                     const LocalCSR<T>& A,
                     const std::vector<T>& b,
                     double tol,
                     int max_iter) {
  scoped_actor self{sys};
  auto solver = sys.spawn<Facade>(kReplyId);

  std::vector<T> x0(A.rows, T{0});

  // GMRES requires a restart parameter (k) which the other solvers do not.
  if constexpr (std::is_same_v<Facade, sparse_gmres_facade<T>>) {
    self->mail(return_mem_ptr_atom_v,
               create_in_arg(A.row_ptr),
               create_in_arg(A.col_ind),
               create_in_arg(A.values),
               create_in_arg(b),
               create_in_out_arg(x0),
               matrix_format::csr,
               A.rows,
               A.nnz,
               static_cast<T>(tol),
               max_iter,
               30, // Restart parameter k
               0,  // device_num
               0)  // stream_id
        .send(solver);
  } else {
    self->mail(return_mem_ptr_atom_v,
               create_in_arg(A.row_ptr),
               create_in_arg(A.col_ind),
               create_in_arg(A.values),
               create_in_arg(b),
               create_in_out_arg(x0),
               matrix_format::csr,
               A.rows,
               A.nnz,
               static_cast<T>(tol),
               max_iter,
               0, // device_num
               0) // stream_id
        .send(solver);
  }

  bool solved = false;
  bool received = false;

  self->receive(
    [&](uint32_t rid, mem_ptr<T> result, solver_result_meta meta) {
      (void)rid;
      (void)result;
      received = true;
      solved = meta.converged;
    },
    [&](const error& err) {
      received = true;
      solved = false;
    },
    after(std::chrono::minutes(20)) >> [&] {
      received = false;
      solved = false;
    }
  );

  self->send_exit(solver, exit_reason::user_shutdown);
  return received && solved;
}

template <class T>
bool run_all_variants(actor_system& sys,
                      const LocalCSR<T>& A,
                      const std::vector<T>& b,
                      double tol,
                      int max_iter) {
  // Try Jacobi variants first, then the base solvers, then GMRES.
  // A file is considered successful if at least one variant converges.
  if (run_one_variant<sparse_cg_jacobi_facade<T>>(sys, A, b, tol, max_iter))
    return true;

  if (run_one_variant<sparse_cg_facade<T>>(sys, A, b, tol, max_iter))
    return true;

  if (run_one_variant<sparse_bicgstab_jacobi_facade<T>>(sys, A, b, tol, max_iter))
    return true;

  if (run_one_variant<sparse_bicgstab_facade<T>>(sys, A, b, tol, max_iter))
    return true;

  if (run_one_variant<sparse_gmres_facade<T>>(sys, A, b, tol, max_iter))
    return true;

  return false;
}

template <class T>
void sweep_one_type(actor_system& sys, const fs::path& matrix_file) {
  try {
    auto A = load_binary_matrix<T>(matrix_file.string());
    if (A.rows <= 0 || A.cols <= 0 || A.nnz <= 0)
      return;
    if (A.rows != A.cols)
      return; // iterative solvers here target square systems

    std::vector<T> x_target(A.rows, T{1});
    std::vector<T> b = compute_rhs(A, x_target);

    constexpr double tol = 1e-10;
    constexpr int max_iter = 5000;

    if (run_all_variants<T>(sys, A, b, tol, max_iter)) {
      std::cout << matrix_file.string() << std::endl;
    }
  } catch (...) {
    // Silent by design: only successful file paths are printed.
  }
}

// ------------------------------------------------------------
// CAF entry point
// ------------------------------------------------------------

void caf_main(actor_system& sys) {
  manager::init(sys, manager_config(true, true));

  const fs::path root = "/scratch/nqr159/matrix-collection/matrix_corpus_v2/matrices";
  const auto files = collect_matrix_files(root);

  for (const auto& file : files) {
    sweep_one_type<float>(sys, file);
    sweep_one_type<double>(sys, file);
  }

  manager::shutdown();
}

CAF_MAIN(id_block::cuda)
