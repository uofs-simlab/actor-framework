#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include "caf/actorSOLVE/sparse-matrix-solvers/sparse-CGS-actor/sparse-CGS-actor.hpp"
#include "caf/actorSOLVE/sparse-matrix-solvers/sparse-BiCGSTAB-actor/sparse-BiCGSTAB-actor.hpp"
#include "caf/actorSOLVE/sparse-matrix-solvers/sparse-GMRES-actor/sparse-GMRES-actor.hpp"

using namespace caf;
using namespace caf::cuda;

// --- Matrix Utilities ---

template <class T = float>
struct LocalCSR {
    int rows, cols, nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<T> values;
};

template <class T = float>
LocalCSR<T> load_binary_matrix(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open " + path);
    int32_t r, c, n;
    file.read(reinterpret_cast<char*>(&r), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&c), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    std::vector<int32_t> rows_coo(n), cols_coo(n);
    std::vector<float> vals_coo(n);
    file.read(reinterpret_cast<char*>(rows_coo.data()), n * sizeof(int32_t));
    file.read(reinterpret_cast<char*>(cols_coo.data()), n * sizeof(int32_t));
    file.read(reinterpret_cast<char*>(vals_coo.data()), n * sizeof(float));
    int max_idx = 0;
    for(auto v : rows_coo) if(v > max_idx) max_idx = v;
    if (max_idx == r || max_idx == c) {
        for(auto& v : rows_coo) v--;
        for(auto& v : cols_coo) v--;
    }
    LocalCSR<T> csr;
    csr.rows = r; csr.cols = c; csr.nnz = n;
    csr.row_ptr.assign(r + 1, 0);
    csr.col_ind.resize(n);
    csr.values.resize(n);
    for (int i = 0; i < n; ++i) csr.row_ptr[rows_coo[i] + 1]++;
    for (int i = 0; i < r; ++i) csr.row_ptr[i + 1] += csr.row_ptr[i];
    std::vector<int> current_pos = csr.row_ptr;
    for (int i = 0; i < n; ++i) {
        int row = rows_coo[i];
        int dest = current_pos[row]++;
        csr.col_ind[dest] = cols_coo[i];
        csr.values[dest] = static_cast<T>(vals_coo[i]);
    }
    return csr;
}

template <class T = float>
std::vector<T> compute_rhs(const LocalCSR<T>& A, const std::vector<T>& x) {
    std::vector<T> b(A.rows, T{0});
    for (int i = 0; i < A.rows; ++i) {
        T sum = T{0};
        for (int j = A.row_ptr[i]; j < A.row_ptr[i+1]; ++j)
            sum += A.values[j] * x[A.col_ind[j]];
        b[i] = sum;
    }
    return b;
}

bool is_valid(const std::vector<float>& x) {
    for (float val : x) {
        if (std::isnan(val) || std::isinf(val)) return false;
    }
    return true;
}

// --- Robust Solver Actor (Facade Orchestrator) ---

struct robust_solver_state {
    LocalCSR<float> A;
    std::vector<float> b;
    float tol;
    int max_iter;
    const char* method = "CGS";
};

behavior robust_solver(stateful_actor<robust_solver_state>* self, 
                       LocalCSR<float> A, std::vector<float> b, float tol, int max_iter) {
    self->state().A = std::move(A);
    self->state().b = std::move(b);
    self->state().tol = tol;
    self->state().max_iter = max_iter;

    auto start_cgs = [=] {
        auto& s = self->state();
        s.method = "CGS";
        auto facade = self->spawn<sparse_cg_facade>(100);
        std::vector<float> x(s.A.rows, 0.0f);
        self->mail(create_in_arg(s.A.row_ptr), create_in_arg(s.A.col_ind), create_in_arg(s.A.values),
                   create_in_arg(s.b), create_in_out_arg(x),
                   matrix_format::csr, s.A.rows, s.A.nnz, s.tol, s.max_iter, 0, 0).send(facade);
    };

    auto start_bicgstab = [=] {
        auto& s = self->state();
        s.method = "BiCGSTAB";
        auto facade = self->spawn<sparse_bicgstab_facade<float>>(100);
        std::vector<float> x(s.A.rows, 0.0f);
        self->mail(create_in_arg(s.A.row_ptr), create_in_arg(s.A.col_ind), create_in_arg(s.A.values),
                   create_in_arg(s.b), create_in_out_arg(x),
                   matrix_format::csr, s.A.rows, s.A.nnz, s.tol, s.max_iter, 0, 0).send(facade);
    };

    auto start_gmres = [=] {
        auto& s = self->state();
        s.method = "GMRES";
        auto facade = self->spawn<sparse_gmres_facade<float>>(100);
        std::vector<float> x(s.A.rows, 0.0f);
        self->mail(create_in_arg(s.A.row_ptr), create_in_arg(s.A.col_ind), create_in_arg(s.A.values),
                   create_in_arg(s.b), create_in_out_arg(x),
                   matrix_format::csr, s.A.rows, s.A.nnz, s.tol, s.max_iter, 30, 0, 0).send(facade);
    };

    return {
        [=](start_atom) {
            std::cout << "[INFO] Attempting solve with CGS..." << std::endl;
            start_cgs();
        },
        [=](uint32_t /*id*/, int /*idx*/, const std::vector<float>& result, solver_result_meta meta) {
            auto& s = self->state();
            if (meta.converged && is_valid(result)) {
                std::cout << "[SUCCESS] " << s.method << " converged. Matrix good." << std::endl;
                self->quit();
            } else {
                std::cout << "[WARNING] " << s.method << " failed or produced NaN. Retrying..." << std::endl;
                if (std::string(s.method) == "CGS") {
                    start_bicgstab();
                } else if (std::string(s.method) == "BiCGSTAB") {
                    start_gmres();
                } else {
                    std::cout << "[ERROR] All facade solvers failed." << std::endl;
                    self->quit();
                }
            }
        }
    };
}

void caf_main(actor_system& sys) {
    manager::init(sys, manager_config(true, true)); 
    scoped_actor self{sys};

    std::string path = "/scratch/nqr159/matrix-collection/matrices/unsymmetric/lnsp3937.bin";
    std::cout << "[INFO] Loading real-world matrix: " << path << std::endl;

    try {
        LocalCSR<float> A = load_binary_matrix<float>(path);
        std::vector<float> x_target(A.rows, 1.0f);
        std::vector<float> b = compute_rhs(A, x_target);

        auto robust = sys.spawn(robust_solver, std::move(A), std::move(b), 1e-5f, 5000);
        self->mail(start_atom_v).send(robust);
        
        self->receive(
            [] { std::cout << "[INFO] Robust solver test complete." << std::endl; }
        );
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
    }

    manager::shutdown();
}

CAF_MAIN(id_block::cuda)