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

    std::cout << "rows=" << csr.rows
              << " cols=" << csr.cols
              << " nnz=" << csr.nnz
              << " row_ptr.back()=" << csr.row_ptr.back()
              << std::endl;

    for (int i = 0; i < csr.nnz; ++i) {
        if (csr.col_ind[i] < 0 || csr.col_ind[i] >= csr.cols)
            throw std::runtime_error("bad col index");
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

bool is_valid(const std::vector<double>& x) {
    for (double val : x) {
        if (std::isnan(val) || std::isinf(val)) return false;
    }
    return true;
}

// --- Robust Solver Actor (Facade Orchestrator) ---

enum class solver_strategy { cgs, bicgstab, gmres };

struct robust_solver_state {
    LocalCSR<double> A;
    std::vector<double> b;
    std::vector<double> x;
    double tol;
    int max_iter;
    std::vector<caf::actor> facades;
    solver_strategy current_strategy = solver_strategy::cgs;
    caf::actor requester;
};

behavior robust_solver(stateful_actor<robust_solver_state>* self, 
                       LocalCSR<double> A, std::vector<double> b, double tol, int max_iter) {
    self->state().A = std::move(A);
    self->state().b = std::move(b);
    self->state().tol = tol;
    self->state().x.assign(self->state().A.rows, 0.0);
    self->state().max_iter = max_iter;

    self->state().facades.push_back(self->spawn<sparse_cg_facade<double>>(100));
    self->state().facades.push_back(self->spawn<sparse_bicgstab_facade<double>>(100));
    self->state().facades.push_back(self->spawn<sparse_gmres_facade<double>>(100));

    auto get_method_name = [](solver_strategy s) {
        switch (s) {
            case solver_strategy::cgs:      return "CGS";
            case solver_strategy::bicgstab: return "BiCGSTAB";
            case solver_strategy::gmres:    return "GMRES";
            default:                        return "Unknown";
        }
    };

    auto start_cgs = [=] {
        auto& s = self->state();
        s.current_strategy = solver_strategy::cgs;
        self->mail(create_in_arg(s.A.row_ptr), create_in_arg(s.A.col_ind), create_in_arg(s.A.values),
                   create_in_arg(s.b), create_in_out_arg(s.x),
                   matrix_format::csr, s.A.rows, s.A.nnz, s.tol, s.max_iter, 0, 0).send(s.facades[0]);
    };

    auto start_bicgstab = [=] {
        auto& s = self->state();
        s.current_strategy = solver_strategy::bicgstab;
        self->mail(create_in_arg(s.A.row_ptr), create_in_arg(s.A.col_ind), create_in_arg(s.A.values),
                   create_in_arg(s.b), create_in_out_arg(s.x),
                   matrix_format::csr, s.A.rows, s.A.nnz, s.tol, s.max_iter, 0, 0).send(s.facades[1]);
    };

    auto start_gmres = [=] {
        auto& s = self->state();
        s.current_strategy = solver_strategy::gmres;
        self->mail(create_in_arg(s.A.row_ptr), create_in_arg(s.A.col_ind), create_in_arg(s.A.values),
                   create_in_arg(s.b), create_in_out_arg(s.x),
                   matrix_format::csr, s.A.rows, s.A.nnz, s.tol, s.max_iter, 30, 0, 0).send(s.facades[2]);
    };

    return {
        [=](start_atom) {
            self->state().requester = actor_cast<caf::actor>(self->current_sender());
            std::cout << "[INFO] Attempting solve with CGS..." << std::endl;
            start_cgs();
        },
        [=](uint32_t /*id*/, int /*idx*/, const std::vector<double>& result, solver_result_meta meta) {
            auto& s = self->state();
            const char* method_name = get_method_name(s.current_strategy);
            if (meta.converged && is_valid(result)) {
                std::cout << "[SUCCESS] " << method_name << " converged. Matrix good." << std::endl;
                if (s.requester)
                    self->mail(true).send(s.requester);
                self->quit();
            } else {
                std::cout << "[WARNING] " << method_name;
                if (!meta.converged) {
                    std::cout << " failed to converge.";
                }
                if (!is_valid(result)) {
                    if (!meta.converged) {
                        std::cout << " and";
                    }
                    std::cout << " produced NaN/Inf values.";
                }
                std::cout << " Retrying..." << std::endl;
                switch (s.current_strategy) {
                    case solver_strategy::cgs:
                        start_bicgstab();
                        break;
                    case solver_strategy::bicgstab:
                        start_gmres();
                        break;
                    default:
                        std::cout << "[ERROR] All facade solvers failed." << std::endl;
                        if (s.requester)
                            self->mail(std::string("All facade solvers failed")).send(s.requester);
                        self->quit();
                        break;
                }
            }
        }
    };
}

void caf_main(actor_system& sys) {
    manager::init(sys, manager_config(true, true)); 
    scoped_actor self{sys};

    std::string path = "/scratch/nqr159/matrix-collection/matrix_corpus_v2/matrices/unsymmetric/jpwh_991.bin";
    std::cout << "[INFO] Loading real-world matrix: " << path << std::endl;

    try {
        LocalCSR<double> A = load_binary_matrix<double>(path);
        std::vector<double> x_target(A.rows, 1.0);
        std::vector<double> b = compute_rhs(A, x_target);

        auto robust = sys.spawn(robust_solver, std::move(A), std::move(b), 1e-10, 5000);
        self->mail(start_atom_v).send(robust);
        
        self->receive(
            [](bool) { std::cout << "[INFO] Robust solver converged successfully." << std::endl; },
            [](std::string err) { std::cout << "[INFO] Robust solver aborted: " << err << std::endl; }
        );
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
    }

    manager::shutdown();
}

CAF_MAIN(id_block::cuda)