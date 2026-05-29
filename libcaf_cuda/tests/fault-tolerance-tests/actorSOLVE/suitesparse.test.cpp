#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <type_traits>
#include <cmath>
#include "caf/actorSOLVE/sparse-matrix-solvers/sparse-CGS-actor/sparse-CGS-actor.hpp"
#include "caf/actorSOLVE/sparse-matrix-solvers/sparse-BiCGSTAB-actor/sparse-BiCGSTAB-actor.hpp"
#include "caf/actorSOLVE/sparse-matrix-solvers/sparse-GMRES-actor/sparse-GMRES-actor.hpp"

using namespace caf;
using namespace caf::cuda;
namespace fs = std::filesystem;

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

template <class T>
bool is_valid(const std::vector<T>& x) {
    for (auto val : x) {
        if (std::isnan(val) || std::isinf(val)) return false;
    }
    return true;
}

struct solver_orchestrator_state {
    std::vector<caf::actor> float_solvers;
    std::vector<caf::actor> double_solvers;
};

behavior solver_orchestrator(stateful_actor<solver_orchestrator_state>* self) {
    // Create solver facades once and reuse them for all matrix solve requests
    self->state().float_solvers = {
        self->spawn<sparse_cg_facade<float>>(100),
        self->spawn<sparse_cg_jacobi_facade<float>>(100),
        self->spawn<sparse_bicgstab_facade<float>>(100),
        self->spawn<sparse_bicgstab_jacobi_facade<float>>(100),
        self->spawn<sparse_gmres_facade<float>>(100)
    };
    self->state().double_solvers = {
        self->spawn<sparse_cg_facade<double>>(100),
        self->spawn<sparse_cg_jacobi_facade<double>>(100),
        self->spawn<sparse_bicgstab_facade<double>>(100),
        self->spawn<sparse_bicgstab_jacobi_facade<double>>(100),
        self->spawn<sparse_gmres_facade<double>>(100)
    };

    auto run_solvers = [self](auto& A, auto tol, int max_iter) -> bool {
        using T = typename std::decay_t<decltype(A.values)>::value_type;
        std::vector<T> x_target(A.rows, T{1});
        std::vector<T> b = compute_rhs(A, x_target);
        std::vector<T> x_init(A.rows, T{0});

        const auto& solvers = std::is_same_v<T, float> ? self->state().float_solvers : self->state().double_solvers;

        for (size_t i = 0; i < solvers.size(); ++i) {
            bool success = false;
            auto& solver = solvers[i];
            
            // GMRES (index 4) needs the restart parameter (e.g., 30)
            if (i == 4) {
                self->mail(create_in_arg(A.row_ptr), create_in_arg(A.col_ind), create_in_arg(A.values),
                           create_in_arg(b), create_in_out_arg(x_init),
                           matrix_format::csr, A.rows, A.nnz, tol, max_iter, 30, 0, 0).send(solver);
            } else {
                self->mail(create_in_arg(A.row_ptr), create_in_arg(A.col_ind), create_in_arg(A.values),
                           create_in_arg(b), create_in_out_arg(x_init),
                           matrix_format::csr, A.rows, A.nnz, tol, max_iter, 0, 0).send(solver);
            }

            self->receive(
                [&](uint32_t, int, const std::vector<T>& result, solver_result_meta meta) {
                    if (meta.converged && is_valid(result)) success = true;
                }
            );
            if (success) return true;
        }
        return false;
    };

    return {
        [=](const std::string& path) -> bool {
            try {
                // Attempt float variants
                LocalCSR<float> Af = load_binary_matrix<float>(path);
                if (run_solvers(Af, 1e-4f, 1000)) return true;

                // Attempt double variants
                LocalCSR<double> Ad = load_binary_matrix<double>(path);
                if (run_solvers(Ad, 1e-8, 1000)) return true;
            } catch (...) {
                return false;
            }
            return false;
        }
    };
}

void caf_main(actor_system& sys) {
    manager::init(sys, manager_config(true, true));
    scoped_actor self{sys};
    auto orchestrator = sys.spawn(solver_orchestrator);

    std::string root = "/scratch/nqr159/matrix-collection/matrix_corpus_v2/matrices";
    if (!fs::exists(root)) return;

    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            std::string path = entry.path().string();
            
            bool ok = false;
            self->mail(path).request(orchestrator, infinite).receive(
                [&](bool result) {
                    ok = result;
                },
                [&](error& err) {
                    std::cerr << "Error encountered for " << path << ": " << sys.render(err) << std::endl;
                }
            );

            if (ok) {
                std::cout << path << std::endl;
            }
        }
    }
    manager::shutdown();
}

CAF_MAIN(id_block::cuda)