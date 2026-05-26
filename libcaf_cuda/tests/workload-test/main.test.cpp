#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <algorithm>
#include "caf/actorSOLVE/sparse-matrix-solvers/sparse-CGS-actor/sparse-CGS-actor.hpp"
#include "sparse_utils.hpp"

using namespace caf;
using namespace caf::cuda;

void caf_main(actor_system& sys) {
    // Initialize GPU Manager with cuBLAS and cuSPARSE enabled
    manager::init(sys, manager_config(true, true));
    std::string filepath = "/scratch/nqr159/matrix-collection/matrices/spd/bcsstk08.bin";

    try {
        std::cout << "Loading binary file: " << filepath << " ...\n";
        SparseMatrixCOO coo_matrix = load_binary_coo(filepath);
        
        std::cout << "-> Matrix Loaded. Components: " 
                  << "Rows=" << coo_matrix.rows 
                  << ", Cols=" << coo_matrix.cols 
                  << ", NNZ=" << coo_matrix.nnz << "\n";

        std::cout << "Converting to Compressed Sparse Row (CSR) format...\n";
        SparseMatrixCSR A = convert_coo_to_csr(coo_matrix);

        // --- PREPARE BENCHMARK VECTORS ---
        std::cout << "Generating test vectors (b = A * x_true)...\n";
        
        // 1. Create a known ideal solution vector x_true (filled with 1.0f)
        std::vector<float> x_true(A.cols, 1.0f);
        
        // 2. Compute true right-hand side b
        std::vector<float> b = compute_rhs_spmv(A, x_true);
        
        // 3. Allocate an initial guess vector x filled with zeros
        std::vector<float> x_guess(A.cols, 0.0f);

        std::cout << "\n=========================================\n";
        std::cout << " Ready for Solver Execution!\n";
        std::cout << "=========================================\n";
        std::cout << "Arrays allocated and verified:\n";
        std::cout << " - A.values size:      " << A.values.size() << " elements\n";
        std::cout << " - A.row_ptr size:     " << A.row_ptr.size() << " elements\n";
        std::cout << " - Vector b size:      " << b.size() << " elements\n";
        std::cout << " - Vector x_guess size:" << x_guess.size() << " elements\n\n";
        
        scoped_actor self{sys};
        float tolerance = 1e-5f;
        int max_iter = 2000;

        // Spawn the Sparse CG Actor
        auto solver = sys.spawn<sparse_cg_actor>(
            create_in_arg(A.row_ptr), create_in_arg(A.col_indices), create_in_arg(A.values),
            create_in_arg(b), create_in_out_arg(x_guess),
            matrix_format::csr, A.rows, A.nnz, tolerance, max_iter, 0, 0, actor_cast<actor>(self));

        std::cout << "[INFO] Starting Solver Actor...\n";
        self->mail(start_atom_v).send(solver);

        self->receive(
            [&](std::vector<float> result_x) {
                std::cout << "\n=========================================\n";
                std::cout << " Solver Finished!\n";
                std::cout << "=========================================\n";
                std::cout << "Verification (Expected values near 1.0):\n";
                std::cout << "First 5 elements: ";
                for (int i = 0; i < std::min(5, (int)result_x.size()); ++i) {
                    std::cout << result_x[i] << " ";
                }
                std::cout << "\n";
            }
        );

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL EXCEPTION: " << e.what() << "\n";
    }
    manager::shutdown();
}
CAF_MAIN(id_block::cuda, id_block::cg_actor)
