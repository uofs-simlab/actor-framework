#pragma once

#include <caf/all.hpp>
#include <deque>
#include <vector>
#include "caf/cuda/all.hpp"
#include "caf/actorBLAS/dot-actor/dot-actor.hpp"
#include "caf/actorSPARSE/spmv-actor/spmv-actor.hpp"
#include "caf/actorBLAS/axpy-actor/axpy-actor.hpp"
#include "caf/actorBLAS/copy-actor/copy-actor.hpp"
#include "caf/cuda/platform.hpp"

// Define a new block for CG specific atoms starting where the cuda block ended
CAF_BEGIN_TYPE_ID_BLOCK(cg_actor, caf::id_block::cuda::end)
  CAF_ADD_ATOM(cg_actor, cg_next_step_atom)
CAF_END_TYPE_ID_BLOCK(cg_actor)

namespace caf::cuda {

// // enum class matrix_format {
//  csr,
//  csc,
//  coo
// };

enum class sparse_cg_step {
 idle,
 init_rho,
 calc_dot_pw,
 check_convergence,
 finished
};

// Reply IDs used to distinguish which actor type is replying
constexpr int id_dot = 100;
constexpr int id_spmv = 200;
constexpr int id_axpy = 300;
constexpr int id_copy = 400;

struct sparse_cg_state {
 // Host Data (stored until start)
 in<int> h_row_ptr, h_col_ind;
 in<float> h_values, h_b;
 in_out<float> h_x;

 // Device Problem data
 mem_ptr<int> A_row_ptr, A_col_ind;
 mem_ptr<float> A_values, b, x;
 matrix_format format;
 int n, nnz;
 float tol;
 int max_iter;
 int device_num;
 int stream_id;
 device_ptr d_ptr;
 caf::actor supervisor;

 // Workspace vectors
 mem_ptr<float> r, p, w, y_tmp;
 mem_ptr<char> spmv_workspace;
 
 // Scalars needed across asynchronous steps
 float rho_val = 0.0f;
 float old_rho_val = 0.0f;
 float alpha_val = 0.0f;
 float beta_val = 0.0f;
 float dot_pw_val = 0.0f;
 int iterations = 0;
 sparse_cg_step step = sparse_cg_step::idle;
};

class sparse_cg_actor : public stateful_actor<sparse_cg_state> {
public:
 sparse_cg_actor(actor_config& cfg, in<int> rp, in<int> ci,
                 in<float> val, in<float> b, in_out<float> x,
                 matrix_format fmt, int n, int nnz, float tol, int max_iter, int device_num, int stream_id,
                 caf::actor supervisor = nullptr)
   : stateful_actor<sparse_cg_state>(cfg) {
   state().h_row_ptr = std::move(rp);
   state().h_col_ind = std::move(ci);
   state().h_values = std::move(val);
   state().h_b = std::move(b);
   state().h_x = std::move(x);
   state().format = fmt;
   state().n = n; state().nnz = nnz;
   state().tol = tol; state().max_iter = max_iter;
   state().device_num = device_num; state().stream_id = stream_id;
   state().supervisor = supervisor;
 }

 behavior make_behavior() override {
   return {
     [this](start_atom) {
       auto& s = state();
       if (!s.supervisor)
         s.supervisor = actor_cast<caf::actor>(this->current_sender());
       start_solve();
     },
     [this](cg_next_step_atom, float val) {
       auto& s = state();
       // Thread-safely update scalars based on the stage that just finished
       switch (s.step) {
         case sparse_cg_step::init_rho:
         case sparse_cg_step::check_convergence:
           s.rho_val = val;
           break;
         case sparse_cg_step::calc_dot_pw:
           s.dot_pw_val = val;
           break;
         default: break;
       }
       perform_cg_step();
     },
     [this](gpu_done_atom, std::vector<float>& solution) {
       if (state().supervisor)
         this->mail(std::move(solution)).send(state().supervisor);
       this->quit();
     }
   };
 }

private:
 void start_solve() {
   auto& s = state();
   command_runner<> runner;

   // Transfer problem data to device
   auto res = runner.transfer_memory(s.device_num, s.stream_id,
                                     s.h_row_ptr,
                                     s.h_col_ind,
                                     s.h_values,
                                     s.h_b,
                                     s.h_x);
  
   s.A_row_ptr = std::get<0>(res);
   s.A_col_ind = std::get<1>(res);
   s.A_values  = std::get<2>(res);
   s.b         = std::get<3>(res);
   s.x         = std::get<4>(res);

   s.d_ptr = platform::create()->schedule(s.stream_id, s.device_num);

   // Allocate workspace
   command_runner<out<float>> work_runner;
   s.r = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
   s.p = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
   s.w = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
   s.y_tmp = work_runner.transfer_memory(s.device_num, s.stream_id, create_out_arg_with_size<float>(1));

   // Allocate SPMV workspace to avoid reallocations in the loop
   size_t ws_size = 0;
   if (s.format == matrix_format::csr)
     ws_size = s.d_ptr->spmv_csr_buffer_size(s.stream_id, s.n, s.n, s.nnz, s.A_row_ptr, s.A_col_ind, s.A_values, s.x, s.w);
   else if (s.format == matrix_format::csc)
     ws_size = s.d_ptr->spmv_csc_buffer_size(s.stream_id, s.n, s.n, s.nnz, s.A_row_ptr, s.A_col_ind, s.A_values, s.x, s.w);
   else if (s.format == matrix_format::coo)
     ws_size = s.d_ptr->spmv_coo_buffer_size(s.stream_id, s.n, s.n, s.nnz, s.A_row_ptr, s.A_col_ind, s.A_values, s.x, s.w);

   if (ws_size > 0) {
     command_runner<out<char>> ws_runner;
     s.spmv_workspace = ws_runner.transfer_memory(s.device_num, s.stream_id, out<char>(static_cast<int>(ws_size)));
   }

   // 1. Initial SpMV: w = A * x
   execute_spmv(s.x, s.w);

   // 2. Initial r = b - w
   s.d_ptr->scopy(s.stream_id, s.n, s.b, s.r);
   s.d_ptr->saxpy(s.stream_id, s.n, -1.0f, s.w, s.r);

   // 3. Initial rho = r * r
   s.d_ptr->sdot(s.stream_id, s.n, s.r, s.r, s.y_tmp);
   
   s.step = sparse_cg_step::init_rho;
   auto self = actor_cast<actor>(this);
   runner.copy_to_host_async(s.y_tmp, [self](std::vector<float> host_data) {    
    anon_mail(cg_next_step_atom_v, host_data[0]).send(self);
   });
 }

 void perform_cg_step() {
   auto& s = state();
   command_runner<> runner;
   auto self = actor_cast<actor>(this);

   switch (s.step) {
     case sparse_cg_step::init_rho:
     case sparse_cg_step::check_convergence: {
       s.iterations++; // Increment for the current iteration

       // Check convergence
       if (s.rho_val <= (s.tol * s.tol) || s.iterations > s.max_iter) {
         finish_solve();
         return;
       }

       if (s.iterations > 1) {
         s.beta_val = s.rho_val / s.old_rho_val;
         s.d_ptr->scopy(s.stream_id, s.n, s.r, s.w);
         s.d_ptr->saxpy(s.stream_id, s.n, s.beta_val, s.p, s.w);
         s.d_ptr->scopy(s.stream_id, s.n, s.w, s.p);
       } else {
         s.d_ptr->scopy(s.stream_id, s.n, s.r, s.p);
       }

       execute_spmv(s.p, s.w);

       s.d_ptr->sdot(s.stream_id, s.n, s.p, s.w, s.y_tmp);
       s.step = sparse_cg_step::calc_dot_pw;
       runner.copy_to_host_async(s.y_tmp, [self](std::vector<float> host_data) {
         anon_mail(cg_next_step_atom_v, host_data[0]).send(self);
       });
       break;
     }

     case sparse_cg_step::calc_dot_pw: {
       s.alpha_val = s.rho_val / s.dot_pw_val;

       s.d_ptr->saxpy(s.stream_id, s.n, s.alpha_val, s.p, s.x);
       s.d_ptr->saxpy(s.stream_id, s.n, -s.alpha_val, s.w, s.r);

       s.old_rho_val = s.rho_val;
       s.d_ptr->sdot(s.stream_id, s.n, s.r, s.r, s.y_tmp);
       s.step = sparse_cg_step::check_convergence;
       runner.copy_to_host_async(s.y_tmp, [self](std::vector<float> host_data) {
         anon_mail(cg_next_step_atom_v, host_data[0]).send(self);
       });
       break;
     }

     default: break;
   }
 }

 void finish_solve() {
   auto& s = state();
   s.step = sparse_cg_step::finished;
   auto self = actor_cast<actor>(this);
   command_runner<> runner;
   runner.copy_to_host_async(s.x, [self](std::vector<float> solution) {
     anon_mail(gpu_done_atom_v, std::move(solution)).send(self);
   });
 }

 void execute_spmv(mem_ptr<float> input_v, mem_ptr<float> output_v) {
   auto& s = state();
   switch (s.format) {
     case matrix_format::csr: s.d_ptr->spmv_csr(s.stream_id, s.n, s.n, s.nnz, 1.0f, s.A_row_ptr, s.A_col_ind, s.A_values, input_v, 0.0f, output_v, s.spmv_workspace); break;
     case matrix_format::csc: s.d_ptr->spmv_csc(s.stream_id, s.n, s.n, s.nnz, 1.0f, s.A_row_ptr, s.A_col_ind, s.A_values, input_v, 0.0f, output_v, s.spmv_workspace); break;
     case matrix_format::coo: s.d_ptr->spmv_coo(s.stream_id, s.n, s.n, s.nnz, 1.0f, s.A_row_ptr, s.A_col_ind, s.A_values, input_v, 0.0f, output_v, s.spmv_workspace); break;
     default: break;
   }
 }
};

} // namespace caf::cuda