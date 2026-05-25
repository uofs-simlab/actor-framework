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

namespace caf::cuda {

enum class matrix_format {
 csr,
 csc,
 coo
};

enum class sparse_cg_step {
 idle,
 init_r,
 init_p,
 init_rho,
 main_spmv_w,
 main_dot_pw,
 main_axpy_x,
 main_axpy_r,
 main_dot_rr,
 update_p_copy_r,
 update_p_axpy_p,
 update_p_final_copy
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
 caf::actor supervisor;

 // Workspace vectors
 mem_ptr<float> r, p, w, y_tmp;
 
 // Scalars
 float rho = 0.0f;
 float old_rho = 0.0f;
 float alpha = 0.0f;
 float beta = 0.0f;
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
       start_setup();
     },
   };
 }

private:
 void start_setup() {
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

   // Allocate workspace
   command_runner<out<float>> work_runner;
   s.r = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
   s.p = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
   s.w = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
   s.y_tmp = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));

   auto dev = platform::create()->schedule(s.stream_id, s.device_num);
   float alpha_const = 1.0f;
   float beta_const = 0.0f;
   float alpham1 = -1.0f;

   // 1. Initial SpMV: w = A * x
   execute_spmv(s.x, s.w);

   // 2. Initial r = b - w
   dev->scopy(s.stream_id, s.n, s.b, s.r);
   dev->saxpy(s.stream_id, s.n, alpham1, s.w, s.r);

   // 3. Initial rho = r * r
   dev->sdot(s.stream_id, s.n, s.r, s.r, s.y_tmp);
   s.rho = s.y_tmp->copy_to_host()[0];

   s.iterations = 1;
   while (s.rho > (s.tol * s.tol) && s.iterations <= s.max_iter) {
     if (s.iterations > 1) {
       s.beta = s.rho / s.old_rho;
       dev->scopy(s.stream_id, s.n, s.r, s.w);
       dev->saxpy(s.stream_id, s.n, s.beta, s.p, s.w);
       dev->scopy(s.stream_id, s.n, s.w, s.p);
     } else {
       dev->scopy(s.stream_id, s.n, s.r, s.p);
     }

     execute_spmv(s.p, s.w);

     dev->sdot(s.stream_id, s.n, s.p, s.w, s.y_tmp);
     float dot_pw = s.y_tmp->copy_to_host()[0];

     s.alpha = s.rho / dot_pw;

     dev->saxpy(s.stream_id, s.n, s.alpha, s.p, s.x);
     dev->saxpy(s.stream_id, s.n, -s.alpha, s.w, s.r);

     s.old_rho = s.rho;
     dev->sdot(s.stream_id, s.n, s.r, s.r, s.y_tmp);
     s.rho = s.y_tmp->copy_to_host()[0];

     s.iterations++;
   }

   auto solution = s.x->copy_to_host();
  
   if (s.supervisor)
     this->mail(std::move(solution)).send(s.supervisor);
 }

 void execute_spmv(mem_ptr<float> input_v, mem_ptr<float> output_v) {
   auto& s = state();
   auto dev = platform::create()->schedule(s.stream_id, s.device_num);
   switch (s.format) {
     case matrix_format::csr: dev->spmv_csr(s.stream_id, s.n, s.n, s.nnz, 1.0f, s.A_row_ptr, s.A_col_ind, s.A_values, input_v, 0.0f, output_v); break;
     case matrix_format::csc: dev->spmv_csc(s.stream_id, s.n, s.n, s.nnz, 1.0f, s.A_row_ptr, s.A_col_ind, s.A_values, input_v, 0.0f, output_v); break;
     case matrix_format::coo: dev->spmv_coo(s.stream_id, s.n, s.n, s.nnz, 1.0f, s.A_row_ptr, s.A_col_ind, s.A_values, input_v, 0.0f, output_v); break;
     default: break;
   }
 }
};

} // namespace caf::cuda