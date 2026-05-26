#pragma once

#include <caf/all.hpp>
#include <vector>
#include "caf/cuda/all.hpp"
#include "caf/actorBLAS/dot-actor/dot-actor.hpp"
#include "caf/actorSPARSE/spmv-actor/spmv-actor.hpp"
#include "caf/actorBLAS/axpy-actor/axpy-actor.hpp"
#include "caf/actorBLAS/copy-actor/copy-actor.hpp"
#include "caf/cuda/platform.hpp"
#include "caf/actorSOLVE/sparse-matrix-solvers/sparse-CGS-actor/sparse-CGS-actor.hpp"

// Define a new block for BiCGSTAB specific atoms starting where the cuda block ended
CAF_BEGIN_TYPE_ID_BLOCK(bicgstab_actor, caf::id_block::cuda::end)
  CAF_ADD_ATOM(bicgstab_actor, bicgstab_next_step_atom)
CAF_END_TYPE_ID_BLOCK(bicgstab_actor)

namespace caf::cuda {

enum class sparse_bicgstab_step {
  idle,
  init_residual_norm_sq, // Calculate initial ||r||^2
  calc_rho_new,          // Calculate rho_new = <r_hat, r>
  calc_alpha_denom,      // Calculate <r_hat, v> for alpha
  calc_omega_num,        // Calculate <t, s> for omega
  calc_omega_denom,      // Calculate <t, t> for omega
  check_convergence,     // Check ||r||^2 for convergence
  finished
};

struct sparse_bicgstab_state {
  // Host Data
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
  mem_ptr<float> r, r_hat, p, v, s_vec, t_vec, y_tmp;
  
  // Scalars needed across asynchronous steps
  float rho_val = 1.0f; // Renamed from rho to avoid conflict with step-specific rho_new
  float alpha_val = 1.0f;
  float omega_val = 1.0f;
  float beta_val = 0.0f;
  float residual_norm_sq_val = 0.0f; // Stores the latest ||r||^2
  float rho_new_val = 0.0f; // Stores the result of <r_hat, r>
  float alpha_denom_val = 0.0f; // Stores the result of <r_hat, v>
  float omega_num_val = 0.0f; // Stores the result of <t, s>
  float omega_denom_val = 0.0f; // Stores the result of <t, t>

  int iterations = 0;
  sparse_bicgstab_step step = sparse_bicgstab_step::idle;
};

class sparse_bicgstab_actor : public stateful_actor<sparse_bicgstab_state> {
public:
  sparse_bicgstab_actor(actor_config& cfg, in<int> rp, in<int> ci,
                        in<float> val, in<float> b, in_out<float> x,
                        matrix_format fmt, int n, int nnz, float tol, int max_iter, int device_num, int stream_id,
                        caf::actor supervisor = nullptr)
    : stateful_actor<sparse_bicgstab_state>(cfg) {
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
      [this](bicgstab_next_step_atom, float val) {
        auto& s = state();
        // Thread-safely update scalars based on the stage that just finished
        switch (s.step) {
          case sparse_bicgstab_step::init_residual_norm_sq:
          case sparse_bicgstab_step::check_convergence:
            s.residual_norm_sq_val = val;
            break;
          case sparse_bicgstab_step::calc_rho_new:
            s.rho_new_val = val;
            break;
          case sparse_bicgstab_step::calc_alpha_denom:
            s.alpha_denom_val = val;
            break;
          case sparse_bicgstab_step::calc_omega_num:
            s.omega_num_val = val;
            break;
          case sparse_bicgstab_step::calc_omega_denom:
            s.omega_denom_val = val;
            break;
          default: break;
        }
        perform_bicgstab_step();
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
                                      s.h_row_ptr, s.h_col_ind,
                                      s.h_values, s.h_b, s.h_x);
   
    s.A_row_ptr = std::get<0>(res);
    s.A_col_ind = std::get<1>(res);
    s.A_values  = std::get<2>(res);
    s.b         = std::get<3>(res);
    s.x         = std::get<4>(res);

    // Allocate workspace
    command_runner<out<float>> work_runner;
    s.r     = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.r_hat = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.p     = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.v     = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.s_vec = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.t_vec = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.y_tmp = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));

    auto dev = platform::create()->schedule(s.stream_id, s.device_num);

    // 1. Initial Residual: r = b - Ax
    execute_spmv(s.x, s.v); 
    dev->scopy(s.stream_id, s.n, s.b, s.r);
    dev->saxpy(s.stream_id, s.n, -1.0f, s.v, s.r);
    
    // 2. Choose r_hat = r
    dev->scopy(s.stream_id, s.n, s.r, s.r_hat);

    s.iterations = 0;
    // Initial calculation of residual_norm_sq
    dev->sdot(s.stream_id, s.n, s.r, s.r, s.y_tmp);
    s.step = sparse_bicgstab_step::init_residual_norm_sq;
    auto self = actor_cast<actor>(this);
    runner.copy_to_host_async(s.y_tmp, [self](std::vector<float> host_data) {
      anon_mail(bicgstab_next_step_atom_v, host_data[0]).send(self);
    });
  }

  void perform_bicgstab_step() {
    auto& s = state();
    auto dev = platform::create()->schedule(s.stream_id, s.device_num);
    command_runner<> runner;
    auto self = actor_cast<actor>(this);

    switch (s.step) {
      case sparse_bicgstab_step::init_residual_norm_sq:
      case sparse_bicgstab_step::check_convergence: {
        s.iterations++; // Increment for the current iteration

        // Check convergence
        if (s.residual_norm_sq_val <= (s.tol * s.tol) || s.iterations > s.max_iter) {
          finish_solve();
          return;
        }

        // rho_i = <r_hat, r>
        dev->sdot(s.stream_id, s.n, s.r_hat, s.r, s.y_tmp);
        s.step = sparse_bicgstab_step::calc_rho_new;
        runner.copy_to_host_async(s.y_tmp, [self](std::vector<float> host_data) {
          anon_mail(bicgstab_next_step_atom_v, host_data[0]).send(self);
        });
        break;
      }

      case sparse_bicgstab_step::calc_rho_new: {
        // Update p based on rho_new_val
        if (s.iterations == 1) { // This is the first iteration
          dev->scopy(s.stream_id, s.n, s.r, s.p);
        } else { // Subsequent iterations
          s.beta_val = (s.rho_new_val / s.rho_val) * (s.alpha_val / s.omega_val);
          // p = r + beta * (p - omega * v)
          dev->saxpy(s.stream_id, s.n, -s.omega_val, s.v, s.p); // p = p - omega*v
          dev->scopy(s.stream_id, s.n, s.r, s.s_vec);      // use s_vec as temporary
          dev->saxpy(s.stream_id, s.n, s.beta_val, s.p, s.s_vec); // s_vec = r + beta*p
          dev->scopy(s.stream_id, s.n, s.s_vec, s.p);
        }
        s.rho_val = s.rho_new_val; // Update old rho

        // v = Ap
        execute_spmv(s.p, s.v);

        // alpha = rho / <r_hat, v> -> calculate denominator
        dev->sdot(s.stream_id, s.n, s.r_hat, s.v, s.y_tmp);
        s.step = sparse_bicgstab_step::calc_alpha_denom;
        runner.copy_to_host_async(s.y_tmp, [self](std::vector<float> host_data) {
          anon_mail(bicgstab_next_step_atom_v, host_data[0]).send(self);
        });
        break;
      }

      case sparse_bicgstab_step::calc_alpha_denom: {
        s.alpha_val = s.rho_val / s.alpha_denom_val;

        // s = r - alpha * v
        dev->scopy(s.stream_id, s.n, s.r, s.s_vec);
        dev->saxpy(s.stream_id, s.n, -s.alpha_val, s.v, s.s_vec);

        // t = As
        execute_spmv(s.s_vec, s.t_vec);

        // omega = <t, s> / <t, t> -> calculate numerator <t, s>
        dev->sdot(s.stream_id, s.n, s.t_vec, s.s_vec, s.y_tmp);
        s.step = sparse_bicgstab_step::calc_omega_num;
        runner.copy_to_host_async(s.y_tmp, [self](std::vector<float> host_data) {
          anon_mail(bicgstab_next_step_atom_v, host_data[0]).send(self);
        });
        break;
      }

      case sparse_bicgstab_step::calc_omega_num: {
        // omega = <t, s> / <t, t> -> calculate denominator <t, t>
        dev->sdot(s.stream_id, s.n, s.t_vec, s.t_vec, s.y_tmp);
        s.step = sparse_bicgstab_step::calc_omega_denom;
        runner.copy_to_host_async(s.y_tmp, [self](std::vector<float> host_data) {
          anon_mail(bicgstab_next_step_atom_v, host_data[0]).send(self);
        });
        break;
      }

      case sparse_bicgstab_step::calc_omega_denom: {
        s.omega_val = s.omega_num_val / s.omega_denom_val;

        // x = x + alpha*p + omega*s
        dev->saxpy(s.stream_id, s.n, s.alpha_val, s.p, s.x);
        dev->saxpy(s.stream_id, s.n, s.omega_val, s.s_vec, s.x);

        // r = s - omega*t
        dev->scopy(s.stream_id, s.n, s.s_vec, s.r);
        dev->saxpy(s.stream_id, s.n, -s.omega_val, s.t_vec, s.r);

        // Calculate new residual_norm_sq for next iteration's convergence check
        dev->sdot(s.stream_id, s.n, s.r, s.r, s.y_tmp);
        s.step = sparse_bicgstab_step::check_convergence;
        runner.copy_to_host_async(s.y_tmp, [self](std::vector<float> host_data) {
          anon_mail(bicgstab_next_step_atom_v, host_data[0]).send(self);
        });
        break;
      }

      case sparse_bicgstab_step::idle:
      case sparse_bicgstab_step::finished:
        // Should not happen if logic is correct
        break;
    }
  }

  void finish_solve() {
    auto& s = state();
    s.step = sparse_bicgstab_step::finished;
    // Final copy to host and send result to supervisor
    auto self = actor_cast<actor>(this);
    command_runner<> runner;
    runner.copy_to_host_async(s.x, [self](std::vector<float> solution) {
      anon_mail(gpu_done_atom_v, std::move(solution)).send(self);
    });
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