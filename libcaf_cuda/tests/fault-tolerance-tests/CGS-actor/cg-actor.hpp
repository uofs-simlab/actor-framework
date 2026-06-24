#pragma once

#include <caf/all.hpp>
#include <deque>
#include <vector>
#include "caf/cuda/all.hpp"
#include "caf/actorBLAS/dot-actor/dot-actor.hpp"
#include "caf/actorBLAS/gemv-actor/gemv-actor.hpp"
#include "caf/actorBLAS/axpy-actor/axpy-actor.hpp"
#include "caf/actorBLAS/copy-actor/copy-actor.hpp"

CAF_BEGIN_TYPE_ID_BLOCK(cg_solver, caf::first_custom_type_id + 50)
  CAF_ADD_ATOM(cg_solver, start_atom)
CAF_END_TYPE_ID_BLOCK(cg_solver)

namespace caf::cuda {

enum class cg_step {
  idle,
  init_r,
  init_p,
  init_rho,
  main_gemv_w,
  main_dot_pw,
  main_axpy_x,
  main_axpy_r,
  main_dot_rr,
  update_p_copy_r,
  update_p_axpy_p,
  update_p_final_copy,
  restart_gemv_y,
  restart_copy_b,
  restart_axpy_r,
  restart_copy_p,
  restart_dot_rho
};

// Reply IDs used to distinguish which actor type is replying
constexpr int id_dot = 100;
constexpr int id_gemv = 200;
constexpr int id_axpy = 300;
constexpr int id_copy = 400;

struct cg_state {
  // Problem data
  mem_ptr<float> A, b, x;
  int n;
  float tol;
  int max_iter;
  int device_num;
  int stream_id;

  // Workspace vectors
  mem_ptr<float> r, p, w, y_tmp;
  
  // Scalars
  float rho = 0.0f;
  float old_rho = 0.0f;
  float cur_norm = 0.0f;
  float alpha = 0.0f;
  float beta = 0.0f;
  int iterations = 0;
  cg_step step = cg_step::idle;

  // Fault Tolerance: Stagnation Detection
  float last_norm = -1.0f;
  int stagnation_count = 0;

  // BLAS Actors
  caf::actor dot_actor, gemv_actor, axpy_actor, copy_actor;

  caf::actor requester;
};

class cg_actor : public stateful_actor<cg_state> {
public:
  cg_actor(actor_config& cfg, mem_ptr<float> A, mem_ptr<float> b, mem_ptr<float> x, 
           int n, float tol, int max_iter, int device_num, int stream_id)
    : stateful_actor<cg_state>(cfg) {
    state().A = A; state().b = b; state().x = x;
    state().n = n; state().tol = tol; state().max_iter = max_iter;
    state().device_num = device_num; state().stream_id = stream_id;
    
    // Initialize workspace (assuming command_runner is used for allocation)
    command_runner<out<float>> runner;
    state().r = runner.transfer_memory(device_num, stream_id, out<float>(n));
    state().p = runner.transfer_memory(device_num, stream_id, out<float>(n));
    state().w = runner.transfer_memory(device_num, stream_id, out<float>(n));
    state().y_tmp = runner.transfer_memory(device_num, stream_id, out<float>(n));

    // Spawn helpers with specific RIDs to distinguish messages
    state().dot_actor = this->system().spawn<caf::cuda::dot_actor>(id_dot);
    state().gemv_actor = this->system().spawn<caf::cuda::gemv_actor>(id_gemv);
    state().axpy_actor = this->system().spawn<caf::cuda::axpy_actor>(id_axpy);
    state().copy_actor = this->system().spawn<caf::cuda::copy_actor>(id_copy);
  }

  behavior make_behavior() override {
    return {
      [this](start_atom) {
        state().requester = actor_cast<caf::actor>(this->current_sender());
        start_setup();
      },
      // Dot product result (Host scalar)
      [this](int rid, float val) {
        if (rid == id_dot) {
          handle_dot_result(val);
        }
      },
      // Matrix-Vector result (Memory handles)
      [this](int rid, mem_ptr<float> A, mem_ptr<float> x, mem_ptr<float> y) { // Assuming GEMV returns A, x, and y
        if (rid == id_gemv) {
          handle_gemv_result();
        }
      },
      // Copy or AXPY result (Memory handles share the same signature)
      [this](int rid, mem_ptr<float> x, mem_ptr<float> y) {
        if (rid == id_copy) {
          handle_copy_result();
        } else if (rid == id_axpy) {
          handle_axpy_result();
        }
      }
    };
  }

private:
  void start_setup() {
    auto& s = state();
    state().step = cg_step::init_r;
    this->mail(return_mem_ptr_atom_v, s.device_num, s.stream_id, s.b, s.r, s.n).send(s.copy_actor);
  }

  void iterate() {
    auto& s = state();
    if (state().iterations >= state().max_iter || state().cur_norm < state().tol) {
      if (state().requester) this->mail(state().x).send(state().requester);
      state().step = cg_step::idle;
      return;
    }
    state().step = cg_step::main_gemv_w;
    this->mail(return_mem_ptr_atom_v, s.device_num, s.stream_id, s.A, s.p, s.w, s.n, s.n).send(s.gemv_actor);
  }

  void perform_restart() {
    auto& s = state();
    std::cout << "[RECOVERY] Triggering Mathematical Restart at iteration " << state().iterations << "\n";
    state().stagnation_count = 0;
    state().step = cg_step::restart_gemv_y;
    this->mail(return_mem_ptr_atom_v, s.device_num, s.stream_id, s.A, s.x, s.y_tmp, s.n, s.n).send(s.gemv_actor);
  }

  void handle_dot_result(float val) {
    auto& s = state();
    switch (s.step) {
      case cg_step::init_rho:
        s.rho = val;
        s.cur_norm = std::sqrt(val);
        iterate();
        break;
      case cg_step::main_dot_pw:
        s.alpha = s.rho / val;
        s.step = cg_step::main_axpy_x;
        this->mail(return_mem_ptr_atom_v, s.device_num, s.stream_id, s.p, s.x, s.n, s.alpha).send(s.axpy_actor);
        break;
      case cg_step::main_dot_rr:
        s.old_rho = s.rho;
        s.rho = val;
        s.cur_norm = std::sqrt(val);
        check_stagnation();
        break;
      case cg_step::restart_dot_rho:
        s.rho = val;
        s.cur_norm = std::sqrt(val);
        iterate();
        break;
      default: break;
    }
  }

  void check_stagnation() {
    auto& s = state();
    // Debugging prints to observe stagnation behavior
    // std::cout << "[CG_DEBUG] Iter: " << s.iterations 
    //           << ", cur_norm: " << s.cur_norm 
    //           << ", last_norm: " << s.last_norm << ", stagnation_count: " << s.stagnation_count << "\n";
    
    bool diverged = s.last_norm > 0 && s.cur_norm > s.last_norm * 1.5f;
    bool stalled = s.last_norm > 0 && s.cur_norm > s.last_norm * 0.999f;

    s.iterations++;
    if (stalled || diverged) {
      s.stagnation_count++;
    } else {
      s.stagnation_count = 0;
    }
    s.last_norm = s.cur_norm;

    if (s.stagnation_count >= 15 || diverged) {
      perform_restart();
    } else {
      s.beta = s.rho / s.old_rho;
      s.step = cg_step::update_p_copy_r;
      this->mail(return_mem_ptr_atom_v, s.device_num, s.stream_id, s.r, s.w, s.n).send(s.copy_actor);
    }
  }

  void handle_gemv_result() {
    auto& s = state();
    if (s.step == cg_step::main_gemv_w) {
      s.step = cg_step::main_dot_pw;
      this->mail(s.device_num, s.stream_id, s.p, s.w, s.y_tmp, s.n).send(s.dot_actor);
    } else if (s.step == cg_step::restart_gemv_y) {
      s.step = cg_step::restart_copy_b;
      this->mail(return_mem_ptr_atom_v, s.device_num, s.stream_id, s.b, s.r, s.n).send(s.copy_actor);
    }
  }

  void handle_axpy_result() {
    auto& s = state();
    if (s.step == cg_step::main_axpy_x) {
      s.step = cg_step::main_axpy_r;
      this->mail(return_mem_ptr_atom_v, s.device_num, s.stream_id, s.w, s.r, s.n, -s.alpha).send(s.axpy_actor);
    } else if (s.step == cg_step::main_axpy_r) {
      s.step = cg_step::main_dot_rr;
      this->mail(s.device_num, s.stream_id, s.r, s.r, s.y_tmp, s.n).send(s.dot_actor);
    } else if (s.step == cg_step::update_p_axpy_p) {
      s.step = cg_step::update_p_final_copy;
      this->mail(return_mem_ptr_atom_v, s.device_num, s.stream_id, s.w, s.p, s.n).send(s.copy_actor);
    } else if (s.step == cg_step::restart_axpy_r) {
      s.step = cg_step::restart_copy_p;
      this->mail(return_mem_ptr_atom_v, s.device_num, s.stream_id, s.r, s.p, s.n).send(s.copy_actor);
    }
  }

  void handle_copy_result() {
    auto& s = state();
    switch (s.step) {
      case cg_step::init_r:
        s.step = cg_step::init_p;
        this->mail(return_mem_ptr_atom_v, s.device_num, s.stream_id, s.r, s.p, s.n).send(s.copy_actor);
        break;
      case cg_step::init_p:
        s.step = cg_step::init_rho;
        this->mail(s.device_num, s.stream_id, s.r, s.r, s.y_tmp, s.n).send(s.dot_actor);
        break;
      case cg_step::update_p_copy_r:
        s.step = cg_step::update_p_axpy_p;
        this->mail(return_mem_ptr_atom_v, s.device_num, s.stream_id, s.p, s.w, s.n, s.beta).send(s.axpy_actor);
        break;
      case cg_step::update_p_final_copy:
        iterate();
        break;
      case cg_step::restart_copy_b:
        s.step = cg_step::restart_axpy_r;
        this->mail(return_mem_ptr_atom_v, s.device_num, s.stream_id, s.y_tmp, s.r, s.n, -1.0f).send(s.axpy_actor);
        break;
      case cg_step::restart_copy_p:
        s.step = cg_step::restart_dot_rho;
        this->mail(s.device_num, s.stream_id, s.r, s.r, s.y_tmp, s.n).send(s.dot_actor);
        break;
      default: break;
    }
  }
};

} // namespace caf::cuda