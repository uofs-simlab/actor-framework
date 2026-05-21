#pragma once

#include <caf/all.hpp>
#include <deque>
#include <vector>
#include "caf/cuda/all.hpp"
#include "caf/actorBLAS/dot-actor/dot-actor.hpp"
#include "caf/actorBLAS/gemv-actor/gemv-actor.hpp"
#include "caf/actorBLAS/axpy-actor/axpy-actor.hpp"
#include "caf/actorBLAS/copy-actor/copy-actor.hpp"

namespace caf::cuda {

struct cg_state {
  // Problem data
  mem_ptr<float> A, b, x;
  int n;
  float tol;
  int max_iter;

  // Workspace vectors
  mem_ptr<float> r, p, w, y_tmp;
  
  // Scalars
  float rho = 0.0f;
  float cur_norm = 0.0f;
  int iterations = 0;

  // Fault Tolerance: Stagnation Detection
  std::deque<float> norm_history;
  const size_t window_size = 15;
  const float stall_threshold = 0.001f; // 0.1%

  // BLAS Actors
  caf::actor dot_ptr, gemv_ptr, axpy_ptr, copy_ptr;

  caf::actor requester;
};

class cg_actor : public stateful_actor<cg_state> {
public:
  cg_actor(actor_config& cfg, mem_ptr<float> A, mem_ptr<float> b, mem_ptr<float> x, 
           int n, float tol, int max_iter)
    : stateful_actor<cg_state>(cfg) {
    state().A = A; state().b = b; state().x = x;
    state().n = n; state().tol = tol; state().max_iter = max_iter;
    
    // Initialize workspace (assuming command_runner is used for allocation)
    command_runner<out<float>> runner;
    state().r = std::get<0>(runner.transfer_memory(0, 0, out<float>(n)));
    state().p = std::get<0>(runner.transfer_memory(0, 0, out<float>(n)));
    state().w = std::get<0>(runner.transfer_memory(0, 0, out<float>(n)));
    state().y_tmp = std::get<0>(runner.transfer_memory(0, 0, out<float>(n)));

    // Spawn helpers
    state().dot_ptr = sys().spawn<dot_actor>();
    state().gemv_ptr = sys().spawn<gemv_actor>();
    state().axpy_ptr = sys().spawn<axpy_actor>();
    state().copy_ptr = sys().spawn<copy_actor>();
  }

  behavior make_behavior() override {
    return {
      [this](atom_value start) {
        state().requester = actor_cast<caf::actor>(this->current_sender());
        initial_setup();
      }
    };
  }

private:
  void initial_setup() {
    // Start with x0 = 0, so r = b
    request(state().copy_ptr, infinite, state().b, state().r, state().n).then([this](int, int, std::vector<float>) {
      // Initial search direction p = r
      request(state().copy_ptr, infinite, state().r, state().p, state().n).then([this](int, int, std::vector<float>) {
        // rho = r^T * r
        request(state().dot_ptr, infinite, state().r, state().r, state().y_tmp, state().n)
          .then([this](int, float res) {
            state().rho = res;
            state().cur_norm = std::sqrt(res);
            iterate();
          });
      });
    });
  }

  void iterate() {
    if (state().iterations >= state().max_iter || state().cur_norm < state().tol) {
      if (state().requester) this->send(state().requester, state().x);
      return;
    }

    // Step 1: w = Ap
    request(state().gemv_ptr, infinite, state().A, state().p, state().w, state().n, state().n, 1.0f, 0.0f)
      .then([this](int, int, std::vector<float>) {
        // Step 2: alpha = rho / (p^T * w)
        request(state().dot_ptr, infinite, state().p, state().w, state().y_tmp, state().n)
          .then([this](int, float p_dot_w) {
            float alpha = state().rho / p_dot_w;

            // Step 3: x = x + alpha*p
            request(state().axpy_ptr, infinite, state().p, state().x, state().n, alpha).then([this](int, int, std::vector<float>) {
              // Step 4: r = r - alpha*w
              request(state().axpy_ptr, infinite, state().w, state().r, state().n, -alpha).then([this](int, int, std::vector<float>) {
                // Step 5: rho_new = r^T * r
                request(state().dot_ptr, infinite, state().r, state().r, state().y_tmp, state().n)
                  .then([this](int, float rho_new) {
                    float old_rho = state().rho;
                    state().rho = rho_new;
                    state().cur_norm = std::sqrt(rho_new);
                    check_for_stall(old_rho);
                  });
              });
            });
          });
      });
  }

  void check_for_stall(float old_rho) {
    state().iterations++;
    state().norm_history.push_back(state().cur_norm);

    if (state().norm_history.size() > state().window_size) {
      float past_norm = state().norm_history.front();
      state().norm_history.pop_front();

      // Check: (Norm_old - Norm_new) / Norm_old < 0.1%
      if ((past_norm - state().cur_norm) / past_norm < state().stall_threshold) {
        this->println("[RECOVERY] Stall detected at iteration {}. Triggering Restart.", state().iterations);
        perform_restart();
        return;
      }
    }

    // Normal Update: beta = rho_new / rho_old
    float beta = state().rho / old_rho;

    // p = r + beta * p
    // Sequence: 1. w = r, 2. w = beta*p + w, 3. p = w
    request(state().copy_ptr, infinite, state().r, state().w, state().n).then([this, beta](int, int, std::vector<float>) {
      request(state().axpy_ptr, infinite, state().p, state().w, state().n, beta).then([this](int, int, std::vector<float>) {
        request(state().copy_ptr, infinite, state().w, state().p, state().n).then([this](int, int, std::vector<float>) {
          iterate();
        });
      });
    });
  }

  void perform_restart() {
    // Step 1: Recalculate y = M*x_k (purging drift)
    request(state().gemv_ptr, infinite, state().A, state().x, state().y_tmp, state().n, state().n, 1.0f, 0.0f)
      .then([this](int, int, std::vector<float>) {
        // Step 2: r = b - y (Clean residual)
        request(state().copy_ptr, infinite, state().b, state().r, state().n)
          .then([this](int, int, std::vector<float>) {
            request(state().axpy_ptr, infinite, state().y_tmp, state().r, state().n, -1.0f)
              .then([this](int, int, std::vector<float>) {
                // Step 3: p = r (Align search direction)
                request(state().copy_ptr, infinite, state().r, state().p, state().n)
                  .then([this](int, int, std::vector<float>) {
                    // Step 4: Resume
                    request(state().dot_ptr, infinite, state().r, state().r, state().y_tmp, state().n)
                      .then([this](int, float new_rho) {
                        state().rho = new_rho;
                        state().cur_norm = std::sqrt(new_rho);
                        state().norm_history.clear(); // Reset tracking
                        iterate();
                      });
                  });
              });
          });
      });
  }
};

} // namespace caf::cuda