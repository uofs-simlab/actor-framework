#pragma once

#include <caf/actor.hpp>
#include <caf/actor_cast.hpp>
#include <caf/actor_config.hpp>
#include <caf/actor_system.hpp>
#include <caf/anon_mail.hpp>
#include <caf/event_based_actor.hpp>

#include "caf/cuda/device.hpp"
#include "caf/cuda/mem_ref.hpp"
#include "caf/cuda/command_runner.hpp"
#include "caf/cuda/platform.hpp"
#include "caf/cuda/types.hpp"

namespace caf::cuda {

/// NRM2 Actor for single-precision vector Euclidean norm.
/// Message Signature: (in<float> x, out<float> res, int n)
class nrm2_actor : public event_based_actor {
public:
  static caf::actor spawn(caf::actor_system& sys, int reply_id = 0) {
    return sys.spawn<nrm2_actor>(reply_id);
  }

  nrm2_actor(caf::actor_config& cfg, int reply_id = 0) 
    : event_based_actor(cfg), reply_id_(reply_id) {
    actor_id_ = static_cast<int>(this->id());
  }

  ~nrm2_actor() override {
    command_runner<> runner;
    runner.release_stream_for_actor(actor_id_);
  }

  caf::behavior make_behavior() override {
    return {
      // Standard host buffer based calls
      [this](in<float> x, out<float> res, int n) {
        enqueue_nrm2(-1, actor_id_, x, res, n, false);
      },
      // Routing control overloads
      [this](int device_num, int stream_id, in<float> x, out<float> res, int n) {
        enqueue_nrm2(device_num, stream_id, x, res, n, false);
      },
      // mem_ptr based calls
      [this](mem_ptr<float> x, mem_ptr<float> res, int n) {
        enqueue_nrm2(-1, actor_id_, x, res, n, false);
      },
      [this](int device_num, int stream_id, mem_ptr<float> x, mem_ptr<float> res, int n) {
        enqueue_nrm2(device_num, stream_id, x, res, n, false);
      },
      // Mem ptr return overloads
      [this](return_mem_ptr_atom, in<float> x, out<float> res, int n) {
        enqueue_nrm2(-1, actor_id_, x, res, n, true);
      },
      [this](return_mem_ptr_atom, int device_num, int stream_id, in<float> x, out<float> res, int n) {
        enqueue_nrm2(device_num, stream_id, x, res, n, true);
      },
      [this](return_mem_ptr_atom, mem_ptr<float> x, mem_ptr<float> res, int n) {
        enqueue_nrm2(-1, actor_id_, x, res, n, true);
      },
    };
  }

private:
  void enqueue_nrm2(int device_num, int stream_id, 
                    in<float> x_arg, out<float> res_arg, 
                    int n, bool return_ptrs) {
    command_runner<in<float>, out<float>> runner;
    auto results = runner.transfer_memory(device_num, stream_id, x_arg, res_arg);
    execute_and_reply(device_num, stream_id, std::get<0>(results), 
                      std::get<1>(results), n, return_ptrs);
  }

  void enqueue_nrm2(int device_num, int stream_id, 
                    mem_ptr<float> x_ptr, mem_ptr<float> res_ptr, 
                    int n, bool return_ptrs) {
    command_runner<mem_ptr<float>, mem_ptr<float>> runner;
    auto results = runner.transfer_memory(device_num, stream_id, x_ptr, res_ptr);
    execute_and_reply(device_num, stream_id, std::get<0>(results), 
                      std::get<1>(results), n, return_ptrs);
  }

  void execute_and_reply(int device_num, int stream_id, 
                         mem_ptr<float> x, mem_ptr<float> res, 
                         int n, bool return_ptrs) {
    auto plat = platform::create();
    device_ptr dev;
    if (device_num == -1)
      dev = plat->schedule(stream_id);
    else
      dev = plat->schedule(stream_id, device_num);

    dev->snrm2(stream_id, n, x, res);
    handle_reply(device_num, stream_id, x, res, return_ptrs);
  }

  void handle_reply(int, int, mem_ptr<float> x_ptr, mem_ptr<float> res_ptr, bool return_ptrs) {
    command_runner<> runner;
    auto sender = actor_cast<actor>(this->current_sender());
    if (!sender) return;
    
    auto r_id = reply_id_;
    if (return_ptrs) {
      caf::anon_mail(r_id, x_ptr, res_ptr).send(sender);
    } else {
      runner.copy_to_host_async(res_ptr, [sender, r_id](std::vector<float>&& data) {
        if (sender) {
          caf::anon_mail(r_id, 1, std::move(data)).send(sender);
        }
      });
    }
  }

  int actor_id_;
  int reply_id_;
};

} // namespace caf::cuda

