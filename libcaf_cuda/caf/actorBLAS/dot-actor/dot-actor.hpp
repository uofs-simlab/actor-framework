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

/// SDOT Actor for result = x^T * y.
class dot_actor : public event_based_actor {
public:
  static caf::actor spawn(caf::actor_system& sys, int reply_id = 0) {
    return sys.spawn<dot_actor>(reply_id);
  }

  dot_actor(caf::actor_config& cfg, int reply_id = 0) 
    : event_based_actor(cfg), reply_id_(reply_id) {
    actor_id_ = static_cast<int>(this->id());
  }

  ~dot_actor() override {
    command_runner<> runner;
    runner.release_stream_for_actor(actor_id_);
  }

  caf::behavior make_behavior() override {
    return {
      [this](in<float> x, in<float> y, out<float> res, int n) {
        enqueue_sdot(-1, actor_id_, x, y, res, n, false);
      },
      [this](int device_num, int stream_id, in<float> x, in<float> y, out<float> res, int n) {
        enqueue_sdot(device_num, stream_id, x, y, res, n, false);
      },
      [this](mem_ptr<float> x, mem_ptr<float> y, mem_ptr<float> res, int n) {
        enqueue_sdot(-1, actor_id_, x, y, res, n, false);
      },
      [this](int device_num, int stream_id, mem_ptr<float> x, mem_ptr<float> y, mem_ptr<float> res, int n) {
        enqueue_sdot(device_num, stream_id, x, y, res, n, false);
      },
      [this](return_mem_ptr_atom, mem_ptr<float> x, mem_ptr<float> y, mem_ptr<float> res, int n) {
        enqueue_sdot(-1, actor_id_, x, y, res, n, true);
      },
      [this](return_mem_ptr_atom, in<float> x, in<float> y, out<float> res, int n) {
        enqueue_sdot(-1, actor_id_, x, y, res, n, true);
      },
      [this](return_mem_ptr_atom,int device, int stream, in<float> x, in<float> y, out<float> res, int n) {
        enqueue_sdot(device,stream, x, y, res, n, true);
      },
       [this](return_mem_ptr_atom,int device_num, int stream_id ,mem_ptr<float> x, mem_ptr<float> y, mem_ptr<float> res, int n) {
        enqueue_sdot(device_num, stream_id, x, y, res, n, true);
      }
    };
  }

private:
  void enqueue_sdot(int device_num, int stream_id, in<float> x, in<float> y, out<float> res, int n, bool return_ptrs) {
    command_runner<in<float>, in<float>, out<float>> runner;
    auto results = runner.transfer_memory(device_num, stream_id, x, y, res);
    execute_and_reply(device_num, stream_id, std::get<0>(results), std::get<1>(results), std::get<2>(results), n, return_ptrs);
  }

  void enqueue_sdot(int device_num, int stream_id, mem_ptr<float> x, mem_ptr<float> y, mem_ptr<float> res, int n, bool return_ptrs) {
    command_runner<mem_ptr<float>, mem_ptr<float>, mem_ptr<float>> runner;
    auto results = runner.transfer_memory(device_num, stream_id, x, y, res);
    execute_and_reply(device_num, stream_id, std::get<0>(results), std::get<1>(results), std::get<2>(results), n, return_ptrs);
  }

  void execute_and_reply(int device_num, int stream_id, mem_ptr<float> x, mem_ptr<float> y, mem_ptr<float> res, int n, bool return_ptrs) {
    auto plat = platform::create();
    device_ptr dev = (device_num == -1) ? plat->schedule(stream_id) : plat->schedule(stream_id, device_num);
    dev->sdot(stream_id, n, x, y, res);
    handle_reply(device_num, stream_id, x, y, res, return_ptrs);
  }

  void handle_reply(int device_num, int stream_id, mem_ptr<float> x, mem_ptr<float> y, mem_ptr<float> res, bool return_ptrs) {
    command_runner<mem_ptr<float>> runner;
    auto sender = actor_cast<actor>(this->current_sender());
    if (!sender) return;
    auto r_id = reply_id_;
    if (return_ptrs) {
      caf::anon_mail(r_id, x, y, res).send(sender);
    } else {
      runner.copy_to_host_async(res, [sender, r_id](std::vector<float>&& data) {
        if (sender && !data.empty()) {
          caf::anon_mail(r_id, data[0]).send(sender);
        }
      });
    }
  }

  int actor_id_;
  int reply_id_;
};

} // namespace caf::cuda
