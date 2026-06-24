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

/// SCOPY Actor for y = x.
class copy_actor : public event_based_actor {
public:
  static caf::actor spawn(caf::actor_system& sys, int reply_id = 0) {
    return sys.spawn<copy_actor>(reply_id);
  }

  copy_actor(caf::actor_config& cfg, int reply_id = 0) 
    : event_based_actor(cfg), reply_id_(reply_id) {
    actor_id_ = static_cast<int>(this->id());
  }

  ~copy_actor() override {
    command_runner<> runner;
    runner.release_stream_for_actor(actor_id_);
  }

  caf::behavior make_behavior() override {
    return {
      [this](in<float> x, out<float> y, int n) {
        enqueue_scopy(-1, actor_id_, x, y, n, false);
      },
      [this](int device_num, int stream_id, in<float> x, out<float> y, int n) {
        enqueue_scopy(device_num, stream_id, x, y, n, false);
      },
      [this](mem_ptr<float> x, mem_ptr<float> y, int n) {
        enqueue_scopy(-1, actor_id_, x, y, n, false);
      },
      [this](int device_num, int stream_id, mem_ptr<float> x, mem_ptr<float> y, int n) {
        enqueue_scopy(device_num, stream_id, x, y, n, false);
      },
      [this](return_mem_ptr_atom, in<float> x, out<float> y, int n) {
        enqueue_scopy(-1, actor_id_, x, y, n, true);
      },
      [this](return_mem_ptr_atom, int device_num, int stream_id, in<float> x, out<float> y, int n) {
        enqueue_scopy(device_num, stream_id, x, y, n, true);
      },
      [this](return_mem_ptr_atom, mem_ptr<float> x, mem_ptr<float> y, int n) {
        enqueue_scopy(-1, actor_id_, x, y, n, true);
      },
      [this](return_mem_ptr_atom, int device_num, int stream_id, mem_ptr<float> x, mem_ptr<float> y, int n) {
        enqueue_scopy(device_num, stream_id, x, y, n, true);
      }
    };
  }

private:
  void enqueue_scopy(int device_num, int stream_id, in<float> x, out<float> y, int n, bool return_ptrs) {
    command_runner<in<float>, out<float>> runner;
    auto results = runner.transfer_memory(device_num, stream_id, x, y);
    execute_and_reply(device_num, stream_id, std::get<0>(results), std::get<1>(results), n, return_ptrs);
  }

  void enqueue_scopy(int device_num, int stream_id, mem_ptr<float> x, mem_ptr<float> y, int n, bool return_ptrs) {
    command_runner<mem_ptr<float>, mem_ptr<float>> runner;
    auto results = runner.transfer_memory(device_num, stream_id, x, y);
    execute_and_reply(device_num, stream_id, std::get<0>(results), std::get<1>(results), n, return_ptrs);
  }

  void execute_and_reply(int device_num, int stream_id, mem_ptr<float> x, mem_ptr<float> y, int n, bool return_ptrs) {
    auto plat = platform::create();
    device_ptr dev = (device_num == -1) ? plat->schedule(stream_id) : plat->schedule(stream_id, device_num);
    dev->scopy(stream_id, n, x, y);
    handle_reply(device_num, stream_id, x, y, return_ptrs);
  }

  void handle_reply(int device_num, int stream_id, mem_ptr<float> x_ptr, mem_ptr<float> y_ptr, bool return_ptrs) {
    command_runner<mem_ptr<float>> runner;
    auto sender = actor_cast<actor>(this->current_sender());
    if (!sender) return;
    auto r_id = reply_id_;
    if (return_ptrs) {
      caf::anon_mail(r_id, x_ptr, y_ptr).send(sender);
    } else {
      runner.copy_to_host_async(y_ptr, [sender, r_id](std::vector<float>&& data) {
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
