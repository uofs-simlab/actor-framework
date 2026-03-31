#pragma once

#include <functional>
#include <queue>

#include <caf/actor.hpp>
#include <caf/actor_cast.hpp>
#include <caf/actor_config.hpp>
#include <caf/actor_system.hpp>
#include <caf/anon_mail.hpp>
#include <caf/event_based_actor.hpp>
#include <caf/response_promise.hpp>

#include <cuda.h>

#include "caf/cuda/command.hpp"
#include "caf/cuda/global.hpp"
#include "caf/cuda/helpers.hpp"
#include "caf/cuda/platform.hpp"
#include "caf/cuda/program.hpp"


namespace caf::cuda {

// ---------------------------------------------------------------------------
// actor_facade
//
// A CAF actor that wraps a single GPU kernel.  Callers send typed messages;
// the facade launches the kernel via the run_async_notify() pattern so that
// the CAF worker thread is NEVER blocked:
//
//   1. On message receipt  → launch kernel (base_enqueue), register a
//      cuLaunchHostFunc callback on the same CUDA stream, store
//      (collector_fn, response_promise) in a FIFO queue, return immediately.
//   2. On gpu_done_atom    → pop the head of the queue, copy results to host,
//      deliver the response_promise.
//
// Because all kernels for a given actor share one CUDA stream (keyed by
// actor_id_), the GPU executes them in order and the gpu_done_atom messages
// therefore arrive in the same FIFO order.
//
// NOTE (L-3 fix): actor_config&& has been removed from create().  The
// caf::actor_system::spawn<T>() overload used internally always constructs
// a fresh config with default options; there is no public CAF API to forward
// an externally-built actor_config into spawn<>.  Callers that need
// non-default spawn options should use the actor_system API directly.
// ---------------------------------------------------------------------------
template <bool PassConfig, class... Ts>
class actor_facade : public event_based_actor {
public:
  static caf::actor create(caf::actor_system& sys,
                           program_ptr program,
                           nd_range dims,
                           Ts&&...) {
    return caf::actor_cast<caf::actor>(
      sys.spawn<actor_facade>(std::move(program), std::move(dims)));
  }

  static caf::actor create(caf::actor_system* sys,
                           program_ptr program,
                           nd_range dims,
                           Ts&&...) {
    return caf::actor_cast<caf::actor>(
      sys->spawn<actor_facade>(std::move(program), std::move(dims)));
  }

  actor_facade(caf::actor_config& cfg, program_ptr program, nd_range dims)
    : caf::event_based_actor(cfg),
      program_(std::move(program)),
      dims_(std::move(dims)) {
    actor_id_ = this->id();
    platform_ = program_->get_platform();
  }

  ~actor_facade() override {
    if (platform_)
      platform_->release_streams_for_actor(this->id());
  }

  caf::behavior make_behavior() override {
    return {
      // ── GPU completion ────────────────────────────────────────────────────
      // Fires from the CUDA host-function callback (via anon_mail) once the
      // kernel stream is idle.  Safe to call copy_to_host() here because
      // the stream is guaranteed idle at this point.
      [this](gpu_done_atom) {
        if (pending_.empty())
          return;
        pending_job job = std::move(pending_.front());
        pending_.pop();
        try {
          job.rp.deliver(job.collector());
        } catch (const std::exception& ex) {
          job.rp.deliver(
            caf::make_error(caf::sec::runtime_error, ex.what()));
        }
      },

      // ── Kernel launch request ─────────────────────────────────────────────
      // Matches the typed argument list (wrapped or raw), launches the kernel
      // asynchronously, and returns a response_promise so the handler returns
      // to CAF immediately without blocking the worker thread.
      [this](const caf::message& msg) -> caf::result<result_vec> {
        auto rp = this->make_response_promise<result_vec>();
        try {
          if (!enqueue_async(msg, rp))
            rp.deliver(caf::make_error(caf::sec::unexpected_message,
                                       "actor_facade received unsupported arguments"));
        } catch (const std::exception& ex) {
          rp.deliver(caf::make_error(caf::sec::runtime_error, ex.what()));
        }
        return rp;
      },
    };
  }

private:
  // ── Pending job ───────────────────────────────────────────────────────────
  // Holds the type-erased result-collection function and the promise that
  // will carry the output back to the original sender.
  using result_vec = std::vector<output_buffer>;
  using promise_t  = caf::typed_response_promise<result_vec>;

  struct pending_job {
    std::function<result_vec()> collector;
    promise_t                   rp;
  };

  // ── Core async launch ─────────────────────────────────────────────────────
  // Instantiated for each concrete set of argument types Us... (derived from
  // the incoming message).  Launches the kernel, registers the completion
  // callback, and pushes the job onto the FIFO queue.
  template <class... Us>
  void enqueue_impl(promise_t& rp, Us&&... xs) {
    using cmd_t = base_command<caf::actor, std::decay_t<Us>...>;
    auto cmd = caf::make_counted<cmd_t>(program_, dims_, actor_id_,
                                        std::forward<Us>(xs)...);

    // Launch kernel asynchronously — returns immediately without waiting for
    // stream synchronisation.
    auto mem_refs = cmd->base_enqueue();

    // Retrieve the stream the kernel was enqueued on so the host-function
    // callback is placed on the same stream (FIFO ordering guaranteed).
    device_ptr dev    = cmd->get_device();
    CUstream   stream = dev->get_stream_for_actor(actor_id_);

    // Type-erase the result collection so we can store it in the queue
    // independent of the concrete mem_ref tuple type.
    auto collector = [dev, mem_refs]() mutable -> std::vector<output_buffer> {
      return dev->collect_output_buffers(mem_refs);
    };

    // Register a CUDA host-function callback.  cuLaunchHostFunc guarantees
    // that the callback fires only after all preceding work on the stream
    // (i.e. the kernel) has completed.  The callback itself runs on CUDA's
    // internal thread — no CUDA API calls are permitted inside it.
    struct cb_data { caf::actor self; };
    auto* d = new cb_data{caf::actor_cast<caf::actor>(this)};

    CUresult res = cuLaunchHostFunc(
      stream,
      [](void* userdata) {
        auto* d = static_cast<cb_data*>(userdata);
        caf::anon_mail(gpu_done_atom_v).send(d->self);
        delete d;
      },
      d);

    if (res != CUDA_SUCCESS) {
      delete d;
      check(res); // throws std::runtime_error
    }

    pending_.push({std::move(collector), std::move(rp)});
  }

  // ── Message-dispatch helpers ──────────────────────────────────────────────

  template <std::size_t... Is>
  void enqueue_wrapped(promise_t& rp, const caf::message& msg,
                       size_t offset, std::index_sequence<Is...>) {
    enqueue_impl(rp, msg.get_as<Ts>(Is + offset)...);
  }

  template <std::size_t... Is>
  void enqueue_raw(promise_t& rp, const caf::message& msg,
                   size_t offset, std::index_sequence<Is...>) {
    enqueue_impl(rp, Ts{msg.get_as<raw_t<Ts>>(Is + offset)}...);
  }

  // Returns false if the message did not match any expected type pattern.
  bool enqueue_async(const caf::message& msg, promise_t& rp) {
    auto indices = std::index_sequence_for<Ts...>{};
    if (msg.match_elements<Ts...>()) {
      enqueue_wrapped(rp, msg, 0, indices);
      return true;
    }
    if (msg.match_elements<raw_t<Ts>...>()) {
      enqueue_raw(rp, msg, 0, indices);
      return true;
    }
    if (msg.match_elements<caf::actor, Ts...>()) {
      enqueue_wrapped(rp, msg, 1, indices);
      return true;
    }
    if (msg.match_elements<caf::actor, raw_t<Ts>...>()) {
      enqueue_raw(rp, msg, 1, indices);
      return true;
    }
    return false;
  }

  program_ptr  program_;
  platform_ptr platform_;
  nd_range     dims_;
  caf::actor_id actor_id_;

  std::queue<pending_job> pending_;
};

} // namespace caf::cuda
