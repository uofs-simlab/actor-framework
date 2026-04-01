#pragma once

#include <caf/actor.hpp>
#include <caf/anon_mail.hpp>
#include "caf/cuda/command.hpp"
#include "caf/cuda/memory_command.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/platform.hpp"
#include "caf/cuda/control-layer/response_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"
#include "caf/cuda/control-layer/memory_response_token.hpp"
#include "caf/cuda/global.hpp"

namespace caf::cuda {

// ===========================================================================
// COMMAND RUNNER
// An Alternative gateway to the gpu, enabling users to create their own custom
// gpu actors if they wish 
// Manages synchronous and asynchronous command execution with overloads
// for actor_id, shared_memory, and device_number.
// ===========================================================================
template <class... Ts>
class command_runner {
public:
  using command_t = command<caf::actor, Ts...>;
  using base_command_t = base_command<caf::actor, Ts...>;

  // -------------------------------
  // Synchronous run: actor_id only
  // returns an output_buffer
  // -------------------------------
  template <class... Us>
  auto run(program_ptr program,
           nd_range dims,
           caf::actor_id actor_id,
           Us&&... xs) 
  {
      if (!platform_) platform_ = program->get_platform();
      auto cmd = caf::make_counted<command_t>(std::move(program),
                                              std::move(dims),
                                              actor_id,
                                              std::forward<Us>(xs)...);
      return cmd->enqueue();
  }

  // -------------------------------
  // Synchronous run: actor_id + shared_memory
  // -------------------------------
  template <class... Us>
  auto run(program_ptr program,
           nd_range dims,
           caf::actor_id actor_id,
           int shared_memory,
           Us&&... xs) 
  {
      if (!platform_) platform_ = program->get_platform();
      auto cmd = caf::make_counted<command_t>(std::move(program),
                                              std::move(dims),
                                              actor_id,
                                              shared_memory,
                                              std::forward<Us>(xs)...);
      return cmd->enqueue();
  }

  // -------------------------------
  // Synchronous run: actor_id + shared_memory + device_number
  // -------------------------------
  template <class... Us>
  auto run(program_ptr program,
           nd_range dims,
           caf::actor_id actor_id,
           int shared_memory,
           int device_number,
           Us&&... xs) 
  {
      if (!platform_) platform_ = program->get_platform();
      auto cmd = caf::make_counted<command_t>(std::move(program),
                                              std::move(dims),
                                              actor_id,
                                              shared_memory,
                                              device_number,
                                              std::forward<Us>(xs)...);
      return cmd->enqueue();
  }


  // -------------------------------

  // Synchronous run using launch_response_token

  // stream comes from token

  // -------------------------------

  template <class... Us>
  auto run(program_ptr program,
         nd_range dims,
         const response_token_ptr& token,
         Us&&... xs)
  {
    if (!platform_) platform_ = program->get_platform();
    return run(std::move(program),
               std::move(dims),
               /* actor_id = */ token ->getStreamId(),
               /* shared_memory = */ 0,
               /* device_number = */ token -> getDeviceNumber(),
               std::forward<Us>(xs)...);
}




  // -------------------------------
  // Asynchronous run: actor_id only
  // returns a tuple of mem_ptrs
  // -------------------------------
  template <class... Us>
  auto run_async(program_ptr program,
                 nd_range dims,
                 caf::actor_id actor_id,
                 Us&&... xs) 
  {
      if (!platform_) platform_ = program->get_platform();
      auto cmd = caf::make_counted<base_command_t>(std::move(program),
                                                   std::move(dims),
                                                   actor_id,
                                                   std::forward<Us>(xs)...);
      return cmd->base_enqueue();
  }

  // -------------------------------
  // Asynchronous run: actor_id + shared_memory
  // -------------------------------
  template <class... Us>
  auto run_async(program_ptr program,
                 nd_range dims,
                 caf::actor_id actor_id,
                 int shared_memory,
                 Us&&... xs) 
  {
      if (!platform_) platform_ = program->get_platform();
      auto cmd = caf::make_counted<base_command_t>(std::move(program),
                                                   std::move(dims),
                                                   actor_id,
                                                   shared_memory,
                                                   std::forward<Us>(xs)...);
      return cmd->base_enqueue();
  }

  // -------------------------------
  // Asynchronous run: actor_id + shared_memory + device_number
  // -------------------------------
  template <class... Us>
  auto run_async(program_ptr program,
                 nd_range dims,
                 caf::actor_id actor_id,
                 int shared_memory,
                 int device_number,
                 Us&&... xs) 
  {
      if (!platform_) platform_ = program->get_platform();
      auto cmd = caf::make_counted<base_command_t>(std::move(program),
                                                   std::move(dims),
                                                   actor_id,
                                                   shared_memory,
                                                   device_number,
                                                   std::forward<Us>(xs)...);
      return cmd->base_enqueue();
  }



   // -------------------------------
   // Asynchronous run using launch_response_token
   // stream comes from token

   // -------------------------------
    template <class... Us>
    auto run_async(program_ptr program,
               nd_range dims,
               const response_token_ptr& token,
               Us&&... xs)
    {
      if (!platform_) platform_ = program->get_platform();
    	return run_async(std::move(program),
                     std::move(dims),
                     /* actor_id = */ token -> getStreamId(),
                     /* shared_memory = */ 0,
                     /* device_number = */ token -> getDeviceNumber(),
                      std::forward<Us>(xs)...);

    }

  // ---------------------------------------------------------------------------
  // Actor-native asynchronous run  (cuLaunchHostFunc variant)
  //
  // Launches the kernel exactly like run_async(), but instead of the caller
  // having to poll with a fixed-delay timer, a CUDA host-function callback is
  // enqueued AFTER the kernel on the same stream.  When the GPU finishes all
  // work up to that point, the CUDA runtime calls the callback from its own
  // internal thread.  The callback sends a `gpu_done_atom` message to
  // `recipient` using caf::anon_mail — no CAF worker thread is ever blocked.
  //
  // Usage:
  //   auto mem_refs = runner.run_async_notify(prog, dim,
  //       caf::actor_cast<caf::actor>(self_), arg1, arg2);
  //   result_ptr_ = std::get<1>(mem_refs);
  //   // ... actor also has a handler: [this](caf::cuda::gpu_done_atom) { ... }
  //
  // The mem_ptrs returned here are identical to those from run_async().
  // copy_to_host() on them is safe inside the gpu_done_atom handler because
  // the stream is guaranteed idle by then.
  // ---------------------------------------------------------------------------
  template <class... Us>
  auto run_async_notify(program_ptr program,
                        nd_range dims,
                        caf::actor recipient,
                        Us&&... xs)
  {
    if (!platform_) platform_ = program->get_platform();
    // Use the full 64-bit actor ID for stream-pool selection.
    caf::actor_id actor_id = recipient.id();

    auto cmd = caf::make_counted<base_command_t>(program,
                                                 std::move(dims),
                                                 actor_id,
                                                 std::forward<Us>(xs)...);
    auto mem_refs = cmd->base_enqueue();

    // Retrieve the same stream that the kernel was launched on.
    CUstream stream = cmd->get_device()->get_stream_for_actor(actor_id);

    // Heap-allocate a tiny context for the callback.  Deleted inside the
    // callback once the message has been dispatched.
    struct cb_data {
      caf::actor recipient;
    };
    auto* d = new cb_data{std::move(recipient)};

    // cuLaunchHostFunc: fires on CUDA's internal thread after all preceding
    // stream work completes.  Rules: no CUDA API calls allowed inside cb.
    // The captureless lambda decays to a plain CUhostFn function pointer.
    CUresult res = cuLaunchHostFunc(
      stream,
      [](void* userdata) {
        auto* d = static_cast<cb_data*>(userdata);
        // Notify the actor — anon_mail is thread-safe from any OS thread.
        caf::anon_mail(caf::cuda::gpu_done_atom_v).send(d->recipient);
        delete d;
      },
      d);

    if (res != CUDA_SUCCESS) {
      // Callback registration failed — clean up and throw so the caller
      // knows.  The kernel is already running; the existing deferred-unload
      // path will still clean up the module safely.
      delete d;
      check(res);
    }

    return mem_refs;
  }




    // -------------------------------------------------------------------------
    // MEMORY TRANSFER
    // Single transfer per command, returns device buffer
    // -------------------------------------------------------------------------
    template <typename T>
    mem_ptr<raw_t<T>> transfer_memory(int device_number,
                                      int stream_id,
                                      T arg)
    {
        if (!platform_)
            throw std::runtime_error("command_runner: platform not set; call run() before transfer_memory()");
        // stack-allocate memory_command and execute transfer
        memory_command<T> cmd(platform_, device_number, stream_id, std::move(arg));
        return cmd.enqueue();
    }

   // -------------------------------------------------------------------------
    // MEMORY TRANSFER
    // Single transfer per command, returns device buffer
    // can transfer memory with a response token
    // -------------------------------------------------------------------------
    template <typename T>
    mem_ptr<raw_t<T>> transfer_memory(const response_token_ptr& token,
                                      T arg)
    {
        // stack-allocate memory_command and execute transfer
        return transfer_memory(token -> getDeviceNumber(),token -> getStreamId(),arg); 
    }



  // -------------------------------
  // Destroy streams for a given actor ID
  // -------------------------------
  void release_stream_for_actor(int actor_id) {
      if (platform_)
          platform_->release_streams_for_actor(actor_id);
  }

private:
  platform_ptr platform_;



};

} // namespace caf::cuda

