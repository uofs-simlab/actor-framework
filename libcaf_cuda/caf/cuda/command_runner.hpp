#pragma once

#include "caf/cuda/command.hpp"
#include "caf/cuda/memory_command.hpp"
#include <thread>
#include "caf/cuda/program.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/platform.hpp"
#include "caf/cuda/control-layer/response_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"
#include "caf/cuda/control-layer/memory_response_token.hpp"

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
           int actor_id,
           Us&&... xs) 
  {
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
           int actor_id,
           int shared_memory,
           Us&&... xs) 
  {
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
           int actor_id,
           int shared_memory,
           int device_number,
           Us&&... xs) 
  {
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
                 int actor_id,
                 Us&&... xs) 
  {
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
                 int actor_id,
                 int shared_memory,
                 Us&&... xs) 
  {
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
                 int actor_id,
                 int shared_memory,
                 int device_number,
                 Us&&... xs) 
  {
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
    	return run_async(std::move(program),
                     std::move(dims),
                     /* actor_id = */ token -> getStreamId(),
                     /* shared_memory = */ 0,
                     /* device_number = */ token -> getDeviceNumber(),
                      std::forward<Us>(xs)...);

    }




    // -------------------------------------------------------------------------
    // MEMORY TRANSFER
    // Single transfer per command, returns device buffer
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    // MEMORY TRANSFER
    // Single transfer per command, returns device buffer
    // -------------------------------------------------------------------------
    template <typename T>
    mem_ptr<raw_t<T>> transfer_memory(int device_number,
                                      int stream_id,
                                      T arg)
    {
        // default: asynchronous transfer
        memory_command<T> cmd(device_number, stream_id, std::move(arg));
        return cmd.enqueue();
    }

    // Synchronous transfer for testing
    template <typename T>
    mem_ptr<raw_t<T>> transfer_memory_sync(int device_number,
                                           int stream_id,
                                           T arg)
    {
        memory_command<T> cmd(device_number, stream_id, std::move(arg));
        return cmd.run_sync();
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
        return transfer_memory(token->getDeviceNumber(), token->getStreamId(), std::move(arg));
    }

    template <typename T>
    mem_ptr<raw_t<T>> transfer_memory_sync(const response_token_ptr& token,
                                           T arg)
    {
        return transfer_memory_sync(token->getDeviceNumber(), token->getStreamId(), std::move(arg));
    }


    // -------------------------------------------------------------------------
    // COPY BACK
    // -------------------------------------------------------------------------

    // Synchronous copy back
    template <typename T>
    std::vector<T> copy_to_host(mem_ptr<T> ptr) {
        copy_back_command<T> cmd(std::move(ptr));
        return cmd.run();
    }

    // Asynchronous copy back (default)
    template <typename T, typename F>
    void copy_to_host_async(mem_ptr<T> ptr, F callback) {
        auto cmd = caf::make_counted<copy_back_command<T>>(std::move(ptr));
        cmd->run_async(std::move(callback));
    }

    // Asynchronous copy back to user-provided buffer
    template <typename T, typename F>
    void copy_to_host_async(mem_ptr<T> ptr, T* dst, size_t count, F callback) {
        auto cmd = caf::make_counted<copy_back_command<T>>(std::move(ptr));
        cmd->run_async(dst, count, std::move(callback));
    }


  // -------------------------------
  // Destroy streams for a given actor ID
  // -------------------------------
  void release_stream_for_actor(int actor_id) {
      auto plat = platform::create();
      plat->release_streams_for_actor(actor_id);
  }






};

} // namespace caf::cuda
