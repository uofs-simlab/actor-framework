#pragma once

#include "caf/cuda/command.hpp"
#include "caf/cuda/memory_command.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/platform.hpp"
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
         kernel_launch_token token,
         Us&&... xs)
  {
    return run(std::move(program),
               std::move(dims),
               /* actor_id = */ token -> getStreamId(),
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
               kernel_launch_token token,
               Us&&... xs)
    {
    	return run_async(std::move(program),
                     std::move(dims),
                     /* actor_id = */ token ->getStreamId(),
                     /* shared_memory = */ 0,
                     /* device_number = */ token -> getDeviceNumber(),
                      std::forward<Us>(xs)...);

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
        // stack-allocate memory_command and execute transfer
        memory_command<T> cmd(device_number, stream_id, std::move(arg));
        return cmd.enqueue();
    }

   // -------------------------------------------------------------------------
    // MEMORY TRANSFER
    // Single transfer per command, returns device buffer
    // can transfer memory with a response token
    // -------------------------------------------------------------------------
    template <typename T>
    mem_ptr<raw_t<T>> transfer_memory(memory_token token,
                                      T arg)
    {
        // stack-allocate memory_command and execute transfer
        return transfer_memory(token -> getDeviceNumber(),token -> getStreamId(),arg); 
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

