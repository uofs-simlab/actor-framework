#pragma once

#include "caf/cuda/command.hpp"
#include "caf/cuda/memory_command.hpp"
#include <cuda.h>
#include <thread>
#include "caf/cuda/program.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/platform.hpp"
#include "caf/cuda/control-layer/response_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"
#include "caf/cuda/control-layer/memory_response_token.hpp"
#include "caf/cuda/event.hpp"

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

    // Bulk asynchronous transfer: returns std::tuple<mem_ptr<raw_t<Us>>...>
    template <class... Us>
    auto transfer_memory(int device_number,
                         int stream_id,
                         Us&&... args)
    {
        auto cmd = caf::make_counted<bulk_memory_command<std::decay_t<Us>...>>(
          device_number, stream_id, std::forward<Us>(args)...);
        return cmd->enqueue();
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

    // Bulk asynchronous transfer using a response token
    template <class... Us>
    auto transfer_memory(const response_token_ptr& token,
                         Us&&... args)
    {
        return transfer_memory(token->getDeviceNumber(), 
                               token->getStreamId(), 
                               std::forward<Us>(args)...);
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

    // Enqueue copy back without any host-side synchronization or callback
    template <typename T>
    void copy_to_host_async(mem_ptr<T> ptr, T* dst, size_t count) {
        copy_back_command<T> cmd(std::move(ptr));
        cmd.enqueue(dst, count);
    }

    // Asynchronous copy back (default)
    template <typename T, typename F>
    void copy_to_host_async(mem_ptr<T> ptr, F callback) {
        auto cmd = caf::make_counted<copy_back_command<T>>(std::move(ptr));
        cmd->run_async(std::move(callback));
    }

    // Asynchronous copy back with explicit stream/actor ID
    template <typename T, typename F>
    void copy_to_host_async(mem_ptr<T> ptr, int stream_id, F callback) {
        auto cmd = caf::make_counted<copy_back_command<T>>(std::move(ptr), stream_id);
        cmd->run_async(std::move(callback));
    }

    // Asynchronous copy back to user-provided buffer
    template <typename T, typename F>
    void copy_to_host_async(mem_ptr<T> ptr, T* dst, size_t count, F callback) {
        auto cmd = caf::make_counted<copy_back_command<T>>(std::move(ptr));
        cmd->run_async(dst, count, std::move(callback));
    }

    // Enqueue an asynchronous free operation on the given stream
    template <typename T>
    void free_memory(mem_ptr<T> ptr, int stream_id = -1) {
        free_memory_command<T> cmd(std::move(ptr), stream_id);
        cmd.enqueue();
    }


  // -------------------------------
  // Destroy streams for a given actor ID
  // -------------------------------
  void release_stream_for_actor(int actor_id) {
      auto plat = platform::create();
      plat->release_streams_for_actor(actor_id);
  }

  // -------------------------------
  // Register a callback on the actor's stream
  // -------------------------------
  template <typename F>
  void add_callback(int stream_id, int device_number, F callback) {
      auto plat = platform::create();
      auto dev = plat->schedule(stream_id, device_number);
      auto stream = dev->get_stream_for_actor(stream_id);
      auto* f_ptr = new F(std::move(callback));
      auto res = cuLaunchHostFunc(stream, [](void* data) {
          auto* f = static_cast<F*>(data);
          (*f)();
          delete f;
      }, f_ptr);
      if (res != CUDA_SUCCESS) { delete f_ptr; check(res, "cuLaunchHostFunc"); }
  }

  // -------------------------------
  // Resets the CUDA context for a given device number.
  // This will force the device to flush its stream pool and create a new context.
  // -------------------------------
  void reset_context(int device_number) {
      auto plat = platform::create();
      plat->reset_device_context(device_number);
  }

  // -------------------------------
  // CUDA Event Management
  // -------------------------------

  /// Creates a new CUDA event on the specified device.
  event_ptr create_event(int device_number, unsigned int flags = CU_EVENT_DEFAULT) {
    auto plat = platform::create();
    auto dev = plat->getDevice(device_number % plat->get_num_devices());
    return dev->create_event(flags);
  }

  /// Records a CUDA event on the stream associated with the given stream_id.
  void record_event(event_ptr e, int stream_id, int device_number) {
    auto plat = platform::create();
    auto dev = plat->schedule(stream_id, device_number);
    dev->record_event(std::move(e), stream_id);
  }

  /// Enqueues a wait on the stream for the specified CUDA event.
  void wait_event(event_ptr e, int stream_id, int device_number) {
    auto plat = platform::create();
    auto dev = plat->schedule(stream_id, device_number);
    dev->wait_event(std::move(e), stream_id);
  }

  /// Returns true if the specified CUDA event has completed.
  bool query_event(event_ptr e, int device_number) {
    auto plat = platform::create();
    auto dev = plat->getDevice(device_number % plat->get_num_devices());
    return dev->query_event(std::move(e));
  }

  /// Blocks until the specified CUDA event has completed.
  void synchronize_event(event_ptr e, int device_number) {
    auto plat = platform::create();
    auto dev = plat->getDevice(device_number % plat->get_num_devices());
    dev->synchronize_event(std::move(e));
  }

  // -------------------------------------------------------------------------
  // CUDA Context and Stream Retrieval
  // -------------------------------------------------------------------------

  /// Returns the CUDA context associated with the given device number.
  CUcontext get_context(int device_number) {
      auto plat = platform::create();
      auto dev = plat->getDevice(device_number);
      return dev->getContext();
  }

  /// Returns the CUDA stream associated with the given stream number (actor ID) and device number.
  CUstream get_stream(int stream_number, int device_number) {
      auto plat = platform::create();
      auto dev = plat->schedule(stream_number, device_number);
      return dev->get_stream_for_actor(stream_number);
  }

};

} // namespace caf::cuda
