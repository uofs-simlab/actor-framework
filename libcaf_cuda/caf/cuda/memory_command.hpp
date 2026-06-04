#pragma once

#include <utility>
#include <type_traits>
#include <tuple>

#include <caf/ref_counted.hpp>
#include <caf/intrusive_ptr.hpp>

#include "caf/cuda/platform.hpp"
#include "caf/cuda/device.hpp"
#include "caf/cuda/mem_ref.hpp"
#include "caf/cuda/types.hpp"

namespace caf::cuda {

// ===========================================================================
// MEMORY COMMAND (single transfer)
// ===========================================================================
template <typename T>
class memory_command : public caf::ref_counted {
public:
  using result_type = mem_ptr<raw_t<T>>;

  // -------------------------------------------------------------------------
  // Constructor
  // -------------------------------------------------------------------------
  memory_command(int device_number,
                 int stream_id,
                 T arg)
      : stream_id_(stream_id),
        arg_(std::move(arg)) {

    dev_ = platform::create()->schedule(stream_id_, device_number);
  }

  // -------------------------------------------------------------------------
  // Execute memory transfer
  // -------------------------------------------------------------------------
  result_type enqueue() {
    CUstream stream = dev_->get_stream_for_actor(stream_id_);
    return dev_->make_arg(arg_, stream);
  }

private:
  int stream_id_;
  device_ptr dev_;
  T arg_;
};

// ===========================================================================
// BULK MEMORY COMMAND
// Handles multiple transfers at once and returns a tuple of mem_ptrs.
// ===========================================================================
template <typename... Ts>
class bulk_memory_command : public caf::ref_counted {
public:
  using result_type = std::tuple<mem_ptr<raw_t<Ts>>...>;

  bulk_memory_command(int device_number,
                      int stream_id,
                      Ts... args)
      : stream_id_(stream_id),
        args_(std::move(args)...) {
    dev_ = platform::create()->schedule(stream_id_, device_number);
  }

  // Asynchronous execution
  result_type enqueue() {
    CUstream stream = dev_->get_stream_for_actor(stream_id_);
    return std::apply([&](auto&&... arg) {
      return std::make_tuple(dev_->make_arg(arg, stream)...);
    }, args_);
  }

private:
  int stream_id_;
  device_ptr dev_;
  std::tuple<Ts...> args_;
};


// ===========================================================================
// COPY BACK COMMAND
// Handles transferring memory from device to host.
// ===========================================================================
template <typename T>
class copy_back_command : public caf::ref_counted {
public:
  copy_back_command(mem_ptr<T> ptr) : ptr_(std::move(ptr)), stream_id_(-1) {
    if (!ptr_)
      throw std::runtime_error("copy_back_command: null mem_ptr");
  }

  copy_back_command(mem_ptr<T> ptr, int stream_id) : ptr_(std::move(ptr)), stream_id_(stream_id) {
    if (!ptr_)
      throw std::runtime_error("copy_back_command: null mem_ptr");
  }

  // Synchronous execution
  std::vector<T> run() {
    if (ptr_->access() == IN)
      throw std::runtime_error("Cannot copy a read-only buffer back to host");
    return ptr_->copy_to_host();
  }

  // Enqueue the transfer on the stream without any host callback or synchronization
  void enqueue(T* dst, size_t count) {
    if (ptr_->access() == IN)
      throw std::runtime_error("Cannot copy a read-only buffer back to host");

    CHECK_CUDA(cuCtxPushCurrent(ptr_->get_ctx()));
    CUstream s = resolve_stream();

    if (ptr_->is_scalar()) {
      // For scalars, the value is already on the host.
      // This assignment happens immediately on the CPU and is NOT stream-ordered.
      // If stream ordering is required for scalars, use run_async with a callback.
      dst[0] = *ptr_->host_scalar_ptr();
    } else {
      size_t bytes = count * sizeof(T);
      // Pure asynchronous copy with no host-side tracking.
      CHECK_CUDA(cuMemcpyDtoHAsync(dst, ptr_->mem(), bytes, s));
    }

    CHECK_CUDA(cuCtxPopCurrent(nullptr));
  }

  // Asynchronous execution with internal buffer allocation
  template <typename F>
  void run_async(F callback) {
    if (ptr_->access() == IN)
      throw std::runtime_error("Cannot copy a read-only buffer back to host");

    struct State {
      std::vector<T> buffer;
      F user_callback;
      bool is_scalar;
      T host_scalar;
    };

    auto* state = new State{std::vector<T>(ptr_->size()), std::move(callback),
                            ptr_->is_scalar(), *ptr_->host_scalar_ptr()};

    CHECK_CUDA(cuCtxPushCurrent(ptr_->get_ctx()));
    CUstream s = resolve_stream();

    if (!ptr_->is_scalar()) {
      size_t bytes = ptr_->size() * sizeof(T);
      CHECK_CUDA(cuMemcpyDtoHAsync(state->buffer.data(), ptr_->mem(), bytes, s));
    }

    auto host_fn = [](void* userData) {
      auto* s_ptr = static_cast<State*>(userData);
      if (s_ptr->is_scalar)
        s_ptr->buffer[0] = s_ptr->host_scalar;

      s_ptr->user_callback(std::move(s_ptr->buffer));
      delete s_ptr;
    };

    CHECK_CUDA(cuLaunchHostFunc(s, host_fn, state));
    CHECK_CUDA(cuCtxPopCurrent(nullptr));
  }

  // Asynchronous execution with user-provided buffer
  template <typename F>
  void run_async(T* dst, size_t count, F callback) {
    if (ptr_->access() == IN)
      throw std::runtime_error("Cannot copy a read-only buffer back to host");

    struct State {
      T* dst;
      size_t count;
      F user_callback;
      bool is_scalar;
      T host_scalar;
    };

    auto* state = new State{dst, count, std::move(callback),
                            ptr_->is_scalar(), *ptr_->host_scalar_ptr()};

    CHECK_CUDA(cuCtxPushCurrent(ptr_->get_ctx()));
    CUstream s = resolve_stream();

    if (!ptr_->is_scalar()) {
      size_t bytes = count * sizeof(T);
      CHECK_CUDA(cuMemcpyDtoHAsync(dst, ptr_->mem(), bytes, s));
    }

    auto host_fn = [](void* userData) {
      auto* s_ptr = static_cast<State*>(userData);
      if (s_ptr->is_scalar)
        s_ptr->dst[0] = s_ptr->host_scalar;

      s_ptr->user_callback(s_ptr->dst, s_ptr->count);
      delete s_ptr;
    };

    CHECK_CUDA(cuLaunchHostFunc(s, host_fn, state));
    CHECK_CUDA(cuCtxPopCurrent(nullptr));
  }

private:
  CUstream resolve_stream() {
    if (stream_id_ == -1)
      return ptr_->stream();
    auto plat = platform::create();
    auto dev = plat->schedule(stream_id_, ptr_->deviceID());
    return dev->get_stream_for_actor(stream_id_);
  }

  mem_ptr<T> ptr_;
  int stream_id_;
};

// ===========================================================================
// FREE MEMORY COMMAND
// Handles freeing memory on a given stream.
// ===========================================================================
template <typename T>
class free_memory_command : public caf::ref_counted {
public:
  free_memory_command(mem_ptr<T> ptr, int stream_id = -1)
      : ptr_(std::move(ptr)), stream_id_(stream_id) {
    if (!ptr_)
      throw std::runtime_error("free_memory_command: null mem_ptr");
  }

  void enqueue() {
    ptr_->free_on(resolve_stream());
  }

private:
  CUstream resolve_stream() {
    if (stream_id_ == -1)
      return ptr_->stream();
    auto plat = platform::create();
    auto dev = plat->schedule(stream_id_, ptr_->deviceID());
    return dev->get_stream_for_actor(stream_id_);
  }

  mem_ptr<T> ptr_;
  int stream_id_;
};

} // namespace caf::cuda
