#pragma once

#include <utility>

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
// COPY BACK COMMAND
// Handles transferring memory from device to host.
// ===========================================================================
template <typename T>
class copy_back_command : public caf::ref_counted {
public:
  copy_back_command(mem_ptr<T> ptr) : ptr_(std::move(ptr)) {
    if (!ptr_)
      throw std::runtime_error("copy_back_command: null mem_ptr");
  }

  // Synchronous execution
  std::vector<T> run() {
    if (ptr_->access() == IN)
      throw std::runtime_error("Cannot copy a read-only buffer back to host");
    return ptr_->copy_to_host();
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
    CUstream s = ptr_->stream();

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
    CUstream s = ptr_->stream();

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
  mem_ptr<T> ptr_;
};

} // namespace caf::cuda
