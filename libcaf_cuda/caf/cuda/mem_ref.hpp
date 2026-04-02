// mem_ref.hpp
#pragma once
#include <cuda.h>
#include <caf/intrusive_ptr.hpp>
#include <caf/ref_counted.hpp>
#include <vector>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <iostream>
#include "caf/cuda/types.hpp"
//#include "caf/cuda/utility.hpp"
#include <cuda.h>
#include <atomic>
#include "caf/cuda/control-layer/memory_actor/mem_token.hpp"

namespace caf::cuda {

//A class that is a handle to gpu memory
template <class T>
class mem_ref : public caf::ref_counted {
public:
  using value_type = T;

  //constructor with CUdeviceptr
  // pool_alloc: set to true when memory was allocated with cuMemAllocAsync.
  // When true, reset() uses cuMemFreeAsync instead of cuMemFree so that
  // de-allocation is non-blocking and memory is returned to the driver pool.
  mem_ref(size_t num_elements,
          CUdeviceptr memory,
          int access,
          int device_id    = 0,
          int context_id   = 0,
	  CUcontext context = nullptr,
          CUstream stream  = nullptr,
          bool pool_alloc  = false)
    : num_elements_(num_elements),
      memory_(memory),
      access_(access),
      device_id(device_id),
      context_id(context_id),
      stream_(stream),
      ctx(context),
      is_scalar_(false),
      pool_alloc_(pool_alloc)
  {
    if (memory_ == 0)
      throw std::runtime_error("mem_ref: null GPU memory pointer");
  }

  // Scalar constructor (new!)
  mem_ref(const T& scalar_value,
          int access,
          int device_id    = 0,
          int context_id   = 0,
	  CUcontext context = nullptr,
          CUstream stream  = nullptr)
    : num_elements_(1),
      memory_(0),                    // no device buffer
      access_(access),
      device_id(device_id),
      context_id(context_id),
      stream_(stream),
      ctx(context),
      is_scalar_(true),
      host_scalar_(scalar_value)
  {
    // no cuMemAlloc, nothing to do
  }

  ~mem_ref() {
    reset();
  }

  mem_ref(mem_ref&&) noexcept = default;
  mem_ref& operator=(mem_ref&&) noexcept = default;
  mem_ref(const mem_ref&) = delete;
  mem_ref& operator=(const mem_ref&) = delete;

  // a bunch of getters
  bool is_scalar() const noexcept {return is_scalar_;}
  const T* host_scalar_ptr() const noexcept {return &host_scalar_;}
  size_t size()  const noexcept { return num_elements_; }
  CUdeviceptr mem()   const noexcept { return memory_; }
  int access()  const noexcept { return access_; }
  CUstream stream() const noexcept { return stream_; }
  int deviceID() const noexcept { return device_id;}
  int deviceNumber() const noexcept { return device_id;}

 //if it is ever needed, you can force synchronization on a mem_ptr
 //to ensure data on the device that the mem_ptr points to
 //is not in the middle of being operated on
 //mostly here to avoid race conditions that can occur between streams 
  void synchronize()  {
    CHECK_CUDA(cuCtxPushCurrent(ctx));
    CUstream s = stream_ ? stream_ : nullptr;
    if (s) CHECK_CUDA(cuStreamSynchronize(s));
    else  CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuCtxPopCurrent(nullptr)); 
  } 

  //Frees the memory on the gpu and 
  //sets all its attributes to null or -1
  void reset() {
    if (!is_scalar_ && memory_) {
      // Clear the pointer first to prevent a second attempt on double-free or
      // re-entry.  failure is not rethrown here because reset() is
      // called from the destructor, and throwing from a destructor causes
      // std::terminate().  Log the failure instead.
      CUdeviceptr mem_to_free = memory_;
      memory_ = 0;
      CUresult err;
      // Botttleneck 3 fix: if memory was allocated with cuMemAllocAsync, free
      // it with cuMemFreeAsync so the release is non-blocking and the memory
      // is immediately returned to the driver pool for reuse.
      if (pool_alloc_ && stream_) {
        err = cuMemFreeAsync(mem_to_free, stream_);
      } else {
        err = cuMemFree(mem_to_free);
      }
      if (err != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(err, &err_str);
        std::cerr << "mem_ref: cuMemFree(Async) failed during cleanup: "
                  << (err_str ? err_str : "unknown error") << "\n";
      }
    }
    num_elements_ = 0;
    access_       = -1;
    stream_       = nullptr;
    ctx = nullptr;
  }

  //copies gpu memory back to cpu memory in the form of an std::vector
  std::vector<T> copy_to_host() const {
	  if (access_ == IN)
	  {
		  throw std::runtime_error("Cannt copy a read only buffer back to device\n");
	  } 
	  if (is_scalar_) {
		  return std::vector<T>{host_scalar_};
	  }
	  std::vector<T> host_data(num_elements_);
	  size_t bytes = num_elements_ * sizeof(T);
	  CHECK_CUDA(cuCtxPushCurrent(ctx));
	  CUstream s = stream_ ? stream_ : nullptr;
	  CHECK_CUDA(cuMemcpyDtoHAsync(host_data.data(), memory_, bytes, s));
	  if (s) CHECK_CUDA(cuStreamSynchronize(s));
	  else  CHECK_CUDA(cuCtxSynchronize());
	  CHECK_CUDA(cuCtxPopCurrent(nullptr));
	  return host_data;
  }

  // Bottleneck 2 fix: unchecked variant — no cuCtxPushCurrent/Pop.
  // cuMemcpyDtoHAsync and cuStreamSynchronize are stream-based operations
  // that do not require the context to be explicitly on the calling thread's
  // context stack.  The caller is responsible for ensuring the correct
  // device is targeted via the stream that was used to allocate this buffer.
  std::vector<T> copy_to_host_unchecked() const {
    if (access_ == IN)
      throw std::runtime_error("Cannot copy a read-only buffer back to host");
    if (is_scalar_)
      return std::vector<T>{host_scalar_};
    std::vector<T> host_data(num_elements_);
    size_t bytes = num_elements_ * sizeof(T);
    CUstream s = stream_ ? stream_ : nullptr;
    CHECK_CUDA(cuMemcpyDtoHAsync(host_data.data(), memory_, bytes, s));
    if (s) CHECK_CUDA(cuStreamSynchronize(s));
    else   CHECK_CUDA(cuCtxSynchronize());
    return host_data;
  }

  //copies buffer back to dst buffer supplied by the user
  //count is number of elements the buffer has
  void copy_to_host(T* dst, size_t count) const {
	  if (access_ == IN)
		  throw std::runtime_error("Cannot copy a read only buffer back to host");

	  if (is_scalar_) {
		  dst[0] = host_scalar_;
		  return;
	  }

	  size_t bytes = count * sizeof(T);

	  CHECK_CUDA(cuCtxPushCurrent(ctx));
	  CUstream s = stream_ ? stream_ : nullptr;

	  CHECK_CUDA(cuMemcpyDtoHAsync(dst, memory_, bytes, s));

	  if (s)
		  CHECK_CUDA(cuStreamSynchronize(s));
	  else
		  CHECK_CUDA(cuCtxSynchronize());

	  CHECK_CUDA(cuCtxPopCurrent(nullptr));
  }

  // Bottleneck 2 fix: unchecked variant for pre-allocated destination buffer.
  // Same semantics as copy_to_host(T*, size_t) but without ctx push/pop.
  void copy_to_host_unchecked(T* dst, size_t count) const {
    if (access_ == IN)
      throw std::runtime_error("Cannot copy a read-only buffer back to host");
    if (is_scalar_) {
      dst[0] = host_scalar_;
      return;
    }
    size_t bytes = count * sizeof(T);
    CUstream s = stream_ ? stream_ : nullptr;
    CHECK_CUDA(cuMemcpyDtoHAsync(dst, memory_, bytes, s));
    if (s) CHECK_CUDA(cuStreamSynchronize(s));
    else   CHECK_CUDA(cuCtxSynchronize());
  }



    //reference counting for auto garabage collection
    friend void intrusive_ptr_add_ref(const mem_ref<T>* p) noexcept {
        p->ref_count_.fetch_add(1, std::memory_order_relaxed);
    }

    friend void intrusive_ptr_release(const mem_ref<T>* p) noexcept {
        if (p->ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1)
            delete p;
    }


    // manual binding of the resource object that denotes memory
    // to the pointer that contains it 
    void bind_token(caf::cuda::mem_token m_token) {
        token = std::move(m_token);
    }


private:
  size_t      num_elements_{0};
  CUdeviceptr memory_{0};
  int         access_{-1};
  int         device_id{0};
  int         context_id{0};
  CUstream    stream_{nullptr};
  CUcontext ctx;
  mutable std::atomic<size_t> ref_count_{0};

  mem_token token;

  bool is_scalar_{false};
  T    host_scalar_{};
  // True when memory was allocated with cuMemAllocAsync (pool-backed).
  // reset() uses cuMemFreeAsync to return memory to the pool non-blocking.
  bool pool_alloc_{false};
};

template <class T>
using mem_ptr = caf::intrusive_ptr<mem_ref<T>>;

} // namespace caf::cuda

