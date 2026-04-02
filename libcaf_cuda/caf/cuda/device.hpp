#pragma once

#include <string>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <shared_mutex>
#include <tuple>
#include <mutex>

#include <caf/adopt_ref.hpp>
#include <caf/intrusive_ptr.hpp>
#include <caf/ref_counted.hpp>

#include <cuda.h>

#include "caf/cuda/global.hpp"
#include "caf/cuda/types.hpp"
#include "caf/cuda/streampool.hpp"
#include "caf/cuda/mem_ref.hpp"

namespace caf::cuda {

class CAF_CUDA_EXPORT device : public caf::ref_counted {
public:
  using device_ptr = caf::intrusive_ptr<device>;

  device(CUdevice device, CUcontext context, const char* name, int id, size_t stream_pool_size = 32)
    : device_(device),
      context_(context),
      id_(id),
      name_(name),
      stream_table_(context, stream_pool_size) {
 	      init_device_properties();
      }

  ~device() {
    check(cuCtxDestroy(context_));
  }

  device(const device&) = delete;
  device& operator=(const device&) = delete;
  device(device&&) noexcept = default;
  device& operator=(device&&) noexcept = default;

  const char* name() const { return name_; }
  CUdevice getDevice() const { return device_; }
  CUcontext getContext() const { return context_; }
  int getId() const { return id_; }
  int getContextId() {return 0;}
  int getStreamId() {return 0;}

  CUcontext getContext(int) { return context_; }


  // Number of streaming multiprocessors (SMs)
  int num_sms() const noexcept { return sm_count_; }

  // Warp size (usually 32)
  int warp_size() const noexcept { return warp_size_; }

  // Maximum threads per SM
  int max_threads_per_sm() const noexcept { return max_threads_per_sm_; }

  // Derived: warps per SM
  int warps_per_sm() const noexcept { return warps_per_sm_; }

  // Derived: total warps on the device
  int total_warps() const noexcept { return total_warps_; }

  // Total device memory in bytes
  std::size_t total_memory_bytes() const noexcept { return total_mem_bytes_; }

  // Convenience: total memory in megabytes
  double total_memory_mb() const noexcept { return static_cast<double>(total_mem_bytes_) / (1024.0 * 1024.0); }

  //total free memory on the device
  std::size_t available_memory_bytes() const;
  // Convenience: returns available memory in megabytes
  double available_memory_mb() const;


  // Short human-readable device summary
  std::string device_summary() const {
	  return std::string(name_) + " (id=" + std::to_string(id_) + ") - SMs: " + std::to_string(sm_count_) +
		  ", warp_size: " + std::to_string(warp_size_) +
		  ", max_threads/SM: " + std::to_string(max_threads_per_sm_) +
		  ", total_mem(MB): " + std::to_string(total_memory_mb());
  }


  //given a program/kernel and dimesions 
  //returns the max blocks that can be on an SM 
  int max_active_blocks_per_sm(const program_ptr& prog, const nd_range& range,
		  size_t dynamic_smem_bytes = 0) const; 




  //returns the CUStream associated with the actor id 
  CUstream get_stream_for_actor(caf::actor_id actor_id) {
    return stream_table_.get_stream(actor_id);
  }

  //releases the CUStream associated with the actor id 
  void release_stream_for_actor(caf::actor_id actor_id) {
    stream_table_.release_stream(actor_id);
  }


  // Overloads for make_arg using actor_id
  template <typename T>
  mem_ptr<T> make_arg(const in<T>& arg, caf::actor_id actor_id) {
    return global_argument(arg, actor_id, IN);
  }

  template <typename T>
  mem_ptr<T> make_arg(const in_out<T>& arg, caf::actor_id actor_id) {
    return global_argument(arg, actor_id, IN_OUT);
  }

  template <typename T>
  mem_ptr<T> make_arg(const out<T>& arg, caf::actor_id actor_id) {
    return scratch_argument(arg, actor_id, OUT);
  }


  // Overloads for make_arg using CUstream directly

  template <typename T>
  mem_ptr<T> make_arg(const in<T>& arg, CUstream stream) {
     return global_argument(arg, stream, IN);
   }


  template <typename T>
  mem_ptr<T> make_arg(const in_out<T>& arg, CUstream stream) {
    return global_argument(arg, stream, IN_OUT);
   }

  template <typename T>
  mem_ptr<T> make_arg(const out<T>& arg, CUstream stream) {
  return scratch_argument(arg, stream, OUT);
  }



  //handling the case that a mem_ref is passed in
  //should I force synchronization onto the same stream always?
  template <typename T>
  mem_ptr<T> make_arg(mem_ptr<T> arg, CUstream stream) {
  
	  if (arg -> deviceID() != id_) {
	  
	throw std::runtime_error("Error memory on device " + std::to_string(arg->deviceID()) +
                         " attempted to be used on a different device, device id was " + std::to_string(id_) + "\n");

	  }
	  //just return the arg back
	  return arg; 
  }




  //given a tuple of mem_ptrs
  //will copy their data back to host and place them in an output buffer
  template <typename... Ts>
  std::vector<output_buffer>  collect_output_buffers_helper(const std::tuple<Ts...>& args) {
    std::vector<output_buffer> result;
    std::apply([&](auto&&... mem) {
      (([&] {
        if (mem && (mem->access() == OUT || mem->access() == IN_OUT)) {
          //using T = typename std::decay_t<decltype(*mem)>::value_type;
          result.emplace_back(output_buffer{buffer_variant{mem->copy_to_host()}});
        }
      })(), ...);
    }, args);
    return result;
  }

  // Bottleneck 2 fix: unchecked variant — uses copy_to_host_unchecked() so
  // no cuCtxPushCurrent/Pop per buffer.  D2H copies are stream-ordered and
  // do not require the context to be explicitly on the calling thread's
  // context stack.
  template <typename... Ts>
  std::vector<output_buffer> collect_output_buffers_helper_unchecked(const std::tuple<Ts...>& args) {
    std::vector<output_buffer> result;
    std::apply([&](auto&&... mem) {
      (([&] {
        if (mem && (mem->access() == OUT || mem->access() == IN_OUT)) {
          result.emplace_back(output_buffer{buffer_variant{mem->copy_to_host_unchecked()}});
        }
      })(), ...);
    }, args);
    return result;
  }

  // Writes the first OUT/IN_OUT mem_ref of type T directly into a caller-supplied
  // buffer, bypassing the internal std::vector allocation in copy_to_host().
  // This removes the page-fault cost from the timed window when the caller
  // pre-allocates `dst` before the timer starts.
  template <typename T, typename... Ts>
  void collect_output_buffers_into(const std::tuple<Ts...>& args, T* dst, size_t count) {
      collect_output_into_seq(args, dst, count, std::index_sequence_for<Ts...>{});
  }

  // Bottleneck 2 fix: unchecked variant \u2014 uses copy_to_host_unchecked(T*, count)
  // so that no cuCtxPushCurrent/Pop is performed per buffer.
  template <typename T, typename... Ts>
  void collect_output_buffers_into_unchecked(const std::tuple<Ts...>& args, T* dst, size_t count) {
      collect_output_into_seq_unchecked(args, dst, count, std::index_sequence_for<Ts...>{});
  }




  //launches a kernel using wrapper types, in, in_out and out as arguments
  //and returns a tuple of mem ref's that hold device memory  
  template <typename... Args>
	std::tuple<mem_ptr<raw_t<Args>>...>
	launch_kernel_mem_ref(CUfunction kernel,
                        const nd_range& range,
                        std::tuple<Args...> args,
                        caf::actor_id actor_id,
		                    int shared_mem = 0) { //in bytes

    // Step 1: Allocate mem_ref<T> for each wrapper type 
    CUstream stream = get_stream_for_actor(actor_id);
    auto mem_refs = std::apply([&](auto&&... arg) {
      return std::make_tuple(make_arg(std::forward<decltype(arg)>(arg), stream)...);
    }, args);

    // Step 2: Prepare kernel argument pointers
    auto kernel_args = prepare_kernel_args(mem_refs);

    // Step 3: Launch kernel
    CHECK_CUDA(cuCtxPushCurrent(getContext()));
    launch_kernel_internal(kernel, range, stream, kernel_args.ptrs.data(),shared_mem);
    CHECK_CUDA(cuCtxPopCurrent(nullptr));

    // Step 4: Clean up kernel argument pointers
    cleanup_kernel_args(kernel_args);

  // Step 5: Return tuple of mem_ref<T>...
  return mem_refs;
}



  // Launch kernel with args that have already been allocated 
  // on the device via mem_ref<T>
  template <typename... Ts>
  std::vector<output_buffer> launch_kernel(CUfunction kernel,
                                           const nd_range& range,
                                           std::tuple<Ts...> args,
                                           caf::actor_id actor_id) {
    CUstream stream = get_stream_for_actor(actor_id);
    CHECK_CUDA(cuCtxPushCurrent(getContext()));

    auto kernel_args = prepare_kernel_args(args);
    launch_kernel_internal(kernel, range, stream, kernel_args.ptrs.data());

    //CHECK_CUDA(cuStreamSynchronize(stream));
    CHECK_CUDA(cuCtxPopCurrent(nullptr));

    auto outputs = collect_output_buffers(args);
    cleanup_kernel_args(kernel_args);

    return outputs;
  }

  // For testing: scalar/buffer detection and cleanup
  struct kernel_arg_pack {
    std::vector<void*> ptrs;
    std::vector<CUdeviceptr*> allocated_device_ptrs; // Buffers only
  };

 

   //given a tuple of mem_refs, turns them into  CUDeviceptrs that 
   //can be used to launch kernels
  template <typename... Ts>
  kernel_arg_pack prepare_kernel_args(const std::tuple<Ts...>& args) {
    kernel_arg_pack pack;
    std::apply([&](auto&&... mem) {
      (([&] {
        if (mem->is_scalar()) {
          pack.ptrs.push_back(const_cast<void*>(
            static_cast<const void*>(mem->host_scalar_ptr())));
        } else {
          CUdeviceptr* dev_ptr = new CUdeviceptr(mem->mem());
          pack.ptrs.push_back(static_cast<void*>(dev_ptr));
          pack.allocated_device_ptrs.push_back(dev_ptr);
        }
      })(), ...);
    }, args);
    return pack;
  }

  //cleans up the cuDevicePtrs that are no longer needed
  void cleanup_kernel_args(kernel_arg_pack& pack) {
    for (auto* ptr : pack.allocated_device_ptrs)
      delete ptr;
    pack.ptrs.clear();
    pack.allocated_device_ptrs.clear();
  }


  //given a tuple of mem_ptrs, collects their data on the gpu and 
  //returns an std::vector<output_buffer>
  template <typename... Ts>
  std::vector<output_buffer> collect_output_buffers(const std::tuple<Ts...>& args) {
   return collect_output_buffers_helper(args);
  }

  // Bottleneck 2 fix: unchecked variant of collect_output_buffers.
  // Uses copy_to_host_unchecked() so that D2H copies are performed without
  // per-buffer cuCtxPushCurrent/Pop round-trips.
  template <typename... Ts>
  std::vector<output_buffer> collect_output_buffers_unchecked(const std::tuple<Ts...>& args) {
    return collect_output_buffers_helper_unchecked(args);
  }

  // Allocates and zero-initialises a std::vector<T> for every OUT/IN_OUT
  // mem_ref in the tuple, returning them wrapped in output_buffer.  Call this
  // immediately after base_enqueue() while the GPU is running H2D copies and
  // the kernel so that page-fault cost overlaps with GPU execution.
  template <typename... Ts>
  std::vector<output_buffer> preallocate_output_buffers(const std::tuple<Ts...>& args) {
    std::vector<output_buffer> result;
    preallocate_output_seq(args, result, std::index_sequence_for<Ts...>{});
    return result;
  }

  // Fills every pre-allocated output_buffer in `pre_alloc` with D2H data from
  // the corresponding OUT/IN_OUT mem_ref in `args`.  Call this when the GPU
  // stream is idle (i.e. inside the gpu_done_atom handler) instead of the
  // full collect_output_buffers() so no new allocation takes place.
  template <typename... Ts>
  void fill_output_buffers(const std::tuple<Ts...>& args,
                           std::vector<output_buffer>& pre_alloc) {
    size_t idx = 0;
    fill_output_seq(args, pre_alloc, idx, std::index_sequence_for<Ts...>{});
  }

  // Bottleneck 2 fix: unchecked variant of fill_output_buffers.
  // Calls copy_to_host_unchecked() so no per-buffer cuCtxPushCurrent/Pop is
  // needed.  Call this after ensuring the correct CUDA device context is known
  // to be reachable via the stream (stream-ordered ops do not need a pushed ctx).
  template <typename... Ts>
  void fill_output_buffers_unchecked(const std::tuple<Ts...>& args,
                                     std::vector<output_buffer>& pre_alloc) {
    size_t idx = 0;
    fill_output_seq_unchecked(args, pre_alloc, idx, std::index_sequence_for<Ts...>{});
  }



  // === Old method for legacy tests ===
  template <typename... Ts>
  std::vector<void*> extract_kernel_args(const std::tuple<Ts...>& t) {
    return extract_kernel_args_impl(t, std::index_sequence_for<Ts...>{});
  }

private:
  CUdevice device_;
  CUcontext context_;
  int id_;
  const char* name_;
  DeviceStreamTable stream_table_;
  std::mutex stream_mutex_;

  // Cached GPU properties (queried once during construction)
  int sm_count_ = 0;
  int warp_size_ = 0;
  int max_threads_per_sm_ = 0;
  int warps_per_sm_ = 0;
  int total_warps_ = 0;
  std::size_t total_mem_bytes_ = 0;

  // Initialize and cache device properties. Called once from constructor.
  void init_device_properties() {
	  CUresult res;
	  int tmp = 0;

	  // Number of SMs
	  res = cuDeviceGetAttribute(&tmp, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device_);
	  if (res == CUDA_SUCCESS) sm_count_ = tmp;
	  else { const char* n = nullptr; cuGetErrorName(res, &n); throw std::runtime_error(std::string("cuDeviceGetAttribute(MULTIPROCESSOR_COUNT) failed: ") + (n ? n : "unknown")); }

	  // Warp size
	  res = cuDeviceGetAttribute(&tmp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device_);
	  if (res == CUDA_SUCCESS) warp_size_ = tmp;
	  else { const char* n = nullptr; cuGetErrorName(res, &n); throw std::runtime_error(std::string("cuDeviceGetAttribute(WARP_SIZE) failed: ") + (n ? n : "unknown")); }

	  // Max threads per SM
	  res = cuDeviceGetAttribute(&tmp, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device_);
	  if (res == CUDA_SUCCESS) max_threads_per_sm_ = tmp;
	  else { const char* n = nullptr; cuGetErrorName(res, &n); throw std::runtime_error(std::string("cuDeviceGetAttribute(MAX_THREADS_PER_MULTIPROCESSOR) failed: ") + (n ? n : "unknown")); }

	  // Derived values
	  if (warp_size_ > 0 && max_threads_per_sm_ > 0) {
		  warps_per_sm_ = std::max(1, max_threads_per_sm_ / warp_size_);
		  total_warps_ = warps_per_sm_ * sm_count_;
	  }

	  // Total memory
	  size_t bytes = 0;
	  res = cuDeviceTotalMem(&bytes, device_);
	  if (res == CUDA_SUCCESS) total_mem_bytes_ = bytes;
	  else { const char* n = nullptr; cuGetErrorName(res, &n); throw std::runtime_error(std::string("cuDeviceTotalMem failed: ") + (n ? n : "unknown")); }
  }


  // === Memory handling ===
  
  //----------------------------------------------
// Helpers for actor_id version
//----------------------------------------------

// allocate a readonly input buffer on the GPU
template <typename T>
mem_ptr<T> global_argument(const in<T>& arg, int actor_id, int access) {
  CUstream stream = get_stream_for_actor(actor_id);
  return global_argument(arg, stream, access);
}

// allocate a read/write input buffer on the GPU
template <typename T>
mem_ptr<T> global_argument(const in_out<T>& arg, int actor_id, int access) {
  CUstream stream = get_stream_for_actor(actor_id);
  return global_argument(arg, stream, access);
}

// allocate an output buffer on the GPU
template <typename T>
mem_ptr<T> scratch_argument(const out<T>& arg, int actor_id, int access) {
  CUstream stream = get_stream_for_actor(actor_id);
  return scratch_argument(arg, stream, access);
}

//----------------------------------------------
// Helpers for CUstream version
//----------------------------------------------

// allocate a readonly input buffer on the GPU
// Bottleneck 2+3 fix: cuMemAllocAsync is stream-based and does not require
// the CUDA context to be current, so no cuCtxPushCurrent/Pop is needed.
// Memory is served from the stream-ordered driver pool (bottleneck 3), which
// avoids blocking system-call overhead of cuMemAlloc.
template <typename T>
mem_ptr<T> global_argument(const in<T>& arg, CUstream stream, int access) {
  if (arg.is_scalar()) {
    return caf::intrusive_ptr<mem_ref<T>>{
      new mem_ref<T>(arg.getscalar(), access, id_, 0, getContext(), stream),
      caf::add_ref};
  }
  size_t bytes = arg.size() * sizeof(T);
  CUdeviceptr dev_ptr;
  CHECK_CUDA(cuMemAllocAsync(&dev_ptr, bytes, stream));
  CHECK_CUDA(cuMemcpyHtoDAsync(dev_ptr, arg.data(), bytes, stream));
  return caf::intrusive_ptr<mem_ref<T>>{
    new mem_ref<T>(arg.size(), dev_ptr, access, id_, 0, getContext(), stream, /*pool_alloc=*/true),
    caf::add_ref};
}

// allocate a read/write input buffer on the GPU
// See global_argument for bottleneck 2+3 notes.
template <typename T>
mem_ptr<T> global_argument(const in_out<T>& arg, CUstream stream, int access) {
  if (arg.is_scalar()) {
    return caf::intrusive_ptr<mem_ref<T>>{
      new mem_ref<T>(arg.getscalar(), access, id_, 0, getContext(), stream),
      caf::add_ref};
  }
  size_t bytes = arg.size() * sizeof(T);
  CUdeviceptr dev_ptr;
  CHECK_CUDA(cuMemAllocAsync(&dev_ptr, bytes, stream));
  CHECK_CUDA(cuMemcpyHtoDAsync(dev_ptr, arg.data(), bytes, stream));
  return caf::intrusive_ptr<mem_ref<T>>{
    new mem_ref<T>(arg.size(), dev_ptr, access, id_, 0, getContext(), stream, /*pool_alloc=*/true),
    caf::add_ref};
}

// allocate an output buffer on the GPU
// See global_argument for bottleneck 2+3 notes.
template <typename T>
mem_ptr<T> scratch_argument(const out<T>& arg, CUstream stream, int access) {
  size_t size =  arg.size();
  CUdeviceptr dev_ptr;
  CHECK_CUDA(cuMemAllocAsync(&dev_ptr, size * sizeof(T), stream));
  return caf::intrusive_ptr<mem_ref<T>>{
    new mem_ref<T>(size, dev_ptr, access, id_, 0, getContext(), stream, /*pool_alloc=*/true),
    caf::add_ref};
}  

  // === Kernel launch core ===
  void launch_kernel_internal(CUfunction kernel,
                              const nd_range& range,
                              CUstream stream,
                              void** args,
			                        int shared_mem = 0) {
    CUresult result = cuLaunchKernel(kernel,
                                     range.getGridDimX(), range.getGridDimY(), range.getGridDimZ(),
                                     range.getBlockDimX(), range.getBlockDimY(), range.getBlockDimZ(),
                                     shared_mem, stream, args, nullptr);
    if (result != CUDA_SUCCESS) {
      const char* err_name = nullptr;
      cuGetErrorName(result, &err_name);
      throw std::runtime_error(std::string("cuLaunchKernel failed: ") +
                               (err_name ? err_name : "unknown error"));
    }
  }
  // === Legacy helper ===
  template <typename Tuple, std::size_t... Is>
  std::vector<void*> extract_kernel_args_impl(const Tuple& t,
                                              std::index_sequence<Is...>) {
    std::vector<void*> args(sizeof...(Is));
    size_t i = 0;
    (([&] {
      auto ptr = std::get<Is>(t);
      if (ptr->is_scalar()) {
        args[i++] = const_cast<void*>(static_cast<const void*>(ptr->host_scalar_ptr()));
      } else {
        CUdeviceptr* slot = new CUdeviceptr(ptr->mem());
        args[i++] = slot;
      }
    }()), ...);
    return args;
  }

  // === collect_output_buffers_into helpers ===
  // Avoiding std::apply + generic lambda + if constexpr (GCC 11 ICE workaround).
  // Uses explicit std::get<I> with an index_sequence instead.

  // Try to write the element at tuple index I into dst if it's OUT/IN_OUT and
  // its value_type matches T.  Returns true when the write happens.
  template <typename T, typename Tuple, std::size_t I>
  static bool try_output_at(const Tuple& t, T* dst, size_t count) {
    const auto& mem = std::get<I>(t);
    using U = typename std::decay_t<decltype(*mem)>::value_type;
    if constexpr (std::is_same_v<U, T>) {
      if (mem && (mem->access() == OUT || mem->access() == IN_OUT)) {
        mem->copy_to_host(dst, count);
        return true;
      }
    }
    return false;
  }

  // Iterate over all tuple indices; stop at the first successful write.
  template <typename T, typename Tuple, std::size_t... Is>
  static void collect_output_into_seq(const Tuple& t, T* dst, size_t count,
                                      std::index_sequence<Is...>) {
    bool done = false;
    ((done = done || try_output_at<T, Tuple, Is>(t, dst, count)), ...);
  }

  // Bottleneck 2 fix: unchecked variant — uses copy_to_host_unchecked.
  template <typename T, typename Tuple, std::size_t I>
  static bool try_output_at_unchecked(const Tuple& t, T* dst, size_t count) {
    const auto& mem = std::get<I>(t);
    using U = typename std::decay_t<decltype(*mem)>::value_type;
    if constexpr (std::is_same_v<U, T>) {
      if (mem && (mem->access() == OUT || mem->access() == IN_OUT)) {
        mem->copy_to_host_unchecked(dst, count);
        return true;
      }
    }
    return false;
  }

  template <typename T, typename Tuple, std::size_t... Is>
  static void collect_output_into_seq_unchecked(const Tuple& t, T* dst, size_t count,
                                                std::index_sequence<Is...>) {
    bool done = false;
    ((done = done || try_output_at_unchecked<T, Tuple, Is>(t, dst, count)), ...);
  }

  // === preallocate_output_buffers / fill_output_buffers helpers ===
  // These two methods allow the actor_facade to pre-allocate (and pre-fault)
  // the host output buffer(s) immediately after the kernel is queued, while
  // the GPU is executing H2D copies and the kernel asynchronously.  The
  // fill step runs when gpu_done_atom fires and does only the D2H DMA.

  // Allocate a zero-initialised std::vector<T> for the element at tuple
  // index I if it is an OUT/IN_OUT buffer.
  template <typename Tuple, std::size_t I>
  static void try_preallocate_at(const Tuple& t, std::vector<output_buffer>& result) {
    const auto& mem = std::get<I>(t);
    if (mem && (mem->access() == OUT || mem->access() == IN_OUT)) {
      using T = typename std::decay_t<decltype(*mem)>::value_type;
      result.emplace_back(output_buffer{buffer_variant{std::vector<T>(mem->size())}});
    }
  }

  template <typename Tuple, std::size_t... Is>
  static void preallocate_output_seq(const Tuple& t, std::vector<output_buffer>& result,
                                     std::index_sequence<Is...>) {
    (try_preallocate_at<Tuple, Is>(t, result), ...);
  }

  // Copy D2H into the pre-allocated vector at slot idx for the element at
  // tuple index I if it is an OUT/IN_OUT buffer.
  template <typename Tuple, std::size_t I>
  static void try_fill_at(const Tuple& t, std::vector<output_buffer>& pre_alloc,
                           size_t& idx) {
    const auto& mem = std::get<I>(t);
    if (mem && (mem->access() == OUT || mem->access() == IN_OUT)) {
      if (idx < pre_alloc.size()) {
        using T = typename std::decay_t<decltype(*mem)>::value_type;
        auto* ptr = std::get_if<std::vector<T>>(&pre_alloc[idx].data);
        if (ptr) mem->copy_to_host(ptr->data(), ptr->size());
        ++idx;
      }
    }
  }

  template <typename Tuple, std::size_t... Is>
  static void fill_output_seq(const Tuple& t, std::vector<output_buffer>& pre_alloc,
                               size_t& idx, std::index_sequence<Is...>) {
    (try_fill_at<Tuple, Is>(t, pre_alloc, idx), ...);
  }

  // Bottleneck 2 fix: unchecked fill variant — uses copy_to_host_unchecked.
  template <typename Tuple, std::size_t I>
  static void try_fill_at_unchecked(const Tuple& t, std::vector<output_buffer>& pre_alloc,
                                    size_t& idx) {
    const auto& mem = std::get<I>(t);
    if (mem && (mem->access() == OUT || mem->access() == IN_OUT)) {
      if (idx < pre_alloc.size()) {
        using T = typename std::decay_t<decltype(*mem)>::value_type;
        auto* ptr = std::get_if<std::vector<T>>(&pre_alloc[idx].data);
        if (ptr) mem->copy_to_host_unchecked(ptr->data(), ptr->size());
        ++idx;
      }
    }
  }

  template <typename Tuple, std::size_t... Is>
  static void fill_output_seq_unchecked(const Tuple& t, std::vector<output_buffer>& pre_alloc,
                                         size_t& idx, std::index_sequence<Is...>) {
    (try_fill_at_unchecked<Tuple, Is>(t, pre_alloc, idx), ...);
  }
};

} // namespace caf::cuda

