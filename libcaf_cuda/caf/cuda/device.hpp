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
#include <memory>

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
      stream_table_(std::make_unique<DeviceStreamTable>(context, stream_pool_size)) {
 	      init_device_properties();
      }

  ~device() {
    check(cuCtxDestroy(context_), "cuCtxDestroy");
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


  // Resets the CUDA context for this device.
  // Destroys the old context and reinitializes the stream pool with the new one.
  void reset_context(CUcontext new_ctx) {
    // Destroy the old context
    if (context_) {
      check(cuCtxDestroy(context_), "cuCtxDestroy in device::reset_context");
    }
    // Set the new context and reinitialize the stream table
    context_ = new_ctx;
    stream_table_ = std::make_unique<DeviceStreamTable>(context_, stream_table_->pool_size());
  }
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
  CUstream get_stream_for_actor(int actor_id) {
    return stream_table_->get_stream(actor_id);
  }

  //releases the CUStream associated with the actor id 
  void release_stream_for_actor(int actor_id) {
    stream_table_->release_stream(actor_id);
  }

  /// Enable cuBLAS support.
  void enable_cublas() {
    if (!cublas_table_)
      cublas_table_ = std::make_unique<DeviceCublasHandleTable>(context_);
  }

  /// Returns the cuBLAS handle associated with the actor id.
  cublasHandle_t get_cublas_handle(int stream_id) {
    if (!cublas_table_) return nullptr;
    return cublas_table_->get_handle(stream_id, get_stream_for_actor(stream_id));
  }

  /// Performs single precision matrix-vector multiplication (y = alpha*A*x + beta*y).
  /// Assumes A is in row-major order of dimensions m x n.
  void sgemv(int stream_id, int m, int n, float alpha, mem_ptr<float> A,
             mem_ptr<float> x, float beta, mem_ptr<float> y) {
    cublasHandle_t handle = get_cublas_handle(stream_id);
    if (!handle)
      throw std::runtime_error("cuBLAS not enabled on device " + std::to_string(id_));

    CHECK_CUDA(cuCtxPushCurrent(context_));
    
    // Row-major matrix A (m x n) is stored as m rows of n elements.
    // Viewed as column-major by cuBLAS, this is a n x m matrix.
    // To compute y = A * x:
    // Op(Memory) * x = (n x m)^T * (n x 1) = (m x n) * (n x 1) = (m x 1).
    // We use CUBLAS_OP_T. LDA is the 'rows' in the column-major view, which is n.
    cublasStatus_t status = cublasSgemv(handle, CUBLAS_OP_T, 
                                        n, m, 
                                        &alpha,
                                        reinterpret_cast<const float*>(A->mem()), n,
                                        reinterpret_cast<const float*>(x->mem()), 1,
                                        &beta,
                                        reinterpret_cast<float*>(y->mem()), 1);

    CHECK_CUDA(cuCtxPopCurrent(nullptr));
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublasSgemv failed on device " + std::to_string(id_));
  }

  /// Performs symmetric rank-k update (C = alpha*A*A^T + beta*C).
  /// Assumes A is in row-major order of dimensions n x k, and C is n x n.
  void ssyrk(int stream_id, int n, int k, float alpha, mem_ptr<float> A,
             float beta, mem_ptr<float> C) {
    cublasHandle_t handle = get_cublas_handle(stream_id);
    if (!handle)
      throw std::runtime_error("cuBLAS not enabled on device " + std::to_string(id_));

    CHECK_CUDA(cuCtxPushCurrent(context_));
    
    // Row-major matrix A (n x k) viewed as column-major is k x n.
    // To compute C = alpha * A * A^T + beta * C:
    // We use CUBLAS_OP_T so that (k x n)^T * (k x n) = (n x k) * (k x n) = n x n.
    // Note: We use CUBLAS_FILL_MODE_UPPER because the upper triangle in 
    // column-major maps to the lower triangle in row-major layout.
    cublasStatus_t status = cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                        n, k,
                                        &alpha,
                                        reinterpret_cast<const float*>(A->mem()), k,
                                        &beta,
                                        reinterpret_cast<float*>(C->mem()), n);

    CHECK_CUDA(cuCtxPopCurrent(nullptr));
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublasSsyrk failed on device " + std::to_string(id_));
  }

  /// Performs single precision vector-vector addition (y = alpha*x + y).
  void saxpy(int stream_id, int n, float alpha, mem_ptr<float> x, mem_ptr<float> y) {
    cublasHandle_t handle = get_cublas_handle(stream_id);
    if (!handle)
      throw std::runtime_error("cuBLAS not enabled on device " + std::to_string(id_));

    CHECK_CUDA(cuCtxPushCurrent(context_));

    cublasStatus_t status = cublasSaxpy(handle, n, &alpha,
                                        reinterpret_cast<const float*>(x->mem()), 1,
                                        reinterpret_cast<float*>(y->mem()), 1);

    CHECK_CUDA(cuCtxPopCurrent(nullptr));
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublasSaxpy failed on device " + std::to_string(id_));
  }

  /// Performs single precision Euclidean norm (result = ||x||2).
  void snrm2(int stream_id, int n, mem_ptr<float> x, mem_ptr<float> result) {
    cublasHandle_t handle = get_cublas_handle(stream_id);
    if (!handle)
      throw std::runtime_error("cuBLAS not enabled on device " + std::to_string(id_));

    CHECK_CUDA(cuCtxPushCurrent(context_));

    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasStatus_t status = cublasSnrm2(handle, n, 
                                        reinterpret_cast<const float*>(x->mem()), 1,
                                        reinterpret_cast<float*>(result->mem()));
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

    CHECK_CUDA(cuCtxPopCurrent(nullptr));
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublasSnrm2 failed on device " + std::to_string(id_));
  }

  /// Performs single precision dot product (result = x^T * y).
  void sdot(int stream_id, int n, mem_ptr<float> x, mem_ptr<float> y, mem_ptr<float> result) {
    cublasHandle_t handle = get_cublas_handle(stream_id);
    if (!handle)
      throw std::runtime_error("cuBLAS not enabled on device " + std::to_string(id_));

    CHECK_CUDA(cuCtxPushCurrent(context_));

    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasStatus_t status = cublasSdot(handle, n, 
                                        reinterpret_cast<const float*>(x->mem()), 1,
                                        reinterpret_cast<const float*>(y->mem()), 1,
                                        reinterpret_cast<float*>(result->mem()));
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

    CHECK_CUDA(cuCtxPopCurrent(nullptr));
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublasSdot failed on device " + std::to_string(id_));
  }

  /// Copies vector x to vector y (y = x).
  void scopy(int stream_id, int n, mem_ptr<float> x, mem_ptr<float> y) {
    cublasHandle_t handle = get_cublas_handle(stream_id);
    if (!handle)
      throw std::runtime_error("cuBLAS not enabled on device " + std::to_string(id_));

    CHECK_CUDA(cuCtxPushCurrent(context_));

    cublasStatus_t status = cublasScopy(handle, n, 
                                         reinterpret_cast<const float*>(x->mem()), 1,
                                         reinterpret_cast<float*>(y->mem()), 1);

    CHECK_CUDA(cuCtxPopCurrent(nullptr));
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublasScopy failed on device " + std::to_string(id_));
  }

  /// Performs single precision matrix-matrix multiplication (C = alpha*A*B + beta*C).
  /// Assumes A is m x k, B is k x n, and C is m x n, all in row-major order.
  void sgemm(int stream_id, int m, int n, int k, float alpha, mem_ptr<float> A,
             mem_ptr<float> B, float beta, mem_ptr<float> C) {
    cublasHandle_t handle = get_cublas_handle(stream_id);
    if (!handle)
      throw std::runtime_error("cuBLAS not enabled on device " + std::to_string(id_));

    CHECK_CUDA(cuCtxPushCurrent(context_));

    // For row-major matrices A(m,k), B(k,n), C(m,n) to compute C = alpha*A*B + beta*C
    // cuBLAS expects column-major. The equivalent column-major operation is:
    // C_col = alpha * B_col * A_col + beta * C_col
    // where X_col = X_row^T.
    // So, we call cublasSgemm with:
    // A_cublas = B (transposed), B_cublas = A (transposed)
    // Dimensions: m_cublas = n, n_cublas = m, k_cublas = k
    cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                        n, m, k,
                                        &alpha,
                                        reinterpret_cast<const float*>(B->mem()), n, // B is k x n row-major, lda = n
                                        reinterpret_cast<const float*>(A->mem()), k, // A is m x k row-major, ldb = k
                                        &beta,
                                        reinterpret_cast<float*>(C->mem()), n); // C is m x n row-major, ldc = n

    CHECK_CUDA(cuCtxPopCurrent(nullptr));
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublasSgemm failed on device " + std::to_string(id_));
  }

  /// Performs double precision matrix-matrix multiplication (C = alpha*A*B + beta*C).
  /// Assumes A is m x k, B is k x n, and C is m x n, all in row-major order.
  void dgemm(int stream_id, int m, int n, int k, double alpha, mem_ptr<double> A,
             mem_ptr<double> B, double beta, mem_ptr<double> C) {
    cublasHandle_t handle = get_cublas_handle(stream_id);
    if (!handle)
      throw std::runtime_error("cuBLAS not enabled on device " + std::to_string(id_));

    CHECK_CUDA(cuCtxPushCurrent(context_));

    // For row-major matrices A(m,k), B(k,n), C(m,n) to compute C = alpha*A*B + beta*C
    // cuBLAS expects column-major. The equivalent column-major operation is:
    // C_col = alpha * B_col * A_col + beta * C_col
    // where X_col = X_row^T.
    // So, we call cublasDgemm with:
    // A_cublas = B (transposed), B_cublas = A (transposed)
    // Dimensions: m_cublas = n, n_cublas = m, k_cublas = k
    cublasStatus_t status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                        n, m, k,
                                        &alpha,
                                        reinterpret_cast<const double*>(B->mem()), n, // B is k x n row-major, lda = n
                                        reinterpret_cast<const double*>(A->mem()), k, // A is m x k row-major, ldb = k
                                        &beta,
                                        reinterpret_cast<double*>(C->mem()), n); // C is m x n row-major, ldc = n

    CHECK_CUDA(cuCtxPopCurrent(nullptr));
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublasDgemm failed on device " + std::to_string(id_));
  }

  /// Performs single precision strided batched matrix-matrix multiplication.
  void sgemm_strided_batched(int stream_id, int m, int n, int k, float alpha, mem_ptr<float> A, long long int strideA,
                             mem_ptr<float> B, long long int strideB, float beta, mem_ptr<float> C, long long int strideC, int batchCount) {
    cublasHandle_t handle = get_cublas_handle(stream_id);
    if (!handle)
      throw std::runtime_error("cuBLAS not enabled on device " + std::to_string(id_));

    CHECK_CUDA(cuCtxPushCurrent(context_));
    
    // Row-major A(m,k), B(k,n), C(m,n) -> cuBLAS (col-major): C = B * A
    cublasStatus_t status = cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                                      n, m, k,
                                                      &alpha,
                                                      reinterpret_cast<const float*>(B->mem()), n, strideB,
                                                      reinterpret_cast<const float*>(A->mem()), k, strideA,
                                                      &beta,
                                                      reinterpret_cast<float*>(C->mem()), n, strideC,
                                                      batchCount);

    CHECK_CUDA(cuCtxPopCurrent(nullptr));
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublasSgemmStridedBatched failed on device " + std::to_string(id_));
  }

  /// Performs double precision strided batched matrix-matrix multiplication.
  void dgemm_strided_batched(int stream_id, int m, int n, int k, double alpha, mem_ptr<double> A, long long int strideA,
                             mem_ptr<double> B, long long int strideB, double beta, mem_ptr<double> C, long long int strideC, int batchCount) {
    cublasHandle_t handle = get_cublas_handle(stream_id);
    if (!handle)
      throw std::runtime_error("cuBLAS not enabled on device " + std::to_string(id_));

    CHECK_CUDA(cuCtxPushCurrent(context_));
    
    cublasStatus_t status = cublasDgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                                      n, m, k,
                                                      &alpha,
                                                      reinterpret_cast<const double*>(B->mem()), n, strideB,
                                                      reinterpret_cast<const double*>(A->mem()), k, strideA,
                                                      &beta,
                                                      reinterpret_cast<double*>(C->mem()), n, strideC,
                                                      batchCount);

    CHECK_CUDA(cuCtxPopCurrent(nullptr));
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublasDgemm failed on device " + std::to_string(id_));
  }

  // Overloads for make_arg using actor_id
  template <typename T>
  mem_ptr<T> make_arg(const in<T>& arg, int actor_id) {
    return global_argument(arg, actor_id, IN);
  }

  template <typename T>
  mem_ptr<T> make_arg(const in_out<T>& arg, int actor_id) {
    return global_argument(arg, actor_id, IN_OUT);
  }

  template <typename T>
  mem_ptr<T> make_arg(const out<T>& arg, int actor_id) {
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




  //launches a kernel using wrapper types, in, in_out and out as arguments
  //and returns a tuple of mem ref's that hold device memory  
  	  template <typename... Args>
	  std::tuple<mem_ptr<raw_t<Args>>...>
	  launch_kernel_mem_ref(CUfunction kernel,
                      const nd_range& range,
                      std::tuple<Args...> args,
                      int actor_id,
		      int shared_mem = 0 //in bytes
		      ) {

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
                                           int actor_id) {
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
  std::unique_ptr<DeviceStreamTable> stream_table_;
  std::unique_ptr<DeviceCublasHandleTable> cublas_table_;
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
template <typename T>
mem_ptr<T> global_argument(const in<T>& arg, CUstream stream, int access) {
  if (arg.is_scalar()) {
    return caf::intrusive_ptr<mem_ref<T>>(
      new mem_ref<T>(arg.getscalar(), access, id_, 0, getContext(), stream));
  }
  size_t bytes = arg.size() * sizeof(T);
  CUdeviceptr dev_ptr;
  CHECK_CUDA(cuCtxPushCurrent(getContext()));
  CHECK_CUDA(cuMemAllocAsync(&dev_ptr, bytes, stream));
  CHECK_CUDA(cuMemcpyHtoDAsync(dev_ptr, arg.data(), bytes, stream));
  CHECK_CUDA(cuCtxPopCurrent(nullptr));
  return caf::intrusive_ptr<mem_ref<T>>(
    new mem_ref<T>(arg.size(), dev_ptr, access, id_, 0, getContext(), stream));
}

// allocate a read/write input buffer on the GPU
template <typename T>
mem_ptr<T> global_argument(const in_out<T>& arg, CUstream stream, int access) {
  if (arg.is_scalar()) {
    return caf::intrusive_ptr<mem_ref<T>>(
      new mem_ref<T>(arg.getscalar(), access, id_, 0, getContext(), stream));
  }
  size_t bytes = arg.size() * sizeof(T);
  CUdeviceptr dev_ptr;
  CHECK_CUDA(cuCtxPushCurrent(getContext()));
  CHECK_CUDA(cuMemAllocAsync(&dev_ptr, bytes, stream));
  CHECK_CUDA(cuMemcpyHtoDAsync(dev_ptr, arg.data(), bytes, stream));
  CHECK_CUDA(cuCtxPopCurrent(nullptr));
  return caf::intrusive_ptr<mem_ref<T>>(
    new mem_ref<T>(arg.size(), dev_ptr, access, id_, 0, getContext(), stream));
}

// allocate an output buffer on the GPU
template <typename T>
mem_ptr<T> scratch_argument(const out<T>& arg, CUstream stream, int access) {
  size_t size =  arg.size();
  CUdeviceptr dev_ptr;
  CHECK_CUDA(cuCtxPushCurrent(getContext()));
  CHECK_CUDA(cuMemAllocAsync(&dev_ptr, size * sizeof(T), stream));
  CHECK_CUDA(cuCtxPopCurrent(nullptr));
  return caf::intrusive_ptr<mem_ref<T>>(
    new mem_ref<T>(size, dev_ptr, access, id_, 0, getContext(), stream));
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
};

} // namespace caf::cuda
