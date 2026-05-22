#include "caf/cuda/streampool.hpp"
#include <iostream> // only used for potential debug; remove if undesired

namespace caf::cuda {

// ---------------------- StreamPool ----------------------

StreamPool::StreamPool(CUcontext ctx, size_t initial_size, size_t max_size)
    : ctx_(ctx), max_size_(max_size) {
  // Create `initial_size` streams up-front so the pool can quickly serve requests.
  // We push them into both `all_streams_` and `available_streams_`.
  for (size_t i = 0; i < initial_size && total_streams_ < max_size_; ++i) {
    CUstream s = create_stream();
    all_streams_.push_back(s);
    available_streams_.push_back(s);
    ++total_streams_;
  }
}

StreamPool::~StreamPool() {
  // Streams are destroyed when the CUDA context is destroyed.
  // If explicit stream destruction is desired, add cuStreamDestroy calls here
  // while pushing the appropriate context. For simplicity and safety we rely on
  // context teardown to free streams (common pattern).
}

CUstream StreamPool::acquire() {
  // Acquire is designed so the common case (available stream exists) is fast:
  // a single mutex lock, pop front, increment in_use_, return.
  std::lock_guard<std::mutex> guard(pool_mutex_);

  // Fast path: reuse an available stream.
  if (!available_streams_.empty()) {
    CUstream s = available_streams_.front();
    available_streams_.pop_front();
    ++in_use_;
    return s;
  }

  // No available streams in the queue. If we can still create new streams
  // (we haven't hit max_size_), do so. This is the second-fast path.
  if (total_streams_ < max_size_) {
    CUstream s = create_stream();
    all_streams_.push_back(s);
    ++total_streams_;
    ++in_use_;
    return s;
  }

  // Pool is exhausted: we must reuse an existing stream from all_streams_.
  // This is intentionally done instead of throwing. Reuse is chosen to avoid
  // unbounded memory growth; callers must understand reuse may collide with
  // in-flight work if they were relying on unique streams per concurrent actor.
  //
  // We use a simple round-robin index (rr_index_) to pick streams from
  // all_streams_. Access to rr_index_ is protected by pool_mutex_.
  if (!all_streams_.empty()) {
    CUstream s = all_streams_[rr_index_];
    rr_index_ = (rr_index_ + 1) % all_streams_.size();
    ++in_use_;
    return s;
  }

  // Defensive: if we somehow reached here with no streams at all (shouldn't happen),
  // signal an error.
  throw std::runtime_error("StreamPool: no streams available and none were created");
}

void StreamPool::release(CUstream s) {
  // Return the stream to the available queue and decrement in_use_.
  // This does not check for duplicates; callers should follow the acquire/release
  // discipline. Duplicate releases will push the same stream multiple times,
  // which can be detected via monitoring if necessary.
  std::lock_guard<std::mutex> guard(pool_mutex_);
  available_streams_.push_back(s);

  // Protect underflow: ensure we don't decrement below zero.
  if (in_use_ > 0) {
    --in_use_;
  } else {
    // This can indicate a logic error (double-release). We don't throw here,
    // but you can enable debug logging or an assert if you prefer strict behavior.
    // std::cerr << "StreamPool::release called when none in use\n";
  }
}

CUstream StreamPool::create_stream() {
  // Create a new CUDA stream while ensuring the correct context is current on
  // this thread. We push the context before creating the stream and pop it
  // immediately after. This mirrors the common driver API usage pattern.
  CUresult err = cuCtxPushCurrent(ctx_);
  if (err != CUDA_SUCCESS) {
    const char* err_str = nullptr;
    cuGetErrorString(err, &err_str);
    throw std::runtime_error(
        std::string("cuCtxPushCurrent failed: ") + (err_str ? err_str : "unknown error"));
  }

  CUstream s;
  err = cuStreamCreate(&s, CU_STREAM_DEFAULT);

  // Pop the context as soon as possible.
  CUcontext popped_ctx;
  CUresult pop_err = cuCtxPopCurrent(&popped_ctx);
  if (pop_err != CUDA_SUCCESS) {
    const char* err_str = nullptr;
    cuGetErrorString(pop_err, &err_str);
    throw std::runtime_error(
        std::string("cuCtxPopCurrent failed: ") + (err_str ? err_str : "unknown error"));
  }

  if (err != CUDA_SUCCESS) {
    const char* err_str = nullptr;
    cuGetErrorString(err, &err_str);
    throw std::runtime_error(
        std::string("cuStreamCreate failed: ") + (err_str ? err_str : "unknown error"));
  }

  return s;
}

size_t StreamPool::num_total() const {
  std::lock_guard<std::mutex> guard(pool_mutex_);
  return total_streams_;
}

size_t StreamPool::num_in_use() const {
  std::lock_guard<std::mutex> guard(pool_mutex_);
  return in_use_;
}

size_t StreamPool::num_available() const {
  std::lock_guard<std::mutex> guard(pool_mutex_);
  return available_streams_.size();
}

// ---------------------- DeviceStreamTable ----------------------

// DeviceStreamTable caches a stream per actor ID. The common case (actor already
// has an assigned stream) is a read-only lookup protected by a shared lock,
// which is cheap and concurrent-friendly.
DeviceStreamTable::DeviceStreamTable(CUcontext ctx, size_t pool_size)
    : pool_(ctx, pool_size) {}

CUstream DeviceStreamTable::get_stream(int actor_id) {
  // Fast read path: shared lock to allow concurrent lookups.
  {
    std::shared_lock<std::shared_mutex> read_lock(table_mutex_);
    auto it = table_.find(actor_id);
    if (it != table_.end()) {
      return it->second; // Common case: actor already has a stream
    }
  }

  // Need to assign a stream for this actor. Upgrade to exclusive lock to modify.
  std::unique_lock<std::shared_mutex> write_lock(table_mutex_);
  // Double-check in case another thread assigned while we were waiting for lock
  auto it = table_.find(actor_id);
  if (it != table_.end()) {
    return it->second;
  }

  CUstream s = pool_.acquire();      // Acquire (or reuse) stream from the pool
  table_[actor_id] = s;              // Cache mapping for fast future lookup
  return s;
}

void DeviceStreamTable::release_stream(int actor_id) {
  std::unique_lock<std::shared_mutex> write_lock(table_mutex_);
  auto it = table_.find(actor_id);
  if (it != table_.end()) {
    pool_.release(it->second); // Return the stream to the pool
    table_.erase(it);
  }
}

// ---------------------- CublasHandlePool ----------------------

CublasHandlePool::CublasHandlePool(CUcontext ctx, size_t max_size)
    : ctx_(ctx), max_size_(std::min(max_size, (size_t)32)) {}

CublasHandlePool::~CublasHandlePool() {
  for (auto h : all_handles_) {
    cublasDestroy(h);
  }
}

cublasHandle_t CublasHandlePool::acquire() {
  std::lock_guard<std::mutex> guard(pool_mutex_);

  if (!available_handles_.empty()) {
    cublasHandle_t h = available_handles_.front();
    available_handles_.pop_front();
    return h;
  }

  if (all_handles_.size() < max_size_) {
    cublasHandle_t h = create_handle();
    all_handles_.push_back(h);
    return h;
  }

  if (!all_handles_.empty()) {
    static size_t rr_idx = 0;
    cublasHandle_t h = all_handles_[rr_idx];
    rr_idx = (rr_idx + 1) % all_handles_.size();
    return h;
  }

  throw std::runtime_error("CublasHandlePool: no handles available");
}

void CublasHandlePool::release(cublasHandle_t h) {
  std::lock_guard<std::mutex> guard(pool_mutex_);
  available_handles_.push_back(h);
}

cublasHandle_t CublasHandlePool::create_handle() {
  CHECK_CUDA(cuCtxPushCurrent(ctx_));
  cublasHandle_t h;
  cublasStatus_t status = cublasCreate(&h);
  CHECK_CUDA(cuCtxPopCurrent(nullptr));

  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cublasCreate failed");
  return h;
}

// ---------------------- DeviceCublasHandleTable ----------------------

DeviceCublasHandleTable::DeviceCublasHandleTable(CUcontext ctx, size_t pool_size)
    : pool_(ctx, pool_size) {}

cublasHandle_t DeviceCublasHandleTable::get_handle(int actor_id, CUstream stream) {
  cublasHandle_t h = nullptr;
  {
    std::shared_lock<std::shared_mutex> read_lock(table_mutex_);
    auto it = table_.find(actor_id);
    if (it != table_.end())
      h = it->second;
  }

  if (!h) {
    std::unique_lock<std::shared_mutex> write_lock(table_mutex_);
    auto it = table_.find(actor_id);
    if (it != table_.end()) {
      h = it->second;
    } else {
      h = pool_.acquire();
      table_[actor_id] = h;
    }
  }

  // Always bind the handle to the caller's stream
  cublasStatus_t status = cublasSetStream(h, stream);
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cublasSetStream failed");

  return h;
}

void DeviceCublasHandleTable::release_handle(int actor_id) {
  std::unique_lock<std::shared_mutex> write_lock(table_mutex_);
  auto it = table_.find(actor_id);
  if (it != table_.end()) {
    pool_.release(it->second);
    table_.erase(it);
  }
}

// ---------------------- CusparseHandlePool ----------------------

CusparseHandlePool::CusparseHandlePool(CUcontext ctx, size_t max_size)
    : ctx_(ctx), max_size_(std::min(max_size, (size_t)32)) {}

CusparseHandlePool::~CusparseHandlePool() {
  for (auto h : all_handles_) {
    cusparseDestroy(h);
  }
}

cusparseHandle_t CusparseHandlePool::acquire() {
  std::lock_guard<std::mutex> guard(pool_mutex_);

  if (!available_handles_.empty()) {
    cusparseHandle_t h = available_handles_.front();
    available_handles_.pop_front();
    return h;
  }

  if (all_handles_.size() < max_size_) {
    cusparseHandle_t h = create_handle();
    all_handles_.push_back(h);
    return h;
  }

  if (!all_handles_.empty()) {
    static size_t rr_idx = 0;
    cusparseHandle_t h = all_handles_[rr_idx];
    rr_idx = (rr_idx + 1) % all_handles_.size();
    return h;
  }

  throw std::runtime_error("CusparseHandlePool: no handles available");
}

void CusparseHandlePool::release(cusparseHandle_t h) {
  std::lock_guard<std::mutex> guard(pool_mutex_);
  available_handles_.push_back(h);
}

cusparseHandle_t CusparseHandlePool::create_handle() {
  CHECK_CUDA(cuCtxPushCurrent(ctx_));
  cusparseHandle_t h;
  cusparseStatus_t status = cusparseCreate(&h);
  CHECK_CUDA(cuCtxPopCurrent(nullptr));

  if (status != CUSPARSE_STATUS_SUCCESS)
    throw std::runtime_error("cusparseCreate failed");
  return h;
}

// ---------------------- DeviceCusparseHandleTable ----------------------

DeviceCusparseHandleTable::DeviceCusparseHandleTable(CUcontext ctx, size_t pool_size)
    : pool_(ctx, pool_size) {}

cusparseHandle_t DeviceCusparseHandleTable::get_handle(int actor_id, CUstream stream) {
  cusparseHandle_t h = nullptr;
  {
    std::shared_lock<std::shared_mutex> read_lock(table_mutex_);
    auto it = table_.find(actor_id);
    if (it != table_.end())
      h = it->second;
  }

  if (!h) {
    std::unique_lock<std::shared_mutex> write_lock(table_mutex_);
    auto it = table_.find(actor_id);
    if (it != table_.end()) {
      h = it->second;
    } else {
      h = pool_.acquire();
      table_[actor_id] = h;
    }
  }

  // Always bind the handle to the caller's stream
  cusparseStatus_t status = cusparseSetStream(h, stream);
  if (status != CUSPARSE_STATUS_SUCCESS)
    throw std::runtime_error("cusparseSetStream failed");

  return h;
}

void DeviceCusparseHandleTable::release_handle(int actor_id) {
  std::unique_lock<std::shared_mutex> write_lock(table_mutex_);
  auto it = table_.find(actor_id);
  if (it != table_.end()) {
    pool_.release(it->second);
    table_.erase(it);
  }
}

} // namespace caf::cuda
