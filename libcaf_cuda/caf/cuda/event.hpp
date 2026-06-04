#pragma once

#include <cuda.h>
#include <iostream>
#include <cstdlib>

#include <caf/ref_counted.hpp>
#include <caf/intrusive_ptr.hpp>

namespace caf::cuda {

class event;
using event_ptr = caf::intrusive_ptr<event>;

/**
 * @brief A smart-pointer managed wrapper for a CUDA event.
 * Ensures that the CUevent handle is destroyed when the last reference is gone.
 */
class event : public caf::ref_counted {
public:
  explicit event(unsigned int flags = CU_EVENT_DEFAULT) {
    // Note: Expects an active CUDA context for creation.
    check_error(cuEventCreate(&event_, flags), "cuEventCreate");
  }

  ~event() {
    check_error(cuEventDestroy(event_), "cuEventDestroy");
  }

  event(const event&) = delete;
  event& operator=(const event&) = delete;

  CUevent get() const {
    return event_;
  }

  /// Returns true if the event has been recorded and the work has completed.
  bool query() const {
    CUresult res = cuEventQuery(event_);
    if (res == CUDA_SUCCESS)
      return true;
    if (res == CUDA_ERROR_NOT_READY)
      return false;
    check_error(res, "cuEventQuery");
    return false;
  }

  /// Blocks the calling thread until the event has completed.
  void synchronize() const {
    check_error(cuEventSynchronize(event_), "cuEventSynchronize");
  }

private:
  static void check_error(CUresult result, const char* msg) {
    if (result != CUDA_SUCCESS) {
      const char* err_str = nullptr;
      cuGetErrorString(result, &err_str);
      std::cerr << "CUDA Driver API Error (" << msg << "): " 
                << (err_str ? err_str : "unknown error") << "\n";
      std::exit(1);
    }
  }

  CUevent event_;
};

} // namespace caf::cuda