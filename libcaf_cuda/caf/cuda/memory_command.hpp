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

} // namespace caf::cuda

