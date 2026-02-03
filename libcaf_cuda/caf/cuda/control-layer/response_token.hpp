#pragma once
#include "caf/cuda/control-layer/token.hpp"
#include <caf/actor.hpp>
#include "caf/cuda/global_export.hpp"

namespace caf::cuda {

// -----------------------------------------------------------------------------
// Base class for all response tokens (memory, launch, etc.)
// Abstracts common attributes like device, stream, and memory usage
// -----------------------------------------------------------------------------
class CAF_CUDA_EXPORT response_token : public token {
public:
   
    response_token() = default;

    response_token(caf::actor receiver, int device_num, int stream_id, int memory_size = 0)
        : receiver_(std::move(receiver)),
          device_number_(device_num),
          stream_id_(stream_id),
          memory_size_(memory_size) {}

    virtual ~response_token() = default;

    // Common getters
    int getDeviceNumber() const { return device_number_; }
    int getStreamId() const { return stream_id_; }
    int memorySize() const { return memory_size_; }
    const caf::actor& getReceiver() const { return receiver_; }
    virtual const std::string& name() const { return default_name;}

    // Pure virtual: children must implement release()
    virtual void release() = 0;

protected:
    caf::actor receiver_;
    int device_number_{0};
    int stream_id_{0};
    int memory_size_{0};  // abstracted memory usage / size
    std::string default_name = "unknown";
};

// Typedef for convenience
using response_token_ptr = caf::intrusive_ptr<response_token>;

} // namespace caf::cuda

