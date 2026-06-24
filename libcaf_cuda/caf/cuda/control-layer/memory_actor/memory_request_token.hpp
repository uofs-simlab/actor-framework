#pragma once


/*
 * This token represents a request for memory allocation
 * on a specific device. All sizes are in bytes.
 */

namespace caf::cuda {

class CAF_CUDA_EXPORT memory_request_token {
public:
  memory_request_token(std::size_t size,
                       int device_number,
                       caf::actor replyActor)
    : 
      size_(size),
      device_number_(device_number),
      replyActor_(replyActor) {}

  // CAF compliance
  memory_request_token() = default;

  int getType() const { return MEMORY; }

  std::size_t getSize() const { return size_; }
  int getDeviceNumber() const { return device_number_; }
  caf::actor getReplyActor() const { return replyActor_; }

private:
  std::size_t size_;
  int device_number_;
  caf::actor replyActor_;
};

} // namespace caf::cuda
