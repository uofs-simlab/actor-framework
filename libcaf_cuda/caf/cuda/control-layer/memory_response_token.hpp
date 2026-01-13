#pragma once

//this class is meant to act as a response issued by the scheduler
//actor authorizing memory transfer as needed 

#pragma once

#include "caf/cuda/control-layer/token.hpp"
#include "caf/cuda/control-layer/memory_transfer_token.hpp"
#include "caf/cuda/global_export.hpp"

#include <caf/actor.hpp>
#include <atomic>

namespace caf::cuda {

class CAF_CUDA_EXPORT memory_response_token : public token {
public:
    // Required by CAF – do not use directly
    memory_response_token() = default;

    // Construct from memory_transfer_token
    memory_response_token(caf::actor receiver,
                          const memory_transfer_token& token,
			  int device_num,
			  int streamId)
        : receiver_(std::move(receiver)),
          size_(token.getSize()),
          direction_(token.getDirection()),
          released_(false),
       	  device_number(device_num),
	  stream_id(streamId) {}


    ~memory_response_token() {
        release();
    }

    int getType() const override {
        return MEMORY_RESPONSE;
    }

    int getSize() const {
        return size_;
    }

    int getDirection() const {
        return direction_;
    }

    int getDeviceNumber() const { return device_number;}
    int getStreamId() const {return stream_id;}

    void release() {
        bool expected = false;

        // ONLY send if we successfully transition false → true
        if (!released_.compare_exchange_strong(expected, true)) {
            return; // already released → do nothing
        }

        // Real message (commented for testing)
        // caf::anon_mail(size_, direction_).urgent().send(receiver_);

        // Test message
        caf::anon_mail("Hello world from memory response")
            .urgent()
            .send(receiver_);
    }

private:
    caf::actor receiver_;
    int size_;
    int direction_;
    std::atomic<bool> released_;
    int device_number;
    int stream_id;
};


using memory_token = caf::intrusive_ptr<memory_response_token>;


} // namespace caf::cuda

