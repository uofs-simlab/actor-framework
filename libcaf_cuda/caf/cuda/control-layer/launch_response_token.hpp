#pragma once
#include "caf/cuda/control-layer/token.hpp"
#include "caf/cuda/control-layer/launch_token.hpp"  // Full include for constructor param
#include <caf/actor.hpp>  // For caf::actor
#include "caf/cuda/nd_range.hpp"  // For nd_range
#include "caf/cuda/global_export.hpp"
#include <atomic>
#include <string>  // For std::string (if not from elsewhere)

namespace caf::cuda {
class CAF_CUDA_EXPORT launch_response_token : public token {
public:

	//only here to be complaint with CAFS type if system DO NOT USE 
	launch_response_token() = default;

    // Construct manually    
launch_response_token(caf::actor receiver,
                          nd_range range,
                          int memory_usage,
                          std::string id)
        : receiver_(std::move(receiver)),
          range_(std::move(range)),
          memory_usage_(memory_usage),
          id_(std::move(id)),
          released_(false),
	  device_number(0),
	  stream_id(0) {}

    // Construct from a launch_token
    launch_response_token(caf::actor receiver, const launch_token& token,int device_num,int streamId)
        : receiver_(std::move(receiver)),
          range_(token.getRange()),
          memory_usage_(token.getMemoryUsage()),
          id_(token.getId()),
          released_(false),
	  device_number(device_num),
	  stream_id(streamId) {}

    ~launch_response_token() {
        release();
    }
    int getType() const override { return LAUNCH_RESPONSE; }
    const nd_range& getRange() const { return range_; }
    int getMemoryUsage() const { return memory_usage_; }
    int getDeviceNumber() const {return device_number;}
    int getStreamId() const {return stream_id;}

    // Return requested number of CUDA blocks
    int getBlocks() const {
        return static_cast<int>(
            range_.getGridDimX() *
            range_.getGridDimY() *
            range_.getGridDimZ()
        );
    }
    const std::string& getId() const { return id_; }
    void release() {
        bool expected = false;
        if (released_.compare_exchange_strong(expected, true)) {
            	//the real message commented out for testing 
		//caf::anon_mail(id_, getBlocks()).urgent().send(receiver_);
		//test message
		caf::anon_mail("Hello world from me").urgent().send(receiver_);
        }
    }
private:
    caf::actor receiver_;
    nd_range range_;
    int memory_usage_;
    std::string id_;
    std::atomic<bool> released_;
    int device_number;
    int stream_id;
};
} // namespace caf::cuda
