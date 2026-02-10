#pragma once

#include <caf/all.hpp>
#include "caf/cuda/program.hpp"
#include "caf/cuda/control-layer/request_token.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/global_export.hpp"

namespace caf::cuda {

class CAF_CUDA_EXPORT launch_token : public request_token {
public:
  // CAF compliance only
  launch_token() = default;

  launch_token(program_ptr prog,
               nd_range range,
               int memory_usage,
               std::string id,
               caf::actor receiver,
               int dependency = INDEPENDENT)
    : request_token(dependency),
      program_(std::move(prog)),
      range_(std::move(range)),
      memory_usage_(memory_usage),
      id_(std::move(id)),
      reply_handle_(receiver) {}

  int getType() const override { return LAUNCH; }

  const program_ptr& getProgram() const { return program_; }
  const nd_range& getRange() const { return range_; }
  int getMemoryUsage() const { return memory_usage_; }
  caf::actor getReplyActor() const { return reply_handle_; }

  int getBlocks() const {
    return static_cast<int>(
      range_.getGridDimX() *
      range_.getGridDimY() *
      range_.getGridDimZ()
    );
  }

  const std::string& getId() const { return id_; }



private:
  program_ptr program_;
  nd_range range_;
  int memory_usage_;
  std::string id_;
  caf::actor reply_handle_;
};

} // namespace caf::cuda

