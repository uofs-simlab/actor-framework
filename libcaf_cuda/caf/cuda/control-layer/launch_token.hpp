#pragma once
#include "caf/scheduler/token.hpp"
#include "caf/cuda/all.hpp"
#include <string>

namespace caf::cuda {

class launch_token : public token {
public:
    launch_token(program_ptr prog,
                 nd_range range,
                 int memory_usage,
                 std::string id)
        : program_(std::move(prog)),
          range_(std::move(range)),
          memory_usage_(memory_usage),
          id_(std::move(id)) {}

    int getType() override { return 1; }

    const program_ptr& getProgram() const { return program_; }
    const nd_range& getRange() const { return range_; }
    int getMemoryUsage() const { return memory_usage_; }

    // Return requested number of CUDA blocks (gridDimX * gridDimY * gridDimZ)
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
};

} // namespace caf::cuda

