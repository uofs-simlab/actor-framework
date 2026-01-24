#pragma once

#include <string>
#include <exception>

#include "caf/cuda/device.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/control-layer/scheduler-functions/heuristic_function.hpp"

namespace caf::cuda {

class core_heuristic_function : public heuristic_function {
public:
  explicit core_heuristic_function(device_ptr dev)
    : dev_(dev) {}

  // Copy from ANY heuristic_function
  core_heuristic_function(device_ptr dev,
                          const heuristic_function& other)
    : heuristic_function(other),
      dev_(dev) {}

  int getCost(const program_ptr& prog,
              const nd_range& range) override {
    try {
      const std::string key =
        prog->getName() + range.to_string();

      auto it = values_.find(key);
      if (it != values_.end())
        return it->second;

      int cost = dev_->max_active_blocks_per_sm(prog, range);
      values_[key] = cost;
      return cost;
    }
    catch (const std::exception&) {
      return ERROR_CODE;
    }
    catch (...) {
      return ERROR_CODE;
    }
  }

private:
  device_ptr dev_;
};

} // namespace caf::cuda

