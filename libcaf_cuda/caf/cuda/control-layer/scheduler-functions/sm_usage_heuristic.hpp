#pragma once

#include <cmath>
#include "caf/cuda/device.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/control-layer/scheduler-functions/heuristic_function.hpp"

namespace caf::cuda {

class sm_usage_heuristic : public heuristic_function {
public:
  explicit sm_usage_heuristic(device_ptr dev)
    : dev_(dev) {}

  int getCost(const program_ptr& prog,
              const nd_range& range) override {
    try {
      const std::string key =
        prog->getName() + range.to_string();

      auto it = values_.find(key);
      if (it != values_.end())
        return it->second;

      int total_blocks =
        static_cast<int>(range.get_num_blocks());

      int blocks_per_sm =
        dev_->max_active_blocks_per_sm(prog, range);

      if (blocks_per_sm <= 0)
        return ERROR_CODE;

      int sms_needed =
        (total_blocks + blocks_per_sm - 1) / blocks_per_sm; // ceil

      int sms_used =
        std::min(dev_->num_sms(), sms_needed);

      values_[key] = sms_used;
      return sms_used;
    }
    catch (...) {
      return ERROR_CODE;
    }
  }

  int getCost(const token_ptr& tok) override {
    return heuristic_function::getCost(tok);
  }

private:
  device_ptr dev_;
};

} // namespace caf::cuda

