#pragma once

#include <cmath>
#include "caf/cuda/device.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/control-layer/scheduler-functions/heuristic_function.hpp"
#include "caf/cuda/control-layer/scheduler-functions/profiler.hpp"

/*
 * will return a value indicating 
 * how much blocks a kernel with dimensions will consume
 */

namespace caf::cuda {

class sm_usage_heuristic : public heuristic_function {
public:
  explicit sm_usage_heuristic(device_ptr dev)
    : dev_(dev) {}

int getCost(const program_ptr& prog,
            const nd_range& range) override {

  scoped_timer timer("sm_usage_heuristic::getCost");
  try {
    int key = prog->getHash() ^ static_cast<int>(range.getHash());

    auto it = values_.find(key);
    if (it != values_.end())
      return it->second;

    int total_blocks = static_cast<int>(range.get_num_blocks());
    int blocks_per_sm = dev_->max_active_blocks_per_sm(prog, range);

    if (blocks_per_sm <= 0) {
      std::cout << "blocks_per_sm = " << blocks_per_sm << "\n"; 
      std::cout << "blocks is less than zero\n";
      return ERROR_CODE;
    }

    int sms_needed = (total_blocks + blocks_per_sm - 1) / blocks_per_sm; // ceil
    int sms_used = std::min(dev_->num_sms(), sms_needed);

    values_[key] = sms_used;
    return sms_used;
  }
  catch (const std::exception& e) {
    std::cerr << "Caught std::exception: " << e.what() << "\n";
    return ERROR_CODE;
  }
  catch (...) {
    std::cerr << "Caught unknown exception!\n";
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

