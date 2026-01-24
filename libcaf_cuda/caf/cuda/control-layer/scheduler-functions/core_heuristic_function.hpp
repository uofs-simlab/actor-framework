#pragma once

#include "caf/cuda/program.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/device.hpp"
#include "caf/cuda/control-layer/scheduler-functions.hpp"

#include <unordered_map>
#include <string>

namespace caf::cuda {

// Concrete heuristic based on GPU occupancy / device properties
class cost_heuristic_function : public heuristic_function {
public:
  /// Construct with device (no prior state)
  explicit cost_heuristic_function(device_ptr dev)
    : dev_(dev) {
    init_device_properties();
  }

  /// Construct with device + copy state from another heuristic
  cost_heuristic_function(device_ptr dev,
                          const heuristic_function& other)
    : heuristic_function(other),
      dev_(dev) {
    init_device_properties();
  }

  /// Virtual destructor
  ~cost_heuristic_function() override = default;

  /// Returns cached or computed cost
  int getCost(const program_ptr& prog,
              const nd_range& range) const override {

    const std::string key =
      prog->getName() + "_" + range.to_string();

    auto it = values_.find(key);
    if (it != values_.end())
      return it->second;

    // Compute cost (occupancy-based for now)
    int cost = dev_->max_active_blocks_per_sm(prog, range);

    values_[key] = cost;
    return cost;
  }

private:
  void init_device_properties() {
    sm_count_            = dev_->sm_count();
    warp_size_           = dev_->warp_size();
    max_threads_per_sm_  = dev_->max_threads_per_sm();
    warps_per_sm_        = max_threads_per_sm_ / warp_size_;
    total_warps_         = sm_count_ * warps_per_sm_;
    total_mem_bytes_     = dev_->total_global_mem();
  }

private:
  device_ptr dev_;

  // Mutable because getCost() is logically const but caches values
  mutable std::unordered_map<std::string, int> values_;

  // Cached GPU properties
  int sm_count_ = 0;
  int warp_size_ = 0;
  int max_threads_per_sm_ = 0;
  int warps_per_sm_ = 0;
  int total_warps_ = 0;
  std::size_t total_mem_bytes_ = 0;
};

} // namespace caf::cuda

