#pragma once

#include <string>
#include <exception>

#include "caf/cuda/device.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/control-layer/scheduler-functions/heuristic_function.hpp"
/*
 * Was originally supposed to estimate core usage 
 * but it just tells how much blocks can fit into 1 SM
 */


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


  //mostly here to please the c++ compiler
  int getCost(const token_ptr& tok) override {
    return heuristic_function::getCost(tok); // call base
  }



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

