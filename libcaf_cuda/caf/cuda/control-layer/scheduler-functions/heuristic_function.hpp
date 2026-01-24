#pragma once

#include "caf/cuda/program.hpp"
#include "caf/cuda/nd_range.hpp"

#include <unordered_map>
#include <string>


#define ERROR_CODE -999999


namespace caf::cuda {

// Abstract heuristic function object.
// Subclasses estimate the "cost" of launching a kernel based on
// the program and execution configuration.
class heuristic_function {
public:
  /// Default constructor
  heuristic_function() = default;

  /// Copy constructor from another heuristic_function
  /// Copies all internal heuristic values 
  heuristic_function(const heuristic_function& other)
    : values_(other.values_) {}

  /// Virtual destructor (required for polymorphic base classes)
  virtual ~heuristic_function() = default;

  /// Abstract cost function
  /// @param prog Kernel program
  /// @param range Kernel execution configuration
  /// @return Cost metric (interpretation left to implementation)
  virtual int getCost(const program_ptr& prog,
                      const nd_range& range) = 0;

protected:
  /// Heuristic-specific values
  std::unordered_map<std::string, int> values_;
};

} // namespace caf::cuda

