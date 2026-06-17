#pragma once

#include <string>
#include <functional>
#include <vector>
#include <unordered_map>
#include <atomic>

#include <caf/ref_counted.hpp>
#include "caf/cuda/global.hpp"
#include "caf/cuda/platform.hpp"
#include "caf/cuda/device.hpp"

namespace caf::cuda {

// Class that represents a CUDA kernel
class CAF_CUDA_EXPORT program : public caf::ref_counted {
public:
  /// Construct a program from binary data or PTX.
  /// Loads the kernel on all devices.
  /// @param name Name of the kernel function.
  /// @param binary Binary data (fatbin or PTX).
  /// @param is_fatbin Whether the binary is a fatbinary (default: false).
  program(std::string name, std::vector<char> binary, bool is_fatbin = false);

  /// Returns the CUfunction for a given device.
  /// @throws std::runtime_error if the kernel was not loaded for the device.
  CUfunction get_kernel(int device_id);

    friend void intrusive_ptr_add_ref([[maybe_unused]] const program* p) noexcept {
        //p->ref_count_.fetch_add(1, std::memory_order_relaxed);
    }
    friend void intrusive_ptr_release(const program* p) noexcept {
        if (p->ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
	   //WARNING TURNING THIS ON FOR SOME REASON, CAUSES SEGFAUTLS
	   //I HAVE NO IDEA WHY 
	   //TODO FIX THIS 
	   // std::cout<< "Deleting\n";
           // delete p;
	}
    }

    std::string getName() {return name_;}

    int getHash() const {return hashValue;}


private:
  /// Internal helper to load the kernel modules on all devices.
  void load_kernels(bool is_fatbin);

  std::string name_;                       ///< Name of the kernel
  std::vector<char> binary_;               ///< The binary or PTX of the program
  std::unordered_map<int, CUfunction> kernels_; ///< Device ID -> CUfunction mapping
  mutable std::atomic<size_t> ref_count_{0};
  std::hash<std::string> hasher;
  int hashValue = 0;

};

/// Alias for an intrusive pointer to a program
using program_ptr = caf::intrusive_ptr<program>;

} // namespace caf::cuda
