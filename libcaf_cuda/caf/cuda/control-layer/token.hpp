#pragma once
#include <atomic>
#include <caf/intrusive_ptr.hpp>
#include <caf/ref_counted.hpp>
#include "caf/cuda/global_export.hpp"
#include <iostream>

//types of tokens
#define LAUNCH 1
#define LAUNCH_RESPONSE 2
#define BEHAVIOR 3
#define MEMORY 4
#define MEMORY_RESPONSE 5
#define TRANSFER 6

//dependency tags
#define INDEPENDENT -1

namespace caf::cuda {

// Base token interface
class CAF_CUDA_EXPORT token : public caf::ref_counted {
public:
   virtual ~token() {
    // Print ref count when destructor runs
    [[maybe_unused]] size_t count = ref_count_.load(std::memory_order_acquire);
    //std::cout << "token object getting deleted, ref_count = " << count << "\n";
}

    //should only be used by caf's type id system
  //  token() = default;

    explicit  token(int dependency = INDEPENDENT)
    : dependency_(dependency) {}

    virtual int getType() const {return -1;}

  // Dependency API
  virtual int getDependency() const { return dependency_; }
  bool isIndependent() const { return dependency_ == INDEPENDENT; }


protected:
 int dependency_ = INDEPENDENT;
 
 friend void intrusive_ptr_add_ref(token* p) noexcept {
    p->ref_count_.fetch_add(1, std::memory_order_relaxed);
  }

  friend void intrusive_ptr_release(token* p) noexcept {
    if (p->ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1)
      delete p;
  
  }

private:
 mutable std::atomic<size_t> ref_count_{0};


};


using token_ptr = caf::intrusive_ptr<token>;

} // namespace caf::cuda
