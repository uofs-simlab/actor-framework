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



namespace caf::cuda {

// Base token interface
class CAF_CUDA_EXPORT token : public caf::ref_counted {
public:
   virtual ~token() {
    // Print ref count when destructor runs
    size_t count = ref_count.load(std::memory_order_acquire);
    std::cout << "token object getting deleted, ref_count = " << count << "\n";
}

    //should only be used by caf's type id system
    token() = default;

    virtual int getType() const {return -1;}

protected:
 mutable std::atomic<size_t> ref_count{0}; // start at 1 for make_counted

friend  void intrusive_ptr_add_ref(const token* p) noexcept {
    std::cout << "[add_ref] old count = " << p->ref_count << "\n";
    ++p->ref_count;
    std::cout << "[add_ref] new count = " << p->ref_count << "\n";
}

friend void intrusive_ptr_release(const token* p) noexcept {
    std::cout << "[release] old count = " << p->ref_count << "\n";
    --p->ref_count;
    std::cout << "[release] new count = " << p->ref_count << "\n";
    if (p->ref_count == 0) std::cout << "token object getting deleted\n";
}
};

using token_ptr = caf::intrusive_ptr<token>;

} // namespace caf::cuda
