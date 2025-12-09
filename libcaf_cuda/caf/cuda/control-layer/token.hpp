#pragma once
#include <atomic>
#include "caf/all.hpp"

//types of tokens
#define LAUNCH 1
#define LAUNCH_RESPONSE 2
#define BEHAVIOR 3



namespace caf::cuda {

// Base token interface
class token : public caf::ref_counted {
public:
    virtual ~token() = default;
    virtual int getType() = 0;

protected:
    std::atomic<size_t> ref_count_{0};

    friend void intrusive_ptr_add_ref(const token* p) noexcept {
        p->ref_count_.fetch_add(1, std::memory_order_relaxed);
    }

    friend void intrusive_ptr_release(const token* p) noexcept {
        if (p->ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1)
            delete p;
    }
};

using token_ptr = caf::intrusive_ptr<token>;

} // namespace caf::cuda
