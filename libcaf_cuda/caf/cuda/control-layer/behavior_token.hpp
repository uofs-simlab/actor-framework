#pragma once
#include "caf/cuda/control-layer/token.hpp"
#include <string>

namespace caf::cuda {

class behavior_token : public token {
public:
    explicit behavior_token(std::string n) : name_(std::move(n)) {}

    behavior_token() = default;
    const std::string& name() const { return name_; }

    // override getType() from token
    int getType() const override { return BEHAVIOR; }

private:
    std::string name_;
};

using behavior_token_ptr = caf::intrusive_ptr<behavior_token>;

} // namespace caf::cuda

