#pragma once
#include "caf/cuda/control-layer/token.hpp"
#include "caf/cuda/control-layer/behavior_token.hpp"
#include <unordered_map>
#include <string>

namespace caf::cuda {
class scheduler_actor_behavior;  // Forward decl
class behavior_token;

class behavior_table {
public:
    behavior_table() = default;
    void add(const std::string& name, scheduler_actor_behavior* beh) {
        table_[name] = beh;
    }
    scheduler_actor_behavior* get(const behavior_token& tok) const {
        auto it = table_.find(tok.name());
        return it != table_.end() ? it->second : nullptr;
    }
    auto& all_behaviors() { return table_; }
private:
    std::unordered_map<std::string, scheduler_actor_behavior*> table_;
};
} // namespace caf::cuda
