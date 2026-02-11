#pragma once
#include <unordered_map>
#include <string>

namespace caf::cuda {

class scheduler_actor_behavior; // Forward declaration is fine here
class behavior_token;
class scheduler_actor_state;

class behavior_table {
public:
       	behavior_table() = default;
      // Constructor that creates all default behaviors for a given actor state
    explicit behavior_table(scheduler_actor_state& state);
    

 
    ~behavior_table();
        

    void add(const std::string& name, scheduler_actor_behavior* beh) {
        table_[name] = beh;
    }

    scheduler_actor_behavior* get(const behavior_token& tok) const;
    auto& all_behaviors() { return table_; }
    const auto& all_behaviors() const { return table_; }

private:
    std::unordered_map<std::string, scheduler_actor_behavior*> table_;
};

} // namespace caf::cuda
