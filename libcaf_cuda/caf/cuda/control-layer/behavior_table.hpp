#include "caf/cuda/control-layer/token.hpp"
#include "caf/cuda/control-layer/behavior.hpp"
#include "caf/cuda/control-layer/behavior_token.hpp"
#include <unordered_map>
#include <string>

namespace caf::cuda {

class behavior_table {
public:
    behavior_table() {
        // TODO insert behaviors here
    }

    /*
     * Given a behavior_token, return a scheduler_actor_behavior*.
     * Returns nullptr if nothing is found.
     */
    scheduler_actor_behavior* getBehavior(behavior_token tok) {

        auto it = table.find(tok.getBehavior());

        if (it != table.end())
            return it->second;

        return nullptr;
    }

private:
    std::unordered_map<std::string, scheduler_actor_behavior*> table;
};

} // namespace caf::cuda

