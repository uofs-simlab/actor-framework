/*
 * A class that is meant to enable users to configure 
 * the caf cuda library
 */

namespace caf::cuda {

class manager_config {
public:
    manager_config() : scheduler_on(false) {}      // initialize the bool
    manager_config(bool scheduler) : scheduler_on(scheduler) {}

    bool getSchedulerOn() const { return scheduler_on; } // should be const

private:
    bool scheduler_on;
};

} // namespace caf::cuda

