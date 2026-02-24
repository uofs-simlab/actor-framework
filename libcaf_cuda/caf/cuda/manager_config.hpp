#pragma once
/*
 * A class that is meant to enable users to configure 
 * the caf cuda library
 */

namespace caf::cuda {

class manager_config {
public:
    manager_config() : scheduler_on(false) {}      // initialize the bool
    manager_config(bool scheduler) : scheduler_on(scheduler) {}
    manager_config(bool scheduler, bool memory_manager) : scheduler_on(scheduler), memory_manager_on(memory_manager) {}
    
    bool getSchedulerOn() const { return scheduler_on; }
    bool getMemoryManagerOn() const { return memory_manager_on; }

private:
    bool scheduler_on;
    bool memory_manager_on = false;
};

} // namespace caf::cuda

