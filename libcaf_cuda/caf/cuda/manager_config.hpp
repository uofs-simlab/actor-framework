#pragma once
/*
 * A class that is meant to enable users to configure 
 * the caf cuda library
 */

namespace caf::cuda {

class manager_config {
public:
    manager_config() : actorBLAS(false), memory_manager_on(false) {}
    manager_config(bool blas) : actorBLAS(blas), memory_manager_on(false) {}
    manager_config(bool blas, bool memory_manager) : actorBLAS(blas), memory_manager_on(memory_manager) {}
    
    bool getActorBLAS() const { return actorBLAS; }
    bool getMemoryManagerOn() const { return memory_manager_on; }

private:
    bool actorBLAS;
    bool memory_manager_on = false;
};

} // namespace caf::cuda
