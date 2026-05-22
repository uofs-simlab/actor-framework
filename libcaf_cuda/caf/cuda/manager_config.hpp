#pragma once
/*
 * A class that is meant to enable users to configure 
 * the caf cuda library
 */

namespace caf::cuda {

class manager_config {
public:
    manager_config() : actorBLAS(false), actorSparse(false) {}
    manager_config(bool blas) : actorBLAS(blas), actorSparse(false) {}
    manager_config(bool blas, bool sparse) : actorBLAS(blas), actorSparse(sparse) {}
    
    bool getActorBLAS() const { return actorBLAS; }
    bool getActorSparse() const { return actorSparse; }

private:
    bool actorBLAS;
    bool actorSparse = false;
};

} // namespace caf::cuda
