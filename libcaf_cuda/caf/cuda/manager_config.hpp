#pragma once
/*
 * A class that is meant to enable users to configure 
 * the caf cuda library
 */

namespace caf::cuda {

class manager_config {
public:
    manager_config() : actorBLAS(false), actorSparse(false), num_scheduler_streams(500), scheduler_stream_depth(1) {}
    manager_config(bool blas) : actorBLAS(blas), actorSparse(false), num_scheduler_streams(500), scheduler_stream_depth(1) {}
    manager_config(bool blas, bool sparse) : actorBLAS(blas), actorSparse(false), num_scheduler_streams(500), scheduler_stream_depth(1) {}
    manager_config(bool blas, bool sparse, int num_streams, int stream_depth) : actorBLAS(blas), actorSparse(sparse), num_scheduler_streams(num_streams), scheduler_stream_depth(stream_depth) {}
    
    bool getActorBLAS() const { return actorBLAS; }
    bool getActorSparse() const { return actorSparse; }
    int getNumSchedulerStreams() const { return num_scheduler_streams; }
    int getSchedulerStreamDepth() const { return scheduler_stream_depth; }

private:
    bool actorBLAS;
    bool actorSparse = false;
    int num_scheduler_streams;
    int scheduler_stream_depth;
};

} // namespace caf::cuda
