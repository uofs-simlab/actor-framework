#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/control-layer/scheduler_actor.hpp"
#include <iostream>

#include <random> // Required for random number generation
#include <algorithm> // Required for std::remove_if if filtering self from neighbors
namespace caf::cuda {

scheduler_actor::scheduler_actor(caf::actor_config& cfg, 
                                 int device_number, int num_streams, 
                                 int stream_depth, bool multi_gpu)
    : caf::event_based_actor(cfg),
      device_number_(device_number),
      num_streams_(num_streams),
      stream_depth_(stream_depth),
      multi_gpu_(multi_gpu),
      rng_(std::random_device{}()) {
    // Initialize the pool of available execution slots
    for (int depth = 0; depth < stream_depth_; ++depth) {
        for (int s = 0; s < num_streams_; ++s) {
            available_streams_.push(s);
        }
    }
}

caf::behavior scheduler_actor::make_behavior() {
    return {
        [this](const token_ptr& tok) {
            on_receive(tok);
        },
        [this](std::vector<token_ptr> tokens) {
            on_receive_batch(std::move(tokens));
        },
        [this](int val, int mem, int time, int dep, int stream_id) {
            on_reclaim(val, mem, time, dep, stream_id);
        },
        [this](std::vector<caf::actor> neighbors) {
            on_set_neighbors(std::move(neighbors));
        },
        [this](int requesting_device) -> std::vector<token_ptr> {
            return on_steal_request(requesting_device);
        },
        [this](std::string word) {
             std::cout << "Scheduler " << device_number_ << " received: " << word << "\n";
        }
    };
}

void scheduler_actor::on_receive(const token_ptr& tok) {
    if (tok->getType() == LAUNCH) {
        queue_.push(tok);
        schedule_work();
    } else if (tok->getType() == MEMORY) {
        const auto& mem = static_cast<const memory_transfer_token&>(*tok);
        auto response = make_memory_response_token(this, mem, device_number_, 0);
        this->mail(response).send(mem.getReplyActor());
    }
}

void scheduler_actor::on_receive_batch(std::vector<token_ptr> tokens) {
    for (auto& tok : tokens) {
        queue_.push(std::move(tok));
    }
    schedule_work();
}

void scheduler_actor::on_reclaim(int /*val*/, int /*mem*/, int /*time*/, int /*dep*/, int stream_id) {
    if (in_flight_ > 0) {
        in_flight_--;
    }
    available_streams_.push(stream_id);
    schedule_work();
}

void scheduler_actor::on_set_neighbors(std::vector<caf::actor> neighbors) {
    schedulers_ = std::move(neighbors);
    victims_.clear();
    auto self_handle = caf::actor_cast<caf::actor>(this);
    for (const auto& neighbor : schedulers_) {
        if (neighbor != self_handle) {
            victims_.push_back(neighbor);
        }
    }
}

std::vector<token_ptr> scheduler_actor::on_steal_request(int requesting_device) {
    // size_t min_giveaway_threshold = static_cast<size_t>(num_streams_) * stream_depth_ * 2;
    size_t min_giveaway_threshold = static_cast<size_t>(num_streams_) * stream_depth_;

    if (queue_.size() > min_giveaway_threshold) {
        size_t to_give = queue_.size() / 2;
        std::vector<token_ptr> batch;
        for (size_t i = 0; i < to_give; ++i) {
            batch.push_back(queue_.front());
            queue_.pop();
        }
        std::cout << "[INFO] Scheduler " << device_number_ 
                  << " sharing " << batch.size() 
                  << " tasks with device " << requesting_device << "\n";
        return batch;
    }
    return {};
}

void scheduler_actor::schedule_work() {
    int capacity = num_streams_ * stream_depth_;

    // Dispatch queued tasks while we have capacity
    while (!queue_.empty() && !available_streams_.empty()) {
        auto tok = queue_.front();
        queue_.pop();

        if (tok->getType() == LAUNCH) {
            in_flight_++;
            const auto& launch = static_cast<const launch_token&>(*tok);

            // Take an available stream ID from the pool
            int stream_id = available_streams_.front();
            available_streams_.pop();
            
            auto response = make_launch_response_token(this, launch, device_number_, stream_id);
            this->mail(response).send(launch.getReplyActor());
        }
    }

    // Work Stealing logic
    if (multi_gpu_ && queue_.empty() && in_flight_ < (capacity / 2)) {
        // Randomly select a neighbor to request work from using the pre-generated victims list
        if (!victims_.empty()) {
            std::uniform_int_distribution<size_t> distrib(0, victims_.size() - 1);
            size_t idx = distrib(rng_);
            this->mail(device_number_).send(victims_[idx]);
        }
    }
}

} // namespace caf::cuda
