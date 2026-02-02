#pragma once
#include <chrono>
#include <iostream>


namespace caf::cuda {
struct scoped_timer {
    const char* name;
    std::chrono::steady_clock::time_point start;

    explicit scoped_timer(const char* n)
        : name(n), start(std::chrono::steady_clock::now()) {}

    ~scoped_timer() {
        auto end = std::chrono::steady_clock::now();
        auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "[PROFILE] " << name << " took "
                  << us << " us\n";
    }
};

} //namespace caf::cuda

