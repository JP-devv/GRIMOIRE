#pragma once

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "b70/cpu_experts.hpp"

namespace b70 {

// Persistent readers for the small, scattered PLE reads on the decode path.
class PleReadPool {
    std::mutex mutex_;
    std::condition_variable ready_, finished_;
    std::vector<std::thread> workers_;
    const std::function<void(size_t, size_t)>* job_ = nullptr;
    std::exception_ptr error_;
    size_t generation_ = 0, rows_ = 0, pending_ = 0;
    bool stopping_ = false;

    void work(size_t index) {
        unpin_self();
        std::unique_lock lock(mutex_);
        size_t generation = 0;
        for (;;) {
            ready_.wait(lock, [&] { return stopping_ || generation_ != generation; });
            if (stopping_) return;
            generation = generation_;
            const size_t begin = rows_ * index / workers_.size();
            const size_t end = rows_ * (index + 1) / workers_.size();
            const auto* job = job_;
            lock.unlock();
            std::exception_ptr error;
            try { (*job)(begin, end); }
            catch (...) { error = std::current_exception(); }
            lock.lock();
            if (error && !error_) error_ = error;
            if (--pending_ == 0) finished_.notify_one();
        }
    }

    void stop() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_) worker.join();
    }

public:
    explicit PleReadPool(size_t count) {
        workers_.reserve(count);
        try {
            for (size_t i = 0; i < count; ++i)
                workers_.emplace_back([this, i] { work(i); });
        } catch (...) {
            stop();
            throw;
        }
    }
    ~PleReadPool() { stop(); }

    void read(size_t rows, const std::function<void(size_t, size_t)>& job) {
        std::unique_lock lock(mutex_);
        rows_ = rows;
        job_ = &job;
        error_ = nullptr;
        pending_ = workers_.size();
        ++generation_;
        ready_.notify_all();
        finished_.wait(lock, [&] { return pending_ == 0; });
        job_ = nullptr;
        if (error_) std::rethrow_exception(error_);
    }
};

} // namespace b70
