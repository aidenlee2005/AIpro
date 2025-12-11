#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <mutex>
#include <iostream>

class MemoryPool {
private:
    // Map from size to a list of free pointers
    std::map<size_t, std::vector<void*>> pool;
    std::mutex mutex;

    MemoryPool() {}

public:
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    static MemoryPool& instance() {
        static MemoryPool pool;
        return pool;
    }

    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = pool.find(size);
        if (it != pool.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            // cudaMemset(ptr, 0, size); // Removed for performance
            return ptr;
        }
        
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed for size " << size << ": " << cudaGetErrorString(err) << std::endl;
            return nullptr; 
        }
        return ptr;
    }

    void deallocate(void* ptr, size_t size) {
        if (ptr == nullptr) return;
        std::lock_guard<std::mutex> lock(mutex);
        pool[size].push_back(ptr);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto& pair : pool) {
            for (void* ptr : pair.second) {
                cudaFree(ptr);
            }
            pair.second.clear();
        }
        pool.clear();
    }
    
    ~MemoryPool() {
        clear();
    }
};

#endif
