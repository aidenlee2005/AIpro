#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <mutex>
#include <iostream>

class MemoryPool {
private:
    std::map<size_t, std::vector<void*>> pool;
    std::mutex mutex;

    MemoryPool() {}

    // Helper to align size to reduce fragmentation and increase reuse
    size_t get_aligned_size(size_t size) {
        if (size <= 4096) {
            size_t aligned = 256;
            while (aligned < size) aligned <<= 1;
            return aligned;
        } else if (size < 1024 * 1024) {
            return (size + 4095) & ~4095;
        } else {
            return (size + 1048575) & ~1048575;
        }
    }

public:
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    static MemoryPool& instance() {
        static MemoryPool pool;
        return pool;
    }

    void* allocate(size_t size) {
        size_t aligned_size = get_aligned_size(size);
        std::lock_guard<std::mutex> lock(mutex);
        auto it = pool.find(aligned_size);
        if (it != pool.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            return ptr;
        }
        
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, aligned_size);
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed for size " << aligned_size << " (requested " << size << "): " << cudaGetErrorString(err) << std::endl;
            return nullptr; 
        }
        return ptr;
    }

    void deallocate(void* ptr, size_t size) {
        if (ptr == nullptr) return;
        size_t aligned_size = get_aligned_size(size);
        std::lock_guard<std::mutex> lock(mutex);
        pool[aligned_size].push_back(ptr);
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
