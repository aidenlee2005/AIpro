#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <unordered_map>
#include <mutex>
#include <iostream>
#include <chrono>
#include <deque>

// Memory pool configuration
#define MAX_POOL_SIZE_MB 1024  // Maximum pool size in MB
#define MAX_BLOCKS_PER_SIZE 100  // Maximum blocks per size class
#define MEMORY_ALIGNMENT 256  // Memory alignment in bytes

class MemoryPool {
private:
    // Use unordered_map for better performance
    std::unordered_map<size_t, std::deque<void*>> pool;
    std::unordered_map<void*, std::pair<size_t, size_t>> allocated_blocks; // ptr -> (size, aligned_size)
    std::mutex mutex;
    size_t total_allocated_bytes = 0;
    size_t max_pool_bytes = MAX_POOL_SIZE_MB * 1024 * 1024;

    MemoryPool() {}

    // Improved alignment strategy for better cache performance
    size_t get_aligned_size(size_t size) {
        if (size == 0) return 0;
        
        // For small allocations, align to cache line (64 bytes) or power of 2
        if (size <= 1024) {
            return (size + 63) & ~63;  // 64-byte alignment
        } else if (size <= 4096) {
            return (size + 255) & ~255;  // 256-byte alignment
        } else if (size <= 65536) {
            return (size + 4095) & ~4095;  // 4KB alignment
        } else {
            return (size + 1048575) & ~1048575;  // 1MB alignment for large blocks
        }
    }

    // Check if we should evict old blocks to prevent unlimited growth
    void evict_if_needed(size_t needed_size) {
        while (total_allocated_bytes + needed_size > max_pool_bytes && !pool.empty()) {
            // Find a size class to evict from (prefer larger sizes)
            size_t max_size = 0;
            for (auto& pair : pool) {
                if (!pair.second.empty() && pair.first > max_size) {
                    max_size = pair.first;
                }
            }
            
            if (max_size > 0) {
                auto& block_list = pool[max_size];
                void* ptr = block_list.front();
                block_list.pop_front();
                cudaFree(ptr);
                total_allocated_bytes -= max_size;
                
                if (block_list.empty()) {
                    pool.erase(max_size);
                }
            } else {
                break; // No blocks to evict
            }
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
        if (size == 0) return nullptr;
        
        size_t aligned_size = get_aligned_size(size);
        std::lock_guard<std::mutex> lock(mutex);
        
        // Try to reuse from pool first
        auto it = pool.find(aligned_size);
        if (it != pool.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            allocated_blocks[ptr] = {size, aligned_size};
            return ptr;
        }
        
        // Check if we need to evict old blocks
        evict_if_needed(aligned_size);
        
        // Allocate new memory
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, aligned_size);
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed for size " << aligned_size << " (requested " << size << "): " << cudaGetErrorString(err) << std::endl;
            return nullptr; 
        }
        
        total_allocated_bytes += aligned_size;
        allocated_blocks[ptr] = {size, aligned_size};
        return ptr;
    }

    void deallocate(void* ptr, size_t size) {
        if (ptr == nullptr) return;
        
        std::lock_guard<std::mutex> lock(mutex);
        auto it = allocated_blocks.find(ptr);
        if (it == allocated_blocks.end()) {
            // Not allocated by this pool, free directly
            cudaFree(ptr);
            return;
        }
        
        size_t aligned_size = it->second.second;
        allocated_blocks.erase(it);
        
        // Check if we should keep this block in pool
        auto pool_it = pool.find(aligned_size);
        if (pool_it != pool.end()) {
            if (pool_it->second.size() < MAX_BLOCKS_PER_SIZE) {
                pool_it->second.push_back(ptr);
                return;
            }
        } else {
            pool[aligned_size].push_back(ptr);
            return;
        }
        
        // Pool is full for this size, free the memory
        cudaFree(ptr);
        total_allocated_bytes -= aligned_size;
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
        allocated_blocks.clear();
        total_allocated_bytes = 0;
    }
    
    // Get pool statistics
    void get_stats(size_t& total_bytes, size_t& num_blocks) {
        std::lock_guard<std::mutex> lock(mutex);
        total_bytes = total_allocated_bytes;
        num_blocks = 0;
        for (auto& pair : pool) {
            num_blocks += pair.second.size();
        }
    }
    
    ~MemoryPool() {
        clear();
    }
};

#endif
