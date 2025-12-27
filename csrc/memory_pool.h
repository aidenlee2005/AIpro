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

#define MAX_POOL_SIZE_MB 1024  // Maximum pool size in MB
#define MAX_BLOCKS_PER_SIZE 100  // Maximum blocks per size class
#define MEMORY_ALIGNMENT 256  // Memory alignment in bytes

// CUDA 内存池类，用于复用内存块减少分配开销
class MemoryPool {
private:
    std::unordered_map<size_t, std::deque<void*>> pool;// 按对齐大小分组的空闲内存块队列
    std::unordered_map<void*, std::pair<size_t, size_t>> allocated_blocks; // ptr -> (size, aligned_size)
    std::mutex mutex;    // 线程同步锁
    size_t total_allocated_bytes = 0;     // 当前池中总分配字节数
    size_t max_pool_bytes = MAX_POOL_SIZE_MB * 1024 * 1024;    // 池的最大字节数

    MemoryPool() {}

    // 计算对齐后的内存大小
    size_t get_aligned_size(size_t size) {
        if (size == 0) return 0;
        
        // 小块对齐到64字节
        if (size <= 1024) {
            return (size + 63) & ~63; 
        // 中块256
        } else if (size <= 4096) {
            return (size + 255) & ~255; 
        // 大块4kb
        } else if (size <= 65536) {
            return (size + 4095) & ~4095;
        // 超大块1MB
        } else {
            return (size + 1048575) & ~1048575; 
        }
    }

    // 如果需要，驱逐大块内存以腾出空间
    void evict_if_needed(size_t needed_size) {
        // 循环直到有足够空间或池为空
        while (total_allocated_bytes + needed_size > max_pool_bytes && !pool.empty()) {
            size_t max_size = 0;
            for (auto& pair : pool) {
                if (!pair.second.empty() && pair.first > max_size) {
                    max_size = pair.first;  // 找到最大的非空大小类别
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
                break; 
            }
        }
    }

public:
    // 删除拷贝构造和赋值，单例模式
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    // 获取单例实例
    static MemoryPool& instance() {
        static MemoryPool pool;
        return pool;
    }

    // 分配内存，优先复用池中块
    void* allocate(size_t size) {
        if (size == 0) return nullptr;
        
        size_t aligned_size = get_aligned_size(size);
        std::lock_guard<std::mutex> lock(mutex);
        
        // 尝试复用
        auto it = pool.find(aligned_size);
        if (it != pool.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            allocated_blocks[ptr] = {size, aligned_size};
            return ptr;
        }
        // 驱逐
        evict_if_needed(aligned_size);
        
        // 新分配内存
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

    // 释放内存，回收到池中复用
    void deallocate(void* ptr, size_t size) {
        if (ptr == nullptr) return;
        
        std::lock_guard<std::mutex> lock(mutex);
        auto it = allocated_blocks.find(ptr);
        if (it == allocated_blocks.end()) {
            // 非池中分配，直接释放
            cudaFree(ptr);
            return;
        }
        
        size_t aligned_size = it->second.second;
        allocated_blocks.erase(it);
        
        // 尝试回收到池中
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
        
        // 池满，直接释放
        cudaFree(ptr);
        total_allocated_bytes -= aligned_size;
    }

    // 清空池，释放所有内存
    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        // 释放池中所有块
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
    
    // 获取池的统计信息
    void get_stats(size_t& total_bytes, size_t& num_blocks) {
        std::lock_guard<std::mutex> lock(mutex);
        total_bytes = total_allocated_bytes;
        num_blocks = 0;
        for (auto& pair : pool) {
            num_blocks += pair.second.size();
        }
    }
    
    // 析构时清空池
    ~MemoryPool() {
        clear();
    }
};

#endif
