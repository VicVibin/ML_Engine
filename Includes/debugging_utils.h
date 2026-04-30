#pragma once
#include <curand_kernel.h>
#include <string>
#include <chrono>
#include <iostream>

using str = std::string;

#define XAVIER 6.0f
#define ADAMW true
#define NORM 1.0f
#define LEARNING_RATE 1e-4f
#define SHOULDNORM false

template <typename T>
void printHeadGPU(const str name, T* d_ptr, int start, int elems, int total)
{
    std::cout << "Printing dimensions for pointer: " <<name << "\n";
    T* CPU = (T *)malloc(total * sizeof(T));
    cudaMemcpy(CPU, d_ptr, sizeof(T)*total, cudaMemcpyDeviceToHost);
    for(int i = start; i < start + elems; ++i){std::cout << CPU[i] << "\t";}
    std::cout << "\n _______________________________ \n";
    free(CPU);

}

template <typename T>
void printHeadGPU(const str name, const T* X, const int ch, const int rows, const int cols, const int total)
{
    T* CPU = (T *)malloc(total * sizeof(T));
    cudaMemcpy(CPU, X, total * sizeof(T), cudaMemcpyDeviceToHost);

    std::cout << "Printing dimensions for node " << name << "->output \n";
    for (int c = 0; c < min(3,ch); ++c)
    {
        std::cout << "Channel " << c << ":\n";
        for (int r = 0; r < min(5,rows); ++r)
        {
            for (int col = 0; col < min(5,cols); ++col)
            {
                int idx = (c * rows * cols) + (r * cols) + col;
                std::cout << CPU[idx] << "\t";
            }
            std::cout << "\n";
        }
        std::cout << "------------------------------\n";
    }

    free(CPU);
}

template <typename T>
void printHeadGrad(const str name, const T* X, const int ch, const int rows, const int cols, const int total)
{
    T* CPU = (T *)malloc(total * sizeof(T));
    cudaMemcpy(CPU, X, total * sizeof(T), cudaMemcpyDeviceToHost);

    std::cout << "Printing dimensions for node " << name << "->grad \n";
    for (int c = 0; c < min_int(3,ch); ++c)
    {
        std::cout << "Channel " << c << ":\n";
        for (int r = 0; r < min_int(5,rows); ++r)
        {
            for (int col = 0; col < min_int(5,cols); ++col)
            {
                int idx = (c * rows * cols) + (r * cols) + col;
                std::cout << CPU[idx] << "\t";
            }
            std::cout << "\n";
        }
        std::cout << "------------------------------\n";
    }

    free(CPU);
}

auto printMem = [](const char* label) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[%s] GPU free: %.2f MB\n", label, free_mem / 1e6);
};

class Timing
{
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> beg, ending;
    str function; 

public:
    Timing(const str reason);
    ~Timing();
    void start();
    void end();

};

void CheckError(const str& reason);

template <typename T>
void SafeCudaMalloc(const str name, T* &pointer, size_t size)
{   
    cudaError_t err;
    err = cudaMalloc((void**)&pointer, size*sizeof(T));
    if (err != cudaSuccess) {
    std::cerr << "Failed to allocate " << name << " of size " << size << ": " << cudaGetErrorString(err) << std::endl;
    size_t freeMem=0, totalMem=0;
    cudaMemGetInfo(&freeMem, &totalMem);
    std::cout << "GPU memory free: " << (freeMem>>20) << " MB of " << (totalMem>>20) << " MB\n";
    cudaFree(pointer);
    pointer = nullptr;
    std::exit(1);}
}


