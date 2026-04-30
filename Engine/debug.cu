#include "includes/debugging_utils.h"

Timing::Timing(const str reason): function(reason){};

Timing::~Timing(){}

void Timing::start(){beg = std::chrono::high_resolution_clock::now();}

void Timing::end()
{   ending = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(ending - beg);
    std::cout << "Time elapsed for " << function << ": " << duration.count() << "ms \n";
}

void CheckError(const str& reason)
{
    cudaError_t launchErr = cudaGetLastError();     // did launch succeed?
    cudaError_t syncErr   = cudaDeviceSynchronize(); // did kernel succeed?

    if (launchErr != cudaSuccess) {
    std::cerr << "Launch error in " << reason << ": " << cudaGetErrorString(launchErr) << std::endl;std::exit(1); }

    if (syncErr != cudaSuccess) {
    std::cerr << "Runtime error in " << reason << ": " << cudaGetErrorString(syncErr) << std::endl; std::exit(1);
    }
}


