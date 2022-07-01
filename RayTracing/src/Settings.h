#pragma once

#include <limits>
#include <string>
#include <chrono>

#include "cuda_runtime.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

// pi
constexpr float pi = 3.14159265358979f;

// The accuracy epsilon
constexpr float eps = 1e-6f;

// Float's upperbound
constexpr float floatmax = std::numeric_limits<float>().max();

// Float's lowerbound
constexpr float floatmin = std::numeric_limits<float>().min();

// sq(x) returns x * x
__host__ __device__ float sq(float x);

/*
    formatDuration(d) formats the time duration into the form xx:xx:xx
    Copied from https://stackoverflow.com/questions/42138599/how-to-format-stdchrono-durations
*/
template<class DurationIn, class FirstDuration, class...RestDurations>
std::string formatDuration(DurationIn d)
{
    auto val = std::chrono::duration_cast<FirstDuration>(d);

    std::string out = std::to_string(val.count());

    if constexpr (sizeof...(RestDurations) > 0) {
        out += ": " + formatDuration<DurationIn, RestDurations...>(d - val);
    }

    return out;
}

template<class DurationIn>
std::string formatDuration(DurationIn) { return {}; } // recursion termination
