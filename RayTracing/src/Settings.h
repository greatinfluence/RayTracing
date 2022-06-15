#pragma once

constexpr float pi = 3.14159265358979f;

#include <string>
#include <chrono>

// Copied from https://stackoverflow.com/questions/42138599/how-to-format-stdchrono-durations
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
