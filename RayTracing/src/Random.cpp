#include "Random.h"

namespace GPURandom {
	__device__ float Rand(float l, float r, curandState& state) {
		return curand_uniform(&state) * (r - l) + l;
	}

	__device__ float Rand(float r, curandState& state) { return curand_uniform(&state) * r; }

	__device__ la::vec3 RandinDisc(float r, curandState& state)
	{
		float theta = Rand(2 * pi, state);
		float rad = Rand(r, state);
		rad = sqrt(rad);
		return la::vec3(rad * cos(theta), sqrt(sq(r) - sq(rad)), rad * sin(theta));
	}

	__device__ la::vec3 RandinSphere(float r, curandState& state)
	{
		float theta = Rand(2 * pi, state), phi = acosf(Rand(2, state) - 1);
		float rad = r * pow(Rand(1, state), 1/3);
		return la::vec3(rad * sin(phi) * cos(theta), rad * sin(phi) * sin(theta), rad * cos(phi));
	}

	__device__ la::vec3 RandinHemisphere(la::vec3 norm, float r, curandState& state)
	{
		auto vec = RandinSphere(r, state);
		if (la::dot(vec, norm) < 0) return -vec;
		return vec;
	}
}