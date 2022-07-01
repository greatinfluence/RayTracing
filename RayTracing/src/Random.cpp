#include "Random.h"

namespace GPURandom {
	__device__ float Rand(float l, float r, curandState& state) {
		return curand_uniform(&state) * (r - l) + l;
	}

	__device__ float Rand(float r, curandState& state) { return curand_uniform(&state) * r; }

	__device__ glm::vec3 RandinDisc(float r, curandState& state)
	{
		float theta = Rand(2 * pi, state);
		float rad = Rand(r, state);
		rad = sqrt(rad);
		return glm::vec3(rad * cos(theta), sqrt(sq(r) - sq(rad)), rad * sin(theta));
	}

	__device__ glm::vec3 RandinSphere(float r, curandState& state)
	{
		float theta = Rand(2 * pi, state), phi = Rand(pi, state);
		float rad = Rand(r, state);
		rad = sqrt(rad);
		return glm::vec3(rad * cos(phi) * cos(theta), rad * cos(phi) * sin(theta), rad * sin(phi));
	}

	__device__ glm::vec3 RandinHemisphere(glm::vec3 norm, float r, curandState& state)
	{
		auto vec = RandinSphere(r, state);
		if (glm::dot(vec, norm) < 0) return -vec;
		return vec;
	}
}