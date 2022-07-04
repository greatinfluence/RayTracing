#pragma once

#include "Ray.h"
#include "Random.h"

namespace Raytracing {
	// RayTracing(ray, lev, coef) traces the ray and returns the color seen by the ray
	__device__ glm::vec3 RayTracing(Ray ray, size_t* root, glm::vec3 m_Background, curandState& state, int lev = 1, glm::vec3 coef = glm::vec3(1.0f));

}
