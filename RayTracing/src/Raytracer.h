#pragma once

#include "Ray.h"
#include "Random.h"

namespace Raytracing {
	// RayTracing(ray, lev, coef) traces the ray and returns the color seen by the ray
	__device__ la::vec3 RayTracing(Ray const& ray, Cuboid* cub, la::vec3 const& m_Background, curandState& state, int lev = 1, la::vec3 coef = la::vec3(1.0f));

}
