#pragma once

#include "glm/vec3.hpp"

#include "Triangle.h"
#include "Cuboid.h"
#include "Ball.h"
#include "device_launch_parameters.h"

class Ray {
public:
	__host__ __device__ Ray(glm::vec3 pos, glm::vec3 dir);

	// Hit(geo) tests the distance before the ray to hit the geometry
	//     if the ray will never hit the geometry, return FLOAT_MAX
	__host__ __device__ float Hit(Geometry const* geo) const;

	__host__ __device__ glm::vec3 GetPos() const { return m_Pos; }
	__host__ __device__ glm::vec3 GetDir() const { return m_Dir; }
private:
	glm::vec3 m_Pos, m_Dir;
};
