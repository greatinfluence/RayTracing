#pragma once

#include "glm/vec3.hpp"

#include "Triangle.h"
#include "Cuboid.h"
#include "Ball.h"

class Ray {
public:
	Ray(glm::vec3 pos, glm::vec3 dir);

	// Hit(geo) tests the distance before the ray to hit the geometry
	//     if the ray will never hit the geometry, return FLOAT_MAX
	float Hit(Geometry const* geo) const;

	glm::vec3 GetPos() const { return m_Pos; }
	glm::vec3 GetDir() const { return m_Dir; }
private:
	glm::vec3 m_Pos, m_Dir;
};
