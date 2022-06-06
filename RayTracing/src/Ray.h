#pragma once

#include "glm/vec3.hpp"

#include "Triangle.h"
#include "Cuboid.h"
#include "Ball.h"

class Ray {
public:
	Ray(glm::vec3 pos, glm::vec3 dir);
	float Hit(Geometry* geo);
	glm::vec3 GetPos() { return m_Pos; }
	glm::vec3 GetDir() { return m_Dir; }
private:
	glm::vec3 m_Pos, m_Dir;
};
