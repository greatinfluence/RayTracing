#pragma once

#include "Geometry.h"

#include "glm/glm.hpp"

class Ball: public Geometry {
public:
	Ball(glm::vec3 cent, float r): m_Center(cent), m_Radius(r) {}
	~Ball() = default;
	GeoType GetType() override { return GeoType::Ball; }
	glm::vec3 GetNorm(glm::vec3 pos) override { return glm::normalize(pos - m_Center); }
	glm::vec3 GetCenter() { return m_Center; }
	float GetRadius() { return m_Radius; }
private:
	glm::vec3 m_Center;
	float m_Radius;
};


