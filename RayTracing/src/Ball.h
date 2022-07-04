#pragma once

#include "Geometry.h"

#include "glm/glm.hpp"

class Ball: public Geometry {
public:
	__host__ __device__ Ball(glm::vec3 cent, float r): m_Center(cent), m_Radius(r) {}
	~Ball() = default;
	__host__ __device__ GeoType GetType() const override { return GeoType::Ball; }
	__host__ __device__ glm::vec3 GetNorm(glm::vec3 pos) const override { return glm::normalize(pos - m_Center); }
	__host__ __device__ glm::vec3 GetCenter() const { return m_Center; }
	__host__ __device__ float GetRadius() const { return m_Radius; }
	size_t GetSize() const override { return sizeof(Ball); };
private:
	glm::vec3 m_Center;
	float m_Radius;
};


