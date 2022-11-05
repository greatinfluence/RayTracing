#pragma once

#include "Geometry.h"

#include "la.h"

class Ball: public Geometry {
public:
	__host__ __device__ Ball(la::vec3 cent, float r): Geometry(GeoType::Ball), m_Center(cent), m_Radius(r) {}
	~Ball() = default;
	__host__ __device__ la::vec3 GetNorm(la::vec3 pos) const override { return la::normalize(pos - m_Center); }
//	__host__ __device__ la::vec3 GetCenter() const { return m_Center; }
	//__host__ __device__ float GetRadius() const { return m_Radius; }
	size_t GetSize() const override { return sizeof(Ball); };
	la::vec3 m_Center;
	float m_Radius;
};


