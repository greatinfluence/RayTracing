#pragma once

#include "Geometry.h"

class Cylindsurf: public Geometry {
public:
	__host__ __device__ Cylindsurf(la::vec3 cent, la::vec3 up, float r, float h):
		Geometry(GeoType::Cylindsurf), m_Cent(cent), m_Up(la::normalize(up)), m_Radius(r), m_Height(h) {}
	~Cylindsurf() = default;
	__host__ __device__ la::vec3 GetNorm(la::vec3 pos) const override {
		return la::normalize(la::perp(pos - m_Cent, m_Up));
	}
	size_t GetSize() const override { return sizeof(Cylindsurf); };
	la::vec3 m_Cent, m_Up;
	float m_Radius, m_Height;
};