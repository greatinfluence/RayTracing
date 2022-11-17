#pragma once

#include "Geometry.h"

class Plate : public Geometry {
public:
	__host__ __device__ Plate(la::vec3 cent, la::vec3 up, float out, float in = 0) :
		Geometry(GeoType::Plate), m_Cent(cent), m_Up(la::normalize(up)), m_Outrad(out), m_Inrad(in) {}
	~Plate() = default;
	__host__ __device__ la::vec3 GetNorm(la::vec3 pos) const override {
		return m_Up;
	}
	size_t GetSize() const override { return sizeof(Plate); };
	la::vec3 m_Cent, m_Up;
	float m_Outrad, m_Inrad;
};
