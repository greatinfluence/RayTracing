#pragma once

#include "Geometry.h"
#include "Cuboid.h"

class Ray;

class Cuboidgpu: public Geometry {
public:
	Cuboidgpu() = delete;
	Cuboidgpu(Cuboid const& cub);
	~Cuboidgpu() = default;

	GeoType GetType() const override { return GeoType::Cuboid; }
	glm::vec3 GetNorm(glm::vec3 pos) const override;

	// GetMin(dim) returns the minimum bound of the dimension dim
	float GetMin(int dim) const { return m_Min[dim]; }

	// GetMax(dim) returns the maximum bound of the dimension dim
	float GetMax(int dim) const { return m_Max[dim]; }

	/*
		TestHit(ray, dist, hitted) does the ray-hit detection among all the sub-geometries
			returns the closest distance and the corresponding sub-geometry
	*/
	void TestHit(Ray const& ray, float& dist, std::shared_ptr<Geometry>& hitted) const;
private:
	float m_Min[3], m_Max[3];
	size_t m_Nsubgeo;
	size_t* m_Subgeo;
};



