#pragma once

#include "Geometry.h"

class Ray;
class World;

class Cuboid: public Geometry {
public:
	Cuboid();
	~Cuboid() = default;

	// AppendSubGeos(subgeos) appends all subgeos as sub-geometry into the cuboid
	void AppendSubGeos(World const& world, std::vector<size_t> const& subgeos);
	__host__ __device__ GeoType GetType() const override { return GeoType::Cuboid; }
	__host__ __device__ glm::vec3 GetNorm(glm::vec3 pos) const override;

	// GetMin(dim) returns the minimum bound of the dimension dim
	__host__ __device__ float GetMin(int dim) const { return m_Min[dim]; }

	// GetMax(dim) returns the maximum bound of the dimension dim
	__host__ __device__ float GetMax(int dim) const { return m_Max[dim]; }

	/*
		TestHit(ray, dist, hitted) does the ray-hit detection among all the sub-geometries
			returns the closest distance and the corresponding sub-geometry
	*/
	__host__ __device__ void TestHit(Ray const& ray, float& dist, Geometry*& hitted) const;
private:
	float m_Min[3], m_Max[3];
	size_t m_Nsubgeo;
	size_t* m_Subgeoid;
};


