#pragma once

#include "Geometry.h"

class Ray;
class World;

class Cuboid: public Geometry {
public:
	__host__ __device__ Cuboid();
	~Cuboid() = default;

	// AppendSubGeos(subgeos) appends all subgeos as sub-geometry into the cuboid
	void AppendSubGeos(World const& world, std::vector<size_t> const& subgeos);
	__host__ __device__ GeoType GetType() const override { return GeoType::Cuboid; }
	__host__ __device__ la::vec3 GetNorm(la::vec3 pos) const override;

	/*
		TestHit(ray, dist, hitted) does the ray-hit detection among all the sub-geometries
			returns the closest distance and the corresponding sub-geometry
	*/
	void TestHit(Ray const& ray, float& dist, Geometry*& hitted) const;

	// TestHitdevice is the device version of TestHit
	__device__ void TestHitdevice(Ray const& ray, float& dist, Geometry*& hitted) const;
	
	size_t GetSize() const override { return sizeof(Cuboid); };

	__host__ __device__ size_t* GetSubgeoids() const { return m_Subgeoid; }
	__host__ __device__ size_t GetNsubgeo() const { return m_Nsubgeo; }
	__host__ __device__ void SetSubgeoid(size_t* subgeoid) { m_Subgeoid = subgeoid; }
	__host__ __device__ void SetNsubgeo(size_t nsubgeo) { m_Nsubgeo = nsubgeo; }
	float m_Min[3], m_Max[3];
	size_t m_Nsubgeo;
	size_t* m_Subgeoid;
};


