#pragma once

#include "Geometry.h"

#include "la.h"

class Triangle: public Geometry {
public:
	__host__ __device__ Triangle(la::vec3 v1, la::vec3 v2, la::vec3 v3, la::vec3 norm = la::vec3(0.0f));
	~Triangle() = default;

	// GetPos(ind) returns the position of the ind-th vertex of the triangle
	__host__ __device__ la::vec3 GetPos(size_t ind) const;

	__host__ __device__ GeoType GetType() const override { return GeoType::Triangle; }
	__host__ __device__ la::vec3 GetNorm(la::vec3 pos) const override { return m_Norm; }

	// OnTriangle(pos) tests if pos is in the triangle
	//   pos should be in the plane of the triangle
	__host__ __device__ bool OnTriangle(la::vec3 pos) const;
	size_t GetSize() const override { return sizeof(Triangle); };
private:
	la::vec3 m_Vertices[3];
	la::vec3 m_Norm;
	__host__ __device__ static la::vec3 ComputeTriangNorm(la::vec3 v1, la::vec3 v2, la::vec3 v3) {
		return la::normalize(la::cross(v2 - v1, v3 - v1));
	}
};

