#pragma once

#include "Geometry.h"

#include "glm/glm.hpp"

class Triangle: public Geometry {
public:
	__host__ __device__ Triangle(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, glm::vec3 norm = glm::vec3(0.0f));
	~Triangle() = default;

	// GetPos(ind) returns the position of the ind-th vertex of the triangle
	__host__ __device__ glm::vec3 GetPos(size_t ind) const;

	__host__ __device__ GeoType GetType() const override { return GeoType::Triangle; }
	__host__ __device__ glm::vec3 GetNorm(glm::vec3 pos) const override { return m_Norm; }

	// OnTriangle(pos) tests if pos is in the triangle
	//   pos should be in the plane of the triangle
	__host__ __device__ bool OnTriangle(glm::vec3 pos) const;
	size_t GetSize() const override { return sizeof(Triangle); };
private:
	glm::vec3 m_Vertices[3];
	glm::vec3 m_Norm;
	__host__ __device__ static glm::vec3 ComputeTriangNorm(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3) {
		return glm::normalize(glm::cross(v2 - v1, v3 - v1));
	}
};

