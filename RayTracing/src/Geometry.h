#pragma once

#include <vector>
#include <memory>

#include "device_launch_parameters.h"

#include "la.h"
#include "Materialrepository.h"

enum class GeoType {
	Unknown = 0, Triangle = 1, Ball = 2, Cuboid = 3
};

class Geometry {
public:
	__host__ __device__ Geometry() : m_Matid{ 0u } {}

	// GetType() returns the type of the geometry
	__host__ __device__ virtual GeoType GetType() const = 0;

	// GetNorm(pos) returns the normal vector of the geometry at position pos
	__host__ __device__ virtual la::vec3 GetNorm(la::vec3 pos) const = 0;

	__host__ __device__ Material* GetMaterial() const { return Materialrepository::GetMat(m_Matid); }

	__host__ __device__ uint32_t GetMatid() const { return m_Matid; }

	virtual size_t GetSize() const = 0;

	// AddMaterial(mat) adds the material to the geometry
	__host__ __device__ void AddMaterial(uint32_t matid) {
		m_Matid = matid;
	}
protected:
	uint32_t m_Matid;
};
