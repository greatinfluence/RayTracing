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

	virtual size_t GetSize() const = 0;

	// AddMaterial(mat) adds the material to the geometry
	__host__ __device__ void AddMaterial(uint32_t matid) {
		m_Matid.id = matid;
	}

	using sizet_or_matptr = union { size_t id; Material* mat; };

	// Transferidtomat() transfers the storage of matid into material 
	__device__ void Transferidtomat() {
		m_Matid.mat = Materialrepository::GetMat(m_Matid.id);
	}

	sizet_or_matptr m_Matid;
};
