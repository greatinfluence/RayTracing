#pragma once

#include "la.h"
#include "Geometry.h"
#include "Settings.h"

class Ray {
public:
	__host__ __device__ Ray(la::vec3&& pos, la::vec3&& dir)
		: m_Pos{ pos }, m_Dir{ dir } {}

	__host__ __device__ Ray(la::vec3 pos, la::vec3 dir)
		: m_Pos{ pos }, m_Dir{ dir } {}

	__host__ __device__ Ray(Ray const& rhs): m_Pos{rhs.m_Pos}, m_Dir{rhs.m_Dir} {}

	__host__ __device__ Ray& operator=(Ray&& rhs) noexcept {
		m_Pos = std::move(rhs.m_Pos);
		m_Dir = std::move(rhs.m_Dir);
		return *this;
	}

	// Hit(geo) tests the distance before the ray to hit the geometry
	//     if the ray will never hit the geometry, return FLOAT_MAX
	__host__ __device__ float Hit(Geometry const* geo) const;

	//__host__ __device__ la::vec3 const& GetPos() const;
	//__host__ __device__ la::vec3 const& GetDir() const;
	la::vec3 m_Pos, m_Dir;
};

