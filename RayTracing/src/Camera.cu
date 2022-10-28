#include "Camera.h"

#include <iostream>

#include "Random.h"

void FishEyeCamera::GenRay(int x, int y, int width, int height, std::vector<Ray>& rays, int num) const {
	double xl = ((double)x) / width, xr = ((double)x + 1) / width;
	double yl = ((double)y) / height, yr = ((double)y + 1) / height;
	double lb = (double)atan((xl - 0.5) * 2 * tan(m_Horang / 2)),
		rb = (double)atan((xr - 0.5) * 2 * tan(m_Horang / 2)),
		db = (double)atan((yl - 0.5) * 2 * tan(m_Perang / 2)),
		ub = (double)atan((yr - 0.5) * 2 * tan(m_Perang / 2));
	// The third direction
	const la::vec3 zaxis = la::cross(m_Front, m_Up);
	for (int i = 0; i < num; ++i) {
		double hora = Random::Rand(lb, rb), perpa = Random::Rand(db, ub);
		la::vec3 dir = m_Front + tan(perpa) * m_Up + tan(hora) * zaxis;
		rays.emplace_back(m_Pos, la::normalize(dir));
	}
}

__device__ Ray FishEyeCamera::GenRay(int x, int y, int width, int height, curandState& state) const
{
//	printf("Pos: %f %f %f Fr: %f %f %f Up: %f %f %f\n", m_Pos.x, m_Pos.y, m_Pos.z, m_Front.x, m_Front.y, m_Front.z, m_Up.x, m_Up.y, m_Up.z);
	double xl = ((double)x) / width, xr = ((double)x + 1) / width;
	double yl = ((double)y) / height, yr = ((double)y + 1) / height;
	double lb = (double)atan((xl - 0.5) * 2 * tan(m_Horang / 2)),
		rb = (double)atan((xr - 0.5) * 2 * tan(m_Horang / 2)),
		db = (double)atan((yl - 0.5) * 2 * tan(m_Perang / 2)),
		ub = (double)atan((yr - 0.5) * 2 * tan(m_Perang / 2));
	// The third direction
	const la::vec3 zaxis = la::cross(m_Front, m_Up);
	double hora = GPURandom::Rand(lb, rb, state), perpa = GPURandom::Rand(db, ub, state);
	la::vec3 dir = m_Front + tan(perpa) * m_Up + tan(hora) * zaxis;
	return Ray(m_Pos, la::normalize(dir));
}

void RegularCamera::GenRay(int x, int y, int width, int height, std::vector<Ray>& rays, int num) const
{
	double xl = ((double)x) / width, xr = ((double)x + 1) / width;
	double yl = ((double)y) / height, yr = ((double)y + 1) / height;
	la::vec3 const zaxis = la::cross(m_Front, m_Up);
	for (int i = 0; i < num; ++i) {
		la::vec3 dir = m_Front + Random::Rand(xl, xr) * m_Per * m_Up + Random::Rand(yl, yr) * m_Hor * zaxis;
		rays.emplace_back(m_Pos, la::normalize(dir));
	}
}

__device__ Ray RegularCamera::GenRay(int x, int y, int width, int height, curandState& state) const
{
	double xl = ((double)x) / width - 0.5, xr = ((double)x + 1) / width - 0.5;
	double yl = ((double)y) / height - 0.5, yr = ((double)y + 1) / height - 0.5;
	la::vec3 const zaxis = la::cross(m_Front, m_Up);
	la::vec3 dir = m_Front + GPURandom::Rand(xl, xr, state) * m_Hor * zaxis + GPURandom::Rand(yl, yr, state) * m_Per * m_Up;
	return Ray(m_Pos, la::normalize(dir));
}
