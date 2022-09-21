#include "Camera.h"

#include <iostream>

#include "Random.h"

Camera::Camera(la::vec3 pos, la::vec3 front, la::vec3 up, float horang, float perang)
: m_Pos(pos), m_Front(la::normalize(front)), m_Up(la::normalize(up)),
		m_Horang(horang), m_Perang(perang) {
	if (abs(la::dot(front, up)) > eps) {
		std::cout << "Error: The camera's up/front vectors are not perpendicular" << std::endl;
	}
}

void Camera::GenRay(int x, int y, int width, int height, std::vector<Ray>& rays, int num) const {
	float xl = ((float)x) / width, xr = ((float)x + 1) / width;
	float yl = ((float)y) / height, yr = ((float)y + 1) / height;
	float lb = (float)atan((xl - 0.5) * 2 * tan(m_Horang / 2)),
		rb = (float)atan((xr - 0.5) * 2 * tan(m_Horang / 2)),
		db = (float)atan((yl - 0.5) * 2 * tan(m_Perang / 2)),
		ub = (float)atan((yr - 0.5) * 2 * tan(m_Perang / 2));
	// The third direction
	const la::vec3 zaxis = la::cross(m_Front, m_Up);
	for (int i = 0; i < num; ++i) {
		float hora = Random::Rand(lb, rb), perpa = Random::Rand(db, ub);
		la::vec3 dir = m_Front + tan(perpa) * m_Up + tan(hora) * zaxis;
		rays.emplace_back(m_Pos, la::normalize(dir));
	}
}

__device__ Ray Camera::GenRay(int x, int y, int width, int height, curandState& state) const
{
//	printf("Pos: %f %f %f Fr: %f %f %f Up: %f %f %f\n", m_Pos.x, m_Pos.y, m_Pos.z, m_Front.x, m_Front.y, m_Front.z, m_Up.x, m_Up.y, m_Up.z);
	float xl = ((float)x) / width, xr = ((float)x + 1) / width;
	float yl = ((float)y) / height, yr = ((float)y + 1) / height;
	float lb = (float)atan((xl - 0.5) * 2 * tan(m_Horang / 2)),
		rb = (float)atan((xr - 0.5) * 2 * tan(m_Horang / 2)),
		db = (float)atan((yl - 0.5) * 2 * tan(m_Perang / 2)),
		ub = (float)atan((yr - 0.5) * 2 * tan(m_Perang / 2));
	// The third direction
	const la::vec3 zaxis = la::cross(m_Front, m_Up);
	float hora = GPURandom::Rand(lb, rb, state), perpa = GPURandom::Rand(db, ub, state);
	la::vec3 dir = m_Front + tan(perpa) * m_Up + tan(hora) * zaxis;
	return Ray(std::move(m_Pos), la::normalize(dir));
}
