#include "Camera.h"

#include <cmath>
#include <iostream>

#include "Random.h"

Camera::Camera(glm::vec3 pos, glm::vec3 front, glm::vec3 up, float horang, float perang)
: m_Pos(pos), m_Front(glm::normalize(front)), m_Up(glm::normalize(up)),
		m_Horang(horang), m_Perang(perang) {
	if (abs(glm::dot(front, up)) > eps) {
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
	const glm::vec3 zaxis = glm::cross(m_Front, m_Up);
	for (int i = 0; i < num; ++i) {
		float hora = Random::Rand(lb, rb), perpa = Random::Rand(db, ub);
		glm::vec3 dir = m_Front + tan(perpa) * m_Up + tan(hora) * zaxis;
		rays.emplace_back(m_Pos, glm::normalize(dir));
	}
}

__device__ Ray Camera::GenRay(int x, int y, int width, int height, curandState& state) const
{
	printf("Pos: %f %f %f Fr: %f %f %f Up: %f %f %f\n", m_Pos.x, m_Pos.y, m_Pos.z, m_Front.x, m_Front.y, m_Front.z, m_Up.x, m_Up.y, m_Up.z);
	float xl = ((float)x) / width, xr = ((float)x + 1) / width;
	float yl = ((float)y) / height, yr = ((float)y + 1) / height;
	float lb = (float)atan((xl - 0.5) * 2 * tan(m_Horang / 2)),
		rb = (float)atan((xr - 0.5) * 2 * tan(m_Horang / 2)),
		db = (float)atan((yl - 0.5) * 2 * tan(m_Perang / 2)),
		ub = (float)atan((yr - 0.5) * 2 * tan(m_Perang / 2));
	// The third direction
	const glm::vec3 zaxis = glm::cross(m_Front, m_Up);
	float hora = GPURandom::Rand(lb, rb, state), perpa = GPURandom::Rand(db, ub, state);
	glm::vec3 dir = m_Front + tan(perpa) * m_Up + tan(hora) * zaxis;
	return Ray(m_Pos, glm::normalize(dir));
}
