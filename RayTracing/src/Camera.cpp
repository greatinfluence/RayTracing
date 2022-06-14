#include "Camera.h"

#include <cmath>
#include <iostream>

#include "Random.h"

Camera::Camera()
	: m_Pos(glm::vec3(0)), m_Front(glm::vec3(1.0f, 0, 0)), m_Up(glm::vec3(0, 1.0f, 0)), m_Horang(1.0f), m_Perang(0.67f) {}

Camera::Camera(glm::vec3 pos, glm::vec3 front, glm::vec3 up, float horang, float perang)
: m_Pos(pos), m_Front(glm::normalize(front)), m_Up(glm::normalize(up)),
		m_Horang(horang), m_Perang(perang) {
	if (fabs(glm::dot(front, up)) > 1e-6) {
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
	for (int i = 0; i < num; ++i) {
		float hora = Random::Rand(lb, rb), perpa = Random::Rand(db, ub);
		glm::vec3 dir = m_Front + tan(perpa) * m_Up + tan(hora) * glm::cross(m_Front, m_Up);
		rays.emplace_back(m_Pos, glm::normalize(dir));
	}
}
