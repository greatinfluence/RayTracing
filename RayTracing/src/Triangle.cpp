#include "Triangle.h"

#include <iostream>

Triangle::Triangle(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, glm::vec3 norm)
	:m_Vertices{v1, v2, v3}, m_Norm(norm) {
	if (norm == glm::vec3(0.0f)) {
		norm = ComputeTriangNorm(v1, v2, v3);
	}
}

glm::vec3 Triangle::GetPos(size_t ind)
{
	if (ind >= 3) {
		std::cout << "Triangle::getpos Out of range" << std::endl;
	}
	return m_Vertices[ind];
}

bool Triangle::OnTriangle(glm::vec3 pos)
{
	auto ed1 = m_Vertices[1] - m_Vertices[0], ed2 = m_Vertices[2] - m_Vertices[0];
	auto vec = pos - m_Vertices[0];
	auto fsthf = glm::cross(ed1, vec), sechf = glm::cross(vec, ed2);
	if (fsthf.x * sechf.x < 0 || fsthf.y * sechf.y < 0 || fsthf.z * sechf.z < 0)
		return false;
	ed1 = -ed1;
	ed2 = m_Vertices[2] - m_Vertices[1];
	vec = pos - m_Vertices[1];
	fsthf = glm::cross(ed1, vec), sechf = glm::cross(vec, ed2);
	if (fsthf.x * sechf.x < 0 || fsthf.y * sechf.y < 0 || fsthf.z * sechf.z < 0)
		return false;
	return true;
}
