#include "Triangle.h"

#include <iostream>

__host__ __device__ Triangle::Triangle(la::vec3 v1, la::vec3 v2, la::vec3 v3, la::vec3 norm)
	:m_Vertices{v1, v2, v3}, m_Norm(norm) {
	if (la::l2Norm(norm) < 1e-8) {
		m_Norm = ComputeTriangNorm(v1, v2, v3);
	}
}

la::vec3 Triangle::GetPos(size_t ind) const {
	if (ind >= 3) {
		printf("Triangle::getpos Out of range\n");
	}
	return *(m_Vertices + ind);
}

bool Triangle::OnTriangle(la::vec3 pos) const
{
	auto ed1 = m_Vertices[1] - m_Vertices[0], ed2 = m_Vertices[2] - m_Vertices[0];
	auto vec = pos - m_Vertices[0];
	auto fsthf = la::cross(ed1, vec), sechf = la::cross(vec, ed2);
	if (fsthf.x * sechf.x < -1e-6 || fsthf.y * sechf.y < -1e-6 || fsthf.z * sechf.z < -1e-6)
		return false;
	ed1 = -ed1;
	ed2 = m_Vertices[2] - m_Vertices[1];
	vec = pos - m_Vertices[1];
	fsthf = la::cross(ed1, vec), sechf = la::cross(vec, ed2);
	if (fsthf.x * sechf.x < -1e-6 || fsthf.y * sechf.y < -1e-6 || fsthf.z * sechf.z < -1e-6)
		return false;
	return true;
}
