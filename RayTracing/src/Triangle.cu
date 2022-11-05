#include "Triangle.h"

#include <iostream>

__host__ __device__ Triangle::Triangle(la::vec3 v1, la::vec3 v2, la::vec3 v3, la::vec3 norm)
	:Geometry(GeoType::Triangle), m_Vertices{v1, v2, v3}, m_Norm(norm) {
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

bool Triangle::OnTriangle(la::vec3 pos) const {
	//	https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
	la::vec3 edge0 = m_Vertices[1] - m_Vertices[0],
			edge1 = m_Vertices[2] - m_Vertices[1],
			edge2 = m_Vertices[0] - m_Vertices[2];
	la::vec3 C0 = pos - m_Vertices[0],
		C1 = pos - m_Vertices[1],
		C2 = pos - m_Vertices[2];
	return la::dot(m_Norm, la::cross(edge0, C0)) > -eps &&
		la::dot(m_Norm, la::cross(edge1, C1)) > -eps &&
		la::dot(m_Norm, la::cross(edge2, C2)) > -eps;

}
