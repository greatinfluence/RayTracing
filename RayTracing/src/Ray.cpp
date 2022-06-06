#include "Ray.h"

#include <iostream>
#include <limits>

#include "glm/gtx/perpendicular.hpp"
#include "glm/gtx/norm.hpp"

Ray::Ray(glm::vec3 pos, glm::vec3 dir)
	: m_Pos(pos), m_Dir(glm::normalize(dir)) {}

float Ray::Hit(Geometry* geo)
{
	if (geo == nullptr) {
		std::cout << "Ray::Hit Error: Received nullptr" << std::endl;
		return -1;
	}
	switch (geo->GetType()) {
	case GeoType::Ball: {
		auto* ball = static_cast<Ball*> (geo);
		float r = ball->GetRadius();
		glm::vec3 cent = ball->GetCenter();
		if (glm::dot(cent - m_Pos, m_Dir) < 0) {
			// leaving the ball
			return std::numeric_limits<float>::max();
		}
		glm::vec3 diff = glm::perp(cent - m_Pos, m_Dir);
		if (glm::l2Norm(diff) > r) {
			// Out of range
			return std::numeric_limits<float>::max();
		}
		glm::vec3 tolen = glm::proj(cent - m_Pos, m_Dir);
		return glm::l2Norm(tolen) - sqrt(r * r - glm::dot(diff, diff));
	}
	case GeoType::Triangle: {
		auto* triangle = static_cast<Triangle*> (geo);
	
		glm::vec3 pp = glm::perp(triangle->GetPos(0) - m_Pos, triangle->GetNorm(m_Pos));
		float cosval = glm::dot(pp, m_Dir) / glm::l2Norm(pp) / glm::l2Norm(m_Dir);
		float sinval = sqrt(1 - cosval * cosval);
		float dist = glm::l2Norm(pp) / sinval;
		if (triangle->OnTriangle(m_Pos + m_Dir * dist))
			return dist;
		else return std::numeric_limits<float>::max();
	}
	default: {
		std::cout << "Not implemented yet!" << std::endl;
	}
	}
	return 0.0f;
}
