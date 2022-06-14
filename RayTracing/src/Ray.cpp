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
		float d = glm::l2Norm(cent - m_Pos);
		if (d < r) {
			// Inside the ball
			if (d < 1e-6) {
				// At the center
				return r;
			}
			double cosine = glm::dot(glm::normalize(cent - m_Pos), m_Dir);
			float ret = d * cosine + sqrt(d * d * cosine * cosine + r * r - d * d);
			float dis = glm::l2Norm(m_Pos + ret * m_Dir - glm::vec3(1, 0, 0));
			return d * cosine + sqrt(d * d * cosine * cosine + r * r - d * d);
		}
		else {
			// Outside the ball
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
	}
	case GeoType::Triangle: {
		assert(fabs(m_Dir.x) < 1.5f);
		assert(fabs(m_Dir.y) < 1.5f);
		assert(fabs(m_Dir.z) < 1.5f);
		auto* triangle = static_cast<Triangle*> (geo);
		glm::vec3 norm = triangle->GetNorm(m_Pos);
		glm::vec3 pp = glm::proj(triangle->GetPos(0) - m_Pos, norm);
		if (glm::l2Norm(pp) < 1e-6) {
			// Too close to the triangle
			return std::numeric_limits<float>::max();
		}
		float cosval = glm::dot(pp, m_Dir) / glm::l2Norm(pp);
		if (cosval <= 1e-6) {
			// Leaving or perpendicular to the plane of the triangle
			return std::numeric_limits<float>::max();
		}
		float dist = glm::l2Norm(pp) / cosval;
		glm::vec3 pos = m_Pos + m_Dir * dist;
		glm::vec3 vec = pos - triangle->GetPos(0);
		float val = glm::l2Norm(glm::proj(vec, norm));
		assert(val < 1e-4);
		if (triangle->OnTriangle(m_Pos + m_Dir * dist)) {
			return dist;
		}
		else return std::numeric_limits<float>::max();
	}
	default: {
		std::cout << "Not implemented yet!" << std::endl;
	}
	}
	return 0.0f;
}
