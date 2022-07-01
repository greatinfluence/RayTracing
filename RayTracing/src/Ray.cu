#include "Ray.h"

#include <iostream>
#include <limits>

#include "glm/gtx/perpendicular.hpp"
#include "glm/gtx/norm.hpp"

#include "Settings.h"

Ray::Ray(glm::vec3 pos, glm::vec3 dir)
	: m_Pos(pos), m_Dir(glm::normalize(dir)) {}

__host__ __device__ float Ray::Hit(Geometry const* geo) const {
	if (geo == nullptr) {
		printf("Ray::Hit error: Received nullptr\n");
		return -1;
	}
	switch (geo->GetType()) {
	case GeoType::Ball: {
		auto* ball = static_cast<Ball const*> (geo);
		float r = ball->GetRadius();
		glm::vec3 cent = ball->GetCenter();
		float d = glm::l2Norm(cent - m_Pos);
		if (d < r) {
			// Inside the ball
			if (d < eps) {
				// At the center
				return r;
			}
			float cosine = glm::dot(glm::normalize(cent - m_Pos), m_Dir);
			return (float) (d * cosine + sqrt(sq(d * cosine) + sq(r) - sq(d)));
		}
		else {
			// Outside the ball
			if (glm::dot(cent - m_Pos, m_Dir) < 0) {
				// leaving the ball
				return floatmax;
			}
			glm::vec3 diff = glm::perp(cent - m_Pos, m_Dir);
			if (glm::l2Norm(diff) > r) {
				// Out of range
				return floatmax;
			}
			glm::vec3 tolen = glm::proj(cent - m_Pos, m_Dir);
			return glm::l2Norm(tolen) - sqrt(sq(r) - glm::dot(diff, diff));
		}
	}
	case GeoType::Triangle: {
		assert(fabs(m_Dir.x) < 1.5f);
		assert(fabs(m_Dir.y) < 1.5f);
		assert(fabs(m_Dir.z) < 1.5f);
		auto* triangle = static_cast<Triangle const*>(geo);
		glm::vec3 norm = triangle->GetNorm(m_Pos);
		glm::vec3 pp = glm::proj(triangle->GetPos(0) - m_Pos, norm);
		if (glm::l2Norm(pp) < eps) {
			// Too close to the triangle
			return floatmax;
		}
		float cosval = glm::dot(pp, m_Dir) / glm::l2Norm(pp);
		if (cosval < eps) {
			// Leaving or perpendicular to the plane of the triangle
			return floatmax;
		}
		float dist = glm::l2Norm(pp) / cosval;
		glm::vec3 pos = m_Pos + m_Dir * dist;
		glm::vec3 vec = pos - triangle->GetPos(0);
		float val = glm::l2Norm(glm::proj(vec, norm));
		assert(val < 1e-4);
		if (triangle->OnTriangle(m_Pos + m_Dir * dist)) {
			return dist;
		}
		else return floatmax;
	}
	case GeoType::Cuboid: {
		/*
			This method is modified from http://tog.acm.org/resources/GraphicsGems/gems/RayBox.c
		*/
		auto* cub = static_cast<Cuboid const*>(geo);
		bool inside = true; // Denote if the ray is inside the box

		// quadrant denote if the starting position is inside the box
		enum class Quadrant {
			Left = 0,
			Middle = 1,
			Right = 2,
		} quadrant[3] = { Quadrant::Middle, Quadrant::Middle, Quadrant::Middle };
		float candidatePlane[3] = {};
		int whichPlane = 0;
		float diff[3] = {};

		for (auto i = 0; i < 3; ++i) {
			float min = cub->GetMin(i), max = cub->GetMax(i);
			if (m_Pos[i] < min) {
				quadrant[i] = Quadrant::Left;
				candidatePlane[i] = min;
				inside = false;
			}
			else if (m_Pos[i] > max) {
				quadrant[i] = Quadrant::Right;
				candidatePlane[i] = max;
				inside = false;
			}
		}

		if (inside) {
			// Ray is inside the box
			return 0.0f;
		}

		// Calculate the dist to the plane
		for (auto i = 0; i < 3; ++i) {
			if (quadrant[i] != Quadrant::Middle && fabs(m_Dir[i]) > 1e-6) {
				diff[i] = (candidatePlane[i] - m_Pos[i]) / m_Dir[i];
			}
			else diff[i] = -1.0f;
		}
		
		// Get largest of the dist for final choice of intersection
		for (auto i = 0; i < 3; ++i) {
			if (diff[whichPlane] < diff[i]) whichPlane = i;
		}

		// Check if the ray really hit the box
		if (diff[whichPlane] < 0.0f) return floatmax;
		for (auto i = 0; i < 3; ++i) {
			if (whichPlane != i) {
				float hitpos = m_Pos[i] + diff[whichPlane] * m_Dir[i];
				if (hitpos < cub->GetMin(i) || hitpos > cub->GetMax(i))
					return floatmax;
			}
		}
		return diff[whichPlane];
	}
	default: {
		printf("Not implemented yet!\n");
	}
	}
	return 0.0f;
}
