#include "Cuboid.h"

#include <limits>
#include <iostream>

#include "Ball.h"
#include "Triangle.h"
#include "Ray.h"
#include "Settings.h"

#include "glm/gtx/norm.hpp"

Cuboid::Cuboid()
	: m_Min{ std::numeric_limits<float>::max(), 
             std::numeric_limits<float>::max(), 
			 std::numeric_limits<float>::max() },
	m_Max{ std::numeric_limits<float>::min(),
		   std::numeric_limits<float>::min(),
		   std::numeric_limits<float>::min() } {}

void Cuboid::AppendSubGeo(std::shared_ptr<Geometry> subgeo)
{
	switch (subgeo->GetType()) {
	case GeoType::Cuboid: {
		auto* subcub = static_cast<Cuboid*>(subgeo.get());
		for (auto i = 0; i < 3; ++i) {
			m_Min[i] = fmin(m_Min[i], subcub->m_Min[i]);
			m_Max[i] = fmax(m_Max[i], subcub->m_Max[i]);
		}
		break;
	}
	case GeoType::Ball: {
		auto* subba = static_cast<Ball*>(subgeo.get());
		glm::vec3 cent = subba->GetCenter();
		float r = subba->GetRadius();
		for (auto i = 0; i < 3; ++i) {
			m_Min[i] = fmin(m_Min[i], cent[i] - r);
			m_Max[i] = fmax(m_Max[i], cent[i] + r);
		}
		break;
	}
	case GeoType::Triangle: {
		auto* subtr = static_cast<Triangle*>(subgeo.get());
		for (auto i = 0; i < 3; ++i) {
			glm::vec3 vert = subtr->GetPos(i);
			for (auto j = 0; j < 3; ++j) {
				m_Min[j] = fmin(m_Min[j], vert[j]);
				m_Max[j] = fmax(m_Max[j], vert[j]);
			}
		}
		break;
	}
	default: {
		std::cout << "Unrecongnized Geometry Type!" << std::endl;
		return;
	}
	}
	m_Subgeo.push_back(std::move(subgeo));
}

glm::vec3 Cuboid::GetNorm(glm::vec3 pos) const
{
	std::cout << "You should never Call Cuboid::GetNorm(), since it's just a structural function" << std::endl;
	return glm::vec3(0.0f);
}

void Cuboid::TestHit(Ray const& ray, float& dist, std::shared_ptr<Geometry>& hitted) const
{
	if (ray.Hit(this) < dist) {
		// It is possible for the ray to hit the box
		for (std::shared_ptr<Geometry> const& geo : m_Subgeo) {
			if (geo->GetType() == GeoType::Cuboid) {
				// A sublevel of Cuboid
				auto* cub = static_cast<Cuboid*>(geo.get());
				cub->TestHit(ray, dist, hitted);
			}
			else {
				// A common object
				float ndist = ray.Hit(geo.get());
				if (ndist < dist) {
					dist = ndist;
					hitted = geo;
				}
			}
		}
	}
}
