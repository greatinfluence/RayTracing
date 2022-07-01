#include "Cuboid.h"

#include <limits>
#include <iostream>

#include "Ball.h"
#include "Triangle.h"
#include "Ray.h"
#include "Settings.h"
#include "Geometryrepository.h"

#include "glm/gtx/norm.hpp"
#include "World.h"

Cuboid::Cuboid()
	: m_Min{ std::numeric_limits<float>::max(), 
             std::numeric_limits<float>::max(), 
			 std::numeric_limits<float>::max() },
	m_Max{ std::numeric_limits<float>::min(),
		   std::numeric_limits<float>::min(),
		   std::numeric_limits<float>::min() }, m_Nsubgeo{ 0 }, m_Subgeoid{nullptr} {}

void Cuboid::AppendSubGeos(World const& world, std::vector<size_t> const& subgeos) {
	m_Subgeoid = new size_t[subgeos.size()];
	for (size_t subgeoid : subgeos) {
		auto subgeo = world.GetGeo(subgeoid);
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
		m_Subgeoid[m_Nsubgeo ++] = subgeoid;
	}
}

__host__ __device__ glm::vec3 Cuboid::GetNorm(glm::vec3 pos) const
{
	printf("You should never call this function, since it's just a structural function\n");
	return glm::vec3(0.0f);
}

__host__ __device__ void Cuboid::TestHit(Ray const& ray, float& dist, Geometry*& hitted) const
{
	if (ray.Hit(this) < dist) {
		// It is possible for the ray to hit the box
		for(size_t i = 0; i < m_Nsubgeo; ++ i) {
			auto geo = Geometryrepository::GetGeo(m_Subgeoid[i]);
			if (geo->GetType() == GeoType::Cuboid) {
				// A sublevel of Cuboid
				auto* cub = static_cast<Cuboid*>(geo);
				cub->TestHit(ray, dist, hitted);
			}
			else {
				// A common object
				float ndist = ray.Hit(geo);
				if (ndist < dist) {
					dist = ndist;
					hitted = geo;
				}
			}
		}
	}
}
