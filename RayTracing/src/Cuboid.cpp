#include "Cuboid.h"

#include <limits>
#include <iostream>

#include "Ball.h"
#include "Triangle.h"

Cuboid::Cuboid()
	: xmin(std::numeric_limits<float>::max()), xmax(std::numeric_limits<float>::min()),
		ymin(xmin), ymax(xmax), zmin(xmin), zmax(xmax) {}

void Cuboid::AppendSubGeo(std::unique_ptr<Geometry> subgeo)
{
	switch (subgeo->GetType()) {
	case GeoType::Cuboid: {
		auto* subcub = static_cast<Cuboid*>(subgeo.get());
		xmin = fmin(xmin, subcub -> xmin);
		ymin = fmin(ymin, subcub -> ymin);
		zmin = fmin(zmin, subcub -> zmin);
		xmax = fmax(xmax, subcub -> xmax);
		ymax = fmax(ymax, subcub -> ymax);
		zmax = fmax(zmax, subcub -> zmax);
		break;
	}
	case GeoType::Ball: {
		auto* subba = static_cast<Ball*>(subgeo.get());
		glm::vec3 cent = subba->GetCenter();
		float r = subba->GetRadius();
		xmin = fmin(xmin, cent.x - r);
		ymin = fmin(ymin, cent.y - r);
		zmin = fmin(zmin, cent.z - r);
		xmax = fmax(xmax, cent.x + r);
		ymax = fmax(ymax, cent.y + r);
		zmax = fmax(zmax, cent.z + r);
		break;
	}
	case GeoType::Triangle: {
		auto* subtr = static_cast<Triangle*>(subgeo.get());
		for (size_t i = 0; i < 3; ++i) {
			glm::vec3 vert = subtr->GetPos(i);
			xmin = fmin(xmin, vert.x);
			ymin = fmin(ymin, vert.y);
			zmin = fmin(zmin, vert.z);
			xmax = fmax(xmax, vert.x);
			ymax = fmax(ymax, vert.y);
			zmax = fmax(zmax, vert.z);
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

glm::vec3 Cuboid::GetNorm(glm::vec3 pos)
{
	std::cout << "You should never Call Cuboid::GetNorm(), since it's just a structural function" << std::endl;
	return glm::vec3(0.0f);
}
