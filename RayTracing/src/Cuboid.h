#pragma once

#include "Geometry.h"

class Cuboid: public Geometry {
public:
	Cuboid();
	~Cuboid() = default;
	void AppendSubGeo(std::unique_ptr<Geometry> subgeo);
	GeoType GetType() override { return GeoType::Cuboid; }
	glm::vec3 GetNorm(glm::vec3 pos) override;
private:
	float xmin, xmax, ymin, ymax, zmin, zmax;
	std::vector<std::unique_ptr<Geometry>> m_Subgeo;
};


