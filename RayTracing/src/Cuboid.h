#pragma once

#include "Geometry.h"

class Ray;

class Cuboid: public Geometry {
public:
	Cuboid();
	~Cuboid() = default;
	void AppendSubGeo(std::shared_ptr<Geometry> subgeo);
	GeoType GetType() const override { return GeoType::Cuboid; }
	glm::vec3 GetNorm(glm::vec3 pos) const override;
	float GetMin(int dim) const { return m_Min[dim]; }
	float GetMax(int dim) const { return m_Max[dim]; }
	void TestHit(Ray const& ray, float& dist, std::shared_ptr<Geometry>& hitted) const;
private:
	float m_Min[3], m_Max[3];
	std::vector<std::shared_ptr<Geometry>> m_Subgeo;
};


