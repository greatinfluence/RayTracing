#pragma once

#include <vector>
#include <memory>

#include "glm/vec3.hpp"

#include "Material.h"

enum class GeoType {
	Unknown = 0, Triangle = 1, Ball = 2, Cuboid = 3
};

class Geometry {
public:
	virtual ~Geometry() = default;
	virtual GeoType GetType() const = 0;
	virtual glm::vec3 GetNorm(glm::vec3 pos) const = 0;
	std::shared_ptr<Material> GetMaterial() const { return m_Mat; }
	void AddMaterial(std::shared_ptr<Material> mat);
protected:
	std::shared_ptr<Material> m_Mat;
};
