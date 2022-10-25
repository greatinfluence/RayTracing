#pragma once

#include <vector>
#include "Material.h"
#include "la.h"
#include "World.h"

class Object {
public:
	virtual void AppendtoWorld(World& world) = 0;
};

class Cube: public Object {
public:
	void AppendtoWorld(World& world) override;
	void SetAll(std::shared_ptr<Material> mat);
	Cube(): transform {} {}
	std::shared_ptr<Material> left, right, front, back, ceil, bot;
	bool uniform;
	la::mat3 transform;
	la::vec3 center;
};
