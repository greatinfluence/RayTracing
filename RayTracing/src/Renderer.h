#pragma once

#include "World.h"
#include "Image.h"

class Renderer {
public:
	void Render(World& world, Image3& output, int nrays = 32, int OutputFreq = 10);
};
