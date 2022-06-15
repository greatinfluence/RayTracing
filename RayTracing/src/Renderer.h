#pragma once

#include "World.h"
#include "Image.h"

#include <mutex>

class Renderer {
public:
	void Render(World& world, Image3& output, int nrays = 32, int OutputFreq = 1000);
private:
	static void DoRender(World& world, Image3& output, uint32_t from, uint32_t to, int narays, int OutputFreq, int nthread, std::mutex& lock);
};
