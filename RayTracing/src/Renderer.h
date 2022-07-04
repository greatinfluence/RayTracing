#pragma once

#include "World.h"
#include "Image.h"
#include "Random.h"

#include <mutex>

namespace GPURenderer {
	void Render(World& world, Image3& output, int nrays = 32);
	__global__ static void DoRender(size_t root, glm::vec3 bg, Camera& cam, int width, int height, glm::vec3* pixels,
		int nrays, curandState& state);
}

class Renderer {
public:

	// Render(world, output, nrays, OutputFreq) will render the world using nrays per pixel,
	//     write the output into outout, and prints the progress rate after rendering every
	//     OutputFreq rays
	void Render(World& world, Image3& output, int nrays = 32, int OutputFreq = 1000);
private:
	static void DoRender(World& world, Image3& output, uint32_t from, uint32_t to, int nrays, int OutputFreq, int nthread, std::mutex& lock);
};
