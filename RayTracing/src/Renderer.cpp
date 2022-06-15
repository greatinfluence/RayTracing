#include "Renderer.h"

#include <iostream>

void Renderer::Render(World& world, Image3& output, int nrays, int OutputFreq) {
	world.CreateHierarchy();
	Camera const& cam = world.GetCam();
	int const Width = output.GetWidth(), Height = output.GetHeight();
	std::vector<Ray> rays;
	for (auto i = 0; i < Width; ++i) {
		for (auto j = 0; j < Height; ++j) {
			cam.GenRay(i, j, Width, Height, rays, nrays);
			auto col = glm::vec3(0.0f);
			for (Ray const& ray : rays) {
				col += world.RayTracing(ray);
			}
			col = col / (float)nrays;
			col = sqrt(col);
			output.Setcol(i, j, glm::clamp(col, glm::vec3(0), glm::vec3(1)), true);
			rays.clear();
		}
		if (i % OutputFreq == 0) {
			std::cout << "Complete Column " << i << std::endl;
		}
	}
}
