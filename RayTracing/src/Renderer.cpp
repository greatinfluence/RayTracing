#include "Renderer.h"

#include <iostream>
#include <thread>

#include "Geometryrepository.h"

void Renderer::Render(World& world, Image3& output, int nrays, int OutputFreq) {
	world.CreateHierarchy();
	Geometryrepository::Initiate(world);
	
	std::vector<std::thread> threads;
	std::mutex OutputLock;

	auto const processor_count = std::thread::hardware_concurrency();
	int const Totpixel = output.GetWidth() * output.GetHeight();
	for (uint32_t i = 0; i < processor_count; ++i) {
		threads.push_back(std::thread(&DoRender, std::ref(world), std::ref(output), Totpixel * i/ processor_count,
			Totpixel * (i + 1) / processor_count, nrays, OutputFreq, i, std::ref(OutputLock)));
	}

	for (auto& th : threads) {
		th.join();
	}
}

void Renderer::DoRender(World& world, Image3& output, uint32_t from, uint32_t to, int nrays, int OutputFreq, int nthread, std::mutex& lock) {
	Camera const& cam = world.GetCam();
	int const Width = output.GetWidth(), Height = output.GetHeight();
	std::vector<Ray> rays;
	std::vector<glm::vec3> cols;
	for (auto ind = from; ind < to; ++ind) {
		auto i = ind / Height, j = ind % Height;
		cam.GenRay(i, j, Width, Height, rays, nrays);
		auto col = glm::vec3(0.0f);
		for (Ray const& ray : rays) {
			col += world.RayTracing(ray);
		}
		col = col / (float)nrays;
		col = sqrt(col); // Gamma correction
		cols.push_back(col);
		rays.clear();
		if ((ind - from) % OutputFreq == 0) {
			std::cout << std::format("Thread {}: Complete {} %\n", nthread, ((float)ind - from) * 100 / (to - from));
		}
	}
	std::cout << std::format("Thread {}: Finish color collecting. Now writing to the output:\n", nthread);
	{
		std::lock_guard<std::mutex> lk(lock);
		for (auto ind = from; ind < to; ++ind) {
			output.Setcol(ind / Height, ind % Height, glm::clamp(cols[ind - from], glm::vec3(0), glm::vec3(1)), true);
		}
	}
	std::cout << std::format("Thread {}: Rendering task completed.\n", nthread);
}
