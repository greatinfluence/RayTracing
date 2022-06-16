#include <iostream>
#include <fstream>

#include "Yamlext.h"
#include "Renderer.h"
#include "Settings.h"

int main() {
	int nrays = 1024;
	YAML::Node config = YAML::LoadFile("102ballsconfig.yaml");
	Image3 img(config["img"].as<Image3>());
	World world(config["world"].as<World>());
	/*
	for (auto i = 0; i < 10; ++i)
		for (auto j = 0; j < 10; ++ j) {
		auto ball = std::shared_ptr<Geometry>(new Ball(glm::vec3(1.4f, i / 5.0f - 0.25, j / 5.0f - 1.0f), 0.1f));
		if ((i + j) % 3 == 0) {
			ball->AddMaterial(
				std::shared_ptr<Material>
				(new Diffuse(glm::vec3(Random::Rand(1.0f), Random::Rand(1.0f), Random::Rand(1.0f)))));
		}
		else if ((i + j) % 3 == 0) {
			ball->AddMaterial(
				std::shared_ptr<Material>
				(new Metal(glm::vec3(Random::Rand(1.0f), Random::Rand(1.0f), Random::Rand(1.0f)), Random::Rand(0.7f))));
		}
		else {
			ball->AddMaterial(
				std::shared_ptr<Material>
				(new Dieletric(Random::Rand(0.67f, 1.5f))));
		}
		world.AddGeo(ball);
	}
	config["world"] = world;
	std::ofstream ofs("102ballsconfig.yaml");
	ofs << config;
	return 0;*/
	using namespace std::chrono;
	Renderer renderer;
	auto start = high_resolution_clock::now();
	renderer.Render(world, img, nrays, 2500);
	img.Write("testpic2.png");
	auto stop = high_resolution_clock::now();
	std::cout << "Rendering finished. Time using: ";

	std::cout << formatDuration<decltype(stop - start), hours, minutes, seconds, milliseconds>(stop - start) << std::endl;
	return 0;
}