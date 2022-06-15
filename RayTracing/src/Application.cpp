#include <iostream>
#include <fstream>

#include "Yamlext.h"
#include "Renderer.h"
#include "Random.h"

int main() {
	int nrays = 1;
	YAML::Node config = YAML::LoadFile("102ballsconfig.yaml");
	Image3 img(config["img"].as<Image3>());
	World world(config["world"].as<World>());
	/*
	for (auto i = 0; i < 10; ++i)
		for (auto j = 0; j < 10; ++ j) {
		auto ball = std::shared_ptr<Geometry>(new Ball(glm::vec3(1.4f, i / 5.0f - 0.35, j / 5.0f - 0.5f), 0.1f));
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
	Renderer renderer;
	renderer.Render(world, img, 800);
	img.Write("testpic2.png");
	return 0;
}