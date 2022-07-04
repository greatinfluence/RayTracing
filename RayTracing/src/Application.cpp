#include <iostream>
#include <fstream>

#include "Fileoperator.h"
#include "Renderer.h"
#include "Settings.h"
#include "Materialrepository.h"
#include "Random.h"

int main() {
	int nrays = 8;
	/*
	Image3 img(512, 256);
	World world(Camera(glm::vec3(0), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0)), glm::vec3(1.0f));

	for (auto i = 0; i < 1000; ++i) {
		if (i % 3 == 0) {
			Materialrepository::AddMat(std::shared_ptr<Material>(new Diffuse(glm::vec3(
				Random::Rand(1.0f),
				Random::Rand(1.0f),
				Random::Rand(1.0f)))));
		}
		else if(i % 3 == 1) {
			Materialrepository::AddMat(std::shared_ptr<Material>(new Metal(glm::vec3(
				Random::Rand(1.0f),
				Random::Rand(1.0f),
				Random::Rand(1.0f)), Random::Rand(0.3f, 1.0f))));
		}
		else {
			Materialrepository::AddMat(std::shared_ptr<Material>(new Dieletric(Random::Rand(1.0f))));
		}
	}

	for (auto i = 0; i < 10; ++i) {
		for (auto j = 0; j < 10; ++j) {
			for (auto k = 0; k < 10; ++k) {
				// The ball (i, j, k) is at position ((k + 5) / 10, (i + 1) / 10, (j - 4.5) / 10)
				//     with radius 0.05
				auto ball = std::shared_ptr<Geometry>(new Ball(
					glm::vec3((k + 5) / 10.0f, (i + 1) / 10.0f, (j - 4.5f) / 10.0f), 0.05f));
				ball->AddMaterial(i * 100 + j * 10 + k);
				world.AddGeo(ball);
			}
		}
	}
	float floorheight = -0.05f;
	float near = -0.1f;
	float far = 5;
	float left = -5;
	float right = 5;
	auto lfloor = std::shared_ptr<Geometry>(new Triangle(
		glm::vec3(near, floorheight, left),
		glm::vec3(near, floorheight, right),
		glm::vec3(far, floorheight, left)));
	lfloor->AddMaterial(0);
	auto rfloor = std::shared_ptr<Geometry>(new Triangle(
		glm::vec3(near, floorheight, right),
		glm::vec3(far, floorheight, right),
		glm::vec3(far, floorheight, left)));
	rfloor->AddMaterial(1);
	world.AddGeo(lfloor);
	world.AddGeo(rfloor);

	YAML::Savescene("1000ballsconfig.yaml", world, img);
	return 0;
	*/
	Image3 img;
	World world;
	YAML::Loadscene("1000ballsconfig.yaml", world, img);
	
	using namespace std::chrono;
	//Renderer renderer;
	auto start = high_resolution_clock::now();
	GPURenderer::Render(world, img, nrays);
	return 0;
	img.Write("testpic2.png");
	auto stop = high_resolution_clock::now();
	std::cout << "Rendering finished. Time using: ";

	std::cout << formatDuration<decltype(stop - start), hours, minutes, seconds, milliseconds>(stop - start) << std::endl;
	return 0;
}