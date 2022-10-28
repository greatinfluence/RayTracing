#include <iostream>
#include <fstream>

#include "Fileoperator.h"
#include "Renderer.h"
#include "Settings.h"
#include "Materialrepository.h"
#include "Random.h"
#include "Object.h"
//#define GENPIC

int main() {
	int nrays = 128;
#ifdef GENPIC
	Image3 img(2048, 1024);
	World world(std::shared_ptr<Camera>(new RegularCamera(la::vec3(0.195f, 0.5f, 0), la::vec3(1.0f, 0, 0), la::vec3(0, 1.0f, 0))), la::vec3(1.0f));
	auto ball = std::shared_ptr<Geometry>(new Ball(la::vec3(0.07f, 0.5f, 2.0f), 0.07f));
	auto mat = std::shared_ptr<Material>(new Diffuse(la::vec3(0.4f, 0.1f, 0.05f), la::vec3(0.8f, 0.2f, 0.1f)));
	ball->AddMaterial(Materialrepository::AddMat(mat));
	world.AddGeo(ball);
	Cube wall;
	wall.SetAll(std::shared_ptr<Material>(new Diffuse(la::vec3(0.1f, 0.2f, 0.8f))));
	wall.center = la::vec3(1.5f, 0.5f, 0);
	wall.transform = la::mat3(0.4f, 1.0f, 10.0f);
	wall.AppendtoWorld(world);
	Cube floor;
	floor.ceil = std::shared_ptr<Material>(new Diffuse(la::vec3(0.1f, 0.8f, 0.2f)));
	floor.transform = la::mat3(20.0f, 1.0f, 15.0f);
	floor.center = la::vec3(0, -0.5f, 0);
	floor.AppendtoWorld(world);
	Cube water;
	water.SetAll(std::shared_ptr<Material>(new Dieletric(1.3333f)));
	water.center = la::vec3(0.1f, 0.5f, 0);
	water.transform = la::mat3(0.2f, 1.0f, 10.0f);
	water.bot = nullptr;
	water.AppendtoWorld(world);

	YAML::Savescene("sscene.yaml", world, img);
	return 0;
#else
	Image3 img;
	World world;
	YAML::Loadscene("sscene.yaml", world, img);
	
	using namespace std::chrono;
//	Renderer renderer;
	auto start = high_resolution_clock::now();
//	renderer.Render(world, img, nrays);
	GPURenderer::Render(world, img, nrays);
	img.Write("testpic2.png");
	auto stop = high_resolution_clock::now();
	std::cout << "Rendering finished. Time using: ";

	std::cout << formatDuration<decltype(stop - start), hours, minutes, seconds, milliseconds>(stop - start) << std::endl;
	return 0;
#endif
}