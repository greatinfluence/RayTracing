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
	int nrays = 1024;
#ifdef GENPIC
	Image3 img(2048, 1024);
	World world(Camera(la::vec3(-0.5f,1.3f,0), la::vec3(0.9f, -0.43589f, 0), la::vec3(0.43589f, 0.9f, 0)), la::vec3(1.0f));
	auto ball = std::shared_ptr<Geometry>(new Ball(la::vec3(1.0f, 0.05f, 0), 0.05f));
	auto mat = std::shared_ptr<Material>(new Metal(la::vec3(0.8f, 0.2f, 0.1f), 0.2f));
	ball->AddMaterial(Materialrepository::AddMat(mat));
	world.AddGeo(ball);
	Cube floor;
	floor.ceil = std::shared_ptr<Material>(new Diffuse(la::vec3(0.1f, 0.2f, 0.8f)));
	floor.transform = la::mat3(20.0f, 1.0f, 20.0f);
	floor.center = la::vec3(0, -0.5f, 0);
	floor.AppendtoWorld(world);
	Cube water;
	water.SetAll(std::shared_ptr<Material>(new Dieletric(1.3333f)));
	water.center = la::vec3(1.0f, 0.54f, 0);
	water.AppendtoWorld(world);
	Cube botglass;
	botglass.SetAll(std::shared_ptr<Material>(new Dieletric(1.52f)));
	botglass.center = la::vec3(1.0f, 0.02f, 0);
	botglass.transform = la::mat3(1.0f, 0.0395f, 1.0f);
	botglass.AppendtoWorld(world);
	Cube frontglass;
	frontglass.SetAll(std::shared_ptr<Material>(new Dieletric(1.52f)));
	frontglass.center = la::vec3(0.48f, 0.551f, 0);
	frontglass.transform = la::mat3(0.0395f, 1.1f, 1.0f);
	frontglass.AppendtoWorld(world);
	Cube backglass;
	backglass.center = la::vec3(1.52f, 0.551f, 0);
	backglass.transform = la::mat3(0.0395f, 1.1f, 1.0f);
	backglass.AppendtoWorld(world);
	Cube leftglass;
	leftglass.SetAll(std::shared_ptr<Material>(new Dieletric(1.52f)));
	leftglass.center = la::vec3(1.0f, 0.551f, -0.52f);
	leftglass.transform = la::mat3(1.08f, 1.1f, 0.0395f);
	leftglass.AppendtoWorld(world);
	Cube rightglass;
	rightglass.SetAll(std::shared_ptr<Material>(new Dieletric(1.52f)));
	rightglass.center = la::vec3(1.0f, 0.551f, 0.52f);
	rightglass.transform = la::mat3(1.08f, 1.1f, 0.0395f);
	rightglass.AppendtoWorld(world);

	YAML::Savescene("sampleconfig.yaml", world, img);
	return 0;
#else
	Image3 img;
	World world;
	YAML::Loadscene("sampleconfig.yaml", world, img);
	
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