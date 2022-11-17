#include <iostream>
#include <fstream>

#include "Framegenerator.h"
#include "Fileoperator.h"
#include "Renderer.h"
#include "Settings.h"
#include "Materialrepository.h"
#include "Random.h"
#include "Object.h"
#include "Cylindsurf.h"
//#define GENPIC

int main() {
	int nrays = 128;
#ifdef GENPIC
	auto fg = new Framegenerator;
	Image3 img(4096, 2304);
	World world(std::shared_ptr<Camera>(new RegularCamera(la::vec3(-0.5f, 0, 0), la::vec3(1.0f, 0, 0), la::vec3(0, 1.0f, 0), 4.0f, 2.25f)), la::vec3(1.0f));
	auto cyl = std::shared_ptr<Geometry>(new Cylindsurf(la::vec3(0.5f, 0, 0), la::vec3(0, 1.0f, 0), 0.1f, 0.2f));
	auto matid = Materialrepository::AddMat(std::shared_ptr<Material>(new Dieletric(1.33f)));
	cyl->AddMaterial(matid);
	world.AddGeo(cyl);
	auto cyli = std::shared_ptr<Geometry>(new Cylindsurf(la::vec3(0.5f, 0, 0), la::vec3(0, 1.0f, 0), 0.095f, 0.195f));
	auto matidi = Materialrepository::AddMat(std::shared_ptr<Material>(new Dieletric(1.0f/1.33f)));
	cyli->AddMaterial(matidi);
	world.AddGeo(cyli);
	auto cap1 = std::shared_ptr<Geometry>(new Triangle(la::vec3(-5.0f, -0.1f, -5.0f), la::vec3(-5.0f, -0.1f, 5.0f), la::vec3(5.0f, -0.1f, -5.0f)));
	auto cap2 = std::shared_ptr<Geometry>(new Triangle(la::vec3(5.0f, -0.1f, -5.0f), la::vec3(-5.0f, -0.1f, 5.0f), la::vec3(5.0f, -0.1f, 5.0f)));
	auto capmat = Materialrepository::AddMat(std::shared_ptr<Material>(new Diffuse(la::vec3(0.8f, 0.2f, 0.1f))));
	auto plt = std::shared_ptr<Geometry>(new Plate(la::vec3(0.5f, 0.1f, 0), la::vec3(0, 1, 0), 0.1f, 0.095f));
	auto pltmat = Materialrepository::AddMat(std::shared_ptr<Material>(new Metal(la::vec3(0.1f, 0.2f, 0.8f), 0.2f)));
	plt->AddMaterial(pltmat);
	world.AddGeo(plt);
//	auto bll = std::shared_ptr<Geometry>(new Ball(la::vec3(0, 0.1f, 0), 0.05f));
//	bll->AddMaterial(pltmat);
//	world.AddGeo(bll);
	cap1->AddMaterial(capmat);
	cap2->AddMaterial(capmat);
	world.AddGeo(cap1);
	world.AddGeo(cap2);
	YAML::Savescene("TestCylinder.yaml", world, img);
	//fg->GenerateFrame(0, pi / 12);
	return 0;
#else
	Image3 img;
	World world;
	YAML::Loadscene("TestCylinder.yaml", world, img);
	printf("After loading the scene\n");
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