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
	int nrays = 1024;
#ifdef GENPIC
	auto fg = new Framegenerator;
	Image3 img(4096, 2304);
//	World world(std::shared_ptr<Camera>(new RegularCamera(la::vec3(0.5f, 0.4f, 0), la::vec3(0, -1, 0), la::vec3(1, 0, 0), 4.0f, 2.25f)), la::vec3(1.0f));
	World world(std::shared_ptr<Camera>(new RegularCamera(la::vec3(0.1f, 0.15f, 0), la::vec3(4, -3.5, 0), la::vec3(3.5, 4, 0), 4.0f, 2.25f)), la::vec3(1.0f));
	auto cyl = std::shared_ptr<Geometry>(new Cylindsurf(la::vec3(0.5f, 0, 0), la::vec3(0, 1.0f, 0), 0.1f, 0.2f));
	auto matid = Materialrepository::AddMat(std::shared_ptr<Material>(new Dieletric(1.2f)));
	cyl->AddMaterial(matid);
	world.AddGeo(cyl);
	auto cylw = std::shared_ptr<Geometry>(new Cylindsurf(la::vec3(0.5f, -0.025f, 0), la::vec3(0, 1.0f, 0), 0.095f, 0.15f));
	auto matidw = Materialrepository::AddMat(std::shared_ptr<Material>(new Dieletric(1.0f/1.2f)));
	cylw->AddMaterial(matidw);
	world.AddGeo(cylw);
	auto cyla = std::shared_ptr<Geometry>(new Cylindsurf(la::vec3(0.5f, 0.075f, 0), la::vec3(0, 1, 0), 0.095f, 0.05f));
	auto matida = Materialrepository::AddMat(std::shared_ptr<Material>(new Dieletric(1.0f/1.2f)));
	cyla->AddMaterial(matida);
	world.AddGeo(cyla);

	auto stk = std::shared_ptr<Geometry>(new Cylindsurf(la::vec3(0.5f, 0.038636f, 0.058138f), la::vec3(0, 2.0f, 1.0f), 0.01f, 0.3f));
	auto matids = Materialrepository::AddMat(std::shared_ptr<Material>(new Metal(la::vec3(0.8f, 0.2f, 0.1f), 0.1f)));
	stk->AddMaterial(matids);
	world.AddGeo(stk);
	auto stkup = std::shared_ptr<Geometry>(new Plate(la::vec3(0.5f, 0.1728f, 0.12522f), la::vec3(0, 2.0f, 1.0f), 0.01f));
	stkup->AddMaterial(matids);
	world.AddGeo(stkup);
//	auto water = std::shared_ptr<Geometry>(new Plate(la::vec3(0.5f, 0.05f, 0), la::vec3(0, 1, 0), 0.095f));
//	auto watermat = Materialrepository::AddMat(std::shared_ptr<Material>(new Dieletric(1.33f)));
//	water->AddMaterial(watermat);
//	world.AddGeo(water);
	auto plscap = std::shared_ptr<Geometry>(new Plate(la::vec3(0.5f, 0.1f, 0), la::vec3(0, 1, 0), 0.1f, 0.095f));
	plscap->AddMaterial(matid);
	world.AddGeo(plscap);
//	auto bll = std::shared_ptr<Geometry>(new Ball(la::vec3(0, 0.1f, 0), 0.05f));
//	bll->AddMaterial(pltmat);
//	world.AddGeo(bll);
	auto cap1 = std::shared_ptr<Geometry>(new Triangle(la::vec3(-5.0f, -0.1f, -5.0f), la::vec3(-5.0f, -0.1f, 5.0f), la::vec3(5.0f, -0.1f, -5.0f)));
	auto cap2 = std::shared_ptr<Geometry>(new Triangle(la::vec3(5.0f, -0.1f, -5.0f), la::vec3(-5.0f, -0.1f, 5.0f), la::vec3(5.0f, -0.1f, 5.0f)));
	auto capmat = Materialrepository::AddMat(std::shared_ptr<Material>(new Diffuse(la::vec3(0.1f, 0.2f, 0.8f))));
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