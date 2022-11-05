#include <iostream>
#include <fstream>

#include "Framegenerator.h"
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
	auto fg = new Framegenerator;
	fg->GenerateFrame(0, pi / 12);
	return 0;
#else
	Image3 img;
	World world;
	YAML::Loadscene("frame0.yaml", world, img);
	
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