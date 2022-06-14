#include <iostream>
#include <fstream>

#include "Yamlext.h"
#include "Renderer.h"

#include "glm/gtx/norm.hpp"

int main() {
	int nrays = 800;
	YAML::Node config = YAML::LoadFile("twoballsconfig.yaml");
	Image3 img(config["img"].as<Image3>());
	World world(config["world"].as<World>());
	Renderer renderer;
	renderer.Render(world, img, 800);
	img.Write("testpic.png");
	return 0;
}