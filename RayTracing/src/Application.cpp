#include <iostream>

#include "Image.h"
#include "World.h"

#include "Geometry.h"

int main() {
	int const Width = 512, Height = 256;
	Image3 img(Width, Height);
	Camera cam(glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f), 1.5f, 1.0472f);
	World world(cam, glm::vec3(1.0f));
	std::shared_ptr<Geometry> ball1(new Ball(glm::vec3(1.0f, 0.26f, 0.0f), 0.25f));
	ball1->AddMaterial(std::shared_ptr<Material>(new Diffuse(glm::vec3(0.2f, 0.8f, 0.1f))));
	world.AddGeo(ball1);
	std::shared_ptr<Geometry> ball2(new Ball(glm::vec3(2.0f, -0.26f, 0.0f), 0.25f));
	ball2->AddMaterial(std::shared_ptr<Material>(new Diffuse(glm::vec3(0.8f, 0.2f, 0.1f))));
	world.AddGeo(ball2);
//	std::shared_ptr<Geometry> fl1(new Triangle(glm::vec3(-1.0f, -1.0f, -1.0f), glm::vec3(1.0f, 1.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f)));
//	fl1->AddMaterial(std::shared_ptr<Material>(new Diffuse(glm::vec3(0.8f, 0.1f, 0.2f))));
//	world.AddGeo(fl1);
	std::vector<Ray> rays;
	int nrays = 32;
	for (int i = 0; i < Width; ++i) {
		for (int j = 0; j < Height; ++j) {
			cam.GenRay(i, j, Width, Height, rays, nrays);
			auto col = glm::vec3(0.0f);
			for (auto ray : rays) {
				col += world.RayTracing(ray);
			}
			img.Setcol(i, j, col / (float)nrays, true);
			if (rays[0].GetDir().y > 0) {
			//std::cout << "In" << i << ',' << j << std::endl;
			//std::cout << rays[0].GetDir().x << ' ' << rays[0].GetDir().y << ' ' << rays[0].GetDir().z << std::endl;
			//std::cout << "Out " << i << ',' << j << std::endl;

			}
			rays.clear();
		}
		std::cout << "Complete Column " << i << std::endl;
	}
	img.Write("testpic.png");
	return 0;
}