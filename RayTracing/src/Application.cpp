#include <iostream>

#include "Image.h"
#include "World.h"

#include "Geometry.h"
#include "glm/gtx/norm.hpp"

int main() {
	int const Width = 512, Height = 256;
	Image3 img(Width, Height);
	Camera cam(glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f), 1.5f, 1.0472f);
	World world(cam, glm::vec3(1.0f));
	std::shared_ptr<Geometry> ball1(new Ball(glm::vec3(1.0f, -0.14f, -0.36f), 0.35f));
	ball1->AddMaterial(std::shared_ptr<Material>(new Diffuse(glm::vec3(0.2f, 0.8f, 0.1f))));
	world.AddGeo(ball1);
	std::shared_ptr<Geometry> ball2(new Ball(glm::vec3(1.0f, -0.14f, 0.36f), 0.35f));
	ball2->AddMaterial(std::shared_ptr<Material>(new Diffuse(glm::vec3(0.8f, 0.2f, 0.1f))));
	world.AddGeo(ball2);
	float fh = -0.5f;
	std::shared_ptr<Geometry> fl1(new Triangle(glm::vec3(-3.0f, fh, -3.0f), glm::vec3(3.0f, fh, 3.0f), glm::vec3(3.0f, fh, -3.0f)));
	fl1->AddMaterial(std::shared_ptr<Material>(new Diffuse(glm::vec3(0.1f, 0.2f, 0.8f))));
	world.AddGeo(fl1);
	std::shared_ptr<Geometry> fl2(new Triangle(glm::vec3(-3.0f, fh, -3.0f), glm::vec3(-3.0f, fh, 3.0f), glm::vec3(3.0f, fh, 3.0f)));
	fl2->AddMaterial(std::shared_ptr<Material>(new Diffuse(glm::vec3(0.1f, 0.2f, 0.8f))));
	world.AddGeo(fl2);
	std::vector<Ray> rays;
	int nrays = 256;
	for (int i = 0; i < Width; ++i) {
		for (int j = 0; j < Height; ++j) {
			cam.GenRay(i, j, Width, Height, rays, nrays);
			auto col = glm::vec3(0.0f);
			//std::cout << i << ' ' << j << std::endl;
			for (auto ray : rays) {
				assert(glm::l2Norm(ray.GetPos()) < 1e-4);
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
		if(i % 10 == 0) std::cout << "Complete Column " << i << std::endl;
	}
	img.Write("testpic.png");
	return 0;
}