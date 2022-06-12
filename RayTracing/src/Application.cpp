#include <iostream>

#include "Image.h"
#include "World.h"

#include "Geometry.h"
#include "glm/gtx/norm.hpp"

int main() {
	int const Width = 256, Height = 128;
	int nrays = 800;
	Image3 img(Width, Height);
	Camera cam(glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f), 1.50f, 1.04f);
	World world(cam, glm::vec3(1.0f, 1.0f, 1.0f));
/*	std::shared_ptr<Geometry> bulb(new Ball(glm::vec3(1.5f, 1.0f, 0.0f), 0.2f));
	auto bmat = std::shared_ptr<Material>(new Diffuse(glm::vec3(0.2f, 0.8f, 0.1f)));
	bmat->SetGlow(glm::vec3(100.0f));
	bulb->AddMaterial(bmat);
	world.AddGeo(bulb);*/
	std::shared_ptr<Geometry> ball1(new Ball(glm::vec3(1.0f, 0.0f, 0.0f), 0.35f));
	auto mat = std::shared_ptr<Material>(new Dieletric(1.5));
	ball1->AddMaterial(mat);
	world.AddGeo(ball1);
//	std::shared_ptr<Geometry> ball2(new Ball(glm::vec3(1.0f, 0.0f, 0.36f), 0.35f));
	//ball2->AddMaterial(std::shared_ptr<Material>(new Metal(glm::vec3(0.8f, 0.1f, 0.2f), 0.1)));
	//world.AddGeo(ball2);
	float fh = -0.35f;
	float fr = -18.0f, bk = 18.0f;
	std::shared_ptr<Geometry> fl1(new Triangle(glm::vec3(fr, fh, -3.0f), glm::vec3(bk, fh, 3.0f), glm::vec3(bk, fh, -3.0f)));
	fl1->AddMaterial(std::shared_ptr<Material>(new Diffuse(glm::vec3(0.1f, 0.2f, 0.8f))));
	world.AddGeo(fl1);
	std::shared_ptr<Geometry> fl2(new Triangle(glm::vec3(fr, fh, -3.0f), glm::vec3(fr, fh, 3.0f), glm::vec3(bk, fh, 3.0f)));
	fl2->AddMaterial(std::shared_ptr<Material>(new Diffuse(glm::vec3(0.1f, 0.2f, 0.8f))));
	world.AddGeo(fl2);
/*	
	float ch = 1.2f;
	std::shared_ptr<Geometry> lwall1(new Triangle(glm::vec3(-3.0f, fh, -3.0f), glm::vec3(3.0f, fh, -3.0f), glm::vec3(-3.0f, ch, -3.0f)));
	lwall1->AddMaterial(std::shared_ptr<Material>(new Diffuse(glm::vec3(1.0f, 1.0f, 1.0f))));
	std::shared_ptr<Geometry> lwall2(new Triangle(glm::vec3(3.0f, fh, -3.0f), glm::vec3(3.0f, ch, -3.0f), glm::vec3(-3.0f, ch, -3.0f)));
	lwall2->AddMaterial(std::shared_ptr<Material>(new Diffuse(glm::vec3(1.0f, 1.0f, 1.0f))));
	world.AddGeo(lwall1);
	world.AddGeo(lwall2);
	std::shared_ptr<Geometry> rwall1(new Triangle(glm::vec3(-3.0f, fh, 3.0f), glm::vec3(3.0f, fh, 3.0f), glm::vec3(-3.0f, ch, 3.0f)));
	rwall1->AddMaterial(std::shared_ptr<Material>(new Diffuse(glm::vec3(1.0f, 1.0f, 1.0f))));
	std::shared_ptr<Geometry> rwall2(new Triangle(glm::vec3(3.0f, fh, 3.0f), glm::vec3(3.0f, ch, 3.0f), glm::vec3(-3.0f, ch, 3.0f)));
	rwall2->AddMaterial(std::shared_ptr<Material>(new Diffuse(glm::vec3(1.0f, 1.0f, 1.0f))));
	world.AddGeo(rwall1);
	world.AddGeo(rwall2);
	std::shared_ptr<Geometry> bwall1(new Triangle(glm::vec3(3.0f, fh, -3.0f), glm::vec3(3.0f, fh, 3.0f), glm::vec3(-3.0f, ch, -3.0f)));
	bwall1->AddMaterial(std::shared_ptr<Material>(new Diffuse(glm::vec3(1.0f, 1.0f, 1.0f))));
	std::shared_ptr<Geometry> bwall2(new Triangle(glm::vec3(3.0f, fh, 3.0f), glm::vec3(3.0f, ch, 3.0f), glm::vec3(-3.0f, ch, -3.0f)));
	bwall2->AddMaterial(std::shared_ptr<Material>(new Diffuse(glm::vec3(1.0f, 1.0f, 1.0f))));
	world.AddGeo(bwall1);
	world.AddGeo(bwall2);
	std::shared_ptr<Geometry> ceil1(new Triangle(glm::vec3(-3.0f, ch, -3.0f), glm::vec3(-3.0f, ch, 3.0f), glm::vec3(3.0f, ch, -3.0f)));
	ceil1->AddMaterial(std::shared_ptr<Material>(new Diffuse(glm::vec3(1.0f, 1.0f, 1.0f))));
	std::shared_ptr<Geometry> ceil2(new Triangle(glm::vec3(-3.0f, ch, 3.0f), glm::vec3(3.0f, ch, 3.0f), glm::vec3(3.0f, ch, -3.0f)));
	ceil2->AddMaterial(std::shared_ptr<Material>(new Diffuse(glm::vec3(1.0f, 1.0f, 1.0f))));
	world.AddGeo(ceil1);
	world.AddGeo(ceil2);*/
	std::vector<Ray> rays;
	for (int i = 0; i < Width; ++i) {
		for (int j = 0; j < Height; ++j) {
			cam.GenRay(i, j, Width, Height, rays, nrays);
			auto col = glm::vec3(0.0f);
			//std::cout << i << ' ' << j << std::endl;
			for (auto ray : rays) {
				assert(glm::l2Norm(ray.GetPos()) < 1e-4);
				col += world.RayTracing(ray);
			}
			col = col / (float)nrays;
			col = sqrt(col);
			img.Setcol(i, j, glm::clamp(col, glm::vec3(0), glm::vec3(1)), true);
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