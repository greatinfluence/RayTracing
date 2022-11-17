#include "Object.h"
#include "Triangle.h"

void Cube::AppendtoWorld(World& world) {
	la::vec3 pos[8];
	for (int i = 0; i < 8; ++i) {
		pos[i] = center + transform * la::vec3((i & 1) - 0.5f, (i >> 1 & 1) - 0.5f, (i >> 2 & 1) - 0.5f);
	}
	uint32_t uniid = 0;
	if (uniform) uniid = Materialrepository::AddMat(left);
	if (left != nullptr) {
		// has left side
		auto matid = uniform ? uniid : Materialrepository::AddMat(left);
		auto ltri = std::shared_ptr<Geometry>(new Triangle(pos[1 | 2], pos[1], pos[2]));
		auto rtri = std::shared_ptr<Geometry>(new Triangle(pos[2], pos[1], pos[0]));
		ltri->AddMaterial(matid);
		rtri->AddMaterial(matid);
		world.AddGeo(ltri);
		world.AddGeo(rtri);
	}
	if (right != nullptr) {
		// has right side
		auto matid = uniform ? uniid : Materialrepository::AddMat(right);
		auto ltri = std::shared_ptr<Geometry>(new Triangle(pos[2 | 4], pos[4], pos[1 | 2 | 4]));
		auto rtri = std::shared_ptr<Geometry>(new Triangle(pos[1 | 2 | 4], pos[4], pos[1 | 4]));
		ltri->AddMaterial(matid);
		rtri->AddMaterial(matid);
		world.AddGeo(ltri);
		world.AddGeo(rtri);
	}
	if (ceil != nullptr) {
		// has ceiling side
		auto matid = uniform ? uniid : Materialrepository::AddMat(ceil);
		auto ltri = std::shared_ptr<Geometry>(new Triangle(pos[1 | 2], pos[2], pos[1 | 2 | 4]));
		auto rtri = std::shared_ptr<Geometry>(new Triangle(pos[1 | 2 | 4], pos[2], pos[2 | 4]));
		ltri->AddMaterial(matid);
		rtri->AddMaterial(matid);
		world.AddGeo(ltri);
		world.AddGeo(rtri);
	}
	if (bot != nullptr) {
		// has bottom side
		auto matid = uniform ? uniid : Materialrepository::AddMat(bot);
		auto ltri = std::shared_ptr<Geometry>(new Triangle(pos[1], pos[1 | 4], pos[0]));
		auto rtri = std::shared_ptr<Geometry>(new Triangle(pos[1 | 4], pos[4], pos[0]));
		ltri->AddMaterial(matid);
		rtri->AddMaterial(matid);
		world.AddGeo(ltri);
		world.AddGeo(rtri);
	}
	if (front != nullptr) {
		// has front side
		auto matid = uniform ? uniid : Materialrepository::AddMat(front);
		auto ltri = std::shared_ptr<Geometry>(new Triangle(pos[1 | 2], pos[1 | 2 | 4], pos[1]));
		auto rtri = std::shared_ptr<Geometry>(new Triangle(pos[1 | 2 | 4], pos[1 | 4], pos[1]));
		ltri->AddMaterial(matid);
		rtri->AddMaterial(matid);
		world.AddGeo(ltri);
		world.AddGeo(rtri);
	}
	if (back != nullptr) {
		// has back side
		auto matid = uniform ? uniid : Materialrepository::AddMat(back);
		auto ltri = std::shared_ptr<Geometry>(new Triangle(pos[2], pos[0], pos[2 | 4]));
		auto rtri = std::shared_ptr<Geometry>(new Triangle(pos[2 | 4], pos[0], pos[4]));
		ltri->AddMaterial(matid);
		rtri->AddMaterial(matid);
		world.AddGeo(ltri);
		world.AddGeo(rtri);
	}
}

void Cube::SetAll(std::shared_ptr<Material> mat) {
	front = mat;
	back = mat;
	left = mat;
	right = mat;
	ceil = mat;
	bot = mat;
	uniform = true;
}
