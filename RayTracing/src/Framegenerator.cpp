#include "Framegenerator.h"
#include "Fileoperator.h"
#include "Object.h"
#include <format>

void Framegenerator::GenerateFrame(int frame, float angle) {
	Image3 img(4096, 2304);

	float waterheight = 0.9f;
	la::vec3 down = la::vec3(-sin(angle), -cos(angle), 0);
	la::vec3 mid = la::vec3(0, waterheight, 0);
	float hflen = 0.4f;
	float hfthk = 0.025f;
	la::vec3 perp = la::vec3(cos(angle), -sin(angle), 0);

	auto sight = la::vec3(sin(0.06 * pi), - cos(0.06 * pi), 0);
	auto pos = -sight + mid;
	auto sightu = la::vec3(cos(0.06 * pi), sin(0.06 * pi), 0);
	World world(std::shared_ptr<Camera>(new RegularCamera(pos, sight, sightu, 4.0f, 2.25f)), la::vec3(1.0f));

	Cube floor;
	floor.ceil = std::shared_ptr<Material>(new Diffuse(la::vec3(0.1f, 0.8f, 0.2f)));
	floor.transform = la::mat3(20.0f, 1.0f, 15.0f);
	floor.center = la::vec3(0, -0.5f, 0);
	floor.AppendtoWorld(world);
	Cube box;
	box.SetAll(std::shared_ptr<Material>(new Diffuse(la::vec3(0.2f, 0.1f, 0.8f))));
	box.ceil = nullptr;
	box.bot = nullptr;
	box.center = la::vec3(0, 0.5f, 0);
	box.AppendtoWorld(world);

	auto water = std::shared_ptr<Material>(new Dieletric(1.33f));
	auto plasticinwater = std::shared_ptr<Material>(new Dieletric(1.2f / 1.33f));
	//auto plasticinwater = std::shared_ptr<Material>(new Diffuse(la::vec3(0.8f, 0.2f, 0.1f)));
	auto plasticinair = std::shared_ptr<Material>(new Dieletric(1.2f));
	//auto plasticinair = std::shared_ptr<Material>(new Diffuse(la::vec3(0.8f, 0.2f, 0.1f)));
	auto redwall = std::shared_ptr<Material>(new Metal(la::vec3(0.8f, 0.2f, 0.1f), 0.7f));
	//auto redwall = std::shared_ptr<Material>(new Diffuse(la::vec3(0.8f, 0.2f, 0.1f)));
	//auto blackout = std::shared_ptr<Material>(new Diffuse(la::vec3(0)));

	la::vec3 nearwater = la::vec3(-hfthk / cos(angle), waterheight, 0);
	la::vec3 farwater = la::vec3(hfthk / cos(angle), waterheight, 0);
	la::vec3 nearinside = down * hflen - perp * hfthk + mid;
	la::vec3 farinside = down * hflen + perp * hfthk + mid;
	auto nearup = -down * hflen - perp * hfthk + mid;
	auto farup = -down * hflen + perp * hfthk + mid;
	la::vec3 left = la::vec3(0, 0, -0.4999f);
	la::vec3 right = la::vec3(0, 0, 0.4999f);
	auto nearend = la::vec3(-0.5f, waterheight, 0);
	auto farend = la::vec3(0.5f, waterheight, 0);

	std::cout << nearup.x << ' ' << nearup.y << ' ' << nearup.z << std::endl;
	std::cout << farup.x << ' ' << farup.y << ' ' << farup.z << std::endl;
	std::cout << la::l2Norm(farup - nearup) << std::endl;

	auto piwid = Materialrepository::AddMat(plasticinwater);
	auto rwid = Materialrepository::AddMat(redwall);
	auto piaid = Materialrepository::AddMat(plasticinair);
//	auto blkid = Materialrepository::AddMat(blackout);
	auto wid = Materialrepository::AddMat(water);

	auto lfrontw = std::shared_ptr<Geometry>(new Triangle(
		nearwater + left,
		nearinside + left,
		nearwater + right
	));
	auto rfrontw = std::shared_ptr<Geometry>(new Triangle(
		nearinside + left,
		nearinside + right,
		nearwater + right
	));
	auto lbot = std::shared_ptr<Geometry>(new Triangle(
		nearinside + left,
		farinside + left,
		farinside + right
	));
	auto rbot = std::shared_ptr<Geometry>(new Triangle(
		nearinside + left,
		farinside + right,
		nearinside + right
	));
	auto lback = std::shared_ptr<Geometry>(new Triangle(
		farup + left,
		farinside + left,
		farinside + right
	));
	auto rback = std::shared_ptr<Geometry>(new Triangle(
		farup + left,
		farinside + right,
		farup + right
	));
	auto lfronta = std::shared_ptr<Geometry>(new Triangle(
		nearup + left,
		nearwater + left,
		nearwater + right
	));
	auto rfronta = std::shared_ptr<Geometry>(new Triangle(
		nearup + left,
		nearwater + right,
		nearup + right
	));
	auto lleft = std::shared_ptr<Geometry>(new Triangle(
		farup + left,
		farwater + left,
		nearup + left
	));
	auto rleft = std::shared_ptr<Geometry>(new Triangle(
		nearup + left,
		farwater + left,
		nearwater + left
	));
	auto blkdot = std::shared_ptr<Geometry>(new Ball(
		farup + left,
		0.02f
	));
	auto lright = std::shared_ptr<Geometry>(new Triangle(
		nearup + right,
		nearwater + right,
		farwater + right
	));
	auto rright = std::shared_ptr<Geometry>(new Triangle(
		nearup + right,
		farwater + right,
		farup + right
	));
	auto lceil = std::shared_ptr<Geometry>(new Triangle(
		farup + left,
		nearup + left,
		nearup + right
	));
	auto rceil = std::shared_ptr<Geometry>(new Triangle(
		farup + left,
		nearup + right,
		farup + right
	));
	auto lfwater = std::shared_ptr<Geometry>(new Triangle(
		nearwater + left,
		nearend + left,
		nearend + right
	));
	auto rfwater = std::shared_ptr<Geometry>(new Triangle(
		nearwater + left,
		nearend + right,
		nearwater + right
	));
	auto lbwater = std::shared_ptr<Geometry>(new Triangle(
		farend + left,
		farwater + left,
		farwater + right
	));
	auto rbwater = std::shared_ptr<Geometry>(new Triangle(
		farend + left,
		farwater + right,
		farend + right
	));

	lfrontw->AddMaterial(piwid);
	rfrontw->AddMaterial(piwid);
	lbot->AddMaterial(piwid);
	rbot->AddMaterial(piwid);
	lback->AddMaterial(rwid);
	rback->AddMaterial(rwid);
	world.AddGeo(lback);
	world.AddGeo(rback);
//	blkdot->AddMaterial(blkid);
	//world.AddGeo(blkdot);
	lfronta->AddMaterial(piaid);
	rfronta->AddMaterial(piaid);
	//lfronta->AddMaterial(blkid);
	//rfronta->AddMaterial(blkid);
	//lleft->AddMaterial(blkid);
	//rleft->AddMaterial(blkid);
	lleft->AddMaterial(piaid);
	rleft->AddMaterial(piaid);
	lright->AddMaterial(piaid);
	rright->AddMaterial(piaid);
	lceil->AddMaterial(piaid);
	rceil->AddMaterial(piaid);
	//lceil->AddMaterial(blkid);
	//rceil->AddMaterial(blkid);
	lfwater->AddMaterial(wid);
	rfwater->AddMaterial(wid);
	lbwater->AddMaterial(wid);
	rbwater->AddMaterial(wid);

	world.AddGeo(lfrontw);
	world.AddGeo(rfrontw);
	world.AddGeo(lbot);
	world.AddGeo(rbot);
	world.AddGeo(lfronta);
	world.AddGeo(rfronta);
	world.AddGeo(lleft);
	world.AddGeo(rleft);
	world.AddGeo(lright);
	world.AddGeo(rright);
	world.AddGeo(lceil);
	world.AddGeo(rceil);
	world.AddGeo(lfwater);
	world.AddGeo(rfwater);
	world.AddGeo(lbwater);
	world.AddGeo(rbwater);

	YAML::Savescene(std::format("frame{}.yaml", frame), world, img);
}
