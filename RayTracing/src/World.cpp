#include "World.h"

#include <limits>
#include <iostream>

#include "Random.h"
#include "Testtool.hpp"

void World::AddGeo(std::shared_ptr<Geometry> geo)
{
	m_Geos.push_back(geo);
}

glm::vec3 World::RayTracing(Ray ray, int lev, glm::vec3 coef)
{
	float dist = std::numeric_limits<float>::max();
	std::shared_ptr<Geometry> hitted = nullptr;
	if (fabs(ray.GetDir().x) > 1.5f) {
		std::cout << "What?" << std::endl;
	}
	for (auto geo : m_Geos) {
		float ndis = ray.Hit(geo.get());
		if (ndis < dist) {
			dist = ndis;
			hitted = geo;
		}
	}
	if (hitted != nullptr) {
		glm::vec3 hitpos = ray.GetPos() + ray.GetDir() * dist;
		glm::vec3 att, wi, norm = hitted->GetNorm(hitpos);
		std::shared_ptr<Material> mat = hitted->GetMaterial();
		float poss = mat->scatter(hitpos, -ray.GetDir(),
			norm, att, wi);
		//std::cout << "Hittttttttttttttttttttttttt" << std::endl;
		//std::cout << coef * att * glm::dot(wi, norm) / poss << std::endl;
		if (fabs(dist) > 1e3 || fabs(ray.GetDir().x) > 1.5f) {
			// Impossible!
			std::cout << "What?" << std::endl;
		}
		/*if (hitted->GetType() == GeoType::Ball && lev == 1) {
			std::cout << "Ray Dir: " << ray.GetDir();
			std::cout << "Norm: " << norm;
			std::cout << "wi: " << wi;
			std::cout << "cosines: " << glm::dot(wi, norm) << ' ' << glm::dot(-ray.GetDir(), norm) << std::endl;
			std::cout << "Pos after hit: " << hitpos + wi * 1e-4f;
			std::cout << "dist1: " << glm::l2Norm(hitpos - glm::vec3(1, 0, 0)) << std::endl;
			std::cout << "dist2: " << glm::l2Norm(hitpos + wi * 1e-4f - glm::vec3(1, 0, 0)) << std::endl;
			std::cout << "Hitpos: " << hitpos;

		//	assert(glm::dot(wi, ray.GetDir()) < 1e-8);
		}*/
		if (poss > 2.0f || poss < -2.0f) {
			std::cout << "What?" << std::endl;
		}
		if(lev > 4) {
			float rr = glm::clamp(fmax(coef.r, fmax(coef.g, coef.b)), 0.0f, 0.95f);
			if (lev > 30 || Random::Rand(1.0f) > rr) return coef * mat->GetGlow();	
			coef = coef / rr;
		//	return coef / rr * mat->GetGlow() +
		//		RayTracing(Ray(hitpos + wi * 1e-4f, wi), lev + 1, coef * att * fabs(glm::dot(wi, norm)) / poss) / rr;
		}
		if (hitted->GetType() == GeoType::Ball) {
			//Debug here
		//	std::cout << "Hit the ball" << std::endl;
		}
		return coef * mat->GetGlow() +
			RayTracing(Ray(hitpos + wi * 1e-4f, wi), lev + 1, coef * att * fabs(glm::dot(wi, norm)) / poss);
	}
	else return coef * m_Background;
}
